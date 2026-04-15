"""Local STT via mlx-whisper — Apple Silicon / MLX, Metal-accelerated."""
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

import re
from typing import Callable, Optional

try:
    import mlx_whisper
except ImportError:
    mlx_whisper = None  # type: ignore[misc, assignment]

from services.learned_corrections import apply_learned_phrase_fixes
from services.stt_postprocess import (
    apply_asr_entity_corrections,
    build_initial_prompt,
)


def _mlx_hf_repo(model_size: str) -> str:
    raw = os.environ.get("MLX_WHISPER_REPO", "").strip()
    if raw:
        return raw
    return f"mlx-community/whisper-{model_size}-mlx"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float | None) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_float_optional(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_temperature_fallback() -> float | tuple[float, ...]:
    """
    mlx-whisper: tuple of temperatures tried in order when compression/logprob fail.
    Env STT_TEMPERATURE_FALLBACK: comma-separated, e.g. "0,0.2,0.4,0.6,0.8,1.0,1.2"
    """
    raw = os.environ.get("STT_TEMPERATURE_FALLBACK", "").strip()
    if not raw:
        return (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    temps: list[float] = []
    for p in parts:
        try:
            temps.append(float(p))
        except ValueError:
            continue
    if not temps:
        return (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
    return tuple(temps) if len(temps) > 1 else temps[0]


# Whisper false-positive language codes → almost certain real language.
# These appear regularly on English/Hindi/French YouTube content.
_WHISPER_LANG_REMAP: dict[str, str] = {
    "la": "en",   # Latin  — virtually never real Latin on YouTube
    "br": "fr",   # Breton — Whisper mis-labels French as Breton frequently
    "jw": "id",   # Javanese — confused with Indonesian
    "sa": "hi",   # Sanskrit — confused with Hindi
    "nn": "no",   # Nynorsk → Norwegian
    "oc": "fr",   # Occitan → French
}

# Languages that are commonly spoken on YouTube; used to catch exotic outliers.
_COMMON_YT_LANGS: frozenset[str] = frozenset({
    "en", "es", "fr", "de", "ar", "pt", "ja", "ko", "zh", "ru",
    "hi", "it", "nl", "tr", "pl", "sv", "id", "vi", "th", "uk",
    "cs", "el", "ro", "hu", "he", "fa", "ur", "bn", "tl", "ms",
    "no", "da", "fi", "sk", "hr", "bg", "sr", "lt", "lv", "sl",
    "ca", "af", "az", "be", "bs", "cy", "et", "gl", "is", "mk",
    "mt", "sq", "sw", "ta", "te", "kn", "ml", "mr", "gu", "pa",
    "si", "km", "my", "lo", "ka", "am", "hy", "ne",
})


def _count_arabic_script_chars(text: str) -> int:
    """Arabic / Nastaliq / presentation forms used for Urdu, Persian, etc."""
    n = 0
    for c in text:
        o = ord(c)
        if (
            0x0600 <= o <= 0x06FF
            or 0x0750 <= o <= 0x077F
            or 0x08A0 <= o <= 0x08FF
            or 0xFB50 <= o <= 0xFDFF
            or 0xFE70 <= o <= 0xFEFF
        ):
            n += 1
    return n


def _latin_letter_count(text: str) -> int:
    return sum(1 for c in text if ("a" <= c <= "z") or ("A" <= c <= "Z"))


def _maybe_roman_english_not_urdu(lang: str, raw_segments: list) -> str | None:
    """
    Whisper often tags **English** YouTube as ``ur`` (Urdu) when the audio is clear
    Roman speech. Real Urdu subtitles are usually Arabic-script; Roman-only + ``ur``
    is almost always a bad ASR language id.
    """
    if os.environ.get("STT_SCRIPT_LANG_FIX", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return None
    if lang not in ("ur", "ps", "sd") or not raw_segments:
        return None
    sample = " ".join((s.get("text") or "") for s in raw_segments[:30])
    lat = _latin_letter_count(sample)
    ar = _count_arabic_script_chars(sample)
    # Short clips: relax Latin count
    if lat >= 18 and ar <= 2:
        return "en"
    if lat >= 40 and ar <= max(3, lat // 80):
        return "en"
    return None


def _validate_detected_language(detected: str, raw_segments: list) -> str:
    """
    Cross-check Whisper's detected language against known false positives
    and against the actual script of the first few transcribed segments.

    Strategy
    --------
    1. Direct remap for well-known Whisper false-positive codes (e.g. la → en).
    1b. If Whisper says Urdu/Pashto/Sindhi but the transcript is almost all Roman
        letters and not Arabic-script, treat as English (common YouTube false id).
    2. If the code is not in the common YouTube language list, inspect the text
       character set: if ≥ 85 % of alpha characters are ASCII/Latin-script and
       the language is not a known Latin-script language, override to ``"en"``.

    Set ``STT_SCRIPT_LANG_FIX=0`` to disable step 1b.
    """
    lang = (detected or "en").lower().strip()

    # Step 1 — known false positives
    if lang in _WHISPER_LANG_REMAP:
        remapped = _WHISPER_LANG_REMAP[lang]
        print(
            f"[STT] Language {lang!r} is a known Whisper false positive "
            f"→ remapped to {remapped!r}"
        )
        return remapped

    # Step 1b — English mis-labeled as Urdu (etc.) on Roman transcripts
    fixed = _maybe_roman_english_not_urdu(lang, raw_segments)
    if fixed:
        print(
            f"[STT] Language {lang!r} inconsistent with Roman-only transcript "
            f"→ remapped to {fixed!r}"
        )
        return fixed

    # Step 2 — unusual code: verify from actual text
    if lang not in _COMMON_YT_LANGS and raw_segments:
        sample = " ".join(
            (s.get("text") or "") for s in raw_segments[:15]
        )
        alpha_chars = [c for c in sample if c.isalpha()]
        if len(alpha_chars) >= 20:
            latin_count = sum(1 for c in alpha_chars if ord(c) < 0x250)
            latin_ratio = latin_count / len(alpha_chars)
            # Known non-Latin-script language codes so we don't wrongly remap them
            non_latin_langs = {
                "ar", "fa", "ur", "he", "hi", "bn", "ta", "te", "kn", "ml",
                "mr", "gu", "pa", "si", "th", "km", "lo", "my", "ka", "am",
                "hy", "zh", "ja", "ko",
            }
            if latin_ratio > 0.85 and lang not in non_latin_langs:
                print(
                    f"[STT] Detected {lang!r} but text is "
                    f"{latin_ratio:.0%} Latin-script → overriding to 'en'"
                )
                return "en"

    return lang


def _clean_transcript_text(text: str) -> str:
    """Reduce Whisper hallucination / noise in segment text."""
    text = re.sub(r"(\b\w+\b)( \1){3,}", r"\1", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    return text.strip()


def _transcribe_faster_whisper(
    path_wav: str,
    model_size: str = "large-v3",
    language: str | None = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> tuple[list[dict], str]:
    """Fallback transcription using faster-whisper (Python 3.9 compatible)."""
    from faster_whisper import WhisperModel

    fw_model = model_size if model_size in ("tiny", "base", "small", "medium", "large-v2", "large-v3") else "large-v3"
    device = "cpu"
    try:
        import torch
        if torch.backends.mps.is_available():
            device = "cpu"  # faster-whisper uses CTranslate2 which doesn't support MPS
    except Exception:
        pass

    progress_callback and progress_callback("loading_whisper")
    print(f"[STT] faster-whisper model: {fw_model} (mlx-whisper unavailable on Python 3.9)")
    model = WhisperModel(fw_model, device=device, compute_type="int8")

    progress_callback and progress_callback("transcribing")
    fw_segments, info = model.transcribe(
        path_wav,
        language=language,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
    )

    detected_lang = info.language or (language or "en")
    segments: list[dict] = []
    for seg in fw_segments:
        words = [{"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                 for w in (seg.words or [])]
        segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "words": words,
            "avg_logprob": seg.avg_logprob,
            "no_speech_prob": seg.no_speech_prob,
            "compression_ratio": seg.compression_ratio,
        })

    segments = [s for s in segments if s["text"]]
    return segments, detected_lang


def transcribe(
    path_wav: str,
    model_size: str = "large-v3",
    language: str | None = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> tuple[list[dict], str]:
    """
    Transcribe with mlx-whisper (Metal on Apple Silicon) or faster-whisper fallback.

    Returns:
        (segments, detected_language_code)
        segments: start, end, text, optional words (with probability when available),
                  optional segment-level avg_logprob, no_speech_prob, compression_ratio, temperature
    """
    if mlx_whisper is None:
        # mlx-whisper unavailable (requires Python >=3.10 via mlx).
        # Fall back to faster-whisper which works on Python 3.9+.
        return _transcribe_faster_whisper(path_wav, model_size, language, progress_callback)

    repo = _mlx_hf_repo(model_size)
    progress_callback and progress_callback("loading_whisper")
    print(f"[STT] mlx-whisper model: {repo}")

    progress_callback and progress_callback("transcribing")

    # mlx-whisper does not implement beam search (beam_size → NotImplementedError).
    # best_of applies when temperature > 0 during fallback decodes (see mlx_whisper.transcribe).
    best_of = _env_int("STT_BEST_OF", 5)
    if best_of < 1:
        best_of = 1
    decode_extras: dict = {}
    if best_of > 1:
        decode_extras["best_of"] = best_of

    temperature = _parse_temperature_fallback()
    logprob_threshold = _env_float("STT_LOGPROB_THRESHOLD", -1.0)
    compression_ratio_threshold = _env_float("STT_COMPRESSION_RATIO_THRESHOLD", 2.4)
    no_speech_threshold = _env_float("STT_NO_SPEECH_THRESHOLD", 0.6)

    hall_silence = _env_float_optional("STT_HALLUCINATION_SILENCE_THRESHOLD")
    hall_kw: dict = {}
    if hall_silence is not None:
        hall_kw["hallucination_silence_threshold"] = hall_silence

    initial_prompt = build_initial_prompt()
    if initial_prompt:
        print(f"[STT] initial_prompt (truncated): {initial_prompt[:120]}...")

    kw: dict = {
        "path_or_hf_repo": repo,
        "word_timestamps": True,
        "verbose": False,
        "temperature": temperature,
        **decode_extras,
        **hall_kw,
    }
    if logprob_threshold is not None:
        kw["logprob_threshold"] = logprob_threshold
    if compression_ratio_threshold is not None:
        kw["compression_ratio_threshold"] = compression_ratio_threshold
    if no_speech_threshold is not None:
        kw["no_speech_threshold"] = no_speech_threshold

    if initial_prompt:
        kw["initial_prompt"] = initial_prompt

    if language is not None and str(language).strip():
        kw["language"] = str(language).strip()

    cond_prev = os.environ.get("STT_CONDITION_ON_PREVIOUS_TEXT", "true").strip().lower()
    if cond_prev in ("0", "false", "no"):
        kw["condition_on_previous_text"] = False

    # Serialize with PyTorch MPS (Demucs, Chatterbox warmup, etc.) — shared Metal queue on macOS.
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.synchronize()
    except Exception:
        pass

    from services.gpu_exclusive import gpu_exclusive

    with gpu_exclusive():
        result = mlx_whisper.transcribe(path_wav, **kw)

    lang_raw = result.get("language") or "en"
    detected_lang = str(lang_raw).strip().lower()[:8]
    detected_lang = _validate_detected_language(detected_lang, result.get("segments") or [])
    print(f"[STT] Detected language: {detected_lang}")

    segments: list[dict] = []
    for seg in result.get("segments") or []:
        raw_text = (seg.get("text") or "").strip()
        text = _clean_transcript_text(raw_text)
        if not text:
            continue
        text, entity_fixed = apply_asr_entity_corrections(text)
        text_after_learned = apply_learned_phrase_fixes(text)
        learned_fixed = text_after_learned != text
        text = text_after_learned
        entry: dict = {
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": text,
        }
        # Segment-level quality signals (mlx-whisper / OpenAI-style)
        for key in (
            "avg_logprob",
            "no_speech_prob",
            "compression_ratio",
            "temperature",
        ):
            if key in seg and seg[key] is not None:
                try:
                    entry[key] = float(seg[key])
                except (TypeError, ValueError):
                    pass

        words_out: list[dict] = []
        for w in seg.get("words") or []:
            if not isinstance(w, dict):
                continue
            ww = (w.get("word") or w.get("text") or "").strip()
            wd: dict = {
                "word": ww,
                "start": float(w.get("start", 0.0)),
                "end": float(w.get("end", 0.0)),
            }
            if w.get("probability") is not None:
                try:
                    wd["probability"] = float(w["probability"])
                except (TypeError, ValueError):
                    pass
            words_out.append(wd)

        # Word timings no longer match if we changed entity spellings substantially
        if (entity_fixed or learned_fixed) and words_out:
            words_out = []

        if words_out:
            entry["words"] = words_out
        segments.append(entry)

    print(f"[STT] {len(segments)} segments")
    return segments, detected_lang
