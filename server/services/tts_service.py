"""
Dubbing: **Hindi** can use Microsoft Edge neural TTS via **edge-tts** (``TTS_USE_EDGE=1``,
default), unless a **clone reference** exists (``{cache_key}_xtts_ref.wav`` from
``voice_extractor``) and ``TTS_HI_USE_XTTS_CLONE=1`` — then **Coqui XTTS** uses that
``speaker_wav`` for all Hindi lines. Other languages / fallbacks: XTTS + MMS-TTS.

**Natural speech:** segments are synthesized at engine-native speed (no fit-to-subtitle
trim), optionally separated by a short gap and crossfaded, then **one** loudness pass.
Optional **gender hint** from the clone ref may apply a light time-stretch + pitch shift
(``TTS_APPLY_GENDER_PROSODY``).

Env (optional):
  ``AUDIO_FORMAT`` / ``AUDIO_BITRATE`` / ``AUDIO_ENHANCE`` — final dub: ``wav`` or ``mp3`` (default),
  MP3 bitrate default ``320`` kbps, optional ``audio_enhancer`` HPF + clarity + peak norm.
  ``TTS_DUB_MASTER_SAMPLE_RATE`` / ``AUDIO_EXPORT_SR`` — concat sample rate (e.g. ``48000``).
  ``HINDI_TTS_ENGINE`` — ``edge`` (default), ``xtts``, or ``mms`` for Hindi only.
  ``TTS_MASTER_SAMPLE_RATE`` — default ``24000`` (pydub resamples segments before concat).
  ``TTS_SEGMENT_GAP_MS`` — silence between subtitle segments (default ``120``).
  ``TTS_CROSSFADE_MS`` — overlap blend between segments (default ``50``; ``0`` = off).
  ``TTS_TARGET_DBFS`` — peak-normalize final mix to this dBFS (default ``-20``).
  ``TTS_HI_USE_XTTS_CLONE`` — app default ``1`` if unset; ``run.sh`` exports ``0`` unless your ``.env`` sets it (Edge-first Hindi for stability).
  ``TTS_CLONE_DEBUG`` — set ``1`` for per-clause ``[CLONE DEBUG]`` logs (``speaker_wav`` resolution, XTTS path).
  ``TTS_APPLY_GENDER_PROSODY`` — default ``1``; set ``0`` to skip light speed/pitch shaping from ``speaker_gender``.
  **Gender / Hindi Edge:** ``VOICE_MALE``, ``VOICE_FEMALE``, ``VOICE_DEFAULT``; ``GENDER_CONFIDENCE_THRESHOLD``
  (default ``0.7``) collapses low-confidence male/female to unknown; ``GENDER_CLONE_VERIFY`` (default on)
  re-checks first XTTS Hindi segment and switches to gender-matched Edge if mismatch;
  ``GENDER_CLONE_VERIFY_MIN_CONF`` (default ``0.55``) minimum confidence on the verification pass.
  **Hindi XTTS inference (optional):** ``XTTS_HI_TEMPERATURE``, ``XTTS_HI_LENGTH_PENALTY``,
  ``XTTS_HI_REPETITION_PENALTY``, ``XTTS_HI_TOP_K``, ``XTTS_HI_TOP_P`` — forwarded to Coqui
  XTTS ``tts_to_file`` for ``language=hi``. ``XTTS_HI_SPLIT_SENTENCES`` applies to non-Hindi
  only; Hindi always uses ``split_sentences=False`` and one TTS unit per subtitle (no clause
  splits) to reduce word repetition.
  **Hindi text safety:** ``HINDI_RAW_MODE=1`` (default) — minimal cleanup only. With
  ``HINDI_RAW_MODE=0``, use ``HINDI_TTS_RAW_MODE=1`` for the same minimal text path.
  ``HINDI_CORRUPTION_CHECK=0`` disables garbage-marker → Edge fallback.
  ``HINDI_USE_EDGE_FALLBACK=1`` — prefer Edge-TTS for Hindi even when a clone ref exists
  (avoids XTTS chunking/repetition). ``TTS_HINDI_CROSSFADE_MS`` — crossfade between subtitle
  segments for Hindi (default ``75``, clamped ``50``–``100``). ``TTS_HINDI_DEBUG=1`` — extra logs.
  ``TTS_CHATTERBOX_FIRST`` — default ``0``: **Edge TTS first** (clearest pronunciation), then
  OpenVoice on top of Edge for timbre. Set ``1`` to try **Chatterbox** before Edge (single-pass
  clone; can sound less clear in some languages).
  ``TTS_CLARITY_MODE`` — default ``1``: **maximum intelligibility** — do not run OpenVoice /
  primitive voice conversion on **Edge** segments (keeps studio-clean neural speech); under
  clarity mode ``TTS_SYNC_STRETCH`` defaults to ``0`` (no fit-to-slot stretch; set ``1`` to
  stretch anyway). Set ``TTS_CLARITY_MODE=0`` for stronger speaker match on Edge (OpenVoice on).
"""
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

import re
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import scipy.io.wavfile as wavfile
import soundfile as sf
import torch
from pydub import AudioSegment
from transformers import AutoTokenizer, VitsModel

try:
    from transformers.models.vits import VitsTokenizer
except ImportError:
    VitsTokenizer = None  # type: ignore[misc, assignment]

SAMPLE_RATE_DEFAULT = 24000  # Coqui XTTS v2 typical output

MMS_MODEL_BY_LANG: dict[str, str] = {
    "en": "facebook/mms-tts-eng",
    "hi": "facebook/mms-tts-hin",
    "ar": "facebook/mms-tts-ara",
    "es": "facebook/mms-tts-spa",
    "fr": "facebook/mms-tts-fra",
    "de": "facebook/mms-tts-deu",
    "pt": "facebook/mms-tts-por",
    "ru": "facebook/mms-tts-rus",
    "ja": "facebook/mms-tts-jpn",
    "ko": "facebook/mms-tts-kor",
    "zh": "facebook/mms-tts-zho",
    "it": "facebook/mms-tts-ita",
    "tr": "facebook/mms-tts-tur",
    "pl": "facebook/mms-tts-pol",
    "nl": "facebook/mms-tts-nld",
    "sv": "facebook/mms-tts-swe",
}
MMS_FALLBACK_ID = "facebook/mms-tts-eng"

XTTS_LANGUAGE_MAP: dict[str, str] = {
    "hi": "hi",
    "ta": "ta",
    "te": "te",
    "bn": "bn",
    "mr": "mr",
    "gu": "gu",
    "kn": "kn",
    "ml": "ml",
    "pa": "pa",
    "ur": "ur",
    "zh": "zh-cn",
    "fr": "fr",
    "de": "de",
    "es": "es",
    "ja": "ja",
    "ko": "ko",
    "ar": "ar",
    "en": "en",
}

UNIVERSAL_XTTS = "tts_models/multilingual/multi-dataset/xtts_v2"

_mms_cache: dict[str, tuple[Any, Any, torch.device]] = {}
_tts_singleton: Optional["TTSService"] = None
# One full XTTS traceback per generate_dubbed_audio run (avoid log spam)
_printed_xtts_traceback: bool = False


def _clone_debug_verbose() -> bool:
    return os.environ.get("TTS_CLONE_DEBUG", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _chatterbox_first() -> bool:
    """If True, try Chatterbox before Edge; default False prioritizes intelligible Edge neural TTS."""
    return os.environ.get("TTS_CHATTERBOX_FIRST", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _chatterbox_first_for_target(target_language: str) -> bool:
    """
    English dubs skip Chatterbox-first unless ``TTS_CHATTERBOX_FIRST_EN=1`` — Edge neural
    English is usually more intelligible than cloned multilingual flow for translated scripts.
    """
    if not _chatterbox_first():
        return False
    if _normalize_lang(target_language) == "en":
        return os.environ.get("TTS_CHATTERBOX_FIRST_EN", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    return True


def _clarity_mode() -> bool:
    """
    When on (default): keep Edge TTS output pristine (skip OpenVoice / voice_convert on Edge
    segments) and default segment time-stretch off for less phase-vocoder smear.
    """
    return os.environ.get("TTS_CLARITY_MODE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _sync_stretch_enabled() -> bool:
    """
    Fit each dubbed clip to its subtitle ``[start,end]`` duration (librosa phase-vocoder).

    Default **on** so cumulative dub tracks the video timeline (fewer skipped tails).
    Set ``TTS_SYNC_STRETCH=0`` for natural TTS length only (no per-segment stretch).
    """
    raw = (os.environ.get("TTS_SYNC_STRETCH") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _sync_stretch_enabled_for_target(target_language: str) -> bool:
    """
    English dubs: per-segment duration stretch is **off** by default (``TTS_SYNC_STRETCH_EN``).

    Fitting TTS into every STT window uses time-stretching, which often makes English
    sound muffled or rushed. Natural Edge length + ``dub_sync`` is clearer; opt in with
    ``TTS_SYNC_STRETCH_EN=1`` when tight timeline fit matters more.
    """
    if not _sync_stretch_enabled():
        return False
    if _normalize_lang(target_language) != "en":
        return True
    return os.environ.get("TTS_SYNC_STRETCH_EN", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _hindi_raw_mode() -> bool:
    """
    Default on: Hindi uses ``minimal_hindi_text_for_tts`` only (no broad regex clean,
    no ``split_into_speech_clauses`` / comma splits).

    Set ``HINDI_RAW_MODE=0`` for full ``clean_text_for_tts`` + clause splitting; you can
    still enable minimal mode with ``HINDI_TTS_RAW_MODE=1``.
    """
    v = os.environ.get("HINDI_RAW_MODE", "1").strip().lower()
    if v in ("0", "false", "no", "off"):
        return os.environ.get("HINDI_TTS_RAW_MODE", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    return True


def _hindi_tts_debug() -> bool:
    return os.environ.get("TTS_HINDI_DEBUG", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


# Subtitle timestamp like [0:01.23 – 0:05.00]
_RE_SUBTITLE_TIME_BRACKET = re.compile(
    r"\[\s*\d+:\d+(?:\.\d+)?\s*[–\-]\s*\d+:\d+(?:\.\d+)?\s*\]"
)
_RE_SUBTITLE_TIME_PAREN = re.compile(
    r"\(\s*\d+:\d+(?:\.\d+)?\s*[–\-]\s*\d+:\d+(?:\.\d+)?\s*\)"
)
# Common Latin STT tags only (never use broad ``[.*?]`` on Hindi — breaks Devanagari / ZWJ).
_RE_STT_LATIN_NOTE = re.compile(
    r"\[(?:music|applause|laughter|noise|silence|inaudible|crosstalk)\]",
    re.IGNORECASE,
)
# ASCII-only bracket / paren notes (e.g. [SOUND], (English aside))
_RE_ASCII_BRACKET_NOTE = re.compile(r"\[[A-Za-z][A-Za-z0-9_,.\s\-]{0,120}\]")
_RE_ASCII_PAREN_NOTE = re.compile(r"\([A-Za-z][A-Za-z0-9_,.\s\-']{0,240}\)")
# e.g. [0:01.23 – 0:05.00] with fractional seconds (subset of bracket pattern; explicit for logs)
_RE_SUBTITLE_TIME_DECIMAL = re.compile(
    r"\[\d+:\d+\.\d+\s*[–\-]\s*\d+:\d+\.\d+\s*\]"
)

_HINDI_CORRUPT_MARKERS = (
    "ड़ई",
    "ड़ड़",
    "हड़शा",
    "ड़झड़",
    "कहड़",
)


def _hindi_corruption_check_enabled() -> bool:
    return os.environ.get("HINDI_CORRUPTION_CHECK", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def minimal_hindi_text_for_tts(text: str) -> str:
    """Strip leaked subtitle timestamps and normalize whitespace; do not alter Devanagari."""
    t = text or ""
    t = _RE_SUBTITLE_TIME_BRACKET.sub("", t)
    t = _RE_SUBTITLE_TIME_PAREN.sub("", t)
    t = _RE_SUBTITLE_TIME_DECIMAL.sub("", t)
    t = " ".join(t.replace("\r", " ").split()).strip()
    return t


def _valid_speaker_ref_path(path: str | None) -> str | None:
    """Return resolved path if file exists and is large enough to be a real WAV ref."""
    if not path or not str(path).strip():
        return None
    try:
        p = Path(path).expanduser().resolve(strict=False)
    except Exception:
        p = Path(path).expanduser()
    if not p.is_file():
        return None
    try:
        if p.stat().st_size <= 1024:
            return None
    except OSError:
        return None
    return str(p)


def _tts_pause_ms(kind: str) -> int:
    defaults = {"period": 340, "comma": 130, "minor": 200}
    env_keys = {
        "period": "TTS_PAUSE_MS_PERIOD",
        "comma": "TTS_PAUSE_MS_COMMA",
        "minor": "TTS_PAUSE_MS_MINOR",
    }
    raw = os.environ.get(env_keys.get(kind, "TTS_PAUSE_MS_MINOR"), "").strip()
    if raw.isdigit():
        return int(raw)
    return defaults.get(kind, 200)


def _target_normalize_dbfs() -> float:
    raw = os.environ.get("TTS_TARGET_DBFS", "-20.0").strip()
    try:
        return float(raw)
    except ValueError:
        return -20.0


def _master_sample_rate() -> int:
    raw = os.environ.get("TTS_MASTER_SAMPLE_RATE", "24000").strip()
    if raw.isdigit() and int(raw) >= 8000:
        return int(raw)
    return 24000


def _dub_master_sample_rate() -> int:
    """Final concat / export rate (``TTS_DUB_MASTER_SAMPLE_RATE`` or ``AUDIO_EXPORT_SR``)."""
    for key in ("TTS_DUB_MASTER_SAMPLE_RATE", "AUDIO_EXPORT_SR"):
        raw = os.environ.get(key, "").strip()
        if raw.isdigit():
            s = int(raw)
            if 8000 <= s <= 96000:
                return s
    return _master_sample_rate()


def dub_audio_format() -> str:
    v = (os.environ.get("AUDIO_FORMAT") or "mp3").strip().lower()
    if v in ("wav", "wave"):
        return "wav"
    return "mp3"


def dub_mp3_bitrate_kbps() -> int:
    raw = (os.environ.get("AUDIO_BITRATE") or "320").strip()
    if raw.isdigit():
        return max(96, min(320, int(raw)))
    return 320


def dub_audio_enhance_enabled() -> bool:
    return os.environ.get("AUDIO_ENHANCE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def resolve_dub_output_path(output_path: str) -> str:
    """Apply ``AUDIO_FORMAT`` (.wav vs .mp3) to the dub output path."""
    p = Path(output_path)
    ext = ".wav" if dub_audio_format() == "wav" else ".mp3"
    return str(p.with_suffix(ext))


def _hindi_tts_engine() -> str:
    """
    Hindi synthesis backend: ``edge`` (default), ``xtts`` (clone when available), ``mms`` only.

    Set ``HINDI_TTS_ENGINE=xtts`` with ``TTS_HI_USE_XTTS_CLONE=1`` for voice cloning.
    """
    v = (os.environ.get("HINDI_TTS_ENGINE") or "").strip().lower()
    if v in ("edge", "xtts", "mms"):
        return v
    return "edge"


def _segment_gap_ms() -> int:
    raw = os.environ.get("TTS_SEGMENT_GAP_MS", "120").strip()
    if raw.isdigit():
        return max(0, int(raw))
    return 120


def _segment_gap_ms_for_target(target_language: str) -> int:
    """Slightly longer pauses for English dubs — easier to parse phrases (``TTS_SEGMENT_GAP_MS_EN``)."""
    if _normalize_lang(target_language) != "en":
        return _segment_gap_ms()
    raw = os.environ.get("TTS_SEGMENT_GAP_MS_EN", "").strip()
    if raw.isdigit():
        return max(0, int(raw))
    try:
        from services.tts_clarity import english_segment_gap_floor_ms

        floor = english_segment_gap_floor_ms()
    except ImportError:
        floor = 200
    return max(_segment_gap_ms(), floor)


def _crossfade_ms() -> int:
    raw = os.environ.get("TTS_CROSSFADE_MS", "50").strip()
    if raw.isdigit():
        return max(0, int(raw))
    return 50


def _hindi_use_edge_fallback() -> bool:
    """Prefer Edge-TTS for Hindi even if XTTS clone ref is available (reduces repetition)."""
    return os.environ.get("HINDI_USE_EDGE_FALLBACK", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _hindi_segment_crossfade_ms() -> int:
    """Overlap when stitching Hindi segment WAVs (50–100 ms recommended)."""
    raw = os.environ.get("TTS_HINDI_CROSSFADE_MS", "75").strip()
    if raw.isdigit():
        return max(50, min(100, int(raw)))
    return 75


def _segment_crossfade_ms(target_language: str) -> int:
    if _normalize_lang(target_language) == "hi":
        return _hindi_segment_crossfade_ms()
    if _normalize_lang(target_language) == "en":
        raw = os.environ.get("TTS_CROSSFADE_MS_EN", "").strip()
        if raw.isdigit():
            return max(0, min(80, int(raw)))
        try:
            from services.tts_clarity import english_crossfade_cap_ms

            cap = english_crossfade_cap_ms()
        except ImportError:
            cap = 24
        # Long overlaps blur consonants between subtitle clips; keep English edges crisp.
        return min(_crossfade_ms(), cap)
    return _crossfade_ms()


def _coqui_gpu_flag() -> bool:
    if os.environ.get("XTTS_FORCE_CPU", "").strip().lower() in ("1", "true", "yes"):
        return False
    if os.environ.get("XTTS_USE_GPU", "").strip().lower() in ("1", "true", "yes"):
        return torch.cuda.is_available()
    if sys.platform == "darwin":
        return False
    return torch.cuda.is_available()


def _normalize_lang(code: str) -> str:
    return (code or "en").lower().strip().split("-")[0]


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _xtts_split_sentences_for_language(language: str) -> bool:
    """Coqui ``split_sentences`` — for Hindi default True (sentence-wise decode inside each clause)."""
    if _normalize_lang(language) != "hi":
        return False
    return os.environ.get("XTTS_HI_SPLIT_SENTENCES", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _xtts_inference_kwargs_for_language(language: str) -> dict[str, float | int]:
    """
    Extra kwargs for Coqui XTTS (passed through ``tts_to_file`` → ``synthesize`` → ``full_inference``).

    Hindi defaults favor steadier decoding (lower temperature, slightly wider nucleus).
    """
    if _normalize_lang(language) != "hi":
        return {}
    return {
        "temperature": _env_float("XTTS_HI_TEMPERATURE", 0.7),
        "length_penalty": _env_float("XTTS_HI_LENGTH_PENALTY", 1.0),
        "repetition_penalty": _env_float("XTTS_HI_REPETITION_PENALTY", 2.0),
        "top_k": _env_int("XTTS_HI_TOP_K", 50),
        "top_p": _env_float("XTTS_HI_TOP_P", 0.95),
    }


def _mms_model_id(target_language: str) -> str:
    return MMS_MODEL_BY_LANG.get(_normalize_lang(target_language), MMS_FALLBACK_ID)


def _infer_reference_wav(output_path: str) -> str | None:
    """Full-download WAV next to dubbed MP3 (``{stem}_audio.wav``)."""
    try:
        p = Path(output_path).expanduser().resolve(strict=False)
    except Exception:
        p = Path(output_path).expanduser()
    if p.suffix.lower() != ".mp3":
        return None
    candidate = p.with_name(f"{p.stem}_audio.wav")
    if candidate.is_file():
        try:
            return str(candidate.resolve(strict=False))
        except Exception:
            return str(candidate)
    return None


def _infer_clone_reference_wav(output_path: str) -> str | None:
    """VAD-trimmed clip for XTTS cloning (``{stem}_xtts_ref.wav``)."""
    try:
        p = Path(output_path).expanduser().resolve(strict=False)
    except Exception:
        p = Path(output_path).expanduser()
    if p.suffix.lower() != ".mp3":
        return None
    candidate = p.with_name(f"{p.stem}_xtts_ref.wav")
    return _valid_speaker_ref_path(str(candidate))


def _resolve_speaker_wav_for_xtts(output_path: str) -> tuple[str | None, str]:
    """
    Prefer extracted clone ref, else full ``*_audio.wav``.

    Returns:
        (path_or_none, kind) where kind is ``\"clone\"``, ``\"full\"``, or ``\"none\"``.
    """
    c = _infer_clone_reference_wav(output_path)
    if c:
        return c, "clone"
    f = _infer_reference_wav(output_path)
    if f:
        return f, "full"
    return None, "none"


def _is_speakable(text: str) -> bool:
    t = (text or "").strip()
    if len(t) <= 1:
        return False
    # Devanagari (+ marks): never use ``[^\w\s]`` — strips U+093C nukta etc. and destroys clusters.
    if re.search(r"[\u0900-\u097F]", t):
        return True
    cleaned = re.sub(r"[^\w\s]", "", t, flags=re.UNICODE).strip()
    return len(cleaned) > 1


_transformers_beam_shim_applied = False


def _patch_transformers_for_coqui_xtts() -> None:
    """
    Coqui XTTS does ``from transformers import BeamSearchScorer``. Newer Hugging Face
    releases omit it from the package root (class remains in generation.beam_search),
    which breaks XTTS import. Mirror the old export before loading Coqui.
    """
    global _transformers_beam_shim_applied
    if _transformers_beam_shim_applied:
        return
    _transformers_beam_shim_applied = True
    import transformers

    if getattr(transformers, "BeamSearchScorer", None) is not None:
        return
    try:
        from transformers.generation.beam_search import BeamSearchScorer as _BeamSearchScorer
    except ImportError:
        print(
            "[TTS] WARNING: Could not import BeamSearchScorer; Coqui XTTS will likely fail. "
            "Pin transformers to >=4.38,<5 (see server/requirements.txt)."
        )
        return
    setattr(transformers, "BeamSearchScorer", _BeamSearchScorer)


class TTSService:
    """XTTS v2 primary (needs reference WAV); MMS-VITS fallback."""

    def __init__(self) -> None:
        self._xtts: Any = None
        self._mms_models: dict[str, tuple[Any, Any, torch.device]] = {}
        self._clause_xtts_ok = 0
        self._clause_mms = 0

    def reset_clause_engine_stats(self) -> None:
        self._clause_xtts_ok = 0
        self._clause_mms = 0

    def _get_voice_parameters_for_gender(self, gender: str) -> dict:
        params: dict = {"speed": 1.0, "pitch": 0}
        g = (gender or "unknown").strip().lower()
        if g == "male":
            params.update({"speed": 0.95, "pitch": -2})
        elif g == "female":
            params.update({"speed": 1.05, "pitch": 2})
        return params

    def _ensure_xtts(self) -> Any:
        if self._xtts is None:
            from services.torch_coqui_compat import (
                apply_torch_load_coqui_compat,
                apply_torchaudio_soundfile_compat,
            )

            apply_torch_load_coqui_compat()
            apply_torchaudio_soundfile_compat()
            _patch_transformers_for_coqui_xtts()
            from TTS.api import TTS as CoquiTTS

            gpu = _coqui_gpu_flag()
            print(f"[TTS] Loading Coqui XTTS v2 (primary), gpu={gpu} …")
            self._xtts = CoquiTTS(
                model_name=UNIVERSAL_XTTS,
                gpu=gpu,
                progress_bar=False,
            )
        return self._xtts

    def clean_text_for_tts(self, text: str, language: str = "hi") -> str:
        """
        Strip subtitle / STT markers. Hindi avoids broad ``[.*?]`` / greedy paren
        patterns that can mangle Devanagari (e.g. nukta U+093C, ZWJ sequences).
        """
        text = text or ""
        lang = _normalize_lang(language)
        text = _RE_SUBTITLE_TIME_BRACKET.sub("", text)
        text = _RE_SUBTITLE_TIME_PAREN.sub("", text)
        text = re.sub(r"_[A-Z]+_[\d_]+", "", text)
        text = re.sub(r"PROPN\d+END", "", text)
        if lang == "hi":
            text = _RE_STT_LATIN_NOTE.sub("", text)
            text = _RE_ASCII_BRACKET_NOTE.sub("", text)
            text = _RE_ASCII_PAREN_NOTE.sub("", text)
        else:
            text = re.sub(r"\[.*?\]|\(.*?\)", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""
        if lang == "hi" and _hindi_raw_mode():
            text = "".join(ch for ch in text if ch == "\n" or ord(ch) >= 32)
            text = " ".join(text.replace("\r", " ").split()).strip()
            return text
        if lang == "hi":
            from services.context_translator import _strip_spurious_leading_latin_before_devanagari

            text = _strip_spurious_leading_latin_before_devanagari(text)
        if lang in ("ar", "fa", "ur", "he", "ps"):
            from services.context_translator import naturalise_rtl_target_for_speech

            text = naturalise_rtl_target_for_speech(text)
        if not text.strip():
            return ""
        # Hindi danda (।), Latin + Arabic sentence ends — avoid double punctuation
        if text[-1] not in "۔।.?!…؟":
            text += "۔" if lang in ("ar", "fa", "ur", "he", "ps") else ("।" if lang == "hi" else ".")
        return text

    def _prepare_hindi_text_for_tts(self, text: str) -> list[str]:
        """Single synthesis unit — avoids comma / कि clause splitting (use with ``HINDI_TTS_RAW_MODE``)."""
        t = " ".join((text or "").replace("\r", " ").split()).strip()
        return [t] if t else []

    def split_into_sentences(self, text: str, language: str = "hi") -> list[str]:
        if _normalize_lang(language) == "hi":
            parts = re.split(r"(?<=[।])\s*", text)
        else:
            parts = re.split(r"(?<=[.!?])\s+", text)
        sentences: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) > 200:
                sub = re.split(r"(?<=[,،])\s+", part)
                sentences.extend(s.strip() for s in sub if s.strip())
            else:
                sentences.append(part)
        return sentences

    def split_into_speech_clauses(self, text: str, language: str = "hi") -> list[str]:
        """
        Finer splits: sentence boundaries, then commas/semicolons when a chunk is long.
        """
        if _normalize_lang(language) == "hi":
            from services.hindi_humanize import split_into_speech_clauses as hi_clauses

            return hi_clauses(text)

        text = text.strip()
        if not text:
            return []
        primary = r"(?<=[।!?…])\s+|\n+|(?<=[.!?])\s+"
        level1 = [p.strip() for p in re.split(primary, text) if p.strip()]
        clauses: list[str] = []
        max_chunk = int(os.environ.get("TTS_CLAUSE_MAX_CHARS", "90") or "90")
        if _normalize_lang(language) == "en":
            try:
                from services.tts_clarity import english_clause_char_cap

                default_en = english_clause_char_cap(58)
            except ImportError:
                default_en = 58
            raw_en = os.environ.get("TTS_CLAUSE_MAX_CHARS_EN", "").strip()
            if raw_en:
                try:
                    cap_en = max(32, int(raw_en))
                    max_chunk = min(max_chunk, cap_en)
                except ValueError:
                    max_chunk = min(max_chunk, default_en)
            else:
                max_chunk = min(max_chunk, default_en)
        for chunk in level1:
            if len(chunk) > max(40, max_chunk):
                subs = re.split(r"(?<=[,،;])\s+", chunk)
                for s in subs:
                    s = s.strip()
                    if s:
                        clauses.append(s)
            else:
                clauses.append(chunk)
        return clauses

    def _pause_ms_after_clause(self, clause: str) -> int:
        t = clause.rstrip()
        if not t:
            return _tts_pause_ms("minor")
        last = t[-1]
        if last in "।.!?…":
            return _tts_pause_ms("period")
        if last in ",,;،":
            return _tts_pause_ms("comma")
        return _tts_pause_ms("minor")

    def _synthesize_clause_numpy(
        self,
        clause: str,
        language: str,
        speaker_wav: str | None,
    ) -> tuple[np.ndarray, int]:
        clause = clause.strip()
        if not clause:
            return np.array([], dtype=np.float32), SAMPLE_RATE_DEFAULT
        if (
            _normalize_lang(language) == "hi"
            and _hindi_corruption_check_enabled()
        ):
            for pattern in _HINDI_CORRUPT_MARKERS:
                if pattern in clause:
                    print(
                        f"[ERROR] Corrupted Hindi text detected (marker {pattern!r}): "
                        f"{clause[:120]!r}"
                    )
                    print("[ERROR] Falling back to Edge-TTS for this clause")
                    try:
                        from services.edge_tts_synth import (
                            DEFAULT_HI_VOICE,
                            synthesize_hindi_for_subtitle_slot,
                        )

                        voice = (
                            os.environ.get("EDGE_TTS_VOICE_HI", "").strip()
                            or DEFAULT_HI_VOICE
                        )
                        edge_out = synthesize_hindi_for_subtitle_slot(
                            clause,
                            voice,
                            tempfile.gettempdir(),
                            5.0,
                            clarity_first=True,
                        )
                        if edge_out[0].size > 0:
                            return edge_out
                    except Exception as ee:
                        print(
                            f"[ERROR] Edge-TTS corruption fallback failed ({ee!r}); "
                            "continuing with XTTS/MMS"
                        )
                    break
        wav: np.ndarray
        sr: int
        refp = _valid_speaker_ref_path(speaker_wav)
        if _clone_debug_verbose():
            ex = (
                os.path.isfile(speaker_wav)
                if speaker_wav
                else False
            )
            print(
                f"[CLONE DEBUG] _synthesize_clause_numpy: speaker_wav={speaker_wav!r} "
                f"validated={refp!r} raw_exists={ex}"
            )
        if refp:
            try:
                out = self._xtts_to_numpy(clause, language, refp)
                self._clause_xtts_ok += 1
                return out
            except Exception as e:
                global _printed_xtts_traceback
                print(f"[TTS] XTTS clause failed ({e!r}), falling back to MMS")
                if not _printed_xtts_traceback:
                    _printed_xtts_traceback = True
                    traceback.print_exc()
                try:
                    self._clause_mms += 1
                    return self._mms_to_numpy(clause, language)
                except Exception as e2:
                    print(f"[TTS] MMS clause failed ({e2})")
                    return np.array([], dtype=np.float32), SAMPLE_RATE_DEFAULT
        try:
            self._clause_mms += 1
            return self._mms_to_numpy(clause, language)
        except Exception as e2:
            print(f"[TTS] MMS clause failed ({e2})")
            return np.array([], dtype=np.float32), SAMPLE_RATE_DEFAULT

    def _synthesize_natural_with_pauses(
        self, text: str, language: str, speaker_wav: str | None
    ) -> tuple[np.ndarray, int]:
        """Full natural length; brief silences between clauses inside one subtitle."""
        # Hindi: always one synthesis unit per subtitle (no comma/humanize splits) to avoid
        # XTTS repetition when rejoining chunks.
        if _normalize_lang(language) == "hi":
            clauses = self._prepare_hindi_text_for_tts(text)
        else:
            clauses = self.split_into_speech_clauses(text, language)
        if _hindi_tts_debug() and _normalize_lang(language) == "hi":
            print(f"[TTS-DEBUG] Hindi clause units ({len(clauses)}): {clauses!r}")
        if not clauses:
            return np.array([], dtype=np.float32), SAMPLE_RATE_DEFAULT

        pieces: list[np.ndarray] = []
        sr0 = SAMPLE_RATE_DEFAULT
        for i, clause in enumerate(clauses):
            wav, sr = self._synthesize_clause_numpy(clause, language, speaker_wav)
            if wav.size == 0:
                continue
            if sr != sr0:
                new_len = int(len(wav) * sr0 / sr)
                if new_len > 0:
                    x = np.linspace(0, len(wav) - 1, new_len)
                    wav = np.interp(x, np.arange(len(wav)), wav).astype(np.float32)
                else:
                    continue
            pieces.append(wav)
            if i < len(clauses) - 1:
                pause_ms = self._pause_ms_after_clause(clause)
                n = int(sr0 * (pause_ms / 1000.0))
                if n > 0:
                    pieces.append(np.zeros(n, dtype=np.float32))

        if not pieces:
            return np.array([], dtype=np.float32), sr0
        result = np.concatenate(pieces)
        return result.astype(np.float32), sr0

    def _xtts_lang(self, language: str) -> str:
        return XTTS_LANGUAGE_MAP.get(_normalize_lang(language), "en")

    def _xtts_to_numpy(
        self,
        text: str,
        language: str,
        speaker_wav: str,
    ) -> tuple[np.ndarray, int]:
        xtts = self._ensure_xtts()
        lang = self._xtts_lang(language)
        sw = _valid_speaker_ref_path(speaker_wav) or speaker_wav
        if _clone_debug_verbose():
            print(f"[CLONE DEBUG] _xtts_to_numpy: speaker_wav={speaker_wav!r} using={sw!r}")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            split_s = (
                False
                if _normalize_lang(language) == "hi"
                else _xtts_split_sentences_for_language(language)
            )
            xkw = _xtts_inference_kwargs_for_language(language)
            xtts.tts_to_file(
                text=text,
                language=lang,
                file_path=tmp_path,
                speaker_wav=sw,
                split_sentences=split_s,
                **xkw,
            )
            data, sr = sf.read(tmp_path, dtype="float32", always_2d=True)
            if data.ndim == 2 and data.shape[1] > 1:
                data = data.mean(axis=1, keepdims=True)
            wav = data.squeeze().astype(np.float32)
            return wav, int(sr)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _load_mms(self, language: str) -> tuple[Any, Any, torch.device]:
        key = _normalize_lang(language)
        if key in self._mms_models:
            return self._mms_models[key]
        model_id = _mms_model_id(language)
        device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        try:
            tok = AutoTokenizer.from_pretrained(model_id)
        except Exception:
            if VitsTokenizer is None:
                raise
            tok = VitsTokenizer.from_pretrained(model_id)
        model = VitsModel.from_pretrained(model_id).to(device).eval()
        self._mms_models[key] = (model, tok, device)
        return self._mms_models[key]

    def _mms_to_numpy(self, text: str, language: str) -> tuple[np.ndarray, int]:
        model, tokenizer, device = self._load_mms(language)
        text = text.replace("*", "")
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs).waveform
        w = out[0].float().cpu().numpy().squeeze()
        sr = int(model.config.sampling_rate)
        return w.astype(np.float32), sr

    def synthesize(
        self,
        text: str,
        language: str = "hi",
        speaker_wav: str | None = None,
    ) -> np.ndarray:
        if _normalize_lang(language) == "hi" and _hindi_raw_mode():
            text = minimal_hindi_text_for_tts(text)
        else:
            text = self.clean_text_for_tts(text, language)
        if not text:
            return np.zeros(SAMPLE_RATE_DEFAULT, dtype=np.float32)
        refp = _valid_speaker_ref_path(speaker_wav)
        if refp:
            try:
                wav, _sr = self._xtts_to_numpy(text, language, refp)
                return wav
            except Exception as e:
                print(f"[TTS] XTTS failed ({e!r}), MMS fallback")
        try:
            wav, _sr = self._mms_to_numpy(text, language)
            return wav
        except Exception as e2:
            print(f"[TTS] MMS failed ({e2}), silence")
            return np.zeros(SAMPLE_RATE_DEFAULT, dtype=np.float32)

    def synthesize_for_segment(
        self,
        text: str,
        target_duration_sec: float,
        language: str = "hi",
        speaker_wav: str | None = None,
        emotion: str = "casual",
    ) -> tuple[np.ndarray, int]:
        """
        Natural-length audio for one subtitle segment. ``target_duration_sec`` and
        ``emotion`` are ignored (back-compat signature).
        """
        del target_duration_sec, emotion
        lang = _normalize_lang(language)
        if _hindi_tts_debug() and lang == "hi":
            print(f"[TTS-DEBUG] synthesize_for_segment raw: {text!r}")
            try:
                print(f"[TTS-DEBUG] UTF-8 prefix: {text.encode('utf-8')[:200]!r}")
            except Exception:
                pass

        if lang == "hi" and _hindi_raw_mode():
            raw_in = text or ""
            print(f"[HINDI-FIX] Raw text: {raw_in!r}")
            text_hi = minimal_hindi_text_for_tts(raw_in)
            print(f"[HINDI-FIX] Cleaned text: {text_hi!r}")
            if _hindi_tts_debug():
                print(
                    f"[TTS-DEBUG] synthesize_for_segment after minimal prep: {text_hi!r}"
                )
            if not text_hi:
                return np.array([], dtype=np.float32), SAMPLE_RATE_DEFAULT
            if _clone_debug_verbose():
                print(
                    f"[CLONE DEBUG] synthesize_for_segment: speaker_wav={speaker_wav!r} "
                    f"lang={language!r} hindi_raw_pipeline=1"
                )
            units = [text_hi]
            sr_out = SAMPLE_RATE_DEFAULT
            arrs: list[np.ndarray] = []
            for unit in units:
                wav, sr_u = self._synthesize_clause_numpy(unit, language, speaker_wav)
                if wav.size == 0:
                    continue
                if sr_u != sr_out:
                    new_len = int(len(wav) * sr_out / sr_u)
                    if new_len > 0:
                        x = np.linspace(0, len(wav) - 1, new_len)
                        wav = np.interp(x, np.arange(len(wav)), wav).astype(np.float32)
                arrs.append(wav)
            if not arrs:
                return np.array([], dtype=np.float32), sr_out
            result = np.concatenate(arrs) if len(arrs) > 1 else arrs[0]
            return result.astype(np.float32), sr_out

        text = self.clean_text_for_tts(text or "", language)
        if _hindi_tts_debug() and lang == "hi":
            print(f"[TTS-DEBUG] synthesize_for_segment after clean: {text!r}")
        if not text:
            return np.array([], dtype=np.float32), SAMPLE_RATE_DEFAULT
        if _clone_debug_verbose():
            print(
                f"[CLONE DEBUG] synthesize_for_segment: speaker_wav={speaker_wav!r} "
                f"lang={language!r}"
            )
        return self._synthesize_natural_with_pauses(text, language, speaker_wav)


def get_tts_service() -> TTSService:
    global _tts_singleton
    if _tts_singleton is None:
        _tts_singleton = TTSService()
    return _tts_singleton


def synthesize(text: str, language: str = "hi") -> np.ndarray:
    return get_tts_service().synthesize(text, language, speaker_wav=None)


def synthesize_for_segment(
    text: str,
    target_duration_sec: float,
    language: str = "hi",
    emotion: str = "casual",
) -> np.ndarray:
    wav, _ = get_tts_service().synthesize_for_segment(
        text, target_duration_sec, language, speaker_wav=None, emotion=emotion
    )
    return wav


def _normalize_audio(
    audio: AudioSegment, target_dbfs: float | None = None
) -> AudioSegment:
    if target_dbfs is None:
        target_dbfs = _target_normalize_dbfs()
    if audio.dBFS == float("-inf"):
        return audio
    return audio.apply_gain(target_dbfs - audio.dBFS)


def _effective_video_duration_s(segments: list[dict], video_duration: float) -> float:
    """YouTube duration from meta when set; else last subtitle end (seconds)."""
    vd = max(0.0, float(video_duration or 0.0))
    if vd > 0:
        return vd
    if not segments:
        return 0.0
    try:
        return max(float(s.get("end", 0) or 0) for s in segments)
    except (TypeError, ValueError):
        return 0.0


def _wav_duration_ms(path: str) -> int:
    try:
        return len(AudioSegment.from_wav(path))
    except Exception:
        return 0


def _gender_prosody_enabled() -> bool:
    return os.environ.get("TTS_APPLY_GENDER_PROSODY", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _gender_clone_verify_enabled() -> bool:
    return os.environ.get("GENDER_CLONE_VERIFY", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _apply_gender_prosody_numpy(
    wav: np.ndarray,
    sr: int,
    gender: str | None,
    params: dict,
) -> tuple[np.ndarray, int]:
    """Light time_stretch + pitch_shift after synthesis (male/female hints only)."""
    if wav.size == 0:
        return wav, sr
    speed = float(params.get("speed", 1.0))
    pitch = float(params.get("pitch", 0))
    if abs(speed - 1.0) < 1e-6 and abs(pitch) < 1e-6:
        return wav, sr
    try:
        import librosa
    except ImportError:
        return wav, sr
    y = wav.astype(np.float32, copy=False)
    if abs(speed - 1.0) >= 1e-6:
        y = librosa.effects.time_stretch(y, rate=speed)
    if abs(pitch) >= 1e-6:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch)
    y = np.clip(y, -1.0, 1.0).astype(np.float32)
    return y, sr


def _append_with_optional_crossfade(
    base: AudioSegment, nxt: AudioSegment, crossfade_ms: int
) -> AudioSegment:
    if len(nxt) == 0:
        return base
    if len(base) == 0:
        return nxt
    cf = min(crossfade_ms, len(base), len(nxt))
    if cf >= 8:
        return base.append(nxt, crossfade=cf)
    return base + nxt


def generate_dubbed_audio(
    segments: list[dict],
    output_path: str,
    target_language: str,
    video_duration: float = 0,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    speaker_ref_wav: str | None = None,
    speaker_gender: str | None = None,
    speaker_gender_meta: dict | None = None,
) -> tuple[str, list[dict]]:
    """
    Build dubbed MP3 + ``dub_sync`` map.

    Per-segment **time-stretch** (when ``TTS_SYNC_STRETCH`` is on, default) aligns TTS
    length to each subtitle window — except **English** targets, where stretch defaults
    off (``TTS_SYNC_STRETCH_EN``) for clearer speech. **Tail silence** pads the mix to
    ``video_duration`` (or last cue end) so the player can map the full YouTube timeline
    without cutting off early. True lip-sync to mouth shape is not performed here.
    """
    global _printed_xtts_traceback
    _printed_xtts_traceback = False
    final_output_path = resolve_dub_output_path(output_path)
    layout_mp3 = str(Path(output_path).with_suffix(".mp3"))

    # Clear per-video cached speaker embeddings so a new video always extracts
    # a fresh embedding from its own reference audio (not the previous video's).
    try:
        from services.openvoice_cloner import clear_reference_cache
        clear_reference_cache()
    except Exception:
        pass

    svc = get_tts_service()
    svc.reset_clause_engine_stats()

    explicit = _valid_speaker_ref_path(speaker_ref_wav)
    ref_wav, ref_kind = _resolve_speaker_wav_for_xtts(layout_mp3)
    if explicit:
        ref_wav, ref_kind = explicit, "clone"

    clone_infer = _infer_clone_reference_wav(layout_mp3)
    hi_clone_env = os.environ.get("TTS_HI_USE_XTTS_CLONE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    hindi_eng = (
        _hindi_tts_engine()
        if _normalize_lang(target_language) == "hi"
        else ""
    )
    hindi_mms_only = hindi_eng == "mms"
    prefer_hi_clone = (
        hindi_eng == "xtts"
        and ref_kind == "clone"
        and ref_wav is not None
        and hi_clone_env
        and not _hindi_use_edge_fallback()
    )
    synth_ref_wav = None if hindi_mms_only else ref_wav

    gnorm = (speaker_gender or "").strip().lower() or None
    # Microsoft Edge voice: prefer ``routing_gender`` (set when confidence gate hid male/female).
    edge_gender = gnorm
    if speaker_gender_meta:
        rg = speaker_gender_meta.get("routing_gender")
        if isinstance(rg, str) and rg.strip().lower() in ("male", "female", "neutral"):
            edge_gender = rg.strip().lower()
    if edge_gender not in ("male", "female", "neutral"):
        edge_gender = None

    gender_params = svc._get_voice_parameters_for_gender(gnorm or "unknown")

    from services.gender_detector import gender_confidence_threshold

    if speaker_gender_meta:
        if speaker_gender_meta.get("gender_gated"):
            rg = speaker_gender_meta.get("raw_gender")
            rc = float(speaker_gender_meta.get("raw_confidence") or 0.0)
            print(
                f"[GENDER] Reference clip: raw={rg!r} (confidence {rc:.2f}) → "
                f"unknown (threshold {gender_confidence_threshold():.2f})"
            )
        else:
            print(
                f"[GENDER] Detected: {gnorm or 'unknown'} "
                f"(confidence {float(speaker_gender_meta.get('confidence') or 0):.2f})"
            )
    elif gnorm:
        print(f"[GENDER] Using gender hint: {gnorm} (no metadata)")
    else:
        print("[GENDER] No gender hint — neutral Edge / default clone prosody")

    print(
        f"[TTS] dub_out={final_output_path!r} format={dub_audio_format()!r} "
        f"mp3_bitrate={dub_mp3_bitrate_kbps()}kbps enhance={dub_audio_enhance_enabled()}\n"
        f"[CLONE] layout_sibling={layout_mp3!r}\n"
        f"[CLONE] speaker_ref_wav_arg={speaker_ref_wav!r} → valid_resolved={explicit!r}\n"
        f"[CLONE] inferred_clone_sibling_of_mp3={clone_infer!r}\n"
        f"[CLONE] effective ref_wav={ref_wav!r} ref_kind={ref_kind} "
        f"hindi_engine={hindi_eng or 'n/a'} hindi_skip_edge={prefer_hi_clone} "
        f"(TTS_HI_USE_XTTS_CLONE ok={hi_clone_env})\n"
        f"[CLONE] speaker_gender={speaker_gender!r} prosody={gender_params} "
        f"(apply={_gender_prosody_enabled()})"
    )

    if ref_kind == "none":
        print("[TTS] No speaker reference WAV — XTTS disabled; MMS / Edge only.")
    elif ref_kind == "clone":
        print(f"[TTS] XTTS voice clone reference: {Path(ref_wav).name}")
    else:
        print(f"[TTS] XTTS speaker reference (full download): {Path(ref_wav).name}")

    print(
        "[TTS] Natural dub: full-speed synthesis, concat + optional gap/crossfade, "
        "single final normalize."
    )
    try:
        from services.tts_clarity import maximum_clarity_enabled as _max_clarity_on

        _mc = _max_clarity_on()
    except ImportError:
        _mc = False
    if _chatterbox_first():
        _spc = "Chatterbox-first (TTS_CHATTERBOX_FIRST=1)"
    else:
        _edge_pristine = _clarity_mode() or _mc
        _spc = (
            "Edge-first; OpenVoice on Edge is OFF"
            if _edge_pristine
            else "Edge-first; OpenVoice on Edge is ON (TTS_CLARITY_MODE=0 — dulls consonants)"
        )
    print(f"[TTS] Speech clarity: {_spc}")
    if _mc:
        print(
            "[TTS] TTS_MAXIMUM_CLARITY=1 — EN Edge ~-20% rate (unless EDGE_TTS_RATE set), "
            "≤44 char clauses, 300ms+ gaps / 10ms crossfade, softer limiter + more presence, "
            "~100Hz HPF; OpenVoice never on Edge; no ref pitch-match on Edge"
        )
    _stretch_en = _sync_stretch_enabled_for_target(target_language)
    print(
        f"[TTS] Segment duration fit (exact samples vs STT window): "
        f"{'on' if _stretch_en else 'off'} "
        f"(TTS_SYNC_STRETCH"
        + (
            f"; English uses natural length unless TTS_SYNC_STRETCH_EN=1"
            if _normalize_lang(target_language) == "en"
            else ""
        )
        + "; see TTS_MAX_STRETCH_RATIO, TTS_EXACT_DURATION_TOL_S, FFmpeg atempo fallback)"
    )

    edge_voice_candidate: str | None = None
    try:
        from services.edge_tts_synth import edge_voice_for_language_and_gender
        # Now supports ALL languages in _EDGE_VOICES (English, Hindi, Spanish, etc.)
        # Gender-aware: male speaker gets male voice, female gets female voice.
        edge_voice_candidate = edge_voice_for_language_and_gender(
            target_language, edge_gender
        )
    except Exception as ex:
        print(f"[TTS] Edge-TTS unavailable ({ex}); using local synthesis.")

    if hindi_eng == "mms":
        edge_voice_candidate = None

    if edge_voice_candidate:
        _eg = edge_gender or "unknown"
        if edge_gender != gnorm and gnorm:
            print(
                f"[EDGE] {target_language} Edge voice: {edge_voice_candidate!r} "
                f"(routing_gender={_eg!r}, canonical={gnorm!r})"
            )
        else:
            print(
                f"[EDGE] {target_language} Edge voice: {edge_voice_candidate!r} "
                f"(gender={_eg!r})"
            )
    if prefer_hi_clone:
        print(
            f"[CLONE] Cloning with gender prosody: speed={gender_params['speed']}, "
            f"pitch={gender_params['pitch']:+d}"
        )

    use_clone_not_edge = bool(prefer_hi_clone)
    force_gender_edge_hindi = False

    edge_hi_fallback = edge_voice_candidate if prefer_hi_clone else None

    if _normalize_lang(target_language) == "hi" and _hindi_use_edge_fallback():
        if edge_voice_candidate:
            print(
                "[TTS] HINDI_USE_EDGE_FALLBACK=1 — Edge-TTS primary for Hindi "
                "(skips XTTS clone to reduce repetition)."
            )
        else:
            print(
                "[TTS] HINDI_USE_EDGE_FALLBACK=1 but Edge voice unavailable "
                "(set TTS_USE_EDGE=1); using XTTS/MMS."
            )

    if prefer_hi_clone:
        print(
            "[TTS] Hindi: using Coqui XTTS with cloned speaker (Edge-TTS skipped). "
            "Set TTS_HI_USE_XTTS_CLONE=0 to use Edge for Hindi."
        )
    elif edge_voice_candidate:
        print(
            f"[TTS] Hindi/Edge neural TTS voice={edge_voice_candidate!r} "
            "(TTS_USE_EDGE=0 for local XTTS/MMS)."
        )
    if hindi_eng == "mms":
        print("[TTS] HINDI_TTS_ENGINE=mms — Edge/XTTS clone disabled; MMS-VITS for Hindi.")

    master_sr = _dub_master_sample_rate()
    gap_ms = _segment_gap_ms_for_target(target_language)
    crossfade_ms = _segment_crossfade_ms(target_language)

    combined = AudioSegment.silent(duration=0)
    total = len(segments)
    raw_wav_by_i: list[str | None] = [None] * total
    dub_sync: list[dict] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, seg in enumerate(segments):
            src_txt = (seg.get("text") or "").strip()
            tt_txt = (seg.get("translated_text") or "").strip()
            if not tt_txt:
                print(
                    f"[TTS] Segment {i + 1}: WARNING translated_text empty "
                    f"(using empty; not falling back to English source)"
                )
                raw_text = ""
            elif tt_txt == src_txt and src_txt:
                print(
                    f"[TTS] Segment {i + 1}: WARNING translated_text same as source "
                    f"(may be untranslated): {tt_txt[:80]!r}…"
                )
                raw_text = tt_txt
            else:
                raw_text = tt_txt

            seg_end_ms = int(float(seg["end"]) * 1000)
            seg_start_ms = int(float(seg["start"]) * 1000)
            seg_duration_ms = max(0, seg_end_ms - seg_start_ms)

            print(
                f"Generating TTS for segment {i + 1}/{total} in [{target_language}] "
                f"— text: {raw_text[:80]!r}…"
            )

            if seg_duration_ms <= 0:
                progress_callback and progress_callback(i + 1, total)
                continue
            if not raw_text or not _is_speakable(raw_text):
                progress_callback and progress_callback(i + 1, total)
                continue

            if _normalize_lang(target_language) == "hi" and _hindi_raw_mode():
                cleaned = minimal_hindi_text_for_tts(raw_text)
                print(f"[HINDI-FIX] seg {i + 1} raw[:120]={raw_text[:120]!r}")
                print(f"[HINDI-FIX] seg {i + 1} minimal={cleaned!r}")
            else:
                cleaned = svc.clean_text_for_tts(raw_text, target_language)
            if _hindi_tts_debug() and _normalize_lang(target_language) == "hi":
                print(
                    f"[TTS-DEBUG] seg {i + 1} translated_text (raw)[:160]={raw_text[:160]!r}"
                )
                print(f"[TTS-DEBUG] seg {i + 1} text_for_tts={cleaned!r}")
            if not cleaned.strip():
                progress_callback and progress_callback(i + 1, total)
                continue

            seg_raw = (seg.get("tts_gender") or "").strip().lower()
            if seg_raw not in ("male", "female", "neutral"):
                seg_raw = ""
            seg_edge_gender = (
                seg_raw
                if seg_raw
                else (edge_gender or gnorm or "neutral")
            )
            if seg_edge_gender not in ("male", "female", "neutral"):
                seg_edge_gender = "neutral"
            seg_clone_gender = (
                seg_raw
                if seg_raw in ("male", "female")
                else (gnorm if gnorm in ("male", "female") else None)
            )
            gender_params_seg = svc._get_voice_parameters_for_gender(
                seg_clone_gender or gnorm or "unknown"
            )

            wav: np.ndarray = np.array([], dtype=np.float32)
            sr: int = 24000
            used_edge = False
            used_chatterbox = False
            xtts_before = svc._clause_xtts_ok
            current_edge: str | None = edge_voice_candidate
            if not hindi_mms_only:
                try:
                    from services.edge_tts_synth import edge_voice_for_language_and_gender

                    current_edge = edge_voice_for_language_and_gender(
                        target_language, seg_edge_gender
                    )
                except Exception:
                    current_edge = edge_voice_candidate

            # ── STAGE 0: Chatterbox — optional first pass (TTS_CHATTERBOX_FIRST=1) ──
            # Default is OFF: Microsoft Edge neural TTS runs first (Stage 1) for the
            # clearest pronunciation; OpenVoice then matches timbre. Chatterbox-first
            # can sound less intelligible in some target languages.
            if (
                _chatterbox_first_for_target(target_language)
                and synth_ref_wav
                and os.path.isfile(synth_ref_wav)
            ):
                try:
                    from services.chatterbox_cloner import (
                        synthesize_with_chatterbox,
                        use_chatterbox_for_target,
                    )
                    if use_chatterbox_for_target(target_language):
                        wav_cb, sr_cb = synthesize_with_chatterbox(
                            cleaned, synth_ref_wav, target_language
                        )
                        if wav_cb.size > 0:
                            wav, sr = wav_cb, sr_cb
                            used_chatterbox = True
                            print(
                                f"[TTS] Segment {i + 1}: Chatterbox OK "
                                f"({len(wav_cb)/sr_cb:.2f}s)"
                            )
                except Exception as _cb_exc:
                    print(f"[TTS] Segment {i + 1}: Chatterbox failed ({_cb_exc!r}); "
                          "falling through to Edge TTS.")

            # ── STAGE 1: Edge TTS — primary synthesis (clearest pronunciation) ──
            # Skipped when Chatterbox already produced audio (used_chatterbox=True).
            _seg_is_hindi = _normalize_lang(target_language) == "hi"
            if not used_chatterbox and current_edge and not hindi_mms_only:
                try:
                    from services.edge_tts_synth import synthesize_hindi_to_numpy

                    wav, sr = synthesize_hindi_to_numpy(
                        cleaned, current_edge, tmp_dir
                    )
                    if wav.size > 0:
                        used_edge = True
                        print(
                            f"[TTS] Segment {i + 1}: Edge TTS OK "
                            f"({len(wav) / sr:.2f}s, voice={current_edge!r})"
                        )
                except Exception as ee:
                    print(
                        f"[TTS] Segment {i + 1}: Edge-TTS failed ({ee}); "
                        "falling back to XTTS/MMS."
                    )

            # ── STAGE 2: XTTS voice clone — fallback when Edge unavailable ──
            # Skipped when Chatterbox already produced audio.
            if (
                not used_chatterbox
                and wav.size == 0
                and synth_ref_wav
                and not force_gender_edge_hindi
                and not hindi_mms_only
            ):
                try:
                    from services.edge_tts_synth import synthesize_with_voice_clone

                    wav, sr = synthesize_with_voice_clone(
                        cleaned,
                        synth_ref_wav,
                        target_language,
                        tmp_dir,
                        gender=seg_clone_gender or gnorm,
                        fallback_to_edge=False,   # Edge was already attempted above
                    )
                    if wav.size > 0:
                        used_edge = False
                        if _clone_debug_verbose():
                            print(
                                f"[CLONE DEBUG] seg {i + 1}: XTTS clone OK "
                                f"({len(wav) / sr:.2f}s)"
                            )
                except Exception as ee:
                    print(
                        f"[TTS] Segment {i + 1}: XTTS clone failed "
                        f"({ee!r}); trying plain XTTS/MMS."
                    )
                    wav = np.array([], dtype=np.float32)
                    sr = 24000

            # ── STAGE 3: Plain XTTS / MMS (last resort) ──
            # Skipped when Chatterbox already produced audio.
            if not used_chatterbox and wav.size == 0:
                eff_ref = synth_ref_wav
                if (
                    force_gender_edge_hindi
                    and _normalize_lang(target_language) == "hi"
                ):
                    eff_ref = None
                wav, sr = svc._synthesize_natural_with_pauses(
                    cleaned, target_language, eff_ref
                )

            if wav.size == 0 and _normalize_lang(target_language) == "hi":
                _hi_fb = (
                    current_edge
                    or edge_hi_fallback
                    or edge_voice_candidate
                )
                if _hi_fb:
                    try:
                        from services.edge_tts_synth import synthesize_hindi_to_numpy

                        wav, sr = synthesize_hindi_to_numpy(
                            cleaned, _hi_fb, tmp_dir
                        )
                        if wav.size > 0:
                            print(
                                f"[TTS] Segment {i + 1}: Edge-TTS fallback after XTTS/MMS empty "
                                f"(voice={_hi_fb!r})"
                            )
                    except Exception as ee:
                        print(f"[TTS] Segment {i + 1}: Edge fallback failed ({ee})")

            if wav.size == 0:
                print(f"[TTS] Segment {i + 1}: empty synthesis; skipping")
                progress_callback and progress_callback(i + 1, total)
                continue

            seg_verify_expect = (
                seg_clone_gender
                if seg_clone_gender in ("male", "female")
                else (gnorm if gnorm in ("male", "female") else None)
            )
            _verify_edge = (
                current_edge
                or edge_voice_candidate
            )
            if (
                _gender_clone_verify_enabled()
                and use_clone_not_edge
                and _normalize_lang(target_language) == "hi"
                and seg_verify_expect in ("male", "female")
                and not used_edge
                and svc._clause_xtts_ok > xtts_before
                and _verify_edge
            ):
                from services.gender_detector import (
                    clone_output_mismatches_expected,
                    detect_gender_from_audio,
                )

                probe = str(Path(tmp_dir) / f"_clone_gender_probe_{i:04d}.wav")
                try:
                    sf.write(probe, wav, sr)
                    det = detect_gender_from_audio(probe)
                except Exception as ex:
                    print(f"[GENDER] Clone verify skipped ({ex!r})")
                    det = {}
                finally:
                    Path(probe).unlink(missing_ok=True)

                if clone_output_mismatches_expected(seg_verify_expect, det):
                    print(
                        f"[GENDER] Clone output mismatch: expected {seg_verify_expect!r}, "
                        f"heard {det.get('gender')!r} (confidence "
                        f"{float(det.get('confidence') or 0):.2f}) — "
                        f"re-synthesizing segment {i + 1} with Edge {_verify_edge!r}"
                    )
                    try:
                        from services.edge_tts_synth import synthesize_hindi_to_numpy

                        wav, sr = synthesize_hindi_to_numpy(
                            cleaned, _verify_edge, tmp_dir
                        )
                        used_edge = wav.size > 0
                    except Exception as ee:
                        print(
                            f"[TTS] Segment {i + 1}: Edge after gender verify failed ({ee})"
                        )
                    if wav.size == 0:
                        print(
                            f"[TTS] Segment {i + 1}: empty after gender verify; skipping"
                        )
                        progress_callback and progress_callback(i + 1, total)
                        continue

            prosody_g = (
                seg_clone_gender
                if seg_clone_gender in ("male", "female")
                else gnorm
            )
            if (
                _gender_prosody_enabled()
                and prosody_g in ("male", "female")
                and not used_edge
            ):
                wav, sr = _apply_gender_prosody_numpy(
                    wav, sr, prosody_g, gender_params_seg
                )

            # ── Neural voice cloning: apply original speaker's voice identity ──
            # Skipped when Chatterbox handled synthesis + voice identity in one pass.
            # TTS_CLARITY_MODE / TTS_MAXIMUM_CLARITY: skip OpenVoice on Edge — it smears
            # consonants vs pristine Edge neural audio (especially for translated English).
            try:
                from services.tts_clarity import maximum_clarity_enabled as _max_clr
            except ImportError:
                def _max_clr() -> bool:
                    return False

            _pristine_edge = used_edge and (_clarity_mode() or _max_clr())
            if (
                not used_chatterbox
                and wav.size > 0
                and synth_ref_wav
                and os.path.isfile(synth_ref_wav)
                and not _pristine_edge
            ):
                _ov_success = False
                try:
                    from services.openvoice_cloner import clone_voice_for_segment, is_enabled as _ov_enabled
                    if _ov_enabled():
                        wav_ov, sr_ov = clone_voice_for_segment(
                            wav, sr, synth_ref_wav
                        )
                        if wav_ov.size > 0 and wav_ov is not wav:
                            wav, sr = wav_ov, sr_ov
                            _ov_success = True
                except Exception as _ov_exc:
                    print(f"[TTS] OpenVoice seg {i+1} failed: {_ov_exc!r}")

                if not _ov_success and not used_edge:
                    # Never run pitch/tilt matching on Edge output — it smears consonants
                    # (especially when TTS_CLARITY_MODE=0 and OpenVoice is skipped).
                    try:
                        from services.voice_converter import convert_voice_from_path
                        wav_cv, sr_cv = convert_voice_from_path(
                            wav, sr, synth_ref_wav,
                            pitch_strength=0.55,
                            tilt_strength=0.25,
                            apply_rms_match=False,
                        )
                        if wav_cv.size > 0:
                            wav, sr = wav_cv, sr_cv
                    except Exception:
                        pass

            # ── Duration alignment (STT [start,end] is ground truth) ─────────────
            # Pitch-preserving stretch (librosa) + optional FFmpeg atempo + exact pad/trim.
            # Set TTS_SYNC_STRETCH=0 to skip and keep natural TTS length per segment.
            if wav.size > 0 and _sync_stretch_enabled_for_target(target_language):
                source_dur = float(seg.get("end", 0)) - float(seg.get("start", 0))
                if source_dur > 0.15:
                    try:
                        from services.audio_master import fit_wav_to_exact_duration

                        wav = fit_wav_to_exact_duration(wav, sr, source_dur)
                    except Exception as _ts_exc:
                        print(f"[TTS] seg {i + 1} duration fit failed: {_ts_exc!r}")

            seg_wav = str(Path(tmp_dir) / f"seg_{i:04d}.wav")
            try:
                sf.write(seg_wav, wav, sr)
            except Exception:
                clip = np.clip(wav, -1.0, 1.0)
                wavfile.write(
                    seg_wav, sr, (clip * 32767.0).astype(np.int16)
                )
            if _wav_duration_ms(seg_wav) > 0:
                raw_wav_by_i[i] = seg_wav

            progress_callback and progress_callback(i + 1, total)

        first_audio = True
        for i, seg in enumerate(segments):
            path = raw_wav_by_i[i]
            if not path:
                dub_sync.append(
                    {
                        "video_start": float(seg["start"]),
                        "video_end": float(seg["end"]),
                        "audio_start": len(combined) / 1000.0,
                        "audio_end": len(combined) / 1000.0,
                    }
                )
                continue

            clip = (
                AudioSegment.from_wav(path)
                .set_channels(1)
                .set_frame_rate(master_sr)
            )

            if not first_audio and gap_ms > 0:
                combined += AudioSegment.silent(duration=gap_ms)

            audio_start_ms = len(combined)
            if first_audio:
                combined = clip
                first_audio = False
            else:
                combined = _append_with_optional_crossfade(
                    combined, clip, crossfade_ms
                )

            dub_sync.append(
                {
                    "video_start": float(seg["start"]),
                    "video_end": float(seg["end"]),
                    "audio_start": audio_start_ms / 1000.0,
                    "audio_end": len(combined) / 1000.0,
                }
            )

    if len(combined) == 0:
        combined = AudioSegment.silent(duration=0)

    combined = _normalize_audio(combined)

    vd_eff = _effective_video_duration_s(segments, float(video_duration or 0.0))
    dub_len_s = len(combined) / 1000.0
    if dub_sync and vd_eff > dub_len_s + 0.08:
        pad_ms = int((vd_eff - dub_len_s) * 1000)
        if pad_ms >= 80:
            a0 = dub_len_s
            combined += AudioSegment.silent(duration=pad_ms)
            a1 = len(combined) / 1000.0
            v0 = float(dub_sync[-1]["video_end"])
            dub_sync.append(
                {
                    "video_start": v0,
                    "video_end": float(vd_eff),
                    "audio_start": a0,
                    "audio_end": a1,
                }
            )
            print(
                f"[TTS] Padded dub to video timeline: +{pad_ms / 1000:.2f}s silence "
                f"(dub {dub_len_s:.2f}s → {a1:.2f}s, video≈{vd_eff:.2f}s)"
            )
    br = dub_mp3_bitrate_kbps()
    fmt = dub_audio_format()
    tmp_mix = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_mix.close()
    tmp_enh = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_enh.close()
    try:
        if fmt == "wav" or dub_audio_enhance_enabled() or True:  # always master
            combined.export(tmp_mix.name, format="wav", parameters=["-ac", "1"])
            working_wav = tmp_mix.name
            # Professional broadcast mastering (always on; replaces simple enhancer)
            try:
                from services.audio_master import master_audio_file
                master_audio_file(tmp_mix.name, tmp_enh.name)
                working_wav = tmp_enh.name
            except Exception as ex:
                print(f"[TTS] Audio mastering failed ({ex!r}); trying legacy enhancer")
                if dub_audio_enhance_enabled():
                    try:
                        from services.audio_enhancer import enhance_audio_file
                        enhance_audio_file(tmp_mix.name, tmp_enh.name)
                        working_wav = tmp_enh.name
                    except Exception as ex2:
                        print(f"[TTS] Legacy enhancer also failed ({ex2!r}); using raw mix")
                        working_wav = tmp_mix.name

            if fmt == "wav":
                Path(final_output_path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(working_wav, final_output_path)
            else:
                seg_mp3 = AudioSegment.from_wav(working_wav)
                Path(final_output_path).parent.mkdir(parents=True, exist_ok=True)
                seg_mp3.export(
                    final_output_path, format="mp3", bitrate=f"{br}k"
                )
        else:
            Path(final_output_path).parent.mkdir(parents=True, exist_ok=True)
            combined.export(
                final_output_path, format="mp3", bitrate=f"{br}k"
            )
    finally:
        Path(tmp_mix.name).unlink(missing_ok=True)
        Path(tmp_enh.name).unlink(missing_ok=True)

    print(
        f"Dubbed audio → {Path(final_output_path).name}: "
        f"{len(combined) / 1000:.1f}s, dub_sync segments: {len(dub_sync)}"
    )
    print(
        f"[CLONE] clause engines: XTTS_ok={svc._clause_xtts_ok}, MMS={svc._clause_mms}"
    )
    if prefer_hi_clone and svc._clause_xtts_ok == 0 and svc._clause_mms > 0:
        print(
            "[CLONE] WARNING: Hindi clone mode was ON but every clause used MMS fallback "
            "(generic VITS). Check the first XTTS traceback above — ref WAV, GPU/CPU, or "
            "Coqui errors are the usual cause."
        )
    return final_output_path, dub_sync
