"""
Chatterbox Multilingual TTS — single-model voice clone + synthesis.

Why this replaces the Edge TTS → OpenVoice two-stage pipeline
--------------------------------------------------------------
The previous pipeline split the problem into two steps:
  1. Edge TTS  → clear pronunciation in the target language
  2. OpenVoice → apply the original speaker's voice identity on top

Chatterbox Multilingual solves BOTH in one forward pass:
  - Flow-matching architecture (Resemble AI, 0.5B params)
  - 23 languages natively including Hindi, Arabic, Chinese, Japanese, etc.
  - Zero-shot voice cloning: provide a 5–15 s reference WAV from the
    original speaker (any language), set ``language_id``, get the
    original speaker's voice speaking the target language naturally.
  - MIT licensed, ``pip install chatterbox-tts``
  - Officially supports ``device="mps"`` (Apple Silicon) and CUDA.

Key parameters
--------------
  exaggeration : 0.0–1.0  — emotion / prosody expressiveness
                 0.0 = flat and neutral, ~0.52 = natural, 1.0 = very expressive
  cfg_weight   : 0.0–1.0  — classifier-free guidance for voice adherence
                 **Set to 0.0 for cross-lingual cloning** — higher values
                 tend to "pull" the output back toward the reference language
                 phonology and can hurt intelligibility in the target language.
  temperature  : 0.6–1.2  — sampling temperature for prosody diversity
                 Lower = more stable but robotic; ~0.72 is a good default.

Env
---
  TTS_CHATTERBOX_FIRST      — dub pipeline: ``1`` to try Chatterbox before Edge TTS;
                              default ``0`` in ``tts_service`` (Edge first = clearer speech).
  CHATTERBOX_ENABLED        — 1 (default) / 0 to disable
  CHATTERBOX_EXAGGERATION   — float 0.0–1.0 (default 0.52)
  CHATTERBOX_CFG_WEIGHT     — float 0.0–1.0 (default 0.0, best for cross-lingual)
  CHATTERBOX_TEMPERATURE    — float 0.5–1.2 (default 0.72)
  CHATTERBOX_CHUNK_CHARS    — max chars per synthesis chunk (default 220)
  CHATTERBOX_CACHE_DIR      — HuggingFace cache override
  CHATTERBOX_ENGLISH_ONLY   — ``1`` to use Chatterbox only for target ``en`` (legacy)
  CHATTERBOX_REF_RMS_MATCH  — ``1`` (default) gently match loudness to reference WAV
  CHATTERBOX_CHUNK_GAP_MS   — silence between long-text chunks (default 80 ms)
"""
from __future__ import annotations

import os
import threading
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Singleton model
# ---------------------------------------------------------------------------

_model = None
_model_lock = threading.Lock()
_load_failed = False

# Language code → Chatterbox language_id mapping
# Chatterbox multilingual supports 23 languages using ISO 639-1 codes.
_LANG_MAP: dict[str, str] = {
    "en": "en", "hi": "hi", "es": "es", "fr": "fr", "de": "de",
    "it": "it", "pt": "pt", "ru": "ru", "zh": "zh", "ja": "ja",
    "ko": "ko", "ar": "ar", "nl": "nl", "pl": "pl", "tr": "tr",
    "sv": "sv", "da": "da", "fi": "fi", "no": "no", "cs": "cs",
    "hu": "hu", "ro": "ro", "uk": "uk",
}


def is_enabled() -> bool:
    return os.environ.get("CHATTERBOX_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )


def use_chatterbox_for_target(target_lang: str) -> bool:
    """
    Whether to try Chatterbox for this dubbing target language.

    Multilingual Chatterbox supports many ISO 639-1 codes (see ``_LANG_MAP``).
    Legacy behaviour: ``CHATTERBOX_ENGLISH_ONLY=1`` restricts to English only
    (old English-only model note in tts_service).
    """
    if not is_enabled():
        return False
    code = (target_lang or "en").strip().lower().split("-")[0][:2]
    if os.environ.get("CHATTERBOX_ENGLISH_ONLY", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return code == "en"
    return code in _LANG_MAP


def _exaggeration() -> float:
    try:
        return max(0.0, min(1.0, float(os.environ.get("CHATTERBOX_EXAGGERATION", "0.52"))))
    except ValueError:
        return 0.52


def _cfg_weight() -> float:
    try:
        return max(0.0, min(1.0, float(os.environ.get("CHATTERBOX_CFG_WEIGHT", "0.0"))))
    except ValueError:
        return 0.0


def _temperature() -> float:
    try:
        return max(0.4, min(1.5, float(os.environ.get("CHATTERBOX_TEMPERATURE", "0.72"))))
    except ValueError:
        return 0.72


def _chunk_chars() -> int:
    try:
        return max(80, min(400, int(os.environ.get("CHATTERBOX_CHUNK_CHARS", "220"))))
    except ValueError:
        return 220


def _chunk_gap_sec() -> float:
    try:
        ms = float(os.environ.get("CHATTERBOX_CHUNK_GAP_MS", "80"))
        return max(0.0, min(0.35, ms / 1000.0))
    except ValueError:
        return 0.08


def _ref_rms_match_enabled() -> bool:
    return os.environ.get("CHATTERBOX_REF_RMS_MATCH", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _align_rms_to_reference(wav: np.ndarray, sr: int, ref_wav_path: str) -> np.ndarray:
    """
    Blend output loudness toward the reference clip's RMS so segments sit in the
    same ballpark as the speaker's level (more natural in the final dub).
    """
    if wav.size == 0 or not ref_wav_path or not Path(ref_wav_path).is_file():
        return wav
    try:
        ref, sr_r = sf.read(ref_wav_path, always_2d=False)
        if ref.ndim > 1:
            ref = np.mean(ref.astype(np.float64), axis=-1)
        else:
            ref = ref.astype(np.float64)
        if ref.size == 0:
            return wav
        if sr_r != sr and sr_r > 0:
            try:
                import librosa

                ref = librosa.resample(ref, orig_sr=sr_r, target_sr=sr)
            except Exception:
                return wav
        ref = ref.astype(np.float32)
        r_ref = float(np.sqrt(np.mean(np.square(ref)) + 1e-12))
        r_out = float(np.sqrt(np.mean(np.square(wav.astype(np.float64))) + 1e-12))
        if r_ref <= 1e-8 or r_out <= 1e-8:
            return wav
        # Gentle match: do not fully snap to ref (avoids pumping / noise boost)
        blend = 0.42
        target = (1.0 - blend) * r_out + blend * r_ref
        scale = target / r_out
        scale = float(np.clip(scale, 0.55, 1.65))
        out = (wav.astype(np.float32) * scale).astype(np.float32)
        peak = float(np.abs(out).max())
        if peak > 0.98:
            out = (out * (0.95 / peak)).astype(np.float32)
        return out
    except Exception:
        return wav


def _device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def _get_model():
    global _model, _load_failed
    if _model is not None:
        return _model
    if _load_failed:
        return None

    with _model_lock:
        if _model is not None:
            return _model
        if _load_failed:
            return None

        try:
            from chatterbox.tts import ChatterboxTTS

            from services.gpu_exclusive import gpu_exclusive

            device = _device()

            print(f"[chatterbox] Loading Chatterbox Multilingual TTS (device={device}) …")

            with gpu_exclusive():
                model = ChatterboxTTS.from_pretrained(device=device)
                if device == "mps":
                    try:
                        import torch

                        torch.mps.synchronize()
                    except Exception:
                        pass
            _model = model
            print(f"[chatterbox] Model ready  device={device}  sr={model.sr}")
            return _model

        except ImportError:
            print(
                "[chatterbox] 'chatterbox-tts' not installed. "
                "Install with: pip install chatterbox-tts\n"
                "Falling back to Edge TTS + OpenVoice pipeline."
            )
            _load_failed = True
            return None
        except Exception as exc:
            print(f"[chatterbox] Load failed: {exc!r}. "
                  "Falling back to Edge TTS + OpenVoice pipeline.")
            _load_failed = True
            return None


# ---------------------------------------------------------------------------
# Text chunker
# ---------------------------------------------------------------------------

def _split_text(text: str, max_chars: int) -> list[str]:
    """
    Split long text into chunks at sentence boundaries so Chatterbox never
    gets an extremely long string (which degrades prosody quality).
    Prefers splitting at  ।  (Devanagari danda) or  .  ?  !  then commas.
    """
    import re
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    # Split on sentence-ending punctuation while keeping the delimiter
    parts = re.split(r'(?<=[।.!?])\s+', text)
    chunks: list[str] = []
    current = ""
    for part in parts:
        if not current:
            current = part
        elif len(current) + 1 + len(part) <= max_chars:
            current += " " + part
        else:
            chunks.append(current.strip())
            current = part
    if current:
        chunks.append(current.strip())

    # If any chunk is still too long, hard-split at commas, then at spaces
    final: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final.append(chunk)
            continue
        sub = re.split(r'(?<=[,،،])\s+', chunk)
        buf = ""
        for s in sub:
            if not buf:
                buf = s
            elif len(buf) + 1 + len(s) <= max_chars:
                buf += " " + s
            else:
                final.append(buf.strip())
                buf = s
        if buf:
            final.append(buf.strip())

    return [c for c in final if c.strip()]


# ---------------------------------------------------------------------------
# Core synthesis
# ---------------------------------------------------------------------------

def synthesize_with_chatterbox(
    text: str,
    ref_wav_path: str,
    target_lang: str,
    *,
    exaggeration: Optional[float] = None,
    cfg_weight: Optional[float]   = None,
    temperature: Optional[float]  = None,
) -> tuple[np.ndarray, int]:
    """
    Synthesise ``text`` in ``target_lang`` using the original speaker's voice
    from ``ref_wav_path``.

    Parameters
    ----------
    text         : Target-language text to synthesise (e.g. Hindi string)
    ref_wav_path : Path to reference WAV of the original speaker (any language,
                   5–15 s recommended for best identity capture)
    target_lang  : ISO 639-1 language code (e.g. "hi", "es", "fr")
    exaggeration : Prosody expressiveness 0.0–1.0 (default from env)
    cfg_weight   : Cross-lingual guidance weight — keep at 0.0 for best results
    temperature  : Sampling temperature 0.4–1.5 (default from env)

    Returns
    -------
    (wav_float32, sample_rate)  on success
    (empty_array, 24000)        on failure — caller falls back to Edge+OpenVoice
    """
    empty = np.array([], dtype=np.float32)

    if not is_enabled():
        return empty, 24000

    text = (text or "").strip()
    if not text:
        return empty, 24000

    if not ref_wav_path or not Path(ref_wav_path).is_file():
        return empty, 24000

    model = _get_model()
    if model is None:
        return empty, 24000

    lang_id = (target_lang or "en").strip().lower()[:2]  # for logging only
    exag    = exaggeration if exaggeration is not None else _exaggeration()
    cfg     = cfg_weight   if cfg_weight   is not None else _cfg_weight()
    temp    = temperature  if temperature  is not None else _temperature()
    max_ch  = _chunk_chars()

    chunks = _split_text(text, max_ch)
    all_wavs: list[np.ndarray] = []
    sr = model.sr

    # chatterbox-tts 0.1.x: no language_id param — model handles language from text
    print(
        f"[chatterbox] Synthesising {len(chunks)} chunk(s)  "
        f"lang={lang_id}  exag={exag:.2f}  cfg={cfg:.2f}  temp={temp:.2f}"
    )

    import torch

    from services.gpu_exclusive import gpu_exclusive

    with gpu_exclusive():
        for idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            try:
                with torch.inference_mode():
                    wav_tensor = model.generate(
                        text=chunk,
                        audio_prompt_path=ref_wav_path,
                        exaggeration=exag,
                        cfg_weight=cfg,
                        temperature=temp,
                    )
                if hasattr(wav_tensor, "numpy"):
                    wav_np = wav_tensor.squeeze().cpu().float().numpy()
                else:
                    wav_np = np.array(wav_tensor, dtype=np.float32).squeeze()

                if wav_np.ndim == 0 or wav_np.size == 0:
                    print(f"[chatterbox] Chunk {idx+1}: empty output")
                    continue

                all_wavs.append(wav_np)

            except Exception as exc:
                print(f"[chatterbox] Chunk {idx+1} failed: {exc!r}")
                return empty, 24000

        if _device() == "mps":
            try:
                torch.mps.synchronize()
            except Exception:
                pass

    if not all_wavs:
        return empty, 24000

    # Concatenate chunks with a short gap (configurable) — tighter than 120 ms for dubs
    gap_sec = _chunk_gap_sec()
    silence = np.zeros(int(sr * gap_sec), dtype=np.float32) if gap_sec > 0 else np.array([], dtype=np.float32)
    combined = all_wavs[0]
    for w in all_wavs[1:]:
        if silence.size:
            combined = np.concatenate([combined, silence, w])
        else:
            combined = np.concatenate([combined, w])

    # Normalise to ~-1 dBFS peak, then gentle loudness ballpark to reference
    peak = np.abs(combined).max()
    if peak > 0.0:
        combined = (combined * (0.891 / peak)).astype(np.float32)
    if _ref_rms_match_enabled():
        combined = _align_rms_to_reference(combined, sr, ref_wav_path)

    combined = combined.astype(np.float32)
    print(
        f"[chatterbox] Done: {len(combined)/sr:.2f}s @ {sr}Hz  "
        f"({len(all_wavs)} chunk(s))"
    )
    return combined, sr


# ---------------------------------------------------------------------------
# Pre-warm
# ---------------------------------------------------------------------------

def warmup() -> bool:
    """
    Pre-load the Chatterbox model so the first pipeline job is fast.
    Returns True if model loaded successfully.
    """
    if not is_enabled():
        return False
    return _get_model() is not None
