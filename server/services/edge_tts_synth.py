"""
Microsoft Edge neural TTS via **edge-tts** (no API key) +
**Coqui XTTS v2 professional voice cloning** (local, no API, zero-shot).

Voice cloning path — ``synthesize_with_voice_clone()``:
  1. ``prepare_reference_audio()`` — find the densest-speech window in the ref clip,
     apply mild noise reduction, peak-normalise to −3 dBFS.
  2. XTTS v2 synthesis with quality-tuned settings
     (temperature, repetition_penalty, top_k / top_p all configurable via env).
  3. Post-processing: RMS loudness-match to reference, noise-gate, 10 ms edge fades.
  4. Automatic fallback to Edge TTS when cloning fails or reference is unavailable.

Edge TTS path (Hindi default / no reference):
  Gender-aware voice selection:
    male   → ``hi-IN-MadhurNeural``
    female → ``hi-IN-SwaraNeural``
    unknown → ``hi-IN-MadhurNeural``

Env (cloning):
  ``CLONE_TEMPERATURE``        — XTTS temperature        (default ``0.65``)
  ``CLONE_REPETITION_PENALTY`` — XTTS repetition penalty (default ``10.0``)
  ``CLONE_TOP_K``              — XTTS top-k              (default ``50``)
  ``CLONE_TOP_P``              — XTTS top-p              (default ``0.85``)
  ``CLONE_REF_TARGET_SR``      — reference WAV sample rate for XTTS (default ``24000``)
  ``CLONE_LOUDNESS_MATCH``     — ``1`` to RMS-match output to reference (default ``1``)
  ``CLONE_NOISE_GATE_DB``      — noise-gate floor in dBFS (default ``-50``)

Env (Edge TTS):
  ``VOICE_MALE`` / ``VOICE_FEMALE`` / ``VOICE_DEFAULT`` — Hindi Edge voice overrides.
  ``EDGE_TTS_VOICE_HI`` — optional fallback when gender-specific env is unset.
  ``EDGE_TTS_RATE`` / ``EDGE_TTS_PITCH`` — SSML-style overrides.
  When ``EDGE_TTS_RATE`` is unset, **Arabic** voices default to ``-5%``; **English** rates
  come from ``services.tts_clarity`` (``TTS_MAXIMUM_CLARITY``, ``EDGE_TTS_EN_RATE*``).
"""
from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

import numpy as np
from pydub import AudioSegment

DEFAULT_HI_VOICE = "hi-IN-SwaraNeural"
DEFAULT_VOICE_MALE_HI = "hi-IN-MadhurNeural"
DEFAULT_VOICE_FEMALE_HI = "hi-IN-SwaraNeural"
DEFAULT_VOICE_NEUTRAL_HI = "hi-IN-MadhurNeural"

# Comprehensive Edge TTS voice map for all supported languages.
# Each entry: {"male": ..., "female": ..., "neutral": ...}
# Voices are Microsoft Cognitive Services Neural voices (free via edge-tts, no API key).
_EDGE_VOICES: dict[str, dict[str, str]] = {
    "en": {
        "male":    "en-US-ChristopherNeural",   # deep, broadcast quality
        "female":  "en-US-JennyNeural",
        # Davis: calm, high intelligibility for dense translated lines (better than Guy for MT dub).
        "neutral": "en-US-DavisNeural",
    },
    "hi": {
        "male":    "hi-IN-MadhurNeural",
        "female":  "hi-IN-SwaraNeural",
        "neutral": "hi-IN-MadhurNeural",
    },
    "es": {
        "male":    "es-ES-AlvaroNeural",
        "female":  "es-ES-ElviraNeural",
        "neutral": "es-ES-AlvaroNeural",
    },
    "fr": {
        "male":    "fr-FR-HenriNeural",
        "female":  "fr-FR-DeniseNeural",
        "neutral": "fr-FR-HenriNeural",
    },
    "de": {
        "male":    "de-DE-ConradNeural",
        "female":  "de-DE-KatjaNeural",
        "neutral": "de-DE-ConradNeural",
    },
    "pt": {
        "male":    "pt-BR-AntonioNeural",
        "female":  "pt-BR-FranciscaNeural",
        "neutral": "pt-BR-AntonioNeural",
    },
    "ja": {
        "male":    "ja-JP-KeitaNeural",
        "female":  "ja-JP-NanamiNeural",
        "neutral": "ja-JP-KeitaNeural",
    },
    "ko": {
        "male":    "ko-KR-InJoonNeural",
        "female":  "ko-KR-SunHiNeural",
        "neutral": "ko-KR-InJoonNeural",
    },
    "zh": {
        "male":    "zh-CN-YunxiNeural",
        "female":  "zh-CN-XiaoxiaoNeural",
        "neutral": "zh-CN-YunxiNeural",
    },
    "ar": {
        "male":    "ar-SA-HamedNeural",
        "female":  "ar-SA-ZariyahNeural",
        "neutral": "ar-SA-HamedNeural",
    },
    "ru": {
        "male":    "ru-RU-DmitryNeural",
        "female":  "ru-RU-SvetlanaNeural",
        "neutral": "ru-RU-DmitryNeural",
    },
    "it": {
        "male":    "it-IT-DiegoNeural",
        "female":  "it-IT-ElsaNeural",
        "neutral": "it-IT-DiegoNeural",
    },
    "nl": {
        "male":    "nl-NL-MaartenNeural",
        "female":  "nl-NL-ColetteNeural",
        "neutral": "nl-NL-MaartenNeural",
    },
    "tr": {
        "male":    "tr-TR-AhmetNeural",
        "female":  "tr-TR-EmelNeural",
        "neutral": "tr-TR-AhmetNeural",
    },
    "pl": {
        "male":    "pl-PL-MarekNeural",
        "female":  "pl-PL-ZofiaNeural",
        "neutral": "pl-PL-MarekNeural",
    },
    "sv": {
        "male":    "sv-SE-MattiasNeural",
        "female":  "sv-SE-SofieNeural",
        "neutral": "sv-SE-MattiasNeural",
    },
}


def _edge_tts_enabled() -> bool:
    return os.environ.get("TTS_USE_EDGE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def edge_voice_for_language_and_gender(language: str, gender: str | None) -> str | None:
    """
    Return the best Edge TTS voice ID for ``language`` + detected ``gender``.

    Precedence (Hindi): ``VOICE_MALE`` / ``VOICE_FEMALE`` / ``VOICE_DEFAULT`` env → map default.
    Precedence (other): ``EDGE_TTS_VOICE_<LANG_UPPER>`` env (e.g. ``EDGE_TTS_VOICE_EN``) → map default.
    Returns ``None`` when Edge TTS is disabled or the language is not in the voice map.
    """
    if not _edge_tts_enabled():
        return None
    lang = (language or "en").lower().strip().split("-")[0]
    voices = _EDGE_VOICES.get(lang)
    if not voices:
        return None
    g = (gender or "").strip().lower()

    # Hindi has legacy per-gender env overrides
    if lang == "hi":
        edge_hi = os.environ.get("EDGE_TTS_VOICE_HI", "").strip()
        if g == "male":
            return os.environ.get("VOICE_MALE", "").strip() or edge_hi or voices["male"]
        if g == "female":
            return os.environ.get("VOICE_FEMALE", "").strip() or edge_hi or voices["female"]
        return os.environ.get("VOICE_DEFAULT", "").strip() or edge_hi or voices["neutral"]

    # Generic per-language env override: EDGE_TTS_VOICE_EN, EDGE_TTS_VOICE_ES, …
    env_key = f"EDGE_TTS_VOICE_{lang.upper()}"
    env_override = os.environ.get(env_key, "").strip()
    if env_override:
        return env_override
    if g == "male":
        return voices["male"]
    if g == "female":
        return voices["female"]
    return voices["neutral"]


def hindi_edge_voice_for_gender(gender: str | None) -> str | None:
    """Hindi-only back-compat wrapper — delegates to edge_voice_for_language_and_gender."""
    return edge_voice_for_language_and_gender("hi", gender)


def edge_tts_voice_for_language(target_language: str, gender: str | None = None) -> str | None:
    """
    Return Edge voice id for ``target_language``, or ``None`` to use local XTTS/MMS.

    Now supports ALL languages in ``_EDGE_VOICES`` (was Hindi-only before).
    """
    return edge_voice_for_language_and_gender(target_language, gender)


async def list_edge_voices_async() -> list[dict]:
    import edge_tts

    return await edge_tts.list_voices()


def list_edge_voices() -> list[dict]:
    """Sync wrapper for tests / CLI (requires network)."""
    return _run_async(list_edge_voices_async())


def _run_async(coro):
    """Run coroutine from sync code (worker thread has no running loop)."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            try:
                loop.close()
            except Exception:
                pass
            asyncio.set_event_loop(None)


async def _save_mp3_async(text: str, voice: str, mp3_path: str) -> None:
    import edge_tts

    rate = os.environ.get("EDGE_TTS_RATE", "").strip()  # e.g. "+0%"
    pitch = os.environ.get("EDGE_TTS_PITCH", "").strip()  # e.g. "+0Hz"
    # Arabic neural lines are dense; a hair slower default reads clearer when env unset.
    if not rate and voice.lower().startswith("ar-"):
        rate = "-5%"
    # English (especially MT): slower default improves intelligibility (see ``tts_clarity``).
    if not rate and voice.lower().startswith("en-"):
        try:
            from services.tts_clarity import english_edge_rate_when_unset

            rate = english_edge_rate_when_unset()
        except ImportError:
            rate = "-16%"
    if rate or pitch:
        comm = edge_tts.Communicate(text, voice, rate=rate or None, pitch=pitch or None)
    else:
        comm = edge_tts.Communicate(text, voice)
    await comm.save(mp3_path)


def _edge_tts_synthesize_once(text: str, voice: str, mp3_path: str) -> tuple[np.ndarray, int]:
    """Single synthesis attempt; cleans up mp3_path regardless of outcome."""
    try:
        _run_async(_save_mp3_async(text, voice, mp3_path))
        if not Path(mp3_path).is_file() or Path(mp3_path).stat().st_size < 64:
            return np.array([], dtype=np.float32), 24000
        audio = AudioSegment.from_mp3(mp3_path)
        audio = audio.set_channels(1)
        sr = int(audio.frame_rate)
        sw = audio.sample_width
        arr = np.array(audio.get_array_of_samples(), dtype=np.float32)
        maxv = float(2 ** (8 * sw - 1))
        return (arr / maxv).astype(np.float32), sr
    finally:
        Path(mp3_path).unlink(missing_ok=True)


def synthesize_hindi_to_numpy(text: str, voice: str, work_dir: str) -> tuple[np.ndarray, int]:
    """
    Synthesize Hindi (or any Edge-supported voice) to mono float32 PCM + sample rate.
    Full natural length — no speed change, trim, or pad.

    Retries once on transient network errors (BrokenPipeError, OSError) that
    can occur when the Microsoft Edge TTS WebSocket is reset mid-stream.
    """
    text = (text or "").strip()
    if not text:
        return np.array([], dtype=np.float32), 24000

    Path(work_dir).mkdir(parents=True, exist_ok=True)

    last_exc: Exception | None = None
    for attempt in range(2):
        mp3_path = str(Path(work_dir) / f"edge_{uuid.uuid4().hex}.mp3")
        try:
            return _edge_tts_synthesize_once(text, voice, mp3_path)
        except (BrokenPipeError, ConnectionError, OSError) as exc:
            last_exc = exc
            if attempt == 0:
                print(f"[edge-tts] network error, retrying: {exc}")
                import time as _time
                _time.sleep(1.5)
        # non-network exceptions propagate immediately (no retry)

    raise last_exc or RuntimeError("edge-tts synthesis failed after retries")


def synthesize_hindi_for_subtitle_slot(
    text: str,
    voice: str,
    work_dir: str,
    target_duration_sec: float,
    *,
    clarity_first: bool,
) -> tuple[np.ndarray, int]:
    """
    Back-compat name: always returns natural-length audio.
    ``target_duration_sec`` and ``clarity_first`` are ignored (no timing fit).
    """
    del target_duration_sec, clarity_first
    return synthesize_hindi_to_numpy(text, voice, work_dir)


# ══════════════════════════════════════════════════════════════════════════════
# Professional Voice Cloning  (XTTS v2 + reference preprocessing + post-proc)
# ══════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Env knobs
# ---------------------------------------------------------------------------

def _clone_temperature() -> float:
    try:
        return float(os.environ.get("CLONE_TEMPERATURE", "0.65") or "0.65")
    except ValueError:
        return 0.65


def _clone_repetition_penalty() -> float:
    try:
        return float(os.environ.get("CLONE_REPETITION_PENALTY", "10.0") or "10.0")
    except ValueError:
        return 10.0


def _clone_top_k() -> int:
    try:
        return max(1, int(os.environ.get("CLONE_TOP_K", "50") or "50"))
    except ValueError:
        return 50


def _clone_top_p() -> float:
    try:
        return float(os.environ.get("CLONE_TOP_P", "0.85") or "0.85")
    except ValueError:
        return 0.85


def _clone_ref_sr() -> int:
    try:
        return max(16000, min(48000, int(os.environ.get("CLONE_REF_TARGET_SR", "24000") or "24000")))
    except ValueError:
        return 24000


def _clone_loudness_match_enabled() -> bool:
    return os.environ.get("CLONE_LOUDNESS_MATCH", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )


def _clone_noise_gate_db() -> float:
    try:
        return float(os.environ.get("CLONE_NOISE_GATE_DB", "-50") or "-50")
    except ValueError:
        return -50.0


# ---------------------------------------------------------------------------
# Reference audio preprocessing
# ---------------------------------------------------------------------------

def _best_speech_window(
    voiced_flags: list,
    samples_per_frame: int,
    target_samples: int,
    total_samples: int,
) -> tuple[int, int]:
    """
    Slide a window and return (start_sample, end_sample) for the segment
    with the highest density of voiced frames.  Better than just taking the
    first speech run — avoids intro music or room-noise segments.
    """
    n = len(voiced_flags)
    win = max(1, target_samples // samples_per_frame)
    if n <= win:
        return 0, total_samples

    best_i = 0
    best_c = sum(voiced_flags[:win])
    running = best_c
    for i in range(1, n - win + 1):
        running -= int(voiced_flags[i - 1])
        running += int(voiced_flags[i + win - 1])
        if running > best_c:
            best_c = running
            best_i = i

    s0 = best_i * samples_per_frame
    s1 = min(total_samples, s0 + target_samples)
    return s0, s1


def prepare_reference_audio(
    wav_path: str,
    work_dir: str,
    *,
    target_duration_sec: float = 10.0,
    target_sr: int | None = None,
) -> str | None:
    """
    Preprocess a speaker reference WAV so XTTS v2 gets the cleanest possible
    voice sample.

    Steps
    -----
    1. Load + mono-mix at native sample rate.
    2. VAD (webrtcvad mode 2) to find voiced frames.
    3. Slide a window → pick the densest-speech segment of ``target_duration_sec``.
    4. Mild noise reduction (noisereduce, optional — skipped if unavailable).
    5. Peak-normalise to −3 dBFS.
    6. Resample to ``target_sr`` and write to a temp WAV.

    Returns path to the prepared WAV, or ``None`` on any failure (caller falls
    back to the original reference or Edge TTS).
    """
    import tempfile
    import uuid as _uuid

    try:
        import librosa
        import soundfile as sf
        import webrtcvad
    except ImportError as exc:
        print(f"[voice-clone] prepare_reference_audio: missing dep ({exc}); using raw ref")
        return None

    if target_sr is None:
        target_sr = _clone_ref_sr()

    src = Path(wav_path)
    if not src.is_file():
        return None

    try:
        y, sr = librosa.load(str(src), sr=None, mono=True)
    except Exception as exc:
        print(f"[voice-clone] librosa.load failed: {exc}")
        return None

    if y.size < int(sr * 1.0):
        return None

    # Normalise for VAD
    peak = float(np.max(np.abs(y)))
    if peak > 1e-6:
        y = (y / peak).astype(np.float32)

    # VAD at 16 kHz (webrtcvad only supports 8 / 16 / 32 / 48 kHz, 10/20/30 ms)
    y16 = librosa.resample(y, orig_sr=sr, target_sr=16000)
    frame_ms = 30
    spf = 480  # 30 ms @ 16 kHz
    y_int = (np.clip(y16, -1.0, 1.0) * 32767.0).astype(np.int16)
    pcm = y_int.tobytes()
    frame_bytes = spf * 2
    flags: list[bool] = []
    try:
        vad = webrtcvad.Vad(2)
        for off in range(0, len(y_int) - spf + 1, spf):
            chunk = pcm[off * 2: off * 2 + frame_bytes]
            try:
                flags.append(vad.is_speech(chunk, 16000))
            except Exception:
                flags.append(False)
    except Exception as exc:
        print(f"[voice-clone] VAD failed: {exc}; treating all frames as voiced")
        flags = [True] * max(1, len(y16) // spf)

    # Pick the best speech window
    need16 = int(16000 * target_duration_sec)
    s0_16, s1_16 = _best_speech_window(flags, spf, need16, len(y16))

    # Map window back to original sample-rate indices
    ratio = sr / 16000.0
    s0 = int(s0_16 * ratio)
    s1 = min(len(y), int(s1_16 * ratio))
    if s1 - s0 < int(sr * 2.0):
        print("[voice-clone] Best speech window too short; using full audio")
        s0, s1 = 0, len(y)

    clip = y[s0:s1].copy()

    # Very gentle stationary noise reduction — ONLY to lift the noise floor,
    # not to reshape the voice.  prop_decrease=0.15 preserves all voice harmonics.
    # stationary=True avoids the over-subtraction artefacts of non-stationary NR.
    try:
        import noisereduce as nr
        clip = nr.reduce_noise(
            y=clip, sr=sr, stationary=True, prop_decrease=0.15
        ).astype(np.float32)
    except Exception:
        pass

    # Peak-normalise to −3 dBFS
    peak_out = float(np.max(np.abs(clip)))
    if peak_out > 1e-6:
        target_peak = 10 ** (-3.0 / 20.0)   # ≈ 0.708
        clip = (clip / peak_out * target_peak).astype(np.float32)

    # Resample to target_sr
    if sr != target_sr:
        clip = librosa.resample(clip, orig_sr=sr, target_sr=target_sr)

    clip = np.clip(clip.astype(np.float32), -1.0, 1.0)

    Path(work_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(work_dir) / f"_clone_ref_{_uuid.uuid4().hex[:8]}.wav")
    try:
        sf.write(out_path, clip, target_sr, subtype="PCM_16")
        print(
            f"[voice-clone] Prepared reference: {len(clip) / target_sr:.2f}s @ {target_sr} Hz "
            f"(window {s0/sr:.1f}s–{s1/sr:.1f}s of source)"
        )
        return out_path
    except Exception as exc:
        print(f"[voice-clone] Could not write prepared ref: {exc}")
        return None


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def _rms_loudness_match(output: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Scale ``output`` so its RMS matches ``reference`` RMS.
    Scale is clamped to [0.5, 1.5] — tight range so we never amplify noise
    substantially above what XTTS already produced.
    """
    if output.size == 0 or reference.size == 0:
        return output
    ref_rms = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2)))
    out_rms = float(np.sqrt(np.mean(output.astype(np.float64) ** 2)))
    if ref_rms < 1e-9 or out_rms < 1e-9:
        return output
    scale = max(0.5, min(1.5, ref_rms / out_rms))
    return np.clip(output.astype(np.float64) * scale, -1.0, 1.0).astype(np.float32)


def _trim_silence(wav: np.ndarray, sr: int, db_threshold: float = -50.0) -> np.ndarray:
    """
    Strip leading and trailing silence below ``db_threshold`` dBFS.
    Uses 20 ms frames; keeps at least 50 ms of audio.

    XTTS sometimes pads output with several seconds of near-silence, making
    dubbed segments far longer than the source — this removes that padding.

    Leading trim is capped (``TRIM_SILENCE_MAX_LEAD_MS``, default 120 ms): soft consonants
    or language onsets can sit below the RMS threshold for the first frames; trimming
    past the cap would clip real speech.
    """
    if wav.size == 0:
        return wav
    frame = max(1, int(sr * 0.02))   # 20 ms
    threshold = 10 ** (db_threshold / 20.0)
    try:
        max_lead_ms = float(os.environ.get("TRIM_SILENCE_MAX_LEAD_MS", "120") or "120")
    except ValueError:
        max_lead_ms = 120.0
    max_lead_ms = max(0.0, min(400.0, max_lead_ms))
    max_lead_samples = int(sr * (max_lead_ms / 1000.0))

    # Find first voiced frame from the start
    start_frame = 0
    for i in range(0, len(wav) - frame + 1, frame):
        rms = float(np.sqrt(np.mean(wav[i:i + frame].astype(np.float64) ** 2)))
        if rms >= threshold:
            start_frame = i
            break
    if max_lead_samples > 0 and start_frame > max_lead_samples:
        start_frame = 0

    # Find last voiced frame from the end
    end_frame = len(wav)
    for i in range(len(wav) - frame, -1, -frame):
        if i < 0:
            break
        rms = float(np.sqrt(np.mean(wav[i:i + frame].astype(np.float64) ** 2)))
        if rms >= threshold:
            end_frame = i + frame
            break

    min_samples = max(frame, int(sr * 0.05))   # keep at least 50 ms
    if end_frame - start_frame < min_samples:
        return wav
    return wav[start_frame:end_frame].copy()


def _fade_edges(wav: np.ndarray, sr: int, fade_ms: int = 10) -> np.ndarray:
    """Linear fade-in / fade-out to eliminate clicks at segment boundaries."""
    n = max(1, int(sr * fade_ms / 1000))
    if wav.size < n * 2:
        return wav
    out = wav.copy()
    ramp_in  = np.linspace(0.0, 1.0, n, dtype=np.float32)
    ramp_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
    out[:n] *= ramp_in
    out[-n:] *= ramp_out
    return out


def _apply_noise_gate(wav: np.ndarray, sr: int) -> np.ndarray:
    """
    Soft noise gate: 20 ms frames below CLONE_NOISE_GATE_DB get smooth gain
    reduction toward silence instead of being hard-zeroed.  A 6 dB knee and
    a ratio of 20:1 mimic a professional downward expander — XTTS inter-word
    hiss is attenuated without the click artefacts of hard zeroing.
    """
    threshold = 10 ** (_clone_noise_gate_db() / 20.0)
    knee_width = threshold * 2.0   # 6 dB knee above threshold
    frame = max(1, int(sr * 0.02))   # 20 ms
    out = wav.copy()
    for i in range(0, len(out) - frame + 1, frame):
        rms = float(np.sqrt(np.mean(out[i:i + frame].astype(np.float64) ** 2)))
        if rms < threshold:
            # Hard floor: gain → ~0.05 (−26 dBFS attenuation)
            gain = max(0.05, rms / threshold) ** 4
            out[i:i + frame] = out[i:i + frame] * gain
        elif rms < knee_width:
            # Soft knee: blend from full reduction toward unity
            blend = (rms - threshold) / (knee_width - threshold)  # 0..1
            gain = 0.05 + blend * 0.95
            out[i:i + frame] = out[i:i + frame] * gain
    return out


# ---------------------------------------------------------------------------
# XTTS synthesis core (lazy import to avoid circular dependency)
# ---------------------------------------------------------------------------

def _xtts_clone_to_numpy(
    text: str,
    language: str,
    speaker_wav: str,
) -> tuple[np.ndarray, int]:
    """
    Synthesise ``text`` with Coqui XTTS v2, cloning the voice from
    ``speaker_wav``.  Uses the TTSService singleton so the model is only
    loaded once across the whole pipeline.

    Quality settings (all tunable via env):
      temperature=0.65  — stable, consistent output (lower = less noise/artifacts)
      repetition_penalty=10.0 — strongly suppresses looping / word repetition
      top_k=50, top_p=0.85   — nucleus sampling for expressive but stable output
    """
    import tempfile
    import soundfile as sf

    # Late import — avoids circular dependency (tts_service imports edge_tts_synth).
    from services.tts_service import get_tts_service, XTTS_LANGUAGE_MAP

    def _norm(code: str) -> str:
        return (code or "en").lower().strip().split("-")[0]

    lang = XTTS_LANGUAGE_MAP.get(_norm(language), "en")
    svc = get_tts_service()
    xtts = svc._ensure_xtts()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        xkw: dict = {
            "temperature": _clone_temperature(),
            "repetition_penalty": _clone_repetition_penalty(),
            "top_k": _clone_top_k(),
            "top_p": _clone_top_p(),
            "length_penalty": 1.0,
        }
        try:
            xtts.tts_to_file(
                text=text,
                language=lang,
                file_path=tmp_path,
                speaker_wav=speaker_wav,
                split_sentences=False,   # one unit → no mid-sentence voice drift
                **xkw,
            )
        except TypeError:
            # Older Coqui build that doesn't accept all kwargs — retry bare
            xtts.tts_to_file(
                text=text,
                language=lang,
                file_path=tmp_path,
                speaker_wav=speaker_wav,
                split_sentences=False,
            )

        data, sr = sf.read(tmp_path, dtype="float32", always_2d=True)
        wav = data.mean(axis=1) if data.shape[1] > 1 else data.squeeze()
        return wav.astype(np.float32), int(sr)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _edge_tts_fallback(
    text: str,
    language: str,
    gender: str | None,
    work_dir: str,
) -> tuple[np.ndarray, int]:
    """Edge TTS synthesis used as quality-assured fallback for any supported language."""
    voice = edge_voice_for_language_and_gender(language, gender)
    if voice:
        try:
            # synthesize_hindi_to_numpy works for any Edge-supported language/voice
            return synthesize_hindi_to_numpy(text, voice, work_dir)
        except Exception as exc:
            print(f"[voice-clone] Edge TTS fallback failed for {language!r}: {exc}")
    return np.array([], dtype=np.float32), 24000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def synthesize_with_voice_clone(
    text: str,
    speaker_wav: str,
    language: str,
    work_dir: str,
    gender: str | None = None,
    *,
    fallback_to_edge: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Professional zero-shot voice cloning synthesis.

    Pipeline
    --------
    1. Validate reference WAV.
    2. ``prepare_reference_audio()`` — densest-speech window, denoise, normalise.
    3. ``_xtts_clone_to_numpy()`` — XTTS v2 with quality-tuned settings.
    4. Post-processing:
       - RMS loudness-match output to reference energy.
       - Noise-gate (remove XTTS hiss floor between words).
       - 10 ms fade-in / fade-out (remove click artefacts).
    5. Fallback to Edge TTS if any step fails or reference is unusable.

    Parameters
    ----------
    text          : Text to synthesise in ``language``.
    speaker_wav   : Path to reference voice WAV (≥ 3 s of clean speech).
    language      : BCP-47 language code, e.g. ``"hi"``, ``"en"``, ``"es"``.
    work_dir      : Directory for temporary files (cleaned up internally).
    gender        : Detected speaker gender — used only for Edge TTS fallback
                    voice selection (``"male"`` / ``"female"`` / ``None``).
    fallback_to_edge : If ``True`` (default), fall back to Edge TTS on failure.

    Returns
    -------
    (wav_float32, sample_rate)
    """
    text = (text or "").strip()
    if not text:
        return np.array([], dtype=np.float32), 24000

    Path(work_dir).mkdir(parents=True, exist_ok=True)

    # ── 1. Validate reference ──────────────────────────────────────────────
    ref_src = _validate_ref_wav(speaker_wav)
    if ref_src is None:
        print(f"[voice-clone] No valid reference WAV at {speaker_wav!r}")
        if fallback_to_edge:
            return _edge_tts_fallback(text, language, gender, work_dir)
        return np.array([], dtype=np.float32), 24000

    # ── 2. Preprocess reference ────────────────────────────────────────────
    prepared_ref = prepare_reference_audio(ref_src, work_dir)
    effective_ref = prepared_ref or ref_src   # use prepared if available
    print(
        f"[voice-clone] Cloning voice from {Path(effective_ref).name!r} "
        f"lang={language!r} temp={_clone_temperature()} rep_pen={_clone_repetition_penalty()}"
    )

    # ── 3. XTTS synthesis ──────────────────────────────────────────────────
    # Pre-load the reference audio NOW — before the finally block deletes the
    # prepared temp file — so loudness matching still works after cleanup.
    ref_mono: np.ndarray | None = None
    if _clone_loudness_match_enabled():
        try:
            import soundfile as sf
            ref_data, _ = sf.read(effective_ref, dtype="float32", always_2d=True)
            ref_mono = (
                ref_data.mean(axis=1) if ref_data.shape[1] > 1 else ref_data.squeeze()
            ).astype(np.float32)
        except Exception as exc:
            print(f"[voice-clone] Could not preload reference for loudness match: {exc}")

    wav: np.ndarray = np.array([], dtype=np.float32)
    sr: int = 24000
    try:
        wav, sr = _xtts_clone_to_numpy(text, language, effective_ref)
    except Exception as exc:
        print(f"[voice-clone] XTTS failed: {exc!r}")
    finally:
        # Safe to delete now — ref_mono already holds the reference data
        if prepared_ref and prepared_ref != ref_src:
            Path(prepared_ref).unlink(missing_ok=True)

    if wav.size == 0:
        print("[voice-clone] XTTS returned empty audio")
        if fallback_to_edge:
            return _edge_tts_fallback(text, language, gender, work_dir)
        return np.array([], dtype=np.float32), 24000

    # Trim leading/trailing silence that XTTS sometimes pads into output,
    # which can make a 5-second sentence appear as 25+ seconds of audio.
    before_trim = len(wav) / sr
    wav = _trim_silence(wav, sr)
    after_trim = len(wav) / sr
    if before_trim - after_trim > 0.5:
        print(
            f"[voice-clone] Trimmed {before_trim - after_trim:.2f}s silence "
            f"({before_trim:.2f}s → {after_trim:.2f}s)"
        )

    # ── 4. Post-processing ────────────────────────────────────────────────
    if ref_mono is not None and ref_mono.size > 0:
        wav = _rms_loudness_match(wav, ref_mono)
    elif _clone_loudness_match_enabled():
        print("[voice-clone] Loudness match skipped: reference not preloaded")

    wav = _apply_noise_gate(wav, sr)
    wav = _fade_edges(wav, sr)

    print(
        f"[voice-clone] Done: {len(wav) / sr:.2f}s @ {sr} Hz "
        f"(text len={len(text)} chars)"
    )
    return wav.astype(np.float32), sr


def _validate_ref_wav(path: str | None) -> str | None:
    """Return the path if it points to a usable WAV file, else None."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        if p.stat().st_size < 2048:   # too small to be a real speech clip
            return None
    except OSError:
        return None
    return str(p)
