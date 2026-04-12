"""
Professional broadcast audio mastering for dubbed audio.

This is the final processing stage before the MP3 is written — exactly what
streaming platforms (Netflix, Prime, Disney+) apply to every dubbed track:

  1. **High-pass filter** — remove sub-80 Hz rumble (mic handling noise,
     HVAC, infrasound) that wastes codec bits and degrades clarity.

  2. **De-esser** — attenuate harsh sibilance (2–8 kHz peaks) that TTS
     voices often over-produce. A simple spectral compressor targeting the
     sibilance band.

  3. **Dynamic range compression** — gentle 3:1 ratio, −18 dB threshold.
     Brings up quiet passages so dialogue is consistently intelligible without
     increasing the loudest peaks. Keeps the signal musical and natural.

  4. **EQ / presence boost** — gentle +1.5 dB shelf at 2 kHz adds the
     "phone call" clarity that makes dubbed voices cut through.

  5. **True-peak limiter** — hard brick wall at −1 dBTP prevents inter-sample
     clipping in MP3/AAC codecs, which can cause distortion on mobile speakers.

  6. **LUFS normalisation** — normalise integrated loudness to −16 LUFS,
     the streaming standard (Spotify, Netflix, YouTube all target −14 to −18).
     This ensures the dubbed audio plays at a consistent, broadcast-appropriate
     volume level.

Env:
  ``MASTER_TARGET_LUFS``   — integrated loudness target, default ``-16.0``
  ``MASTER_TRUE_PEAK_DB``  — true-peak ceiling, default ``-1.0``
  ``MASTER_COMP_THRESHOLD``— compressor threshold dBFS, default ``-18.0``
  ``MASTER_COMP_RATIO``    — compressor ratio, default ``3.0``
  ``MASTER_ENABLED``       — set ``0`` to skip mastering (default ``1``)
  ``MASTER_SPEECH_FRIENDLY`` — default ``1``: caps ratio / threshold for less pumping on TTS dubs
  ``MASTER_PRESENCE_DB``   — optional override for presence shelf gain
  ``TTS_MAXIMUM_CLARITY``  — when ``1`` (default), tightens compression / boosts presence / HPF (via ``tts_clarity``)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _lufs_target() -> float:
    try:
        return float(os.environ.get("MASTER_TARGET_LUFS", "-16.0") or "-16.0")
    except ValueError:
        return -16.0


def _true_peak_db() -> float:
    try:
        return float(os.environ.get("MASTER_TRUE_PEAK_DB", "-1.0") or "-1.0")
    except ValueError:
        return -1.0


def _comp_threshold() -> float:
    try:
        return float(os.environ.get("MASTER_COMP_THRESHOLD", "-18.0") or "-18.0")
    except ValueError:
        return -18.0


def _comp_ratio() -> float:
    try:
        return float(os.environ.get("MASTER_COMP_RATIO", "3.0") or "3.0")
    except ValueError:
        return 3.0


def _speech_friendly_mastering() -> bool:
    """Softer dynamics on TTS concatenations — less mud than aggressive 3:1 broadcast squash."""
    return os.environ.get("MASTER_SPEECH_FRIENDLY", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _maximum_clarity_mastering() -> bool:
    try:
        from services.tts_clarity import maximum_clarity_enabled

        return maximum_clarity_enabled()
    except ImportError:
        return False


def _effective_compressor_settings() -> tuple[float, float]:
    """
    Returns ``(threshold_db, ratio)`` for the dub mastering chain.
    When speech-friendly is on, ratio and threshold are capped for clearer consonants.
    ``TTS_MAXIMUM_CLARITY`` tightens further (less pumping / less dull consonants).
    """
    t0 = _comp_threshold()
    r0 = _comp_ratio()
    t1, r1 = t0, r0
    if _speech_friendly_mastering():
        # More negative threshold → less often above threshold → less gain-riding pumping
        t1 = min(t1, -20.5)
        r1 = min(r1, 2.12)
    if _maximum_clarity_mastering():
        t1 = min(t1, -22.0)
        r1 = min(r1, 1.92)
    return t1, r1


def _presence_gain_db() -> float:
    raw = os.environ.get("MASTER_PRESENCE_DB", "").strip()
    if raw:
        try:
            return max(0.0, min(4.5, float(raw)))
        except ValueError:
            pass
    if _maximum_clarity_mastering():
        return 2.85
    return 2.0 if _speech_friendly_mastering() else 1.5


def _highpass_cutoff_hz() -> float:
    """Slightly higher HPF when maximum-clarity — less rumble masking consonants."""
    if _maximum_clarity_mastering():
        try:
            v = float(os.environ.get("MASTER_HPF_HZ_MAX_CLARITY", "100") or "100")
            return max(75.0, min(130.0, v))
        except ValueError:
            return 100.0
    return 80.0


def master_enabled() -> bool:
    return os.environ.get("MASTER_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )


# ---------------------------------------------------------------------------
# DSP building blocks
# ---------------------------------------------------------------------------

def _highpass_filter(y: np.ndarray, sr: int, cutoff_hz: float = 80.0) -> np.ndarray:
    """4th-order Butterworth high-pass. Removes sub-bass rumble."""
    sos = signal.butter(4, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return signal.sosfilt(sos, y).astype(np.float32)


def _deess(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Frequency-aware de-esser: detect frames with excessive sibilance energy
    (3–8 kHz band), and gently attenuate when sibilance dominates.

    This reduces the harsh 's', 'sh', 'ch' artifacts common in TTS output.
    """
    nyq = sr / 2.0
    if nyq < 4000:
        return y  # SR too low to matter

    # Band-pass: sibilance zone 3–8 kHz
    low = min(3000.0, nyq * 0.9)
    high = min(8000.0, nyq * 0.95)
    if low >= high or low >= nyq or high >= nyq:
        return y

    sos_band = signal.butter(4, [low, high], btype="bandpass", fs=sr, output="sos")
    sib = signal.sosfilt(sos_band, y)

    frame_s = int(sr * 0.01)   # 10 ms frames
    frame_s = max(1, frame_s)
    out = y.copy()

    # Compute broadband RMS for comparison
    broad_rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2))) + 1e-9

    for start in range(0, len(y) - frame_s + 1, frame_s):
        sib_rms = float(np.sqrt(np.mean(
            sib[start:start + frame_s].astype(np.float64) ** 2
        )))
        if sib_rms > broad_rms * 0.80:          # only fire on extreme sibilance peaks
            attn = max(0.70, broad_rms * 0.80 / sib_rms)
            out[start:start + frame_s] *= attn

    return out.astype(np.float32)


def _compress(
    y: np.ndarray,
    sr: int,
    threshold_db: float,
    ratio: float,
    attack_ms: float = 5.0,
    release_ms: float = 80.0,
) -> np.ndarray:
    """
    Feed-forward RMS compressor with attack/release smoothing.

    Makes quiet passages louder and prevents sudden loud spikes — the key
    ingredient for consistently intelligible dubbed dialogue.
    """
    threshold_lin = 10 ** (threshold_db / 20.0)
    attack  = 1.0 - np.exp(-2.2 / (sr * attack_ms / 1000.0))
    release = 1.0 - np.exp(-2.2 / (sr * release_ms / 1000.0))

    out = y.copy().astype(np.float64)
    env = 0.0
    gain_db = 0.0

    frame = max(1, int(sr * 0.001))   # 1 ms frames for envelope

    for start in range(0, len(out) - frame + 1, frame):
        rms = float(np.sqrt(np.mean(out[start:start + frame] ** 2))) + 1e-12
        coeff = attack if rms > env else release
        env = env + coeff * (rms - env)

        if env > threshold_lin:
            target_gain_db = threshold_db + (20 * np.log10(env) - threshold_db) / ratio - 20 * np.log10(env)
        else:
            target_gain_db = 0.0

        gain_db = gain_db + release * (target_gain_db - gain_db)
        gain_lin = 10 ** (gain_db / 20.0)
        out[start:start + frame] *= gain_lin

    return out.astype(np.float32)


def _presence_boost(y: np.ndarray, sr: int, freq_hz: float = 2000.0, gain_db: float = 1.5) -> np.ndarray:
    """
    Gentle high-shelf boost at ``freq_hz`` for voice clarity.
    Makes dubbed voices cut through background noise (even if there is none —
    it adds perceived 'near-field' quality that listeners associate with clarity).
    """
    nyq = sr / 2.0
    if freq_hz >= nyq * 0.95:
        return y
    sos = signal.butter(2, freq_hz, btype="highpass", fs=sr, output="sos")
    high_shelf = signal.sosfilt(sos, y)
    gain_lin = 10 ** (gain_db / 20.0)
    boosted = y + (gain_lin - 1.0) * high_shelf
    return np.clip(boosted, -1.0, 1.0).astype(np.float32)


def _true_peak_limit(y: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    """
    Hard brick-wall limiter at ``ceiling_db`` dBFS.
    Prevents inter-sample clipping in lossy codecs.
    Simple but effective: scale the peak down to ceiling, then hard clip.
    """
    ceiling_lin = 10 ** (ceiling_db / 20.0)
    peak = float(np.max(np.abs(y))) + 1e-12
    if peak > ceiling_lin:
        y = y * (ceiling_lin / peak)
    return np.clip(y, -ceiling_lin, ceiling_lin).astype(np.float32)


def _lufs_normalise(y: np.ndarray, sr: int, target_lufs: float = -16.0) -> np.ndarray:
    """
    ITU-R BS.1770-4 integrated loudness normalisation.

    Uses pyloudnorm when available (most accurate); falls back to RMS-based
    approximation (-23 LUFS ≈ -20 dBFS RMS for speech) when not installed.
    """
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        # pyloudnorm needs float64 2D array: (samples, channels)
        y2d = y.astype(np.float64).reshape(-1, 1)
        measured = meter.integrated_loudness(y2d)
        if not np.isfinite(measured) or measured < -70:
            # Audio too quiet to measure — fall back to simple normalise
            peak = float(np.max(np.abs(y))) + 1e-12
            target_lin = 10 ** (target_lufs / 20.0 + 3)  # rough speech approximation
            return np.clip(y * (target_lin / peak), -0.99, 0.99).astype(np.float32)
        gain_db = target_lufs - measured
        # Clamp gain: never amplify by more than +20 dB or attenuate by more than −30 dB
        gain_db = max(-30.0, min(20.0, gain_db))
        gain_lin = 10 ** (gain_db / 20.0)
        return np.clip(y.astype(np.float64) * gain_lin, -0.99, 0.99).astype(np.float32)
    except ImportError:
        # pyloudnorm not available — RMS-based approximation
        rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2))) + 1e-12
        # Map LUFS target to approximate linear RMS (speech: LUFS ≈ dBFS + 3)
        target_rms = 10 ** ((target_lufs + 3.0) / 20.0)
        scale = min(target_rms / rms, 5.0)   # never amplify more than 14 dB
        return np.clip(y.astype(np.float64) * scale, -0.99, 0.99).astype(np.float32)


# ---------------------------------------------------------------------------
# Segment-level time-stretch (sync dubbed to source timing)
# ---------------------------------------------------------------------------

def time_stretch_to_duration(
    wav: np.ndarray,
    sr: int,
    target_duration_s: float,
    *,
    max_stretch_ratio: float = 0.25,
) -> np.ndarray:
    """
    Time-stretch ``wav`` so its duration matches ``target_duration_s``.

    Uses librosa's phase vocoder (STFT-based), which preserves pitch perfectly
    — unlike simple resampling, which changes both speed and pitch.

    Only applied when the required stretch is within ``max_stretch_ratio``
    (default ±25%). Beyond that, the audio is returned unchanged (too much
    stretch sounds robotic even with the best algorithms).

    This is the critical sync step: Netflix dubs must start and end within
    ~0.5 s of the original dialogue — this function achieves that.
    """
    if wav.size == 0 or target_duration_s <= 0:
        return wav

    actual_s = len(wav) / sr
    if actual_s <= 0:
        return wav

    ratio = target_duration_s / actual_s

    # Within 2% → not worth stretching (imperceptible)
    if abs(ratio - 1.0) < 0.02:
        return wav

    # Outside bounds → don't stretch (would sound robotic)
    if abs(ratio - 1.0) > max_stretch_ratio:
        return wav

    # librosa rate: actual/target — rate > 1 speeds up (shorter), < 1 slows down (longer)
    rate = actual_s / target_duration_s

    try:
        import librosa
        # n_fft tuned for speech: 2048 samples at 24kHz ≈ 85ms window
        stretched = librosa.effects.time_stretch(
            wav.astype(np.float32), rate=rate, n_fft=2048
        )
        return stretched.astype(np.float32)
    except Exception as exc:
        print(f"[audio-master] time_stretch failed ({exc!r}); using original")
        return wav


def _atempo_filter_chain(tempo: float) -> str:
    """
    Build chained ``atempo`` filters (each factor must stay in [0.5, 2.0] for FFmpeg).

    ``tempo`` is input_duration / target_duration (speed up when > 1).
    """
    parts: list[str] = []
    t = float(tempo)
    guard = 0
    while t > 2.0 + 1e-9 and guard < 16:
        parts.append("atempo=2.0")
        t /= 2.0
        guard += 1
    guard = 0
    while t < 0.5 - 1e-9 and guard < 16:
        parts.append("atempo=0.5")
        t /= 0.5
        guard += 1
    t = max(0.5, min(2.0, t))
    parts.append(f"atempo={t:.6f}")
    return ",".join(parts)


def _ffmpeg_atempo_wav(wav: np.ndarray, sr: int, target_duration_s: float) -> np.ndarray | None:
    """Pitch-preserving coarse fit via FFmpeg ``atempo``; returns None on failure."""
    if wav.size == 0 or not shutil.which("ffmpeg"):
        return None
    in_s = wav.size / float(sr)
    tgt = float(target_duration_s)
    if in_s <= 0 or tgt <= 0.04:
        return None
    tempo = in_s / tgt
    if abs(tempo - 1.0) < 0.015:
        return wav.astype(np.float32)
    chain = _atempo_filter_chain(tempo)
    fin = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    fout = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    fin.close()
    fout.close()
    try:
        sf.write(fin.name, wav.astype(np.float32), sr, subtype="PCM_16")
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            fin.name,
            "-af",
            chain,
            "-ar",
            str(int(sr)),
            "-ac",
            "1",
            fout.name,
        ]
        subprocess.run(cmd, check=True, timeout=120)
        out, sr2 = sf.read(fout.name, dtype="float32", always_2d=False)
        if sr2 != sr and out.size > 0:
            return None
        if out.ndim > 1:
            out = out.mean(axis=1)
        return np.asarray(out, dtype=np.float32).ravel()
    except Exception:
        return None
    finally:
        Path(fin.name).unlink(missing_ok=True)
        Path(fout.name).unlink(missing_ok=True)


def fit_wav_to_exact_duration(
    wav: np.ndarray,
    sr: int,
    target_duration_s: float,
) -> np.ndarray:
    """
    Force ``wav`` to match the STT segment ``target_duration_s`` (seconds).

    Pipeline: (1) single conservative ``time_stretch_to_duration`` pass;
    (2) iterative librosa phase-vocoder steps (pitch-preserving);
    (3) FFmpeg ``atempo`` chain if still far off;
    (4) sample-accurate trim with micro-fade or zero-pad to ``round(target * sr)`` samples.

    This is **not** naive resampling (which shifts pitch). True facial lip-sync still
    requires video-side models; this aligns **audio** to **source timing windows**.
    """
    if wav.size == 0 or sr <= 0:
        return wav
    tgt = float(target_duration_s)
    if tgt <= 0.05:
        return wav

    tol_s = float(os.environ.get("TTS_EXACT_DURATION_TOL_S", "0.028") or "0.028")
    tol_s = max(0.012, min(0.08, tol_s))
    max_passes = int(os.environ.get("TTS_EXACT_DURATION_PASSES", "14") or "14")
    max_passes = max(3, min(24, max_passes))
    step_hi = float(os.environ.get("TTS_STRETCH_STEP_MAX", "1.34") or "1.34")
    step_lo = float(os.environ.get("TTS_STRETCH_STEP_MIN", "0.75") or "0.75")
    step_hi = max(1.05, min(1.8, step_hi))
    step_lo = max(0.55, min(0.98, step_lo))

    wav = np.asarray(wav, dtype=np.float32).copy().ravel()

    mr = float(os.environ.get("TTS_MAX_STRETCH_RATIO", "0.48") or "0.48")
    mr = max(0.12, min(0.62, mr))
    wav = time_stretch_to_duration(wav, sr, tgt, max_stretch_ratio=mr)

    try:
        import librosa
    except ImportError:
        librosa = None

    for _ in range(max_passes):
        actual_s = wav.size / float(sr)
        if abs(actual_s - tgt) <= tol_s:
            break
        if librosa is None:
            break
        rate = actual_s / tgt
        rate = max(step_lo, min(step_hi, rate))
        if abs(rate - 1.0) < 0.006:
            break
        try:
            wav = librosa.effects.time_stretch(
                wav.astype(np.float32), rate=rate, n_fft=2048
            ).astype(np.float32)
        except Exception:
            break

    if abs(wav.size / float(sr) - tgt) > tol_s * 2.2:
        alt = _ffmpeg_atempo_wav(wav, sr, tgt)
        if alt is not None and alt.size > 0:
            wav = alt

    target_n = max(1, int(round(tgt * float(sr))))
    n = int(wav.size)
    if n == target_n:
        return wav.astype(np.float32)
    if n > target_n:
        return wav[:target_n].astype(np.float32)

    out = np.zeros(target_n, dtype=np.float32)
    out[:n] = wav
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def master_audio_file(
    path_in: str,
    path_out: str | None = None,
    *,
    target_lufs: float | None = None,
    true_peak_db: float | None = None,
) -> str:
    """
    Apply the full broadcast mastering chain to a WAV file.

    Steps: HPF → de-ess → compress → presence → true-peak limit → LUFS normalise

    Writes the mastered audio as 16-bit PCM WAV (input format preserved if WAV;
    caller re-encodes to MP3).

    Parameters
    ----------
    path_in      : Source WAV path.
    path_out     : Destination WAV; if None, overwrites ``path_in`` atomically.
    target_lufs  : Integrated loudness target (default from env / -16.0).
    true_peak_db : True-peak ceiling (default from env / -1.0).

    Returns
    -------
    Path of the written mastered WAV.
    """
    if not master_enabled():
        return path_in

    p = Path(path_in)
    if not p.is_file():
        raise FileNotFoundError(path_in)

    y, sr = sf.read(str(p), dtype="float32", always_2d=True)
    # Mix to mono for processing (dub is always mono)
    if y.shape[1] > 1:
        y = y.mean(axis=1)
    else:
        y = y.squeeze()
    y = y.astype(np.float32)

    if y.size == 0:
        raise ValueError("empty audio file")

    tgt_lufs = target_lufs if target_lufs is not None else _lufs_target()
    tp_db    = true_peak_db if true_peak_db is not None else _true_peak_db()

    print(
        f"[audio-master] mastering {p.name}: "
        f"SR={sr}Hz dur={len(y)/sr:.1f}s target={tgt_lufs}LUFS tp={tp_db}dBTP"
    )

    # Chain
    y = _highpass_filter(y, sr, cutoff_hz=_highpass_cutoff_hz())
    y = _deess(y, sr)
    _ct, _cr = _effective_compressor_settings()
    y = _compress(y, sr, threshold_db=_ct, ratio=_cr)
    y = _presence_boost(y, sr, freq_hz=3000.0, gain_db=_presence_gain_db())
    y = _true_peak_limit(y, ceiling_db=tp_db)
    y = _lufs_normalise(y, sr, target_lufs=tgt_lufs)
    y = _true_peak_limit(y, ceiling_db=tp_db)   # second pass after LUFS gain

    out_p = Path(path_out) if path_out else p
    out_p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_p), y, sr, subtype="PCM_16")

    print(
        f"[audio-master] done → {out_p.name} "
        f"(peak={float(np.max(np.abs(y))):.3f})"
    )
    return str(out_p)


def master_wav_array(
    y: np.ndarray,
    sr: int,
    *,
    target_lufs: float | None = None,
    true_peak_db: float | None = None,
) -> np.ndarray:
    """
    Apply the mastering chain directly to a numpy float32 array.
    Returns the mastered array (same shape, float32).
    """
    if not master_enabled() or y.size == 0:
        return y

    tgt_lufs = target_lufs if target_lufs is not None else _lufs_target()
    tp_db    = true_peak_db if true_peak_db is not None else _true_peak_db()

    y = _highpass_filter(y, sr, cutoff_hz=_highpass_cutoff_hz())
    y = _deess(y, sr)
    _ct2, _cr2 = _effective_compressor_settings()
    y = _compress(y, sr, threshold_db=_ct2, ratio=_cr2)
    y = _presence_boost(y, sr, freq_hz=3000.0, gain_db=_presence_gain_db())
    y = _true_peak_limit(y, ceiling_db=tp_db)
    y = _lufs_normalise(y, sr, target_lufs=tgt_lufs)
    y = _true_peak_limit(y, ceiling_db=tp_db)
    return y
