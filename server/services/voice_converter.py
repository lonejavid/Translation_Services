"""
Voice conversion post-processor for dubbed speech.

After generating speech in the target language (Edge TTS or XTTS), apply
voice-characteristic matching from the original speaker's reference audio.

The fundamental problem with cross-lingual dubbing:
  - XTTS / Edge TTS gives correct *pronunciation* in the target language
  - but the voice *identity* (pitch, timbre, resonance) is generic
  - a real dubbing studio would use voice actors who deliberately match
    the original speaker's pitch range and vocal quality

This module bridges the gap automatically by:

  1. **F0 (pitch) analysis** — extract voiced-frame F0 statistics from
     both the reference (original speaker) and the synthesized audio using
     librosa.pyin (probabilistic YIN, much more accurate than basic YIN).

  2. **Pitch shift** — shift the synthesized audio's median F0 toward the
     reference speaker's median F0 using librosa.effects.pitch_shift
     (phase-vocoder based, no chipmunk artefacts within ±6 semitones).
     ``VOICE_CONVERT_PITCH_STRENGTH`` controls blend (0 = no shift, 1 = full).

  3. **Spectral tilt matching** — compute spectral centroid ratio between
     reference and synthesis. Apply a gentle shelving EQ to shift the
     tonal balance (bright ↔ dark) toward the reference voice character.
     ``VOICE_CONVERT_TILT_STRENGTH`` controls blend (default 0.4).

  4. **RMS energy normalisation** — optionally match output loudness to
     reference (already done in synthesize_with_voice_clone; skipped here
     if already applied to avoid double normalisation).

Result: the dubbed audio has the target language pronunciation from Edge/XTTS
and the voice *identity* characteristics from the original speaker.

Env knobs:
  ``VOICE_CONVERT_ENABLED``          — ``1`` (default) to apply post-processing
  ``VOICE_CONVERT_PITCH_STRENGTH``   — 0.0–1.0, default ``0.75``
  ``VOICE_CONVERT_MAX_SEMITONES``    — clamp applied shift (default ``5.0``)
  ``VOICE_CONVERT_TILT_STRENGTH``    — 0.0–1.0 spectral tilt, default ``0.35``
  ``VOICE_CONVERT_RMS_MATCH``        — ``1`` (default) to RMS-match output to ref
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

def _enabled() -> bool:
    return os.environ.get("VOICE_CONVERT_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )


def _pitch_strength() -> float:
    try:
        return max(0.0, min(1.0, float(os.environ.get("VOICE_CONVERT_PITCH_STRENGTH", "0.75") or "0.75")))
    except ValueError:
        return 0.75


def _max_semitones() -> float:
    try:
        return max(0.5, min(12.0, float(os.environ.get("VOICE_CONVERT_MAX_SEMITONES", "5.0") or "5.0")))
    except ValueError:
        return 5.0


def _tilt_strength() -> float:
    try:
        return max(0.0, min(1.0, float(os.environ.get("VOICE_CONVERT_TILT_STRENGTH", "0.35") or "0.35")))
    except ValueError:
        return 0.35


def _rms_match_enabled() -> bool:
    return os.environ.get("VOICE_CONVERT_RMS_MATCH", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )


# ---------------------------------------------------------------------------
# Speaker profile
# ---------------------------------------------------------------------------

@dataclass
class SpeakerProfile:
    """Voice characteristics extracted from a reference audio clip."""
    median_f0_hz: float = 0.0          # median voiced fundamental frequency
    mean_f0_hz: float = 0.0
    f0_std_hz: float = 0.0
    voiced_ratio: float = 0.0          # fraction of frames that are voiced
    spectral_centroid_hz: float = 0.0  # mean spectral centroid (brightness)
    rms_energy: float = 0.0            # root-mean-square signal energy
    sample_rate: int = 24000


def analyze_speaker(wav: np.ndarray, sr: int) -> SpeakerProfile:
    """
    Extract voice characteristics from ``wav`` (float32 mono, [-1, 1]).

    Returns a ``SpeakerProfile`` with F0 stats and spectral properties.
    Falls back gracefully if any analysis step fails.
    """
    profile = SpeakerProfile(sample_rate=sr)
    if wav.size < sr * 0.5:          # need at least 0.5 s
        return profile

    y = wav.astype(np.float32, copy=False)

    # --- F0 analysis via probabilistic YIN (much more robust than basic YIN) ---
    try:
        import librosa

        # pyin returns (f0_array, voiced_flag, voiced_probs)
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),   # ~65 Hz  (low bass)
            fmax=librosa.note_to_hz("C7"),   # ~2093 Hz (top of normal speech)
            sr=sr,
            frame_length=2048,
            hop_length=256,
            fill_na=0.0,
        )
        voiced_f0 = f0[voiced_flag & (f0 > 50)]  # drop sub-50 Hz artifacts
        if len(voiced_f0) >= 5:
            profile.median_f0_hz = float(np.median(voiced_f0))
            profile.mean_f0_hz = float(np.mean(voiced_f0))
            profile.f0_std_hz = float(np.std(voiced_f0))
            profile.voiced_ratio = float(np.sum(voiced_flag) / max(1, len(voiced_flag)))
    except Exception as exc:
        print(f"[voice-convert] F0 analysis failed: {exc!r}")

    # --- Spectral centroid (tonal brightness) ---
    try:
        import librosa
        centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=256)[0]
        profile.spectral_centroid_hz = float(np.mean(centroids[centroids > 0]))
    except Exception:
        pass

    # --- RMS energy ---
    profile.rms_energy = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))

    return profile


# ---------------------------------------------------------------------------
# Core conversion steps
# ---------------------------------------------------------------------------

def _pitch_shift_to_match(
    wav: np.ndarray,
    sr: int,
    ref_median_f0: float,
    synth_profile: SpeakerProfile,
    strength: float,
    max_semitones: float,
) -> np.ndarray:
    """
    Shift the pitch of ``wav`` so its median F0 moves toward ``ref_median_f0``.
    ``strength`` in [0, 1] blends between no-shift and full-shift.
    """
    if wav.size == 0 or ref_median_f0 <= 0 or synth_profile.median_f0_hz <= 0:
        return wav

    # How many semitones separates reference from synthesis?
    raw_semitones = 12.0 * np.log2(ref_median_f0 / synth_profile.median_f0_hz)
    applied = raw_semitones * strength
    applied = float(np.clip(applied, -max_semitones, max_semitones))

    if abs(applied) < 0.15:          # below perceptual threshold — skip
        return wav

    print(
        f"[voice-convert] Pitch: ref={ref_median_f0:.1f}Hz "
        f"synth={synth_profile.median_f0_hz:.1f}Hz "
        f"shift={applied:+.2f} semitones (strength={strength:.2f})"
    )
    try:
        import librosa
        shifted = librosa.effects.pitch_shift(wav, sr=sr, n_steps=applied)
        return np.clip(shifted, -1.0, 1.0).astype(np.float32)
    except Exception as exc:
        print(f"[voice-convert] Pitch shift failed: {exc!r}")
        return wav


def _spectral_tilt_match(
    wav: np.ndarray,
    sr: int,
    ref_centroid: float,
    synth_centroid: float,
    strength: float,
) -> np.ndarray:
    """
    Apply a gentle shelving EQ to shift the tonal balance of ``wav`` toward
    the reference speaker's brightness (spectral centroid).

    If reference is darker (lower centroid) → roll off highs.
    If reference is brighter (higher centroid) → lift highs.

    Uses a first-order IIR shelf filter for minimal artefacts.
    """
    if wav.size == 0 or ref_centroid <= 0 or synth_centroid <= 0:
        return wav

    ratio = ref_centroid / synth_centroid
    if abs(ratio - 1.0) < 0.05:     # < 5% difference — negligible
        return wav

    # Blend ratio toward 1.0 by (1 - strength) to control how much we apply
    blended_ratio = 1.0 + (ratio - 1.0) * strength
    # Clamp to reasonable range [0.6, 1.7]
    blended_ratio = float(np.clip(blended_ratio, 0.6, 1.7))
    if abs(blended_ratio - 1.0) < 0.03:
        return wav

    print(
        f"[voice-convert] Spectral tilt: ref_centroid={ref_centroid:.0f}Hz "
        f"synth={synth_centroid:.0f}Hz ratio={blended_ratio:.3f} "
        f"(strength={strength:.2f})"
    )

    try:
        from scipy.signal import butter, sosfilt

        # Design a high-shelf or low-shelf filter
        # Shelf frequency = midpoint of speech spectrum (~1500 Hz)
        shelf_hz = 1500.0
        nyq = sr / 2.0
        normalized_shelf = min(0.99, shelf_hz / nyq)

        if blended_ratio < 1.0:
            # Darker reference → low-pass tendency (reduce high-freq energy)
            sos = butter(1, normalized_shelf, btype="low", output="sos")
            filtered = sosfilt(sos, wav)
            # Blend: original * (1-t) + filtered * t where t scales with ratio
            t = float(np.clip(1.0 - blended_ratio, 0.0, 0.4))
            result = wav * (1.0 - t) + filtered.astype(np.float32) * t
        else:
            # Brighter reference → high-pass tendency (reduce low-freq muddiness)
            sos = butter(1, normalized_shelf, btype="high", output="sos")
            filtered = sosfilt(sos, wav)
            t = float(np.clip(blended_ratio - 1.0, 0.0, 0.4))
            result = wav * (1.0 - t) + filtered.astype(np.float32) * t

        return np.clip(result, -1.0, 1.0).astype(np.float32)

    except Exception as exc:
        print(f"[voice-convert] Spectral tilt failed: {exc!r}")
        return wav


def _rms_match(
    wav: np.ndarray,
    ref_rms: float,
) -> np.ndarray:
    """Scale ``wav`` so its RMS matches ``ref_rms``. Clamped to ±6 dB."""
    if wav.size == 0 or ref_rms < 1e-9:
        return wav
    synth_rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    if synth_rms < 1e-9:
        return wav
    scale = float(np.clip(ref_rms / synth_rms, 0.5, 2.0))   # ±6 dB max
    return np.clip(wav.astype(np.float64) * scale, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_voice(
    synth_wav: np.ndarray,
    synth_sr: int,
    ref_wav: np.ndarray,
    ref_sr: int,
    *,
    pitch_strength: Optional[float] = None,
    tilt_strength: Optional[float] = None,
    apply_rms_match: Optional[bool] = None,
) -> tuple[np.ndarray, int]:
    """
    Apply reference speaker characteristics to synthesized audio.

    Parameters
    ----------
    synth_wav     : Synthesized audio (float32 mono, [-1, 1]).
    synth_sr      : Sample rate of ``synth_wav``.
    ref_wav       : Original speaker's reference audio (float32 mono, [-1, 1]).
    ref_sr        : Sample rate of ``ref_wav``.
    pitch_strength: Override for ``VOICE_CONVERT_PITCH_STRENGTH`` (0.0–1.0).
    tilt_strength : Override for ``VOICE_CONVERT_TILT_STRENGTH`` (0.0–1.0).
    apply_rms_match: Override for ``VOICE_CONVERT_RMS_MATCH``.

    Returns
    -------
    (converted_wav, synth_sr)  — same sample rate as input synth.
    """
    if not _enabled():
        return synth_wav, synth_sr
    if synth_wav.size == 0 or ref_wav.size == 0:
        return synth_wav, synth_sr

    ps = pitch_strength if pitch_strength is not None else _pitch_strength()
    ts = tilt_strength if tilt_strength is not None else _tilt_strength()
    rms_match = apply_rms_match if apply_rms_match is not None else _rms_match_enabled()
    max_st = _max_semitones()

    # Analyse both speakers
    print("[voice-convert] Analysing reference speaker characteristics …")
    ref_profile = analyze_speaker(ref_wav, ref_sr)
    synth_profile = analyze_speaker(synth_wav, synth_sr)

    if ref_profile.median_f0_hz > 0:
        print(
            f"[voice-convert] Ref speaker: F0={ref_profile.median_f0_hz:.1f}Hz "
            f"centroid={ref_profile.spectral_centroid_hz:.0f}Hz "
            f"voiced={ref_profile.voiced_ratio:.1%}"
        )
    if synth_profile.median_f0_hz > 0:
        print(
            f"[voice-convert] Synthesis:   F0={synth_profile.median_f0_hz:.1f}Hz "
            f"centroid={synth_profile.spectral_centroid_hz:.0f}Hz"
        )

    out = synth_wav.copy()

    # Step 1: Pitch matching
    if ps > 0.01 and ref_profile.median_f0_hz > 0 and synth_profile.median_f0_hz > 0:
        out = _pitch_shift_to_match(
            out, synth_sr,
            ref_profile.median_f0_hz,
            synth_profile,
            strength=ps,
            max_semitones=max_st,
        )

    # Step 2: Spectral tilt
    if ts > 0.01 and ref_profile.spectral_centroid_hz > 0 and synth_profile.spectral_centroid_hz > 0:
        # Recompute synth centroid after pitch shift
        try:
            import librosa
            new_cent = float(np.mean(
                librosa.feature.spectral_centroid(y=out, sr=synth_sr, hop_length=256)[0]
            ))
        except Exception:
            new_cent = synth_profile.spectral_centroid_hz
        out = _spectral_tilt_match(
            out, synth_sr,
            ref_profile.spectral_centroid_hz,
            new_cent,
            strength=ts,
        )

    # Step 3: RMS energy match
    if rms_match and ref_profile.rms_energy > 0:
        out = _rms_match(out, ref_profile.rms_energy)

    return out, synth_sr


def convert_voice_from_path(
    synth_wav: np.ndarray,
    synth_sr: int,
    ref_wav_path: str,
    **kwargs,
) -> tuple[np.ndarray, int]:
    """
    Convenience wrapper: load reference from file path, then call ``convert_voice``.
    """
    if not _enabled():
        return synth_wav, synth_sr
    try:
        import soundfile as sf
        ref_data, ref_sr = sf.read(ref_wav_path, dtype="float32", always_2d=True)
        ref_mono = ref_data.mean(axis=1) if ref_data.shape[1] > 1 else ref_data.squeeze()
        return convert_voice(synth_wav, synth_sr, ref_mono.astype(np.float32), ref_sr, **kwargs)
    except Exception as exc:
        print(f"[voice-convert] Could not load reference {ref_wav_path!r}: {exc!r}")
        return synth_wav, synth_sr
