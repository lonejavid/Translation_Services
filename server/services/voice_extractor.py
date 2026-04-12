"""
Extract a high-quality voice reference clip from YouTube audio for Coqui XTTS voice cloning.

Professional approach (Netflix/studio standard):
  - Instead of taking the FIRST speech run, scan the ENTIRE audio and find
    the BEST speech windows by SNR (signal-to-noise ratio).
  - Rank all speech runs, select top N by clarity, concatenate up to the
    target duration.
  - Apply very gentle stationary noise reduction (preserve voice harmonics).
  - Peak-normalize and resample to XTTS's expected sample rate.

This produces dramatically better XTTS conditioning — the model gets a clean,
representative voice sample from the best-quality speech in the video rather
than whatever happens to be at the start (which might be intro music or noise).

Env:
  ``VOICE_REF_DURATION_SEC`` — target clip length (default ``20``; was ``12``).
  ``VOICE_REF_VAD_MODE`` — webrtcvad aggressiveness ``0``–``3`` (default ``1``).
  ``VOICE_REF_MIN_SPEECH_SEC`` — minimum speech to accept (default ``2.0``).
  ``DISABLE_VOICE_REF_EXTRACTION`` — ``1`` to skip extraction entirely.
  ``VOICE_REF_OUTPUT_SR`` — output sample rate (default ``24000``).
  ``DISABLE_VOICE_REF_GENDER`` — ``1`` to skip pitch-based gender detection.
  ``GENDER_CONFIDENCE_THRESHOLD`` — confidence gate (default ``0.6``).
"""
from __future__ import annotations

import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import webrtcvad


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


def _webrtc_speech_frames(
    y16: np.ndarray, sample_rate: int, vad_mode: int, frame_ms: int = 30
) -> tuple[list[bool], int]:
    """Return per-frame speech flags and samples per frame (16 kHz only)."""
    if sample_rate != 16000:
        raise ValueError("webrtcvad path expects 16 kHz")
    vad = webrtcvad.Vad(max(0, min(3, vad_mode)))
    samples_per_frame = int(sample_rate * frame_ms / 1000)
    if samples_per_frame not in (160, 320, 480):
        raise ValueError(f"Unsupported frame_ms={frame_ms} for 16 kHz")

    y_int = (np.clip(y16.astype(np.float64), -1.0, 1.0) * 32767.0).astype(np.int16)
    pcm = y_int.tobytes()
    frame_bytes = samples_per_frame * 2
    flags: list[bool] = []
    n = len(y_int)
    for offset in range(0, n - samples_per_frame + 1, samples_per_frame):
        chunk = pcm[offset * 2 : offset * 2 + frame_bytes]
        flags.append(vad.is_speech(chunk, sample_rate))
    return flags, samples_per_frame


def _all_speech_runs(
    flags: list[bool],
    samples_per_frame: int,
    max_gap_frames: int = 10,
    min_run_frames: int = 10,
) -> list[tuple[int, int]]:
    """
    Find ALL contiguous speech runs (not just the first).
    Returns list of (start_sample, end_sample_exclusive) in 16 kHz indices,
    sorted by duration descending.
    """
    runs: list[tuple[int, int]] = []
    n = len(flags)
    i = 0
    while i < n:
        # Skip silence
        while i < n and not flags[i]:
            i += 1
        if i >= n:
            break
        start_f = i
        last_speech = i
        gap = 0
        i += 1
        while i < n:
            if flags[i]:
                last_speech = i
                gap = 0
            else:
                gap += 1
                if gap > max_gap_frames:
                    break
            i += 1
        run_frames = last_speech - start_f + 1
        if run_frames >= min_run_frames:
            runs.append((start_f * samples_per_frame, (last_speech + 1) * samples_per_frame))
    return runs


def _snr_score(y16: np.ndarray, start: int, end: int) -> float:
    """
    Estimate signal-to-noise ratio for a speech window.
    Uses the 10th percentile energy as a noise floor estimate (Ephraim-Malah
    inspired heuristic — good for short clips without a separate noise reference).
    Higher = cleaner speech.
    """
    clip = y16[start:end]
    if clip.size < 160:
        return 0.0
    # Frame-level RMS
    frame = 160  # 10 ms at 16 kHz
    rms_vals = []
    for j in range(0, len(clip) - frame + 1, frame):
        rms = float(np.sqrt(np.mean(clip[j:j + frame].astype(np.float64) ** 2)))
        rms_vals.append(rms)
    if not rms_vals:
        return 0.0
    rms_arr = np.array(rms_vals)
    noise_floor = float(np.percentile(rms_arr, 10)) + 1e-9
    signal = float(np.percentile(rms_arr, 75))
    return signal / noise_floor


def _select_best_windows(
    y16: np.ndarray,
    runs: list[tuple[int, int]],
    target_samples: int,
    max_runs: int = 8,
) -> list[tuple[int, int]]:
    """
    Score each speech run by SNR and select the best ones (up to ``max_runs``)
    that together fill ``target_samples`` of clean speech.

    Returns selected (start, end) pairs in temporal order.
    """
    if not runs:
        return []

    # Score all runs
    scored = sorted(
        [(run, _snr_score(y16, run[0], run[1])) for run in runs],
        key=lambda x: -x[1],
    )

    selected: list[tuple[int, int]] = []
    total_samples = 0
    for (start, end), score in scored[:max_runs]:
        run_len = end - start
        if total_samples >= target_samples:
            break
        # Take only as many samples as we need from this run
        take = min(run_len, target_samples - total_samples)
        selected.append((start, start + take))
        total_samples += take

    # Return in temporal order (so concatenated audio sounds natural)
    selected.sort(key=lambda x: x[0])
    return selected


def extract_reference_voice(
    audio_path: str,
    output_path: str | None = None,
    duration_sec: float | None = None,
    vad_mode: int | None = None,
    detect_gender: bool | None = None,
) -> tuple[str | None, dict | None]:
    """
    Extract the best-quality voice reference clip for XTTS speaker_wav.

    Unlike the naive "take first speech" approach, this function:
    1. Runs VAD across the ENTIRE audio to find ALL speech segments.
    2. Scores each segment by SNR (signal clarity).
    3. Selects the top N cleanest segments.
    4. Concatenates them (temporal order) up to the target duration.

    This gives XTTS a representative, high-quality voice sample from the
    most clearly-recorded speech in the entire video — dramatically better
    conditioning for voice cloning.

    Returns:
        ``(path, gender_info)`` — path is ``None`` if extraction failed / disabled.
    """
    if os.environ.get("DISABLE_VOICE_REF_EXTRACTION", "").strip().lower() in (
        "1", "true", "yes",
    ):
        print("[voice-ref] Extraction disabled (DISABLE_VOICE_REF_EXTRACTION=1)")
        return None, None

    src = Path(audio_path)
    if not src.is_file():
        print(f"[voice-ref] Missing audio: {audio_path}")
        return None, None

    # 60s default — more reference = richer voice identity for embedding extraction
    dur = duration_sec if duration_sec is not None else _env_float("VOICE_REF_DURATION_SEC", 60.0)
    dur = max(3.0, min(120.0, float(dur)))
    mode = vad_mode if vad_mode is not None else _env_int("VOICE_REF_VAD_MODE", 1)
    min_speech = _env_float("VOICE_REF_MIN_SPEECH_SEC", 2.0)
    out_sr = _env_int("VOICE_REF_OUTPUT_SR", 24000)
    out_sr = max(8000, min(48000, out_sr))

    if output_path is None:
        output_path = str(src.with_name(f"{src.stem}_xtts_ref.wav"))
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    try:
        y, sr = librosa.load(str(src), sr=None, mono=True)
    except Exception as e:
        print(f"[voice-ref] librosa.load failed: {e}")
        return None, None

    if y.size == 0:
        print("[voice-ref] Empty audio")
        return None, None

    y = np.asarray(y, dtype=np.float32)

    # Resample to 16 kHz for VAD (webrtcvad requirement)
    y16 = librosa.resample(y, orig_sr=sr, target_sr=16000)

    # Run VAD
    try:
        flags, spf = _webrtc_speech_frames(y16, 16000, mode, frame_ms=30)
    except Exception as e:
        print(f"[voice-ref] VAD failed: {e}")
        return None, None

    # Find ALL speech runs
    runs = _all_speech_runs(flags, spf, max_gap_frames=10, min_run_frames=10)

    if not runs:
        print("[voice-ref] No speech detected by VAD")
        return None, None

    total_speech_s = sum(e - s for s, e in runs) / 16000
    print(
        f"[voice-ref] Found {len(runs)} speech run(s), "
        f"total {total_speech_s:.1f}s of speech"
    )

    # Select best windows by SNR
    target_samples = int(16000 * dur)
    best_windows = _select_best_windows(y16, runs, target_samples)

    if not best_windows:
        print("[voice-ref] No suitable speech windows found")
        return None, None

    # Concatenate selected windows
    chunks: list[np.ndarray] = []
    total_s = 0
    for start, end in best_windows:
        chunk = y16[start:end].copy()
        chunks.append(chunk)
        total_s += len(chunk) / 16000

    clip16 = np.concatenate(chunks).astype(np.float32)

    if clip16.size < int(16000 * min_speech):
        print(f"[voice-ref] Extracted clip too short ({clip16.size/16000:.2f}s)")
        return None, None

    print(
        f"[voice-ref] Selected {len(best_windows)} window(s), "
        f"{total_s:.2f}s total from best-SNR speech"
    )

    # Gentle stationary noise reduction — only lift the noise floor,
    # preserve all voice harmonics (XTTS needs them for accurate cloning)
    try:
        import noisereduce as nr
        # Higher prop_decrease (0.25) safe here: reference is already demucs-
        # cleaned vocals; cleaner signal → richer OpenVoice embedding extraction.
        clip16 = nr.reduce_noise(
            y=clip16, sr=16000, stationary=True, prop_decrease=0.25
        ).astype(np.float32)
    except Exception as e:
        print(f"[voice-ref] noisereduce skipped: {e}")

    # Peak normalize to -1 dBFS
    peak = float(np.max(np.abs(clip16))) + 1e-9
    clip16 = clip16 / peak * 0.95

    # Resample to XTTS output SR
    clip_out = librosa.resample(clip16, orig_sr=16000, target_sr=out_sr)
    clip_out = np.clip(clip_out.astype(np.float32), -1.0, 1.0)

    try:
        sf.write(str(out_p), clip_out, out_sr, subtype="PCM_16")
    except Exception as e:
        print(f"[voice-ref] soundfile.write failed: {e}")
        return None, None

    print(
        f"[voice-ref] Wrote {out_p.name} (~{len(clip_out) / out_sr:.2f}s @ {out_sr} Hz)"
    )

    # Gender detection
    gender_info: dict | None = None
    if detect_gender is None:
        detect_gender = os.environ.get("DISABLE_VOICE_REF_GENDER", "").strip().lower() not in (
            "1", "true", "yes",
        )
    if detect_gender:
        try:
            from services.gender_detector import detect_gender_from_audio
            gender_info = detect_gender_from_audio(str(out_p))
            print(
                f"[voice-ref] Gender: {gender_info['gender']} "
                f"(confidence {gender_info.get('confidence', 0):.2f})"
            )
        except Exception as e:
            print(f"[voice-ref] Gender detection failed: {e}")
            gender_info = {"gender": "unknown", "confidence": 0.0}

    return str(out_p), gender_info
