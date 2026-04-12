"""Reference audio checks for few-shot cloning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import soundfile as sf


@dataclass
class RefAudioReport:
    path: str
    duration_sec: float
    sample_rate: int
    channels: int
    warnings: list[str]


class RefAudioError(ValueError):
    pass


def analyze_reference_audio(
    ref_audio_path: str | Path,
    *,
    min_duration_sec: float = 3.0,
    max_duration_sec: float = 60.0,
    recommended_min_sec: float = 5.0,
    recommended_max_sec: float = 10.0,
    min_sample_rate: int = 8_000,
    max_sample_rate: int = 96_000,
) -> RefAudioReport:
    """
    Validate reference clip duration and sample rate.

    Raises:
        RefAudioError: missing file, unloadable audio, duration out of bounds, SR out of bounds.
    """
    p = Path(ref_audio_path).expanduser().resolve()
    if not p.is_file():
        raise RefAudioError(f"Reference audio not found: {p}")

    try:
        info = sf.info(str(p))
    except Exception as e:
        raise RefAudioError(f"Cannot read audio (soundfile): {e}") from e

    sr = int(info.samplerate)
    ch = int(info.channels)
    dur = float(info.duration)

    if dur < min_duration_sec:
        raise RefAudioError(
            f"Reference too short: {dur:.2f}s < minimum {min_duration_sec}s "
            "(few-shot quality needs at least ~3–5s of clean speech)."
        )
    if dur > max_duration_sec:
        raise RefAudioError(
            f"Reference too long: {dur:.2f}s > maximum {max_duration_sec}s "
            "(trim to a clean window; upstream WebUI slices similarly)."
        )

    warnings: list[str] = []
    if dur < recommended_min_sec:
        warnings.append(
            f"Duration {dur:.2f}s is under the recommended {recommended_min_sec}s–{recommended_max_sec}s "
            "window; cloning may be less stable."
        )
    if dur > recommended_max_sec:
        warnings.append(
            f"Duration {dur:.2f}s exceeds the typical {recommended_max_sec}s few-shot window; "
            "consider trimming to the clearest segment."
        )

    if sr < min_sample_rate or sr > max_sample_rate:
        raise RefAudioError(
            f"Unsupported sample rate {sr} Hz (expected {min_sample_rate}–{max_sample_rate} Hz)."
        )
    if sr not in (16_000, 22_050, 24_000, 32_000, 44_100, 48_000):
        warnings.append(
            f"Sample rate {sr} Hz is non-standard for TTS; GPT-SoVITS often uses 16k–48k. "
            "Resampling before inference is recommended."
        )

    return RefAudioReport(
        path=str(p),
        duration_sec=dur,
        sample_rate=sr,
        channels=ch,
        warnings=warnings,
    )
