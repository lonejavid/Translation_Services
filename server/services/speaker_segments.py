"""
Per-segment speaker routing for dubbed TTS (no full diarization yet).

Uses each subtitle ``[start, end]`` as a time window, slices the clean source WAV,
and runs the same pitch-based :class:`GenderDetector` as the global reference clip.
Windows are **median-smoothed** across neighbors to reduce frame-to-frame jitter.

**Limitation:** two speakers of the same sex are not separated (both map to one
male or one female Edge voice). True multi-speaker identity needs pyannote / ASR
diarization (optional follow-up).

Disable with ``SPEAKER_AWARE_TTS=0``.
"""
from __future__ import annotations

import os

import librosa
import numpy as np

from services.gender_detector import get_gender_detector


def _speaker_aware_enabled() -> bool:
    return os.environ.get("SPEAKER_AWARE_TTS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _median_smooth_labels(labels: list[str], window: int = 3) -> list[str]:
    if window < 2 or len(labels) <= 1:
        return list(labels)
    half = window // 2
    out: list[str] = []
    for i in range(len(labels)):
        lo = max(0, i - half)
        hi = min(len(labels), i + half + 1)
        sub = labels[lo:hi]
        out.append(sorted(sub)[len(sub) // 2])
    return out


def _routing_gender_from_detection(g: str) -> str:
    g = (g or "").strip().lower()
    if g in ("male", "female", "neutral"):
        return g
    return "neutral"


def _speaker_ids_stable_by_gender(smoothed: list[str]) -> list[int]:
    """
    Male segments → speaker 0, female → 1, neutral/unknown → previous bucket.
    Same speaker returning later with the same detected sex keeps the same id.
    """
    out: list[int] = []
    last = 0
    for g in smoothed:
        if g == "male":
            last = 0
            out.append(0)
        elif g == "female":
            last = 1
            out.append(1)
        else:
            out.append(last)
    return out


def enrich_segments_with_speaker_voice(
    wav_path: str,
    segments: list[dict],
    *,
    fallback_routing_gender: str | None = None,
) -> None:
    """
    Mutates each segment in place with:

    - ``tts_gender``: ``male`` | ``female`` | ``neutral`` (for Edge / clone routing)
    - ``speaker_id``: small int (0 = male bucket, 1 = female bucket in two-voice mode)
    - ``segment_pitch_hz``: optional diagnostic when F0 was estimated
    """
    if not _speaker_aware_enabled() or not segments:
        return
    if not wav_path or not os.path.isfile(wav_path):
        return

    fb = (fallback_routing_gender or "").strip().lower()
    if fb not in ("male", "female", "neutral"):
        fb = ""

    try:
        y_full, sr = librosa.load(wav_path, sr=16000, mono=True)
    except Exception as ex:
        print(f"[speaker-seg] skip enrich (load failed): {ex!r}")
        return

    det = get_gender_detector()
    raw_labels: list[str] = []
    pitches: list[float] = []

    for seg in segments:
        try:
            t0 = float(seg.get("start", 0))
            t1 = float(seg.get("end", t0))
        except (TypeError, ValueError):
            raw_labels.append("unknown")
            pitches.append(0.0)
            continue
        if t1 <= t0 + 0.04:
            raw_labels.append("unknown")
            pitches.append(0.0)
            continue
        i0 = max(0, int(t0 * sr))
        i1 = min(len(y_full), int(t1 * sr))
        chunk = y_full[i0:i1]
        r = det.detect_gender_numpy(chunk, sr)
        g = (r.get("gender") or "unknown").strip().lower()
        if g not in ("male", "female", "neutral", "unknown"):
            g = "unknown"
        raw_labels.append(g)
        try:
            pitches.append(float(r.get("avg_pitch_hz") or 0.0))
        except (TypeError, ValueError):
            pitches.append(0.0)

    smoothed = _median_smooth_labels(raw_labels, window=3)
    speaker_ids = _speaker_ids_stable_by_gender(smoothed)

    n_m = n_f = n_n = 0
    for seg, g_sm, sid, p in zip(segments, smoothed, speaker_ids, pitches):
        route = _routing_gender_from_detection(g_sm)
        if route == "neutral" and fb in ("male", "female"):
            route = fb
        seg["tts_gender"] = route
        seg["speaker_id"] = int(sid)
        if p and p > 1.0:
            seg["segment_pitch_hz"] = round(p, 1)
        elif "segment_pitch_hz" in seg:
            del seg["segment_pitch_hz"]
        if route == "male":
            n_m += 1
        elif route == "female":
            n_f += 1
        else:
            n_n += 1

    print(
        f"[speaker-seg] Per-window gender routing: male={n_m} female={n_f} "
        f"neutral/other={n_n} (wav={os.path.basename(wav_path)})"
    )
