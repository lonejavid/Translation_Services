"""
Maximum-effort intelligibility for **dubbed** (translated) speech.

``TTS_MAXIMUM_CLARITY`` (default ``1``) coordinates Edge rate, shorter English
chunks, wider subtitle gaps, minimal crossfades, gentler mastering, and
keeping Microsoft Edge neural output free of OpenVoice / spectral ref matching.

Set ``TTS_MAXIMUM_CLARITY=0`` to disable this bundle (per-language envs still apply).
"""
from __future__ import annotations

import os


def maximum_clarity_enabled() -> bool:
    return os.environ.get("TTS_MAXIMUM_CLARITY", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def english_edge_rate_when_unset() -> str:
    """
    Speaking rate for ``en-*`` Edge voices when ``EDGE_TTS_RATE`` is not set.

    Slower playback materially improves comprehension on machine-translated lines.
    """
    if maximum_clarity_enabled():
        return (os.environ.get("EDGE_TTS_EN_RATE_MAX", "-20%").strip() or "-20%")
    return (os.environ.get("EDGE_TTS_EN_RATE", "-16%").strip() or "-16%")


def english_clause_char_cap(default_if_unset: int) -> int:
    """Max characters per Edge clause for English when env ``TTS_CLAUSE_MAX_CHARS_EN`` unset."""
    if maximum_clarity_enabled():
        return min(default_if_unset, 44)
    return default_if_unset


def english_segment_gap_floor_ms() -> int:
    """Minimum silence between English subtitle clips when ``TTS_SEGMENT_GAP_MS_EN`` unset."""
    return 300 if maximum_clarity_enabled() else 200


def english_crossfade_cap_ms() -> int:
    """Upper bound for English crossfade when ``TTS_CROSSFADE_MS_EN`` unset."""
    return 10 if maximum_clarity_enabled() else 24
