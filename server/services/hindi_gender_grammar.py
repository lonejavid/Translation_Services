"""
Adjust Hindi subtitle text for detected speaker gender (first-person grammar).

Machine translation from English ("I ...") often defaults to masculine forms
(करता हूँ) even for female speakers. When we already know ``speaker_gender``
from the voice reference, we apply safe phrase-level replacements toward
feminine or masculine agreement for common patterns.

This is **heuristic**, not full morphological analysis. Disable with
``HINDI_GENDER_GRAMMAR=0``.
"""
from __future__ import annotations

import os
from typing import Iterable


def _enabled() -> bool:
    return os.environ.get("HINDI_GENDER_GRAMMAR", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _normalize_lang(code: str) -> str:
    return (code or "").strip().lower().split("-")[0]


# Verb stems for habitual present / simple past (1st person) — high frequency in dubs
_STEMS = (
    "कर",
    "देख",
    "सुन",
    "बोल",
    "चाह",
    "सोच",
    "समझ",
    "खा",
    "पी",
    "ले",
    "दे",
    "मिल",
    "जा",
    "आ",
    "रह",
    "पढ़",
    "लिख",
    "कह",
    "पूछ",
    "चल",
    "बैठ",
    "उठ",
    "मान",
    "लग",
    "रख",
    "भूल",
    "खेल",
    "बना",
    "दिख",
    "सीख",
    "बच",
    "नाच",
    "गा",
    "बता",
    "ढूंढ",
    "चुन",
    "जोड़",
    "तोड़",
)


def _build_male_to_female_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    for stem in _STEMS:
        pairs.append((f"{stem}ता हूँ", f"{stem}ती हूँ"))
        pairs.append((f"{stem}ता था", f"{stem}ती थी"))
        pairs.append((f"{stem}ता हूँगा", f"{stem}ती हूँगी"))
        pairs.append((f"{stem}ता रहा", f"{stem}ती रही"))

    # Continuous auxiliary (after any verb stem in MT output)
    pairs.extend(
        [
            (" कर रहा हूँ", " कर रही हूँ"),
            (" कर रहा था", " कर रही थी"),
            (" जा रहा हूँ", " जा रही हूँ"),
            (" जा रहा था", " जा रही थी"),
            (" आ रहा हूँ", " आ रही हूँ"),
            (" आ रहा था", " आ रही थी"),
            (" दे रहा हूँ", " दे रही हूँ"),
            (" दे रहा था", " दे रही थी"),
            (" ले रहा हूँ", " ले रही हूँ"),
            (" ले रहा था", " ले रही थी"),
            (" सोच रहा हूँ", " सोच रही हूँ"),
            (" सोच रहा था", " सोच रही थी"),
            (" बोल रहा हूँ", " बोल रही हूँ"),
            (" बोल रहा था", " बोल रही थी"),
            # Modals / ability
            (" सकता हूँ", " सकती हूँ"),
            (" सकता था", " सकती थी"),
            (" सकता हूँगा", " सकती हूँगी"),
            (" चाहता हूँ", " चाहती हूँ"),
            (" चाहता था", " चाहती थी"),
            (" पाता हूँ", " पाती हूँ"),
            (" पाता था", " पाती थी"),
            # Future (spoken style)
            (" करूंगा", " करूंगी"),
            (" जाऊंगा", " जाऊंगी"),
            (" आऊंगा", " आऊंगी"),
            (" दूंगा", " दूंगी"),
            (" लूंगा", " लूंगी"),
            (" कहूंगा", " कहूंगी"),
            (" देखूंगा", " देखूंगी"),
            (" करूँगा", " करूँगी"),
            (" जाऊँगा", " जाऊँगी"),
            # Motion compound past (common MT for "I went / came")
            (" चला गया", " चली गई"),
            (" चला गया हूँ", " चली गई हूँ"),
            (" आया हूँ", " आई हूँ"),
            (" गया हूँ", " गई हूँ"),
            (" रहा हूँ", " रही हूँ"),
            (" रहा था", " रही थी"),
        ]
    )

    # Longest first so longer phrases win
    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    return pairs


_M2F: list[tuple[str, str]] | None = None
_F2M: list[tuple[str, str]] | None = None


def _pairs_m2f() -> list[tuple[str, str]]:
    global _M2F
    if _M2F is None:
        _M2F = _build_male_to_female_pairs()
    return _M2F


def _pairs_f2m() -> list[tuple[str, str]]:
    global _F2M
    if _F2M is None:
        _F2M = [(f, m) for m, f in _pairs_m2f()]
        _F2M.sort(key=lambda x: len(x[0]), reverse=True)
    return _F2M


def adjust_hindi_line(text: str, speaker_gender: str) -> str:
    """
    Return text with first-person gender tweaks for Hindi.

    ``speaker_gender``: ``male`` | ``female`` (other values → unchanged).
    """
    if not text or not _enabled():
        return text
    g = (speaker_gender or "").strip().lower()
    if g == "female":
        pairs = _pairs_m2f()
    elif g == "male":
        pairs = _pairs_f2m()
    else:
        return text

    out = text
    for a, b in pairs:
        if a in out:
            out = out.replace(a, b)
    return out


def apply_gender_to_translated_segments(
    segments: Iterable[dict],
    speaker_gender: str | None,
    target_language: str,
) -> int:
    """
    Mutate each segment's ``translated_text`` / ``translated`` in place.
    Returns number of segments whose text changed.
    """
    if not _enabled():
        return 0
    if _normalize_lang(target_language) != "hi":
        return 0
    g = (speaker_gender or "").strip().lower()
    if g not in ("male", "female"):
        return 0

    changed = 0
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        key = "translated_text"
        raw = seg.get(key) or seg.get("translated") or ""
        if not isinstance(raw, str) or not raw.strip():
            continue
        new = adjust_hindi_line(raw, g)
        if new != raw:
            seg[key] = new
            if "translated" in seg:
                seg["translated"] = new
            changed += 1
    return changed
