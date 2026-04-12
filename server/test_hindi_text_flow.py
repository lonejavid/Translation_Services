#!/usr/bin/env python3
"""Verify Hindi Devanagari is not mangled by TTS text cleanup / clause prep."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Raw Hindi pipeline: no auto-danda / broad clean
os.environ["HINDI_RAW_MODE"] = "1"
os.environ["HINDI_TTS_RAW_MODE"] = "0"

from services.tts_service import TTSService  # noqa: E402


def main() -> None:
    test_hindi = "मैं एक मनमौजी लड़की बनकर बड़ी हुई हूं।"
    nukta = "लड़की"  # must survive cleaning (U+093C nukta)

    print(f"Original: {test_hindi}")
    print(f"Original bytes: {test_hindi.encode('utf-8')!r}")

    tts = TTSService()

    cleaned = tts.clean_text_for_tts(test_hindi, "hi")
    print(f"After clean (raw mode): {cleaned}")
    assert nukta in cleaned, "Nukta cluster missing after clean_text_for_tts"
    assert cleaned == test_hindi, f"Unexpected change: {cleaned!r} vs {test_hindi!r}"

    prepared = tts._prepare_hindi_text_for_tts(cleaned)
    print(f"After prepare: {prepared}")
    assert len(prepared) == 1, "Should be one unit"
    assert prepared[0] == cleaned, "Prepare changed text"

    os.environ["HINDI_RAW_MODE"] = "0"
    os.environ["HINDI_TTS_RAW_MODE"] = "0"
    tts2 = TTSService()
    normal_clean = tts2.clean_text_for_tts(test_hindi, "hi")
    assert nukta in normal_clean, "Nukta lost in normal (non-raw) clean path"
    print(f"After clean (normal mode, adds danda if needed): {normal_clean!r}")

    print("✓ Hindi text handling OK (nukta / Devanagari preserved)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
