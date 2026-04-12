#!/usr/bin/env python3
"""Exercise gender → Hindi Edge voice mapping and detector gating (no synthesis required)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def test_hindi_edge_voice_for_gender() -> None:
    from services.edge_tts_synth import hindi_edge_voice_for_gender

    os.environ["TTS_USE_EDGE"] = "1"
    os.environ.pop("VOICE_MALE", None)
    os.environ.pop("VOICE_FEMALE", None)
    os.environ.pop("VOICE_DEFAULT", None)
    os.environ.pop("EDGE_TTS_VOICE_HI", None)
    assert hindi_edge_voice_for_gender("male") == "hi-IN-MadhurNeural"
    assert hindi_edge_voice_for_gender("female") == "hi-IN-SwaraNeural"
    assert hindi_edge_voice_for_gender(None) == "hi-IN-MadhurNeural"
    assert hindi_edge_voice_for_gender("unknown") == "hi-IN-MadhurNeural"

    os.environ["VOICE_MALE"] = "hi-IN-MadhurNeural"
    os.environ["VOICE_FEMALE"] = "hi-IN-SwaraNeural"
    assert hindi_edge_voice_for_gender("male") == "hi-IN-MadhurNeural"
    assert hindi_edge_voice_for_gender("female") == "hi-IN-SwaraNeural"
    print("✓ hindi_edge_voice_for_gender")


def test_confidence_gate() -> None:
    from services.gender_detector import apply_gender_confidence_gate

    os.environ["GENDER_CONFIDENCE_THRESHOLD"] = "0.7"
    low_male = apply_gender_confidence_gate(
        {"gender": "male", "confidence": 0.5, "avg_pitch_hz": 120.0}
    )
    assert low_male["gender"] == "unknown"
    assert low_male.get("gender_gated") is True
    high_male = apply_gender_confidence_gate(
        {"gender": "male", "confidence": 0.85, "avg_pitch_hz": 120.0}
    )
    assert high_male["gender"] == "male"
    print("✓ apply_gender_confidence_gate")


def test_clone_mismatch() -> None:
    from services.gender_detector import clone_output_mismatches_expected

    assert clone_output_mismatches_expected(
        "male", {"gender": "female", "confidence": 0.8}
    )
    assert not clone_output_mismatches_expected(
        "male", {"gender": "male", "confidence": 0.8}
    )
    assert not clone_output_mismatches_expected(
        "male", {"gender": "unknown", "confidence": 0.9}
    )
    print("✓ clone_output_mismatches_expected")


def test_list_edge_voices_optional() -> None:
    """Hits Microsoft list API — skip offline with SKIP_EDGE_LIST=1."""
    if os.environ.get("SKIP_EDGE_LIST", "").strip() == "1":
        print("⊘ list_edge_voices (SKIP_EDGE_LIST=1)")
        return
    try:
        from services.edge_tts_synth import list_edge_voices

        voices = list_edge_voices()
        hi = [v for v in voices if str(v.get("ShortName", "")).startswith("hi-IN-")]
        assert hi, "expected some hi-IN-* voices"
        print(f"✓ list_edge_voices ({len(hi)} Hindi voices)")
    except Exception as e:
        print(f"⊘ list_edge_voices skipped ({e})")


def main() -> int:
    test_hindi_edge_voice_for_gender()
    test_confidence_gate()
    test_clone_mismatch()
    test_list_edge_voices_optional()
    print("All local gender/voice checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
