#!/usr/bin/env python3
"""Smoke test: XTTS, gender detector, voice extractor imports."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

print("Testing voice cloning setup...")


def _shim_beam_search_scorer() -> None:
    try:
        import transformers
    except ImportError:
        return
    if getattr(transformers, "BeamSearchScorer", None) is not None:
        return
    try:
        from transformers.generation.beam_search import BeamSearchScorer as _BSS
    except ImportError:
        return
    setattr(transformers, "BeamSearchScorer", _BSS)


_shim_beam_search_scorer()

try:
    from services.torch_coqui_compat import (
        apply_torch_load_coqui_compat,
        apply_torchaudio_soundfile_compat,
    )

    apply_torch_load_coqui_compat()
    apply_torchaudio_soundfile_compat()
    from TTS.api import TTS

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    print("✓ XTTS loaded")
except Exception as e:
    print(f"✗ XTTS failed: {e}")
    sys.exit(1)

try:
    from services.gender_detector import detect_gender_from_audio

    print("✓ Gender detector ready")
except Exception as e:
    print(f"✗ Gender detector failed: {e}")

try:
    from services.voice_extractor import extract_reference_voice

    print("✓ Voice extractor ready")
except Exception as e:
    print(f"✗ Voice extractor failed: {e}")

print("\n✓ All components ready!")
print("Run your main app to test voice cloning.")
