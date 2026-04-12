#!/usr/bin/env python3
"""Verify transformers / torch / Coqui XTTS load (voice cloning)."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _shim_beam_search_scorer() -> None:
    """Coqui XTTS expects BeamSearchScorer on transformers root; some 4.x builds omit it."""
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


def check_versions() -> bool:
    print("=" * 60)
    print("XTTS VOICE CLONING COMPATIBILITY CHECK")
    print("=" * 60)

    try:
        import transformers

        print(f"✓ transformers: {transformers.__version__}")
        if transformers.__version__ != "4.38.0":
            print("  ⚠ Expected 4.38.0 for Coqui XTTS stability — run: pip install transformers==4.38.0")
    except ImportError:
        print("✗ transformers not installed")
        return False

    try:
        import torch

        print(f"✓ torch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("✗ torch not installed")
        return False

    _shim_beam_search_scorer()

    try:
        from services.torch_coqui_compat import (
            apply_torch_load_coqui_compat,
            apply_torchaudio_soundfile_compat,
        )

        apply_torch_load_coqui_compat()
        apply_torchaudio_soundfile_compat()
        from TTS.api import TTS

        print("✓ TTS imported")
    except ImportError as e:
        print(f"✗ TTS import failed: {e}")
        return False

    try:
        print("\nLoading XTTS model...")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
        print("✓ XTTS loaded successfully")
        print("✓ Voice cloning ready")
        return True
    except Exception as e:
        print(f"✗ XTTS loading failed: {e}")
        return False


if __name__ == "__main__":
    success = check_versions()
    sys.exit(0 if success else 1)
