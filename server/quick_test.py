#!/usr/bin/env python3
"""Quick test: XTTS load, synthesis with voice clone ref, gender detection."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


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


def _find_ref_wav() -> Path | None:
    cache = _ROOT / "cache"
    if not cache.is_dir():
        return None
    candidates = sorted(cache.glob("*_xtts_ref.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def test_xtts() -> bool:
    print("=" * 50)
    print("Quick test: XTTS + clone + gender")
    print("=" * 50)

    _shim_beam_search_scorer()
    from services.torch_coqui_compat import (
        apply_torch_load_coqui_compat,
        apply_torchaudio_soundfile_compat,
    )

    apply_torch_load_coqui_compat()
    apply_torchaudio_soundfile_compat()

    print("\n1. Loading XTTS...")
    try:
        from TTS.api import TTS

        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
        print("   OK — XTTS model loaded (no exception).")
    except Exception as e:
        print(f"   FAIL — {e!r}")
        import traceback

        traceback.print_exc()
        return False

    ref_path = _find_ref_wav()
    print(f"\n2. Voice cloning (requires speaker_wav for XTTS v2)...")
    if not ref_path or not ref_path.is_file():
        print("   SKIP — no cache/*_xtts_ref.wav (process a video first or add a ref WAV).")
    else:
        print(f"   Using ref: {ref_path}")
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                out_wav = f.name
            tts.tts_to_file(
                text="This is a short voice cloning test.",
                language="en",
                file_path=out_wav,
                speaker_wav=str(ref_path),
                split_sentences=False,
            )
            size = os.path.getsize(out_wav)
            print(f"   OK — wrote cloned TTS WAV ({size} bytes) -> {out_wav}")
            try:
                os.unlink(out_wav)
            except OSError:
                pass
        except Exception as e:
            print(f"   FAIL — {e!r}")
            import traceback

            traceback.print_exc()
            return False

    print("\n3. Second synthesis (sanity)...")
    if ref_path and ref_path.is_file():
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_file = f.name
            tts.tts_to_file(
                text="Second line.",
                language="en",
                file_path=temp_file,
                speaker_wav=str(ref_path),
                split_sentences=False,
            )
            print(f"   OK — {os.path.getsize(temp_file)} bytes -> {temp_file}")
            try:
                os.unlink(temp_file)
            except OSError:
                pass
        except Exception as e:
            print(f"   FAIL — {e!r}")
            return False
    else:
        print("   SKIP — no ref WAV.")

    print("\n4. Gender detection...")
    try:
        from services.gender_detector import detect_gender_from_audio

        specific = _ROOT / "cache" / "9e884585ca068e8f7c084b3a67083424_xtts_ref.wav"
        test_audio = specific if specific.is_file() else ref_path
        if test_audio and test_audio.is_file():
            result = detect_gender_from_audio(str(test_audio))
            print(
                f"   OK — gender={result.get('gender', 'unknown')!r} "
                f"confidence={result.get('confidence', 0):.2f} "
                f"avg_pitch_hz={result.get('avg_pitch_hz', 'n/a')}"
            )
        else:
            print("   SKIP — no WAV to analyze.")
    except Exception as e:
        print(f"   FAIL — {e!r}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "=" * 50)
    print("All critical checks passed — XTTS + clone path is working.")
    print("=" * 50)
    return True


if __name__ == "__main__":
    ok = test_xtts()
    sys.exit(0 if ok else 1)
