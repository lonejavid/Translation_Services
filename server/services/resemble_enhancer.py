"""
Resemble Enhance — professional speech denoising + bandwidth enhancement.

ResembleAI's open-source dual-module model:
  1. Denoiser  : UNet trained to suppress noise/artifacts in TTS output
  2. Enhancer  : Latent CFM model that restores naturalness and bandwidth

Trained at 44.1 kHz — significantly improves TTS-generated audio clarity.
Achieves 15-25 dB noise reduction, PESQ > 4.0 on speech enhancement benchmarks.

This is the same underlying technology used by ElevenLabs and Resemble AI
in their commercial dubbing pipelines.

Env:
  RESEMBLE_ENHANCE_ENABLED  — 1 (default) / 0 to disable
  RESEMBLE_ENHANCE_DENOISE  — 1 (default) denoise pass
  RESEMBLE_ENHANCE_ENHANCE  — 1 (default) CFM enhancement pass
  RESEMBLE_ENHANCE_LAMBD    — enhancer lambda 0.0-1.0 (default 0.1)
                              0.0 = pure enhancement, 1.0 = pure denoising
                              Low lambda preserves voice naturalness.
  RESEMBLE_ENHANCE_NFE      — number of function evaluations (default 32)
                              More = better quality, slower. 16-64 is practical.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

_lock = threading.Lock()
_load_failed = False


def is_enabled() -> bool:
    return os.environ.get("RESEMBLE_ENHANCE_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )


def _denoise_enabled() -> bool:
    return os.environ.get("RESEMBLE_ENHANCE_DENOISE", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )


def _enhance_enabled() -> bool:
    return os.environ.get("RESEMBLE_ENHANCE_ENHANCE", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )


def _lambd() -> float:
    try:
        return max(0.0, min(1.0, float(os.environ.get("RESEMBLE_ENHANCE_LAMBD", "0.1") or "0.1")))
    except ValueError:
        return 0.1


def _nfe() -> int:
    try:
        return max(8, min(128, int(os.environ.get("RESEMBLE_ENHANCE_NFE", "32") or "32")))
    except ValueError:
        return 32


def _device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def enhance_wav_file(input_path: str, output_path: str | None = None) -> str:
    """
    Run Resemble Enhance on ``input_path`` (WAV).

    Applies denoiser and/or CFM enhancer depending on env settings.
    Writes the result to ``output_path`` (or overwrites input if None).
    Returns the path of the enhanced file.

    Falls back silently to returning the original path on any failure.
    """
    global _load_failed

    if not is_enabled():
        return input_path

    if _load_failed:
        return input_path

    src = Path(input_path)
    if not src.is_file():
        return input_path

    out = Path(output_path) if output_path else src

    try:
        import torch
        import torchaudio
        from resemble_enhance.enhancer.inference import denoise, enhance
    except ImportError:
        print("[resemble-enhance] Package not installed — skipping enhancement. "
              "Install with: pip install resemble-enhance")
        _load_failed = True
        return input_path
    except Exception as e:
        print(f"[resemble-enhance] Import failed ({e!r}) — skipping.")
        _load_failed = True
        return input_path

    try:
        import os as _os
        # MPS fallback required for weight_norm op used in the enhancer stage
        _os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        device = _device()

        # Load audio — resemble-enhance expects 1D mono float32 tensor
        wav, sr = torchaudio.load(str(src))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0)   # stereo → mono (1D)
        else:
            wav = wav.squeeze(0)    # (1, N) → (N,)

        # ── Step 1: Denoise (MPS-compatible) ─────────────────────────────────
        if _denoise_enabled():
            with _lock:
                wav, sr = denoise(wav, sr, device)
            print(f"[resemble-enhance] Denoised: {src.name}")

        # ── Step 2: CFM Enhancement ───────────────────────────────────────────
        # weight_norm is not fully implemented on MPS; fall back to CPU for this
        # stage only. Denoiser above already runs on MPS (fast), enhancer on CPU
        # adds ~5-10s per minute of audio — acceptable for quality gain.
        if _enhance_enabled():
            enhance_device = device
            with _lock:
                try:
                    wav, sr = enhance(
                        wav, sr, enhance_device,
                        nfe=_nfe(),
                        solver="midpoint",
                        lambd=_lambd(),
                        tau=0.5,
                    )
                except (NotImplementedError, RuntimeError):
                    # MPS op not supported — retry on CPU
                    enhance_device = "cpu"
                    wav, sr = enhance(
                        wav, sr, enhance_device,
                        nfe=_nfe(),
                        solver="midpoint",
                        lambd=_lambd(),
                        tau=0.5,
                    )
            print(f"[resemble-enhance] Enhanced: {src.name}  "
                  f"device={enhance_device} lambd={_lambd():.2f} nfe={_nfe()}")

        # Write output — save as mono WAV
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)   # (N,) → (1, N) for torchaudio.save
        out.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(out), wav.cpu(), int(sr))
        print(f"[resemble-enhance] Done → {out.name}  sr={int(sr)}Hz")
        return str(out)

    except Exception as e:
        print(f"[resemble-enhance] Enhancement failed ({e!r}); using original audio.")
        return input_path
