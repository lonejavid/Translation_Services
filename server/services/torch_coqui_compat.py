"""
Coqui / XTTS compatibility shims:

- PyTorch 2.6+ defaults ``torch.load(..., weights_only=True)``; Coqui checkpoints need ``False``.
- Newer ``torchaudio`` may require **torchcodec** for ``torchaudio.load``; we prefer **soundfile**
  for local paths (same as the main TTS service).
"""
from __future__ import annotations

import inspect

_torch_load_done = False
_torchaudio_done = False


def apply_torch_load_coqui_compat() -> None:
    global _torch_load_done
    if _torch_load_done:
        return
    import torch

    _orig = torch.load

    def _patched(*args, **kwargs):
        if "weights_only" not in kwargs:
            try:
                if "weights_only" in inspect.signature(_orig).parameters:
                    kwargs["weights_only"] = False
            except (TypeError, ValueError):
                pass
        return _orig(*args, **kwargs)

    torch.load = _patched  # type: ignore[method-assign]
    _torch_load_done = True


def apply_torchaudio_soundfile_compat() -> None:
    """Route ``torchaudio.load`` for file paths through soundfile (avoids torchcodec)."""
    global _torchaudio_done
    if _torchaudio_done:
        return
    import os

    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    _orig = torchaudio.load

    def _sf_load(
        uri,
        frame_offset: int = 0,
        num_frames: int = -1,
        normalize: bool = True,
        channels_first: bool = True,
        format=None,
        buffer_size: int = 4096,
        backend=None,
    ):
        if isinstance(uri, (str, bytes, os.PathLike)):
            path = os.fsdecode(uri) if isinstance(uri, bytes) else os.fspath(uri)
            try:
                data, sr = sf.read(path, dtype="float32", always_2d=True)
            except Exception:
                return _orig(
                    uri,
                    frame_offset=frame_offset,
                    num_frames=num_frames,
                    normalize=normalize,
                    channels_first=channels_first,
                    format=format,
                    buffer_size=buffer_size,
                    backend=backend,
                )
            if frame_offset > 0:
                data = data[frame_offset:]
            if num_frames is not None and num_frames >= 0:
                data = data[:num_frames]
            w = np.ascontiguousarray(data.T)
            tensor = torch.from_numpy(w)
            if not channels_first:
                tensor = tensor.t()
            return tensor, int(sr)
        return _orig(
            uri,
            frame_offset=frame_offset,
            num_frames=num_frames,
            normalize=normalize,
            channels_first=channels_first,
            format=format,
            buffer_size=buffer_size,
            backend=backend,
        )

    torchaudio.load = _sf_load  # type: ignore[method-assign]
    _torchaudio_done = True
