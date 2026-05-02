"""
Serialize GPU-heavy work on macOS across PyTorch MPS, MLX (mlx-whisper), and Demucs.

Running Demucs on MPS concurrently with mlx-whisper, or with background Chatterbox /
OpenVoice warmup, can trigger Metal::

    failed assertion `A command encoder is already encoding to this command buffer'

``gpu_exclusive`` is a re-entrant context manager (RLock) so nested calls from the
same thread do not deadlock (e.g. warmup loading multiple models).

On non-macOS platforms this is a no-op so Linux/CUDA throughput is unchanged.
"""
from __future__ import annotations

import sys
import threading
from contextlib import contextmanager

_rlock: threading.RLock | None
if sys.platform == "darwin":
    _rlock = threading.RLock()
else:
    _rlock = None


@contextmanager
def gpu_exclusive():
    if _rlock is None:
        yield
        return
    with _rlock:
        yield
