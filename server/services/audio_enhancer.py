"""Light post-processing on the final dub mix (HPF, presence, peak normalize)."""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy import signal


def enhance_audio_file(path_in: str, path_out: str | None = None) -> str:
    """
    Improve clarity: remove sub-bass rumble, add gentle high-mid emphasis, normalize peak.

    Writes PCM WAV. If ``path_out`` is None, overwrites ``path_in`` (via a temp swap).
    """
    path_in_p = Path(path_in)
    if not path_in_p.is_file():
        raise FileNotFoundError(path_in)

    y, sr = librosa.load(str(path_in_p), sr=None, mono=True)
    if y.size == 0:
        raise ValueError("empty audio")

    y = np.asarray(y, dtype=np.float32)

    sos_hp = signal.butter(4, 80.0, btype="highpass", fs=sr, output="sos")
    y = signal.sosfilt(sos_hp, y)

    nyq = sr / 2.0
    lp_cut = min(9000.0, max(2000.0, nyq - 500.0))
    sos_lp = signal.butter(2, lp_cut, btype="lowpass", fs=sr, output="sos")
    low = signal.sosfilt(sos_lp, y)
    bright = np.clip(y - low, -1.0, 1.0)
    y = np.clip(y + 0.12 * bright, -1.0, 1.0)

    peak = float(np.max(np.abs(y))) + 1e-9
    target_linear = 10 ** (-1.0 / 20.0)
    y = (y / peak * target_linear).astype(np.float32)

    out_p = Path(path_out) if path_out else path_in_p
    out_p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_p), y, sr, subtype="PCM_16")
    return str(out_p)
