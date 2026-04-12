"""
Speech denoising with Mozilla RNNoise (local, no API).

Install one of:
  pip install rnnoise-python   # when available / your index; exposes ``import rnnoise``
  pip install pyrnnoise        # PyPI default; used automatically as fallback

Apply before Whisper / mlx-whisper so STT sees cleaner speech.
"""
from __future__ import annotations

import os
from math import gcd
from typing import Optional

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

try:
    import rnnoise
except ImportError:
    rnnoise = None  # type: ignore[misc, assignment]

RNNOISE_SAMPLE_RATE = 48000

_noise_canceller_instance: Optional["NoiseCanceller"] = None


def _get_pyrnnoise_frame_api():
    """Return (create, destroy, process_mono_frame, FRAME_SIZE) or None."""
    try:
        from pyrnnoise.rnnoise import (
            FRAME_SIZE,
            create,
            destroy,
            process_mono_frame,
        )

        return create, destroy, process_mono_frame, int(FRAME_SIZE)
    except ImportError:
        return None


class NoiseCanceller:
    """
    RNNoise-based suppression at 48 kHz; input/output resampled to the file’s rate.
    """

    def __init__(self) -> None:
        if rnnoise is not None and hasattr(rnnoise, "RNNoise"):
            self._backend = "rnnoise"
            self.denoiser = rnnoise.RNNoise()
            self.frame_size = int(getattr(rnnoise.RNNoise, "FRAME_SIZE", 480))
            print("[RNNoise] Noise canceller ready (rnnoise).")
            return

        api = _get_pyrnnoise_frame_api()
        if api is None:
            raise ImportError(
                "RNNoise bindings not found. Install: pip install pyrnnoise "
                "(or rnnoise-python if your environment provides import rnnoise)."
            )
        self._backend = "pyrnnoise"
        self._create, self._destroy, self._process_mono_frame = api[0], api[1], api[2]
        self.frame_size = api[3]
        self.denoiser = None
        print("[RNNoise] Noise canceller ready (pyrnnoise).")

    def _load_wav(self, path: str) -> tuple[np.ndarray, int]:
        sample_rate, data = wavfile.read(path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128.0) / 128.0
        else:
            data = data.astype(np.float32)
        return data, int(sample_rate)

    def _resample(
        self, audio: np.ndarray, orig_rate: int, target_rate: int
    ) -> np.ndarray:
        if orig_rate == target_rate:
            return audio
        g = gcd(orig_rate, target_rate)
        return resample_poly(audio, target_rate // g, orig_rate // g)

    def _to_int16(self, audio: np.ndarray) -> np.ndarray:
        audio = np.clip(audio, -1.0, 1.0)
        return (audio * 32767.0).astype(np.int16)

    def _to_float32(self, audio: np.ndarray) -> np.ndarray:
        return audio.astype(np.float32) / 32767.0

    def denoise_array(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Denoise mono float32 audio in [-1, 1]; returns same rate and length (approx)."""
        if audio.size == 0:
            return audio.astype(np.float32)

        n_orig = len(audio)
        audio_48k = self._resample(
            audio.astype(np.float32), sample_rate, RNNOISE_SAMPLE_RATE
        ).astype(np.float32, copy=False)
        total = len(audio_48k)

        if self._backend == "rnnoise":
            audio_int16 = self._to_int16(audio_48k)
            denoised_frames: list[np.ndarray] = []
            fs = self.frame_size
            for i in range(0, len(audio_int16), fs):
                frame = audio_int16[i : i + fs]
                if len(frame) < fs:
                    frame = np.pad(
                        frame,
                        (0, fs - len(frame)),
                        mode="constant",
                    )
                try:
                    denoised_frame = self.denoiser.process_frame(frame)
                    denoised_frames.append(
                        np.asarray(denoised_frame, dtype=np.int16).ravel()
                    )
                except Exception:
                    denoised_frames.append(frame.astype(np.int16).ravel())
            denoised_48k = np.concatenate(denoised_frames).astype(np.int16)[:total]
            denoised_float = self._to_float32(denoised_48k)
        else:
            denoised_frames = []
            state = self._create()
            try:
                for i in range(0, total, self.frame_size):
                    chunk = audio_48k[i : i + self.frame_size]
                    try:
                        out_int16, _speech = self._process_mono_frame(state, chunk)
                        denoised_frames.append(
                            np.asarray(out_int16, dtype=np.int16).ravel()
                        )
                    except Exception:
                        pad = self.frame_size - len(chunk)
                        if pad > 0:
                            chunk = np.pad(
                                chunk, (0, pad), mode="constant"
                            )
                        denoised_frames.append(self._to_int16(chunk).ravel())
            finally:
                self._destroy(state)
            denoised_48k = np.concatenate(denoised_frames).astype(np.int16)
            denoised_48k = denoised_48k[:total]
            denoised_float = self._to_float32(denoised_48k)

        out = self._resample(
            denoised_float, RNNOISE_SAMPLE_RATE, sample_rate
        ).astype(np.float32)
        if len(out) > n_orig:
            out = out[:n_orig]
        elif len(out) < n_orig:
            out = np.pad(out, (0, n_orig - len(out)), mode="constant")
        return out

    def denoise_file(self, input_path: str, output_path: str | None = None) -> str:
        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_denoised{ext}"

        audio, sample_rate = self._load_wav(input_path)
        denoised = self.denoise_array(audio, sample_rate)
        out_int16 = self._to_int16(denoised)
        wavfile.write(output_path, sample_rate, out_int16)
        print(f"[RNNoise] Denoised audio saved: {output_path}")
        return output_path


def get_noise_canceller() -> NoiseCanceller:
    global _noise_canceller_instance
    if _noise_canceller_instance is None:
        _noise_canceller_instance = NoiseCanceller()
    return _noise_canceller_instance


def denoise_file(input_path: str, output_path: str | None = None) -> str:
    return get_noise_canceller().denoise_file(input_path, output_path)


def denoise_array(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    return get_noise_canceller().denoise_array(audio, sample_rate)
