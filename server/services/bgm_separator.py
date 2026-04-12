"""
Background music / ambience separation and re-mixing for Netflix-quality dubbing.

Professional dubbing workflow (Netflix, Prime, Disney+):
  1. Separate the original audio into:
       - vocals track  (the voice we'll replace with TTS)
       - background track (music, ambience, FX — we KEEP this)
  2. Generate the TTS dubbed voice track
  3. Mix dubbed voice + original background at calibrated levels

This is the critical step that separates "AI dubbing" from "professional dubbing".
Without it: dubbed audio sounds sterile / naked compared to the original.
With it:   the dubbed voice sits in the same acoustic space as the original.

Uses Facebook/Meta Demucs htdemucs model (state-of-the-art, fully local, no API).

Env:
  ``BGM_SEPARATION_ENABLED``  — ``1`` (default) / ``0`` to disable
  ``BGM_MODEL``               — demucs model name (default ``htdemucs``)
  ``BGM_VOICE_GAIN_DB``       — gain applied to dubbed voice before mix (default ``0.0``)
  ``BGM_BG_GAIN_DB``          — gain applied to background before mix (default ``-6.0``)
  ``BGM_SEPARATION_DEVICE``   — ``cpu`` or ``cuda`` or ``mps`` (default: auto-detect).
                                On macOS, Demucs shares Metal with mlx-whisper; the server
                                serializes GPU work via ``gpu_exclusive`` (set ``cpu`` here if
                                you still see Metal crashes).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def bgm_separation_enabled() -> bool:
    return os.environ.get("BGM_SEPARATION_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )


def _bgm_model() -> str:
    return os.environ.get("BGM_MODEL", "htdemucs").strip() or "htdemucs"


def _bgm_device() -> str:
    raw = os.environ.get("BGM_SEPARATION_DEVICE", "").strip().lower()
    if raw in ("cpu", "cuda", "mps"):
        return raw
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _db_to_lin(db: float) -> float:
    return 10 ** (db / 20.0)


def separate_vocals(
    audio_path: str,
    output_dir: str | None = None,
    *,
    model_name: str | None = None,
) -> tuple[str | None, str | None]:
    """
    Separate ``audio_path`` into vocals and background (no-vocals) tracks.

    Returns:
        ``(vocals_path, background_path)`` — WAV files in ``output_dir``.
        Either can be ``None`` if separation fails.
    """
    src = Path(audio_path)
    if not src.is_file():
        print(f"[bgm-sep] Source file not found: {audio_path}")
        return None, None

    out_dir = Path(output_dir) if output_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    model = model_name or _bgm_model()
    device = _bgm_device()

    print(f"[bgm-sep] Separating {src.name} with {model} on {device} …")

    try:
        import torch
        from demucs.apply import apply_model
        from demucs.audio import AudioFile
        from demucs.pretrained import get_model

        from services.gpu_exclusive import gpu_exclusive

        with gpu_exclusive():
            separator = get_model(model)
            separator.eval()
            if device != "cpu":
                try:
                    separator = separator.to(device)
                except Exception:
                    device = "cpu"

            wav_obj = AudioFile(str(src))
            wav_tensor = wav_obj.read(
                seek_time=0,
                duration=None,
                streams=0,
                samplerate=separator.samplerate,
                channels=separator.audio_channels,
            )
            if wav_tensor.dim() == 1:
                wav_tensor = wav_tensor.unsqueeze(0)
            if wav_tensor.shape[0] == 1 and separator.audio_channels == 2:
                wav_tensor = wav_tensor.repeat(2, 1)

            wav_input = wav_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                sources = apply_model(separator, wav_input, device=device)[0]

            if device == "mps":
                try:
                    torch.mps.synchronize()
                except Exception:
                    pass

            sources = sources.detach().cpu()

        source_names = separator.sources  # e.g. ['drums', 'bass', 'other', 'vocals']

        sr = separator.samplerate

        # Vocals stem
        if "vocals" in source_names:
            vidx = source_names.index("vocals")
            vocals_np = sources[vidx].numpy().mean(axis=0)  # mono
            vocals_path = str(out_dir / f"{src.stem}_vocals.wav")
            sf.write(vocals_path, vocals_np, sr, subtype="PCM_16")
        else:
            vocals_path = None

        # Background = all stems except vocals
        bg_stems = [i for i, n in enumerate(source_names) if n != "vocals"]
        if bg_stems:
            bg_np = sum(sources[i].numpy().mean(axis=0) for i in bg_stems)
            bg_np = np.clip(bg_np / max(1, len(bg_stems)), -1.0, 1.0).astype(np.float32)
            bg_path = str(out_dir / f"{src.stem}_background.wav")
            sf.write(bg_path, bg_np, sr, subtype="PCM_16")
        else:
            bg_path = None

        print(
            f"[bgm-sep] Done: vocals={Path(vocals_path).name if vocals_path else 'N/A'} "
            f"bg={Path(bg_path).name if bg_path else 'N/A'}"
        )
        return vocals_path, bg_path

    except Exception as e:
        print(f"[bgm-sep] Separation failed: {e!r}")
        return None, None


def mix_dub_with_background(
    dubbed_wav: str,
    background_wav: str,
    output_path: str,
    *,
    voice_gain_db: float | None = None,
    bg_gain_db: float | None = None,
) -> str:
    """
    Mix the dubbed voice track with the original background audio.

    The background is trimmed/padded to match dubbed duration, then mixed
    at calibrated gain levels. Result is peak-normalized and written as WAV.

    Returns ``output_path`` on success, ``dubbed_wav`` if mixing fails.
    """
    vg = voice_gain_db if voice_gain_db is not None else float(
        os.environ.get("BGM_VOICE_GAIN_DB", "0.0") or "0.0"
    )
    bg = bg_gain_db if bg_gain_db is not None else float(
        os.environ.get("BGM_BG_GAIN_DB", "-6.0") or "-6.0"
    )

    try:
        dub_data, dub_sr = sf.read(dubbed_wav, dtype="float32", always_2d=True)
        dub_mono = dub_data.mean(axis=1) if dub_data.shape[1] > 1 else dub_data.squeeze()

        bg_data, bg_sr = sf.read(background_wav, dtype="float32", always_2d=True)
        bg_mono = bg_data.mean(axis=1) if bg_data.shape[1] > 1 else bg_data.squeeze()

        # Resample background to match dubbed sample rate if needed
        if bg_sr != dub_sr:
            import librosa
            bg_mono = librosa.resample(bg_mono, orig_sr=bg_sr, target_sr=dub_sr)

        dub_len = len(dub_mono)
        bg_len = len(bg_mono)

        # Trim or loop background to match dubbed duration
        if bg_len < dub_len:
            repeats = (dub_len // bg_len) + 1
            bg_mono = np.tile(bg_mono, repeats)
        bg_mono = bg_mono[:dub_len].astype(np.float32)

        # Apply gain
        dub_mixed = dub_mono * _db_to_lin(vg)
        bg_mixed = bg_mono * _db_to_lin(bg)

        # Mix
        mixed = dub_mixed + bg_mixed

        # Peak-normalize to -1 dBFS
        peak = float(np.max(np.abs(mixed))) + 1e-9
        if peak > 0.891:  # -1 dBFS
            mixed = mixed / peak * 0.891

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, mixed.astype(np.float32), dub_sr, subtype="PCM_16")

        print(
            f"[bgm-sep] Mixed: voice@{vg:+.1f}dB + bg@{bg:+.1f}dB "
            f"→ {Path(output_path).name} ({dub_len/dub_sr:.1f}s)"
        )
        return output_path

    except Exception as e:
        print(f"[bgm-sep] Mix failed: {e!r}; returning original dubbed audio")
        return dubbed_wav
