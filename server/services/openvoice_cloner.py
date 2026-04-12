"""
Neural voice cloning via OpenVoice v2 (MyShell AI).

WHY this works better than pitch-shifting or XTTS alone
-------------------------------------------------------
XTTS is a *text-to-speech* model that tries to do voice cloning as a side-
effect of conditioning on a reference.  It works acceptably for mono-lingual
cloning but degrades badly cross-lingually (English voice → Hindi speech)
because the decoder must simultaneously handle phoneme inventory changes AND
voice identity — it often prioritises one at the cost of the other.

Pitch-shifting / spectral tilt (voice_converter.py) improves nothing
perceptible because voice identity is NOT just pitch:  it lives in formant
resonances, voice quality (breathiness, creak), nasal coupling, tract length
normalisation — none of which a simple filter can reproduce.

OpenVoice v2 architecture
--------------------------
  ┌──────────────────────────────────────────────────────┐
  │  1. Speaker Encoder  (SE)                            │
  │     CNN → 256-d "tone color" embedding               │
  │     Language-agnostic: same embedding for EN/HI/ES   │
  ├──────────────────────────────────────────────────────┤
  │  2. ToneColorConverter  (flow-based VITS decoder)    │
  │     Takes: (source audio waveform,                   │
  │             src_speaker_embedding,                   │
  │             tgt_speaker_embedding)                   │
  │     Outputs: source waveform with tgt voice identity │
  └──────────────────────────────────────────────────────┘

Accuracy enhancements in this version
--------------------------------------
  1. AVERAGED reference embedding — the reference WAV is split into
     multiple overlapping chunks; an embedding is extracted for each and
     they are mean-averaged.  A single chunk is vulnerable to noise or
     atypical phoneme content; averaging gives a stable centroid of the
     speaker's actual voice identity.

  2. CACHED Edge TTS src_se — instead of extracting src_se from a short
     per-segment clip (which Whisper can't transcribe, causing extraction
     failures and noisy fallbacks), we synthesise a longer reference clip
     for the Edge TTS voice once, extract src_se from that, and reuse it
     for every segment.  The Edge TTS voice is deterministic, so one
     extraction is sufficient.

  3. SPECTRAL ENVELOPE MATCHING — after OpenVoice converts the audio, we
     compare the short-time spectral centroid of the output vs the
     reference and apply a lightweight IIR tilt to correct residual
     spectral differences.  This closes the remaining timbre gap between
     the converted voice and the reference speaker.

  4. LOW TAU (default 0.15) — tau controls how strongly the target
     embedding is applied.  0.3 (old default) is a blend; 0.15 pushes
     the output firmly toward the reference speaker with minimal
     artefacts.

Pipeline for dubbing
--------------------
  1. Edge TTS synthesises clear target-language speech (pronunciation only).
  2. _build_averaged_embedding(reference_wav) → tgt_se  (stable reference)
  3. _get_edge_src_se(edge_voice_name)         → src_se  (cached TTS voice)
  4. converter.convert(edge_wav, src_se, tgt_se, tau=0.15)
  5. _spectral_match_output(output, reference) → final refined audio

Result:  Trump speaking in Hindi — clear Hindi pronunciation, Trump's voice.

Auto-downloads ~50 MB checkpoints from HuggingFace on first use.
Cached in HuggingFace default cache (or OPENVOICE_CACHE_DIR).

Env:
  OPENVOICE_ENABLED     — 1 (default) / 0 to disable
  OPENVOICE_TAU         — 0.0–1.0 (default 0.15); lower = stronger reference
                          matching; 0.1 = very close to reference, 0.3 = blend
  OPENVOICE_N_CHUNKS    — number of reference chunks to average (default 5)
  OPENVOICE_CHUNK_SEC   — seconds per reference chunk (default 4.0)
  OPENVOICE_SPEC_MATCH  — 1 (default) post-process spectral envelope matching
  OPENVOICE_CACHE_DIR   — HuggingFace cache path override
"""
from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_converter = None
_converter_lock = threading.Lock()
_load_failed = False

# Cache for the averaged reference embedding (keyed by path)
_ref_embedding_cache: dict[str, object] = {}
_ref_emb_lock = threading.Lock()

# Cache for the Edge TTS source embedding (keyed by voice name string)
_src_se_cache: dict[str, object] = {}
_src_se_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

def is_enabled() -> bool:
    return os.environ.get("OPENVOICE_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )


def _tau() -> float:
    try:
        return max(0.05, min(1.0, float(os.environ.get("OPENVOICE_TAU", "0.15") or "0.15")))
    except ValueError:
        return 0.15


def _n_chunks() -> int:
    try:
        return max(1, int(os.environ.get("OPENVOICE_N_CHUNKS", "5") or "5"))
    except ValueError:
        return 5


def _chunk_sec() -> float:
    try:
        return max(2.0, float(os.environ.get("OPENVOICE_CHUNK_SEC", "4.0") or "4.0"))
    except ValueError:
        return 4.0


def _spectral_match_enabled() -> bool:
    return os.environ.get("OPENVOICE_SPEC_MATCH", "1").strip().lower() not in (
        "0", "false", "no", "off"
    )


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


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def _get_converter():
    """Return the ToneColorConverter singleton, loading it on first call."""
    global _converter, _load_failed

    if _converter is not None:
        return _converter
    if _load_failed:
        return None

    with _converter_lock:
        if _converter is not None:
            return _converter
        if _load_failed:
            return None

        try:
            from openvoice.api import ToneColorConverter
            from huggingface_hub import snapshot_download

            print("[openvoice] Loading OpenVoice v2 checkpoints from HuggingFace …")
            cache_dir = os.environ.get("OPENVOICE_CACHE_DIR", "").strip() or None

            ckpt_dir = snapshot_download(
                repo_id="myshell-ai/OpenVoiceV2",
                cache_dir=cache_dir,
                ignore_patterns=[
                    "*.ipynb", "demo*", "example*",
                    "*.mp3", "*.mp4", "*.txt", "*.md",
                    "resources/*",
                ],
            )

            candidates = [
                (os.path.join(ckpt_dir, "converter", "config.json"),
                 os.path.join(ckpt_dir, "converter", "checkpoint.pth")),
                (os.path.join(ckpt_dir, "checkpoints_v2", "converter", "config.json"),
                 os.path.join(ckpt_dir, "checkpoints_v2", "converter", "checkpoint.pth")),
            ]
            config_path, ckpt_path = next(
                (c, p) for c, p in candidates if Path(c).is_file() and Path(p).is_file()
            )

            from services.gpu_exclusive import gpu_exclusive

            with gpu_exclusive():
                device = _device()
                converter = ToneColorConverter(config_path, device=device)
                converter.load_ckpt(ckpt_path)
                if device == "mps":
                    try:
                        import torch

                        torch.mps.synchronize()
                    except Exception:
                        pass

            _converter = converter
            print(f"[openvoice] ToneColorConverter ready  device={device}")
            return _converter

        except ImportError as exc:
            print(
                f"[openvoice] openvoice package not installed ({exc!r}). "
                "Voice cloning will fall back to pitch/tilt matching."
            )
            _load_failed = True
            return None
        except Exception as exc:
            print(f"[openvoice] Load failed: {exc!r}")
            _load_failed = True
            return None


# ---------------------------------------------------------------------------
# Averaged reference embedding
# ---------------------------------------------------------------------------

def _extract_embedding_from_chunks(wav: np.ndarray, sr: int, converter) -> "torch.Tensor | None":
    """
    Split ``wav`` into overlapping chunks and return the mean embedding.

    Averaging across multiple chunks of the reference audio produces a stable,
    representative centroid of the speaker's voice identity — much more robust
    than a single-clip extraction that may be dominated by one phoneme or a
    moment of noise.
    """
    import torch
    import soundfile as sf

    chunk_samples = int(_chunk_sec() * sr)
    n_chunks = _n_chunks()

    if len(wav) < chunk_samples:
        # Audio shorter than one chunk — use the whole thing
        chunks = [wav]
    else:
        # Evenly spaced chunks across the full reference
        step = max(1, (len(wav) - chunk_samples) // max(1, n_chunks - 1))
        offsets = [i * step for i in range(n_chunks)]
        # Always include the last chunk to capture full speaker range
        offsets = sorted(set(offsets + [max(0, len(wav) - chunk_samples)]))
        chunks = [wav[o: o + chunk_samples] for o in offsets if o + chunk_samples <= len(wav)]
        if not chunks:
            chunks = [wav]

    embeddings = []
    with tempfile.TemporaryDirectory() as tmp:
        for idx, chunk in enumerate(chunks):
            chunk_path = os.path.join(tmp, f"ref_chunk_{idx}.wav")
            sf.write(chunk_path, chunk.astype(np.float32), sr)
            try:
                # Call extract_se directly (bypasses Whisper segmentation)
                # This is the internal OpenVoice method that accepts a list of wav paths.
                se = converter.extract_se([chunk_path])
                if se is not None:
                    embeddings.append(se)
            except Exception as exc:
                print(f"[openvoice] Chunk {idx} embedding failed: {exc!r}")
                continue

    if not embeddings:
        return None

    # Mean-average all embeddings — this is the robust speaker centroid
    stacked = torch.stack([e.squeeze(0) if e.dim() == 3 else e for e in embeddings])
    mean_se = stacked.mean(dim=0, keepdim=True)
    print(
        f"[openvoice] Reference embedding: averaged {len(embeddings)}/{len(chunks)} chunks"
    )
    return mean_se


def _build_reference_embedding(ref_wav_path: str, converter):
    """
    Build the averaged reference speaker embedding for ``ref_wav_path``.
    Falls back to single-clip extraction if chunk averaging fails.
    """
    import soundfile as sf

    try:
        data, sr = sf.read(ref_wav_path, dtype="float32", always_2d=True)
        wav = data.mean(axis=1) if data.shape[1] > 1 else data.squeeze()
        wav = wav.astype(np.float32)
    except Exception as exc:
        print(f"[openvoice] Could not load reference {Path(ref_wav_path).name!r}: {exc!r}")
        return None

    se = _extract_embedding_from_chunks(wav, sr, converter)
    if se is not None:
        return se

    # Fallback: single-file extraction via se_extractor
    print("[openvoice] Chunk averaging failed; falling back to single-clip extraction")
    try:
        from openvoice import se_extractor
        with tempfile.TemporaryDirectory() as tmp:
            se, _ = se_extractor.get_se(ref_wav_path, converter, target_dir=tmp, vad=False)
        return se
    except Exception as exc:
        print(f"[openvoice] Single-clip extraction failed: {exc!r}")
        return None


def get_reference_embedding(ref_wav_path: str, converter):
    """Return a cached averaged speaker embedding for ``ref_wav_path``."""
    with _ref_emb_lock:
        if ref_wav_path in _ref_embedding_cache:
            return _ref_embedding_cache[ref_wav_path]

    se = _build_reference_embedding(ref_wav_path, converter)

    with _ref_emb_lock:
        if se is not None:
            _ref_embedding_cache[ref_wav_path] = se
    return se


def clear_reference_cache() -> None:
    """Call this between pipeline runs to free GPU memory held by embeddings."""
    with _ref_emb_lock:
        _ref_embedding_cache.clear()
    with _src_se_lock:
        _src_se_cache.clear()


# ---------------------------------------------------------------------------
# Cached Edge TTS source embedding
# ---------------------------------------------------------------------------

def _synthesize_edge_reference_clip(voice_name: str, work_dir: str) -> str | None:
    """
    Synthesise a ~6 second Edge TTS clip for ``voice_name`` to use as
    a stable source embedding reference.  The content doesn't matter —
    we just need a long enough pure-TTS clip for reliable embedding
    extraction.
    """
    try:
        from services.edge_tts_synth import synthesize_hindi_to_numpy
        import soundfile as sf

        # Fixed reference sentence long enough for Whisper to transcribe
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "She sells sea shells by the sea shore. "
            "How much wood would a woodchuck chuck."
        )
        wav, sr = synthesize_hindi_to_numpy(text, voice_name, work_dir)
        if wav.size == 0:
            return None

        out_path = os.path.join(work_dir, f"edge_ref_{voice_name.replace('-', '_')}.wav")
        sf.write(out_path, wav.astype(np.float32), sr)
        return out_path
    except Exception as exc:
        print(f"[openvoice] Edge TTS reference synthesis failed: {exc!r}")
        return None


def get_src_embedding(voice_name: str | None, converter) -> "torch.Tensor | None":
    """
    Return a cached source embedding for the Edge TTS voice ``voice_name``.

    Extracting src_se from a short per-segment clip often fails because
    Whisper can't transcribe 1–2 second clips, yielding a noisy zero-fallback.
    Instead we synthesise a longer reference clip for this voice once, extract
    src_se from that, and cache it permanently for the session.
    """
    import torch

    key = voice_name or "__default__"

    with _src_se_lock:
        if key in _src_se_cache:
            return _src_se_cache[key]

    se = None
    if voice_name:
        with tempfile.TemporaryDirectory() as tmp:
            ref_path = _synthesize_edge_reference_clip(voice_name, tmp)
            if ref_path:
                se = _extract_embedding_from_chunks(
                    *_load_wav(ref_path), converter
                )
                if se is None:
                    # Fallback: direct extract_se
                    try:
                        se = converter.extract_se([ref_path])
                    except Exception:
                        pass

    with _src_se_lock:
        if se is not None:
            _src_se_cache[key] = se
            print(f"[openvoice] src_se cached for voice {voice_name!r}")
        else:
            print(
                f"[openvoice] src_se extraction failed for {voice_name!r}; "
                "will use zero embedding"
            )

    return se


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _load_wav(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data.squeeze()
    return mono.astype(np.float32), int(sr)


def _save_wav(wav: np.ndarray, sr: int, path: str) -> None:
    import soundfile as sf
    sf.write(path, wav.astype(np.float32), sr, subtype="PCM_16")


# ---------------------------------------------------------------------------
# Spectral envelope matching
# ---------------------------------------------------------------------------

def _spectral_envelope_match(
    converted: np.ndarray,
    converted_sr: int,
    reference: np.ndarray,
    reference_sr: int,
) -> np.ndarray:
    """
    Apply a lightweight spectral tilt to ``converted`` so its spectral
    centroid matches the reference speaker's centroid.

    OpenVoice captures formant *identity* well but can leave a residual
    spectral tilt difference — especially when the TTS voice (e.g. a
    bright female US voice) has very different energy distribution from
    the reference speaker (e.g. a deep male voice).  This step corrects
    that residual gap.

    The correction is a single-pole IIR shelf filter whose gain is
    proportional to the centroid difference — gentle enough to be
    imperceptible as processing, but meaningful enough to close the
    perceptual gap.
    """
    try:
        from scipy import signal as sig

        def centroid(y, sr):
            # Short-time spectral centroid averaged across frames
            frame = min(2048, len(y))
            hop = frame // 4
            freqs = np.fft.rfftfreq(frame, 1.0 / sr)
            cents = []
            for start in range(0, len(y) - frame + 1, hop):
                mag = np.abs(np.fft.rfft(y[start:start + frame] * np.hanning(frame)))
                total = mag.sum() + 1e-12
                cents.append(float((freqs * mag).sum() / total))
            return float(np.median(cents)) if cents else 2000.0

        ref_sr_c = centroid(reference, reference_sr)
        out_sr_c = centroid(converted, converted_sr)

        if ref_sr_c <= 0 or out_sr_c <= 0:
            return converted

        ratio = ref_sr_c / out_sr_c
        # Only apply if there is a meaningful difference (> 5%)
        if abs(ratio - 1.0) < 0.05:
            return converted

        # Clamp correction to ±6 dB to avoid over-processing
        gain_db = max(-6.0, min(6.0, 20.0 * np.log10(ratio)))
        gain_lin = 10 ** (gain_db / 20.0)

        # High-shelf: boost or cut everything above 1 kHz
        nyq = converted_sr / 2.0
        shelf_freq = min(1000.0, nyq * 0.8)
        sos = sig.butter(2, shelf_freq, btype="highpass", fs=converted_sr, output="sos")
        high = sig.sosfilt(sos, converted)
        corrected = converted + (gain_lin - 1.0) * high

        print(
            f"[openvoice] Spectral match: ref={ref_sr_c:.0f}Hz out={out_sr_c:.0f}Hz "
            f"gain={gain_db:+.1f}dB"
        )
        return np.clip(corrected, -1.0, 1.0).astype(np.float32)

    except Exception as exc:
        print(f"[openvoice] Spectral match skipped: {exc!r}")
        return converted


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def clone_voice_for_segment(
    synth_wav: np.ndarray,
    synth_sr: int,
    ref_wav_path: str,
    *,
    tau: Optional[float] = None,
    edge_voice_name: Optional[str] = None,
) -> tuple[np.ndarray, int]:
    """
    Apply the reference speaker's voice characteristics to ``synth_wav``.

    Parameters
    ----------
    synth_wav       : Synthesized speech from Edge TTS (float32 mono)
    synth_sr        : Sample rate of ``synth_wav``
    ref_wav_path    : Path to the extracted voice reference WAV
    tau             : Tone color blending strength (overrides OPENVOICE_TAU env)
    edge_voice_name : Edge TTS voice used for synthesis (e.g.
                      ``"hi-IN-MadhurNeural"``).  Used to fetch the cached
                      src_se so short-segment extraction failures are avoided.

    Returns
    -------
    (converted_wav, converted_sr)  on success
    (synth_wav, synth_sr)          on any failure (no-op fallback)
    """
    if not is_enabled():
        return synth_wav, synth_sr
    if synth_wav.size == 0:
        return synth_wav, synth_sr
    if not ref_wav_path or not Path(ref_wav_path).is_file():
        return synth_wav, synth_sr

    converter = _get_converter()
    if converter is None:
        return synth_wav, synth_sr

    effective_tau = tau if tau is not None else _tau()

    try:
        import torch
        import soundfile as sf

        # ── 1. Get averaged reference embedding (cached) ──────────────────
        tgt_se = get_reference_embedding(ref_wav_path, converter)
        if tgt_se is None:
            print("[openvoice] Could not build reference embedding; skipping clone")
            return synth_wav, synth_sr

        # ── 2. Get source (Edge TTS) embedding (cached per voice) ─────────
        # Using a long cached clip avoids Whisper failure on short segments.
        src_se = get_src_embedding(edge_voice_name, converter)
        if src_se is None:
            # Last resort: zero embedding maps output directly onto tgt_se
            src_se = torch.zeros_like(tgt_se)

        # ── 3. Run ToneColorConverter ──────────────────────────────────────
        with tempfile.TemporaryDirectory() as tmp:
            src_path = os.path.join(tmp, "src.wav")
            out_path = os.path.join(tmp, "out.wav")
            _save_wav(synth_wav, synth_sr, src_path)

            from services.gpu_exclusive import gpu_exclusive

            # message must be non-empty: string_to_bits("") → shape (0,)
            # which can't be broadcast to (0,8) in openvoice/utils.py
            with gpu_exclusive():
                converter.convert(
                    audio_src_path=src_path,
                    src_se=src_se,
                    tgt_se=tgt_se,
                    output_path=out_path,
                    tau=effective_tau,
                    message="@MyShell",
                )
                if _device() == "mps":
                    try:
                        torch.mps.synchronize()
                    except Exception:
                        pass

            if not Path(out_path).is_file() or Path(out_path).stat().st_size < 1024:
                print("[openvoice] Converter produced empty output; using original")
                return synth_wav, synth_sr

            out_wav, out_sr = _load_wav(out_path)

        if out_wav.size == 0:
            return synth_wav, synth_sr

        # Sanity: output must be at least 10% of input duration
        if out_wav.size < int(len(synth_wav) * 0.1):
            print(
                f"[openvoice] Output too short ({out_wav.size} vs "
                f"{len(synth_wav)}); using original"
            )
            return synth_wav, synth_sr

        # ── 4. Spectral envelope matching ─────────────────────────────────
        if _spectral_match_enabled():
            try:
                ref_wav, ref_sr = _load_wav(ref_wav_path)
                out_wav = _spectral_envelope_match(out_wav, out_sr, ref_wav, ref_sr)
            except Exception as exc:
                print(f"[openvoice] Spectral match error: {exc!r}")

        print(
            f"[openvoice] Clone done: {len(out_wav)/out_sr:.2f}s @ {out_sr}Hz "
            f"(tau={effective_tau:.2f})"
        )
        return out_wav, out_sr

    except Exception as exc:
        print(f"[openvoice] Segment conversion failed: {exc!r}")
        return synth_wav, synth_sr


# ---------------------------------------------------------------------------
# Pre-warm
# ---------------------------------------------------------------------------

def warmup() -> bool:
    """
    Pre-load the ToneColorConverter so the first real segment is fast.
    Call from FastAPI startup event in a background thread.
    Returns True if the model loaded successfully.
    """
    if not is_enabled():
        return False
    return _get_converter() is not None
