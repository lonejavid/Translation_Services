"""
GPT-SoVITS (RVC-Boss / MIT) few-shot voice cloning — programmatic inference.

Requires a local checkout of https://github.com/RVC-Boss/GPT-SoVITS with its
dependencies installed (see upstream ``install.sh`` / ``requirements.txt``).

Cross-lingual cloning (e.g. English reference → non-English speech) is supported
for languages the **installed** GPT-SoVITS build lists in ``tts_infer.yaml``.
**Hindi (`hi`) is not officially listed upstream**; we map it to the closest
available frontend with a runtime warning (quality may be poor unless you use
a fork / transliteration / fine-tuned model).
"""

from __future__ import annotations

import copy
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml

from voice_cloner.audio_validate import RefAudioError, analyze_reference_audio
from voice_cloner.hf_download import download_v4_pretrained

logger = logging.getLogger(__name__)

# Upstream language tags accepted by api_v2 ``check_params`` (extend if your fork adds more).
_FALLBACK_LANG_FOR_UNSUPPORTED = "en"


def _resolve_gpt_sovits_root(explicit: str | Path | None) -> Path:
    raw = explicit or os.environ.get("GPT_SOVITS_HOME", "").strip()
    if not raw:
        raise RuntimeError(
            "Set GPT_SOVITS_HOME to the root of a RVC-Boss/GPT-SoVITS git checkout, "
            "or pass gpt_sovits_root= to VoiceCloner()."
        )
    root = Path(raw).expanduser().resolve()
    if not (root / "GPT_SoVITS" / "TTS_infer_pack").is_dir():
        raise RuntimeError(
            f"GPT_SOVITS_HOME={root} does not look like GPT-SoVITS "
            "(missing GPT_SoVITS/TTS_infer_pack/). Clone the real repo and point to it, e.g.\n"
            "  git clone https://github.com/RVC-Boss/GPT-SoVITS.git ~/GPT-SoVITS\n"
            "  export GPT_SOVITS_HOME=\"$HOME/GPT-SoVITS\"\n"
            "Do not use placeholder paths like .../path/to/your/GPT-SoVITS."
        )
    return root


def _pick_device() -> tuple[str, bool]:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", True
    except Exception:
        pass
    if sys.platform == "darwin":
        try:
            import torch

            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps", False
        except Exception:
            pass
    return "cpu", False


def _map_target_language(code: str, supported: set[str]) -> tuple[str, str | None]:
    """
    Map BCP-47-ish codes to GPT-SoVITS ``text_lang`` / ``prompt_lang``.

    Returns (text_lang, warning_or_none).
    """
    c = (code or "en").strip().lower().split("-")[0]
    if c in supported:
        return c, None
    if c == "hi":
        return _FALLBACK_LANG_FOR_UNSUPPORTED, (
            "Target language 'hi' is not in upstream GPT-SoVITS text frontends; "
            f"using '{_FALLBACK_LANG_FOR_UNSUPPORTED}' tokenization — expect weak or broken "
            "Devanagari. Prefer a Hindi-capable fork, transliteration to Latin, or fine-tuning."
        )
    return _FALLBACK_LANG_FOR_UNSUPPORTED, (
        f"Language '{c}' not in supported set {sorted(supported)}; "
        f"falling back to '{_FALLBACK_LANG_FOR_UNSUPPORTED}'."
    )


def _build_runtime_infer_yaml(
    src: Path, device: str, is_half: bool, profile: str
) -> Path:
    """
    TTS_Config loads the ``custom`` block from tts_infer.yaml (else falls back to v2).
    We inject ``custom`` from the chosen profile (e.g. v4 for native 48 kHz stack).
    """
    with open(src, encoding="utf-8") as f:
        full = yaml.safe_load(f)
    if profile not in full or not isinstance(full[profile], dict):
        raise ValueError(
            f"Profile {profile!r} missing in {src} (expected v1/v2/v3/v4/v2Pro/...)."
        )
    custom = copy.deepcopy(full[profile])
    custom["device"] = device
    if device in ("cpu", "mps"):
        custom["is_half"] = False
    else:
        custom["is_half"] = bool(is_half)
    full["custom"] = custom
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    yaml.dump(full, tmp, default_flow_style=False, allow_unicode=True)
    tmp.close()
    return Path(tmp.name)


def _resample_audio_to_48k(wav: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    target = 48_000
    if sr == target:
        return wav.astype(np.float32, copy=False), target
    try:
        import torch
        import torchaudio

        t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sr, target)
        out = resampler(t).squeeze(0).numpy()
        return out, target
    except Exception as e:
        logger.warning("torchaudio resample failed (%s); using scipy fallback", e)
        from scipy import signal

        g = np.gcd(sr, target)
        up, down = target // g, sr // g
        y = signal.resample_poly(wav.astype(np.float64), up, down)
        return y.astype(np.float32), target


class VoiceCloner:
    """
    Few-shot TTS using GPT-SoVITS (v4 by default — native 48 kHz in upstream v4 stack).

    Methods
    -------
    clone_and_save(ref_audio_path, target_text, output_path, ...)
        Run inference and write a 48 kHz WAV (or other format via soundfile).
    """

    def __init__(
        self,
        gpt_sovits_root: str | Path | None = None,
        *,
        infer_profile: str = "v4",
        output_sample_rate: int = 48_000,
        auto_download_weights: bool = True,
        hf_token: str | None = None,
    ) -> None:
        # Set before any step that can raise so __del__ never sees a half-built instance.
        self._patched_yaml: Path | None = None
        self._tts: Any = None
        self._tts_config: Any = None
        self._supported_langs: set[str] = set()

        self._root = _resolve_gpt_sovits_root(gpt_sovits_root)
        self.infer_profile = infer_profile
        self.output_sample_rate = int(output_sample_rate)
        self.auto_download_weights = auto_download_weights
        self.hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        )

        self._device, self._half = _pick_device()

        upstream_yaml = self._root / "GPT_SoVITS" / "configs" / "tts_infer.yaml"
        if not upstream_yaml.is_file():
            raise RuntimeError(f"Missing {upstream_yaml} — update your GPT-SoVITS checkout.")

        self._patched_yaml = _build_runtime_infer_yaml(
            upstream_yaml, self._device, self._half, infer_profile
        )
        logger.info(
            "VoiceCloner device=%s half=%s profile=%s root=%s",
            self._device,
            self._half,
            infer_profile,
            self._root,
        )

    def __del__(self) -> None:
        p = getattr(self, "_patched_yaml", None)
        if p is not None and Path(p).is_file():
            try:
                Path(p).unlink()
            except OSError:
                pass

    def ensure_pretrained_weights(self) -> None:
        """Download v4 pretrained tensors into ``GPT_SoVITS/pretrained_models`` if missing."""
        pm = self._root / "GPT_SoVITS" / "pretrained_models"
        v4_g = pm / "gsv-v4-pretrained" / "s2Gv4.pth"
        if v4_g.is_file():
            return
        if not self.auto_download_weights:
            raise FileNotFoundError(
                f"Missing {v4_g} and auto_download_weights=False. "
                "Download lj1995/GPT-SoVITS v4 assets or run ensure_pretrained_weights manually."
            )
        logger.info("Downloading GPT-SoVITS pretrained weights from Hugging Face …")
        download_v4_pretrained(self._root, token=self.hf_token)

    def _import_upstream_tts(self) -> None:
        root = str(self._root)
        if root not in sys.path:
            sys.path.insert(0, root)
        gs = str(self._root / "GPT_SoVITS")
        if gs not in sys.path:
            sys.path.insert(0, gs)

        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

        cfg_path = self._patched_yaml
        assert cfg_path is not None
        self._tts_config = TTS_Config(str(cfg_path.resolve()))
        langs = getattr(self._tts_config, "languages", None) or []
        self._supported_langs = {str(x).lower() for x in langs}
        if not self._supported_langs:
            self._supported_langs = {"zh", "en", "ja", "ko", "yue"}
        self._tts = TTS(self._tts_config)

    def _lazy_init(self) -> None:
        if self._tts is not None:
            return
        self.ensure_pretrained_weights()
        prev = os.getcwd()
        try:
            os.chdir(self._root)
            self._import_upstream_tts()
        finally:
            os.chdir(prev)

    def clone_and_save(
        self,
        ref_audio_path: str | Path,
        target_text: str,
        output_path: str | Path,
        *,
        target_language: str = "en",
        prompt_text: str = "",
        prompt_language: str = "en",
        text_split_method: str = "cut5",
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        speed_factor: float = 1.0,
        repetition_penalty: float = 1.35,
        validate_reference: bool = True,
    ) -> Path:
        """
        Synthesize ``target_text`` with timbre from ``ref_audio_path``; save to ``output_path``.

        Parameters
        ----------
        target_language :
            Logical language of ``target_text`` (e.g. ``hi``, ``en``). Mapped to GPT-SoVITS codes.
        prompt_language / prompt_text :
            Language and transcript of the reference clip (``en`` for English reference).
            Empty ``prompt_text`` is allowed (upstream often still clones).
        """
        if validate_reference:
            rep = analyze_reference_audio(ref_audio_path)
            for w in rep.warnings:
                logger.warning("[ref-audio] %s", w)

        ref_abs = str(Path(ref_audio_path).expanduser().resolve())
        out_p = Path(output_path).expanduser().resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        self._lazy_init()
        assert self._tts is not None

        tl, warn = _map_target_language(target_language, self._supported_langs)
        if warn:
            logger.warning("%s", warn)

        pl = prompt_language.strip().lower().split("-")[0]
        if pl not in self._supported_langs:
            logger.warning(
                "prompt_language=%r not in supported %s — using 'en'",
                pl,
                self._supported_langs,
            )
            pl = "en"

        req: dict[str, Any] = {
            "text": target_text.strip(),
            "text_lang": tl,
            "ref_audio_path": ref_abs,
            "aux_ref_audio_paths": [],
            "prompt_text": prompt_text or "",
            "prompt_lang": pl,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "text_split_method": text_split_method,
            "batch_size": 1,
            "batch_threshold": 0.75,
            "split_bucket": True,
            "speed_factor": float(speed_factor),
            "fragment_interval": 0.3,
            "seed": -1,
            "media_type": "wav",
            "streaming_mode": False,
            "parallel_infer": True,
            "repetition_penalty": float(repetition_penalty),
            "sample_steps": 32,
            "super_sampling": False,
            "overlap_length": 2,
            "min_chunk_length": 16,
        }

        if not req["text"]:
            raise ValueError("target_text is empty")

        prev = os.getcwd()
        try:
            os.chdir(self._root)
            gen = self._tts.run(req)
            sr_native, audio = next(gen)
        finally:
            os.chdir(prev)

        wav = np.asarray(audio, dtype=np.float32).squeeze()
        if wav.ndim > 1:
            wav = np.mean(wav, axis=-1)

        if self.output_sample_rate and self.output_sample_rate != int(sr_native):
            wav, sr_out = _resample_audio_to_48k(wav, int(sr_native))
        else:
            sr_out = int(sr_native)

        # 48 kHz float WAV for editing / mastering; use PCM_16 if you need smaller files.
        sf.write(str(out_p), wav, sr_out, subtype="FLOAT", format="WAV")
        logger.info("Wrote %s (sr=%d, samples=%d)", out_p, sr_out, wav.shape[0])
        return out_p


