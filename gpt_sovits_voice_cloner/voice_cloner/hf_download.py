"""Download GPT-SoVITS pretrained assets from Hugging Face (MIT upstream weights)."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


REPO_ID = os.environ.get("GPT_SOVITS_HF_REPO", "lj1995/GPT-SoVITS")


def default_pretrained_dir(gpt_sovits_root: str | Path) -> Path:
    root = Path(gpt_sovits_root).resolve()
    return root / "GPT_SoVITS" / "pretrained_models"


def download_v4_pretrained(
    gpt_sovits_root: str | Path,
    *,
    token: str | None = None,
) -> Path:
    """
    Ensure v4 SoVITS + shared text encoders exist under ``GPT_SoVITS/pretrained_models``.

    Pulls (subset) from ``lj1995/GPT-SoVITS``. Hub layouts vary; we try a flat snapshot under
    ``pretrained_models`` first (matches most HF trees). If that yields nothing, fall back to
    nested ``GPT_SoVITS/pretrained_models/...`` into the repo root.
    """
    root = Path(gpt_sovits_root).resolve()
    dest = default_pretrained_dir(gpt_sovits_root)
    dest.mkdir(parents=True, exist_ok=True)

    patterns_flat = [
        "chinese-roberta-wwm-ext-large/**",
        "chinese-hubert-base/**",
        "gsv-v4-pretrained/**",
        "s1v3.ckpt",
    ]
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        token=token,
        allow_patterns=patterns_flat,
    )
    v4_ckpt = dest / "gsv-v4-pretrained" / "s2Gv4.pth"
    if not v4_ckpt.is_file():
        # Alternate layout: weights live under GPT_SoVITS/pretrained_models on the Hub
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=str(root),
            local_dir_use_symlinks=False,
            token=token,
            allow_patterns=[f"GPT_SoVITS/pretrained_models/{p}" for p in patterns_flat],
        )
    return dest


def download_v2final_pretrained(
    gpt_sovits_root: str | Path,
    *,
    token: str | None = None,
) -> Path:
    """Optional: v2 final bundle (24k path) if you switch config to v2."""
    dest = default_pretrained_dir(gpt_sovits_root)
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        token=token,
        allow_patterns=[
            "chinese-roberta-wwm-ext-large/**",
            "chinese-hubert-base/**",
            "gsv-v2final-pretrained/**",
        ],
    )
    return dest


def ensure_single_file(
    repo_file: str,
    gpt_sovits_root: str | Path,
    *,
    token: str | None = None,
) -> Path:
    """Download one file from the HF repo preserving relative path under pretrained_models."""
    dest = default_pretrained_dir(gpt_sovits_root)
    dest.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=REPO_ID,
        filename=repo_file,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        token=token,
    )
    out = dest / repo_file
    if not out.is_file():
        raise FileNotFoundError(f"Expected file after download: {out}")
    return out
