#!/usr/bin/env python3
"""
Example: few-shot clone with GPT-SoVITS (upstream repo required).

Prerequisites
-------------
1. Clone:  git clone https://github.com/RVC-Boss/GPT-SoVITS.git
2. Install upstream deps from that repo (see its README / install.sh).
3. export GPT_SOVITS_HOME=/path/to/GPT-SoVITS
4. pip install -r requirements.txt   # this folder (wrapper deps)

Run
---
python main.py /path/to/ref_english.wav "नमस्ते, यह एक परीक्षण है।" out_hi.wav --target-language hi
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voice_cloner import VoiceCloner

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> int:
    p = argparse.ArgumentParser(description="GPT-SoVITS VoiceCloner example")
    p.add_argument("ref_audio", type=Path, help="5–10s clean reference WAV/FLAC (English OK)")
    p.add_argument("text", type=str, help="Text to synthesize in target language")
    p.add_argument("output", type=Path, help="Output WAV path (48 kHz float)")
    p.add_argument(
        "--gpt-sovits-home",
        type=Path,
        default=None,
        help="Override GPT_SOVITS_HOME",
    )
    p.add_argument(
        "--target-language",
        default="en",
        help="BCP-47 language of ``text`` (hi falls back to en tokenizer — see logs)",
    )
    p.add_argument(
        "--prompt-text",
        default="",
        help="Optional: transcript of reference audio (same language as --prompt-language)",
    )
    p.add_argument(
        "--prompt-language",
        default="en",
        help="Language of reference speech (e.g. en)",
    )
    p.add_argument(
        "--profile",
        default="v4",
        choices=["v1", "v2", "v3", "v4", "v2Pro", "v2ProPlus"],
        help="tts_infer.yaml profile (v4 = upstream native 48 kHz path)",
    )
    p.add_argument("--no-download", action="store_true", help="Fail if pretrained missing")
    args = p.parse_args()

    cloner = VoiceCloner(
        gpt_sovits_root=args.gpt_sovits_home,
        infer_profile=args.profile,
        output_sample_rate=48_000,
        auto_download_weights=not args.no_download,
    )
    out = cloner.clone_and_save(
        args.ref_audio,
        args.text,
        args.output,
        target_language=args.target_language,
        prompt_text=args.prompt_text,
        prompt_language=args.prompt_language,
    )
    print(f"OK → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
