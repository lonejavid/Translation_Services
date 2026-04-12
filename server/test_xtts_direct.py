#!/usr/bin/env python3
"""Call Coqui XTTS v2 directly with Hindi text (isolates pipeline vs model)."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import soundfile as sf

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

test_text = (
    "मैं एक मनमौजी लड़की बनकर बड़ी हुई हूं। मेरे माता-पिता हमेशा मुझसे कहते थे, "
    "यदि आपकी कोई राय है, तो उसे साझा करने से न डरें।"
)

print("Testing XTTS with Hindi text:")
print(f"Text: {test_text}")
print(f"Text bytes: {test_text.encode('utf-8')!r}")

try:
    from TTS.api import TTS
    import torch

    use_gpu = torch.cuda.is_available()
    print(f"\nLoading XTTS (gpu={use_gpu})…")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=use_gpu)

    print("Generating audio…")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_file = f.name

    # XTTS v2 requires a speaker reference WAV
    ref_wav = os.environ.get("XTTS_TEST_SPEAKER_WAV", "").strip()
    if not ref_wav or not Path(ref_wav).is_file():
        cache = _ROOT / "cache"
        if cache.is_dir():
            found = sorted(cache.glob("*_xtts_ref.wav"))
            if found:
                ref_wav = str(found[0])
                print(f"Using speaker_wav={ref_wav!r}")
    if not ref_wav or not Path(ref_wav).is_file():
        print(
            "Need a reference WAV: set XTTS_TEST_SPEAKER_WAV or add "
            "server/cache/{{key}}_xtts_ref.wav"
        )
        sys.exit(2)

    tts.tts_to_file(
        text=test_text,
        language="hi",
        file_path=temp_file,
        speaker_wav=ref_wav,
        split_sentences=True,
    )

    print(f"✓ Wrote {temp_file}")
    data, sr = sf.read(temp_file)
    duration = len(data) / float(sr)
    print(f"Duration: {duration:.2f}s sample_rate={sr}")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
