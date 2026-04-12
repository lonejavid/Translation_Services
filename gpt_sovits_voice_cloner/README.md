# GPT-SoVITS `VoiceCloner` (inference bridge)

Python wrapper around **[RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)** (MIT) for programmatic few-shot TTS with optional **Hugging Face** weight download.

## Layout

```
gpt_sovits_voice_cloner/
├── README.md
├── requirements.txt      # Wrapper + torch/torchaudio; upstream has its own stack
├── main.py               # CLI example
└── voice_cloner/
    ├── __init__.py
    ├── cloner.py         # VoiceCloner.clone_and_save(...)
    ├── audio_validate.py # Reference duration / sample-rate checks
    └── hf_download.py    # snapshot_download lj1995/GPT-SoVITS (v4 subset)
```

## Setup

1. **Clone upstream** (heavy; follow their docs):

   ```bash
   git clone https://github.com/RVC-Boss/GPT-SoVITS.git
   cd GPT-SoVITS && bash install.sh --device CU126 --source HF   # or CPU / MPS
   ```

2. **Point the wrapper** (use your **real** clone path — no placeholders; put the comment on its own line):

   ```bash
   # Example only — change to where you cloned the repo, e.g. $HOME/code/GPT-SoVITS
   export GPT_SOVITS_HOME="$HOME/GPT-SoVITS"
   ```

   **zsh note:** If you see `export: not valid in this context`, you pasted a bad character or merged lines. Run only the `export` line, with no `#` on the same line, or use:

   `export GPT_SOVITS_HOME="$HOME/GPT-SoVITS"`

3. **Install this folder** in a **dedicated venv** (recommended). Installing with the system
   `python3 -m pip` can change **numpy/torch** for your whole user account and break other tools
   (e.g. Coqui **TTS** wants `numpy==1.22.0` on Python ≤3.10).

   ```bash
   cd gpt_sovits_voice_cloner
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -r requirements.txt
   ```

   Or use the **same** conda/venv you already created for GPT-SoVITS upstream.

4. **Weights**: first run downloads v4 checkpoints into  
   `$GPT_SOVITS_HOME/GPT_SoVITS/pretrained_models/`  
   (requires `HF_TOKEN` only for gated assets; public weights usually work without).

## Usage

```bash
python main.py ./ref_en.wav "Hello, this is a test." ./out.wav --target-language en --prompt-language en
```

- **Cross-lingual**: English reference + `prompt_language=en` + Hindi (or other) `text` — see logs; **Hindi is not a first-class `text_lang` upstream**; the wrapper maps `hi` → `en` tokenization with a warning. For production Hindi, use a fork, transliteration, or fine-tuned models.

## API

```python
from voice_cloner import VoiceCloner

cloner = VoiceCloner(infer_profile="v4", output_sample_rate=48_000)
cloner.clone_and_save(
    "ref.wav",
    "Target line in chosen language",
    "out.wav",
    target_language="en",
    prompt_language="en",
    prompt_text="Optional transcript of ref.wav",
)
```

## Notes

- **Device**: CUDA when available; else Apple **MPS** (if `torch` supports it); else **CPU**.
- **Output**: resampled to **48 kHz** float WAV (configurable via `output_sample_rate`).
- **Fish-Speech V1.5** is not bundled here; this module targets GPT-SoVITS only.
