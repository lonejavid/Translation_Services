# YouTube Video Translator

Local **YouTube → transcribe (Whisper) → translate (Google via [deep-translator](https://github.com/nidhaloff/deep-translator) by default, or optional on-device NLLB/IndicTrans2) → dub** stack: **Hindi** uses **[edge-tts](https://github.com/rany2/edge-tts)** (Microsoft neural voices, free, needs internet) by default; other languages / `TTS_USE_EDGE=0` use **XTTS / MMS-TTS**. **React** UI + **FastAPI** backend.

## Quick start

**Backend** (port **8000**):

```bash
cd server
./run.sh
```

**Frontend** (port **3000**):

```bash
cd client
npm install
npm start
```

**Translation (default):** **`TRANSLATION_BACKEND=google`** — uses **`deep-translator`** (`GoogleTranslator`) on **timestamp-free** plain text (`cache/*_source_plain.txt` → `*_translated_plain.txt`), then rebuilds timed subtitles for TTS. **Requires internet.**

**Offline / local:** **`TRANSLATION_BACKEND=local`** — PyTorch models; set **`TRANSLATION_ENGINE=nllb`** or **`indictrans2`**, and **`HF_TOKEN`** for gated models where needed.

See **`server/run.sh`** for Coqui XTTS license env.

## Documentation

**Full system description** (pipeline, libraries, API, data formats, env vars):

→ **[`docs/SYSTEM_GUIDE.md`](docs/SYSTEM_GUIDE.md)**

## Project layout

| Area | Path |
|------|------|
| API & orchestration | `server/main.py` |
| Download / STT / translate / TTS | `server/services/` |
| Generated files | `server/cache/` |
| Web UI | `client/` |

## License / compliance

- **Coqui XTTS**: follow [CPML](https://coqui.ai/cpml) or a commercial license where required.
- **Hugging Face models**: accept model terms on HF where applicable.
