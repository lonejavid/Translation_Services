# YouTube Video Translator — System Guide

This document describes **what the project does**, **which libraries and tools are used**, **how data flows**, and **where each step is implemented** in code. Read this to understand your system end-to-end.

---

## 1. What the system does (one paragraph)

A **React** web app sends a **YouTube URL** and **target language** to a **FastAPI** server. The server **downloads audio**, optionally **denoises** it, **transcribes** speech with **mlx-whisper** (Apple Silicon), **translates** segments using **`deep-translator` (Google)** by default (`TRANSLATION_BACKEND=google`: timestamped `original.txt`, then **timestamp-free** `source_plain.txt` → `translated_plain.txt`, then timed `translated.txt` for TTS), or **on-device** NLLB / IndicTrans2 when `TRANSLATION_BACKEND=local`, then **synthesizes dubbed speech** — for **Hindi**, **Microsoft Edge neural TTS** via **`edge-tts`** by default (free, no API key, **internet**), otherwise **Coqui XTTS v2** or **Facebook MMS-TTS** — exports an **MP3** and **subtitle JSON**, and returns URLs. The browser plays the **original YouTube video** (muted) together with the **dubbed MP3** and shows **bilingual-style subtitles**, using a **`dub_sync` map** so long translated lines are not skipped in audio.

Core models run **locally**; **default translation** and **default Hindi dub** use **free** online services (Google Translate via deep-translator, Microsoft Edge TTS). Set **`TTS_USE_EDGE=0`** for **fully local** Hindi dub (XTTS/MMS only).

---

## 2. High-level architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Browser (React, port 3000)                                              │
│  Home → paste URL / language → navigate to Player                        │
│  Player → POST /api/process-video → SSE /api/process-video/stream        │
│  VideoPlayer: YouTube iframe + <audio> dub + SubtitleDisplay             │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ HTTP / SSE (CORS :3000 ↔ :8000)
┌───────────────────────────────▼─────────────────────────────────────────┐
│  FastAPI (server/main.py, port 8000)                                     │
│  Orchestrates _run_pipeline in a thread; pushes progress to job buffer   │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
  server/cache/          Hugging Face models      System: ffmpeg,
  *.mp3, *.json,         (Whisper, IndicTrans2,   yt-dlp CLI, Rubber Band
  *_meta.json            XTTS, MMS-VITS)         (via pyrubberband)
```

---

## 3. Repository layout (what lives where)

| Path | Purpose |
|------|---------|
| `client/` | React app (Create React App): `Home`, `Player`, `VideoPlayer`, `SubtitleDisplay` |
| `client/src/App.js` | Router; **`API_BASE`** defaults to `http://127.0.0.1:8000` |
| `server/main.py` | FastAPI app, routes, **`_run_pipeline`**, cache key, job/SSE buffers |
| `server/services/downloader.py` | YouTube → WAV (**yt-dlp**, **ffmpeg**) |
| `server/services/noise_canceller.py` | Optional **RNNoise** (**pyrnnoise**) before STT |
| `server/services/transcriber.py` | **mlx-whisper** transcription + language id |
| `server/services/stt_postprocess.py` | Optional STT **initial_prompt** + **entity corrections** |
| `server/config/stt_entity_map.json` | Configurable terms and `asr_corrections` for STT |
| `server/services/google_translate_service.py` | **deep-translator** / Google (default path): plain `.txt` → batch translate → segments |
| `server/services/translator.py` | **NLLB** or **IndicTrans2** when `TRANSLATION_BACKEND=local` |
| `server/services/tts_service.py` | **XTTS** / **MMS-TTS**, **pydub**, **pyrubberband**, **`dub_sync`** build |
| `server/run.sh` | Activates venv, loads `.env`, sets `PYTHONPATH`, runs **uvicorn** |
| `server/requirements.txt` | Python dependencies |
| `server/.env` | **`HF_TOKEN`**, etc. (do not commit) |
| `server/cache/` | Generated `{cache_key}.mp3`, `.json`, `_meta.json`, optional transcripts |

---

## 4. User flow (browser)

1. **`/` (Home)**  
   - User enters URL, selects target language, optional “include transcripts”.  
   - Submit → **`navigate('/player?url=...&lang=...[&tx=1]')`** — no API call yet.

2. **`/player` (Player)**  
   - **`POST /api/process-video`** with `{ youtube_url, target_language, include_transcripts }`.  
   - If **disk cache** hits (see §10): response includes **`cached: true`** and full **`result`** → done.  
   - Else response has **`job_id`**.  
   - Client opens **EventSource** on **`GET /api/process-video/stream?job_id=...`**.  
   - Server pushes JSON events: **`step`**, **`total_steps`**, **`message`**, optional **`progress`**, then **`result`** or **`error`**, then stream ends.

3. **Playback**  
   - **`VideoPlayer`**: YouTube **IFrame API** (video), hidden **`<audio>`** (dub URL).  
   - Subtitles from **`result.subtitles`**; **current time** follows **YouTube** clock.  
   - If **`result.dub_sync`** exists, **audio `currentTime`** is mapped from **video time** (§9.4).

---

## 5. Backend API (exact routes)

| Method | Path | Role |
|--------|------|------|
| `GET` | `/health` | Liveness |
| `GET` | `/api/languages` | Languages for the UI dropdown |
| `POST` | `/api/process-video` | Start job or return cached result |
| `GET` | `/api/process-video/stream?job_id=` | **SSE** progress + final `result` / `error` |
| `GET` | `/api/audio/{filename}` | Dubbed MP3 (e.g. `{cache_key}.mp3`) |
| `GET` | `/api/subtitles/{filename}` | Subtitles JSON |
| `GET` | `/api/transcript/{cache_key}/original.txt` | Optional plain transcript (source text) |
| `GET` | `/api/transcript/{cache_key}/translated.txt` | Optional plain transcript (translated) |
| `POST` | `/api/clear-cache` | Clear server cache (see implementation for scope) |
| `GET` | `/` | Minimal API root (if configured) |

---

## 6. Pipeline: exact steps (`_run_pipeline` in `main.py`)

`TOTAL_STEPS = 5` for UI messaging. Sub-steps (e.g. denoise) are not a separate numbered step.

### Step 1 — Download

- **Code:** `download_audio()` in `services/downloader.py`.  
- **Tools:** **yt-dlp** (Python package / CLI) to fetch best audio; **ffmpeg** to extract **WAV**.  
- **Output:** Path to WAV (e.g. `cache/{cache_key}_audio.wav`), **title**, **duration** (seconds).

### Step 1b — Optional denoise (before STT)

- **Code:** `denoise_file()` in `services/noise_canceller.py`.  
- **Library:** **pyrnnoise** (RNNoise-style denoising).  
- **Skip if:** `DISABLE_RNNOISE=1` (or `true`/`yes`).  
- **On failure:** pipeline continues with the **original** WAV.

### Step 2 — Transcribe + language detection

- **Code:** `transcribe()` in `services/transcriber.py`.  
- **Library:** **mlx-whisper** → loads MLX Whisper weights from Hugging Face (e.g. `mlx-community/whisper-large-v3-mlx`).  
- **Behavior:** Produces **segments** with `start`, `end`, `text`; optional **`words`** (timings + probability); optional segment quality fields.  
- **Post-processing:** `stt_postprocess.py` can add **initial_prompt** (terms from JSON) and **dictionary corrections** (`server/config/stt_entity_map.json`).  
- **Env (examples):** `WHISPER_MODEL_SIZE`, `STT_LANGUAGE`, `TTS_STRICT_TIMING` does *not* apply here; see `transcriber.py` / docs in code for STT env vars.

### Step 3 — Translate

- **Code:** `translate_segments()` in `services/translator.py`.  
- **Model:** **`ai4bharat/indictrans2-en-indic-1B`** via **transformers** + **PyTorch** (CPU for stability in this project).  
- **Tokenizer rule:** **IndicTrans2** expects strings like `eng_Latn hin_Deva <text>` (three space-separated parts). **NLLB** uses `src_lang=eng_Latn` and `forced_bos_token_id` for the target FLORES code (e.g. `hin_Deva`).  
- **Chunking:** Multiple segments batched with markers; on failure → per-segment fallback.  
- **Output:** Same segment list with **`translated_text`** per row.  
- **Then:** `_merge_word_timestamps_into_translated` copies STT **words** onto translated rows; `_subtitles_export` builds the JSON shape for the client.

### Step 4 — Text-to-speech (dub)

- **Code:** `generate_dubbed_audio()` in `services/tts_service.py` (+ **`services/edge_tts_synth.py`** for Hindi Edge path).  
- **Hindi (default):** **`edge-tts`** — Microsoft **neural** voices (e.g. **`hi-IN-MadhurNeural`**, **`hi-IN-SwaraNeural`** via **`EDGE_TTS_VOICE_HI`**). No API key; **requires internet**. **`TTS_USE_EDGE=0`** skips this and uses local models only.  
- **Otherwise / fallback:** **Coqui TTS** — **XTTS v2** when a **reference speaker WAV** exists next to the output (`{cache_key}_audio.wav`), then **transformers** **VitsModel** — **Facebook MMS-TTS** per language.  
- **Assembly:** **pydub** (`AudioSegment`) builds the timeline; **soundfile** / **scipy.io.wavfile** for WAV I/O.  
- **Optional strict timing:** **pyrubberband** time-stretch when **`TTS_STRICT_TIMING=1`** (also used to fit Edge audio into subtitle slots when not in clarity-first mode).  
- **Default (clarity-first):** natural phrase length (local path uses clause splits); Edge path uses **one synthesis per subtitle segment** at natural length; **`dub_sync`** records each segment’s `video_*` and `audio_*` positions in the final MP3.

### Step 5 — Write artifacts

- **`{cache_key}.json`** — subtitles array for UI.  
- **`{cache_key}_meta.json`** — `title`, `duration`, `source_language`, `target_language`, **`dub_sync`**.  
- Optional **`_original.txt` / `_translated.txt`** if `include_transcripts` was true.  
- **`push({"result": ...})`** to SSE — includes `audio_url`, `subtitles`, `dub_sync`, etc.

### Cleanup

- **Original download WAV** is **removed** in `finally` after the run; MP3/JSON/meta remain under `cache/`.

---

## 7. Libraries and tools (what each is for)

### 7.1 Python (`server/requirements.txt`)

| Package | Role in this project |
|---------|----------------------|
| **fastapi** | HTTP API framework |
| **uvicorn** | ASGI server |
| **python-dotenv** | Load `server/.env` |
| **yt-dlp** | Download YouTube audio |
| **mlx-whisper** | Apple Silicon Whisper STT (MLX) |
| **transformers** | IndicTrans2, MMS-VITS TTS models |
| **torch** / **torchaudio** | Deep learning runtime; audio loading for some paths |
| **TTS** | Coqui XTTS v2; **Hindi:** **edge-tts** (Microsoft neural) by default |
| **pydub** | Audio timeline / MP3 export |
| **pyrubberband** | Python wrapper for **Rubber Band** — time-stretch (strict timing mode) |
| **huggingface_hub** | Model download/auth (used via transformers; **`HF_TOKEN`** for gated models) |
| **soundfile** | Read/write WAV |
| **scipy** | Scientific stack (dependency / audio helpers) |
| **numpy** | Arrays for audio |
| **pyrnnoise** | RNNoise denoising |
| **sentencepiece** / **sacremoses** | Tokenization-related deps for NLP models |
| **spacy** / **indic-nlp-library** | Present for NLP utilities / future use |
| **httpx** / **aiofiles** / **python-multipart** | HTTP / async file helpers as needed |

### 7.2 System binaries (not pip)

| Tool | Role |
|------|------|
| **ffmpeg** | Used by yt-dlp audio extract; startup checks for it in `main.py` |
| **Rubber Band** | Native library used by **pyrubberband** for time-stretching |

### 7.3 Frontend (`client/package.json`)

| Package | Role |
|---------|------|
| **react** / **react-dom** | UI |
| **react-router-dom** | Routes `/`, `/player` |
| **react-scripts** | CRA build/dev server |

---

## 8. Important data structures

### 8.1 Segment (internal, after translation)

Roughly: `start`, `end`, `text` (source), `translated_text`, optional `words`, optional STT quality keys.

### 8.2 Subtitle row (API / JSON file)

Includes timing, `text` / `original`, `translated_text` / `translated`, optional `words`, optional `stt_quality`.

### 8.3 `dub_sync` (in `result` and `_meta.json`)

List of objects, one per subtitle segment:

- **`video_start`**, **`video_end`** — seconds on the **YouTube** timeline.  
- **`audio_start`**, **`audio_end`** — seconds into the **exported MP3**.

The **React** player uses this in `client/src/utils/dubAudioSync.js` so **`audio.currentTime`** tracks the correct part of the dub when TTS is longer than the subtitle window.

---

## 9. Behavior details (TTS, STT, cache)

### 9.1 Clarity-first vs strict TTS

- **Default:** **`TTS_STRICT_TIMING`** unset → **clarity-first**: full phrases, pauses at punctuation boundaries (see `split_into_speech_clauses` in `tts_service.py`), **no** rubberband compression to fit each subtitle slot.  
- **`TTS_STRICT_TIMING=1`:** per-segment duration fitting with **pyrubberband** (legacy “timeline fit”).

### 9.2 Dub MP3 length vs video duration

- By default the dub is **not** trimmed to `video_duration` (so speech is not cut at the file end).  
- **`TTS_TRIM_DUB_TO_VIDEO=1`** restores trimming to video length.

### 9.3 STT accuracy helpers

- **`server/config/stt_entity_map.json`**: `initial_prompt_terms`, `asr_corrections`.  
- **`STT_ENTITY_MAP_PATH`**, **`STT_DISABLE_ENTITY_CORRECTIONS`**, **`STT_DISABLE_INITIAL_PROMPT`**, etc. (see `transcriber.py` / `stt_postprocess.py`).

### 9.4 Player sync

- **Subtitles** track **YouTube `getCurrentTime()`**.  
- **Dub audio** tracks **`videoTimeToDubAudioTime(ytTime, dub_sync)`** when `dub_sync` is present; otherwise falls back to matching wall-clock 1:1 (can skip audio for long clauses).

---

## 10. Caching

- **`cache_key`** = hash of **canonical YouTube URL + target language** (see `_cache_key` in `main.py`).  
- If **`ENABLE_VIDEO_DISK_CACHE`** is enabled and both **`{cache_key}.mp3`** and **`{cache_key}.json`** exist, **`POST /api/process-video`** may return immediately with **`cached: true`** (no pipeline run).  
- Older cache entries may lack **`dub_sync`** in `_meta.json`; re-run the pipeline to regenerate.

---

## 11. Environment variables (cheat sheet)

| Variable | Purpose |
|----------|---------|
| **`HF_TOKEN`** | Hugging Face token (gated models: IndicTrans2, XTTS, etc.; **not** required for NLLB) |
| **`TRANSLATION_BACKEND`** | `google` (default, **internet**) or `local` (PyTorch) |
| **`GOOGLE_TRANSLATE_CHUNK`** | Segments per `translate_batch` batch (default `8`) |
| **`GOOGLE_TRANSLATE_DELAY_S`** | Pause between batches to reduce HTTP 429 (default `0.25`) |
| **`TRANSLATION_ENGINE`** | When local: `nllb` or `indictrans2` |
| **`NLLB_MODEL_NAME`** | When local + NLLB: checkpoint override |
| **`COQUI_TOS_AGREED`** | Set in `run.sh` to skip interactive Coqui license prompt (you must comply with CPML) |
| **`DISABLE_RNNOISE`** | Skip denoise |
| **`WHISPER_MODEL_SIZE`** / **`MLX_WHISPER_REPO`** | Whisper model selection |
| **`STT_LANGUAGE`** | Force STT language (e.g. `en`) |
| **`TTS_USE_EDGE`** | `1` (default) = Hindi dub via **edge-tts**; `0` = local XTTS/MMS only |
| **`EDGE_TTS_VOICE_HI`** | Hindi neural voice id (default `hi-IN-MadhurNeural`; e.g. `hi-IN-SwaraNeural`) |
| **`EDGE_TTS_RATE`** / **`EDGE_TTS_PITCH`** | Optional Edge SSML-style overrides (see edge-tts docs) |
| **`TTS_STRICT_TIMING`** | `1` = rubberband fit to subtitle slots |
| **`TTS_TRIM_DUB_TO_VIDEO`** | `1` = trim final MP3 to video duration |
| **`TTS_TARGET_DBFS`**, **`TTS_PAUSE_MS_*`**, **`TTS_CLAUSE_MAX_CHARS`** | Dub loudness / pauses / clause splitting |
| **`ENABLE_VIDEO_DISK_CACHE`** | Reuse cached MP3+JSON |
| **`REACT_APP_API_URL`** | Override API base on the client (default `http://127.0.0.1:8000`) |

---

## 12. How to run (short)

**Terminal 1 — backend**

```bash
cd server
./run.sh
```

**Terminal 2 — frontend**

```bash
cd client
npm install
npm start
```

Open **http://localhost:3000**.

---

## 13. Limitations (honest)

- **English → Indic** only for the built-in language list; other pairs may passthrough or need different models.  
- **mlx-whisper** is **Apple Silicon / MLX** oriented; other platforms need a different STT backend.  
- **“Lip sync”** in this project means **timeline alignment** of dubbed audio to subtitles/video clock, **not** viseme/facial animation.  
- **YouTube** playback uses the **official iframe**; dub is a **separate** audio track — sync relies on **`dub_sync`** and player logic.

---

## 14. Document history

- **`SYSTEM_ARCHITECTURE_AND_PIPELINE.md`** was **removed** and replaced by this file (**`SYSTEM_GUIDE.md`**) as the single detailed architecture document.

If you change the pipeline, update **this file** and **`server/requirements.txt`** together so newcomers stay aligned.
