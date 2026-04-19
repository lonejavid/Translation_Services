#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Loading environment..."

# Load .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | xargs)
    echo "Environment loaded from .env"
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg not found. Install with: brew install ffmpeg"
  exit 1
fi

# Coqui TTS needs Python <3.12. Prefer 3.11, then 3.10 / 3.9 (see server/requirements.txt).
if command -v python3.11 >/dev/null 2>&1; then
  PY=python3.11
elif command -v python3.10 >/dev/null 2>&1; then
  PY=python3.10
elif command -v python3.9 >/dev/null 2>&1; then
  PY=python3.9
else
  PY=python3
fi

# When launched from the Electron app, VENV_DIR and CACHE_DIR are set to
# ~/Library/Application Support/Video Translator/{venv,cache} so the venv
# survives updates and is writable even when the .app is in /Applications.
# In dev mode (npm run server), these fall back to the local server/ directory.
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/venv}"
CACHE_DIR="${CACHE_DIR:-$SCRIPT_DIR/cache}"

if [ ! -d "$VENV_DIR" ]; then
  "$PY" -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip
  pip install -r requirements.txt
else
  source "$VENV_DIR/bin/activate"
  # venv exists but deps missing (failed install or empty venv) — install now.
  if ! python -c "import uvicorn" 2>/dev/null; then
    echo "venv present but dependencies missing — running pip install -r requirements.txt …"
    pip install --upgrade pip
    pip install -r requirements.txt
  fi
fi

# See requirements.txt for transformers / chatterbox-tts pins.

mkdir -p "$CACHE_DIR"
export PYTHONPATH="$SCRIPT_DIR"
export CACHE_DIR="$CACHE_DIR"

# Coqui XTTS v2 opens an interactive CPML license prompt unless this is set.
# By using the app with XTTS you must follow https://coqui.ai/cpml (non-commercial)
# or hold a commercial license from Coqui (licensing@coqui.ai).
export COQUI_TOS_AGREED="${COQUI_TOS_AGREED:-1}"

# XTTS v2 voice cloning quality tuning.
# Lower temperature = more stable output, fewer phonetic artifacts.
# High repetition_penalty = prevents XTTS looping / repeating words.
# These are empirically tuned for cross-lingual cloning (e.g. English voice → Hindi speech).
export CLONE_TEMPERATURE="${CLONE_TEMPERATURE:-0.55}"
export CLONE_REPETITION_PENALTY="${CLONE_REPETITION_PENALTY:-8.0}"
export CLONE_TOP_K="${CLONE_TOP_K:-50}"
export CLONE_TOP_P="${CLONE_TOP_P:-0.85}"

# OpenVoice v2 neural voice cloning.
# PRIMARY voice cloning engine: extracts a 256-d speaker embedding from the
# reference audio and applies it to every TTS segment via a flow-based converter.
# This is what makes a dubbed voice sound like the ORIGINAL speaker across languages.
#
# OPENVOICE_TAU: tone-color blending strength.
#   0.1 = very strongly matches reference speaker (sounds most like original)
#   0.3 = balanced (default) — clear speech + recognisable speaker identity
#   0.6 = light tint — if the original voice sounds over-processed, increase this
#
# Auto-downloads ~50 MB checkpoints from HuggingFace on first run.
# ── CLARITY MODE: maximum intelligibility ─────────────────────────────────────
# All voice-conversion steps (OpenVoice, Chatterbox, primitive pitch-shift,
# phase-vocoder time-stretch) are disabled so users hear clean Microsoft Neural
# TTS directly.  The dubbed voice will not match the original speaker's timbre,
# but every word will be crystal-clear with no robotic / metallic artifacts.
#
# To re-enable speaker voice matching at the cost of some clarity, set:
#   OPENVOICE_ENABLED=1  VOICE_CONVERT_ENABLED=1  TTS_SYNC_STRETCH=1
# in your .env file.

# OpenVoice v2 voice-timbre conversion — OFF for clarity.
# When enabled it applies a neural tone-color conversion on top of Edge TTS,
# which can introduce metallic / robotic artifacts.
export OPENVOICE_ENABLED="${OPENVOICE_ENABLED:-0}"
export OPENVOICE_TAU="${OPENVOICE_TAU:-0.30}"
export OPENVOICE_N_CHUNKS="${OPENVOICE_N_CHUNKS:-8}"
export OPENVOICE_CHUNK_SEC="${OPENVOICE_CHUNK_SEC:-6.0}"

# Chatterbox flow-matching TTS — OFF.  Produces less intelligible output than
# Edge TTS, especially for translated (non-native) text.
export CHATTERBOX_ENABLED="${CHATTERBOX_ENABLED:-0}"
export TTS_CHATTERBOX_FIRST="${TTS_CHATTERBOX_FIRST:-0}"
export CHATTERBOX_CFG_WEIGHT="${CHATTERBOX_CFG_WEIGHT:-0.0}"
export CHATTERBOX_EXAGGERATION="${CHATTERBOX_EXAGGERATION:-0.52}"
export CHATTERBOX_TEMPERATURE="${CHATTERBOX_TEMPERATURE:-0.72}"

# Clarity mode ON — Edge TTS output is never post-processed by OpenVoice or
# primitive voice conversion.  This is the single biggest clarity lever.
export TTS_CLARITY_MODE="${TTS_CLARITY_MODE:-1}"

# Phase-vocoder time-stretch — OFF.  Fitting dubbed clips to subtitle slots
# via librosa causes muffled / robotic sound on translated speech.
# Segments play at their natural Edge TTS length instead.
export TTS_SYNC_STRETCH="${TTS_SYNC_STRETCH:-0}"

# Primitive pitch-shift + spectral tilt fallback — OFF.
# Only used when OpenVoice is unavailable; also degrades clarity.
export VOICE_CONVERT_ENABLED="${VOICE_CONVERT_ENABLED:-0}"
export VOICE_CONVERT_PITCH_STRENGTH="${VOICE_CONVERT_PITCH_STRENGTH:-0.75}"
export VOICE_CONVERT_MAX_SEMITONES="${VOICE_CONVERT_MAX_SEMITONES:-5.0}"
export VOICE_CONVERT_TILT_STRENGTH="${VOICE_CONVERT_TILT_STRENGTH:-0.35}"

# Voice reference extraction: use 60 s of the best-quality speech from the video.
# Extractor picks highest-SNR windows from the full audio — more = richer embedding.
export VOICE_REF_DURATION_SEC="${VOICE_REF_DURATION_SEC:-60}"

# TTS synthesis strategy: Edge TTS primary (clearest pronunciation), XTTS fallback.
# Edge TTS (Microsoft Neural TTS) produces the clearest most intelligible audio for
# all supported languages. OpenVoice v2 then applies the original speaker's voice
# identity on top — this is the two-stage approach for clear + voice-matched dubbing.
# XTTS is kept as a fallback for languages where Edge TTS is unavailable.
export HINDI_RAW_MODE="${HINDI_RAW_MODE:-1}"
export TTS_HI_USE_XTTS_CLONE="${TTS_HI_USE_XTTS_CLONE:-0}"
export HINDI_TTS_ENGINE="${HINDI_TTS_ENGINE:-edge}"


PORT="${PORT:-8000}"
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "ERROR: port $PORT is already in use. Stop the other process (see: lsof -nP -iTCP:$PORT -sTCP:LISTEN) or run: PORT=8001 $0"
  exit 1
fi

echo "Starting server on port $PORT..."
# Single worker: the pipeline uses in-memory job tracking and mutex locks that
# must not be split across OS processes.  Multi-worker requires an external
# broker (Redis/celery) — not needed for local use.
# --reload is intentionally omitted (development-only; causes double-import and
# model re-downloads on save).
exec python -m uvicorn main:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --workers 1 \
  --log-level info \
  --timeout-keep-alive 120 \
  --loop asyncio
