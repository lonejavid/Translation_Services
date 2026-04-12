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

# Prefer Python 3.11+ so pip can resolve TTS / transformers / mlx-whisper without
# Python 3.9-only pins (e.g. old SpaCy/thinc stacks).
if command -v python3.12 >/dev/null 2>&1; then
  PY=python3.12
elif command -v python3.11 >/dev/null 2>&1; then
  PY=python3.11
else
  PY=python3
fi

if [ ! -d "venv" ]; then
  "$PY" -m venv venv
  source venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
else
  source venv/bin/activate
fi

# XTTS compatibility check skipped: Chatterbox requires transformers==5.2.0
# which is incompatible with transformers==4.38.0 (XTTS requirement).
# Chatterbox (Stage 0) handles voice cloning when a reference WAV is present.
# Edge TTS (Stage 1) handles synthesis without a reference. XTTS is not needed.

mkdir -p cache
export PYTHONPATH="$SCRIPT_DIR"

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
export OPENVOICE_ENABLED="${OPENVOICE_ENABLED:-1}"
# TAU=0.30: balanced — clear natural speech + recognisable speaker identity.
# 0.15 was too aggressive and caused robotic/metallic artifacts in the output.
export OPENVOICE_TAU="${OPENVOICE_TAU:-0.30}"
# 8 chunks × 6s each from the 60s reference = very stable averaged embedding
export OPENVOICE_N_CHUNKS="${OPENVOICE_N_CHUNKS:-8}"
export OPENVOICE_CHUNK_SEC="${OPENVOICE_CHUNK_SEC:-6.0}"

# ── Chatterbox Multilingual TTS (optional first pass for dubbing) ─────────────
# By default the dub pipeline uses Edge neural TTS first (clearest speech), then
# OpenVoice for timbre. Chatterbox is a single-pass alternative when enabled below.
# Architecture: flow-matching (Resemble AI, 0.5B params, MIT license, 23 languages)
# Download: ~1.5 GB from HuggingFace on first run.
#
# TTS_CHATTERBOX_FIRST=0    : DEFAULT — Edge TTS first, then OpenVoice (easiest to understand)
# TTS_CHATTERBOX_FIRST=1    : try Chatterbox before Edge when a reference WAV exists
export TTS_CHATTERBOX_FIRST="${TTS_CHATTERBOX_FIRST:-0}"
# TTS_CLARITY_MODE=1 (default): keep Microsoft Edge speech crystal-clear — skip OpenVoice /
# pitch-tilt conversion on Edge segments (those steps trade intelligibility for timbre match).
# TTS_CLARITY_MODE=0: run OpenVoice on Edge audio again for closer speaker match (less clear).
# With clarity on, TTS_SYNC_STRETCH defaults to 0 (no phase-vocoder fit-to-slot); set
# TTS_SYNC_STRETCH=1 explicitly to re-enable stretching while keeping clarity elsewhere.
export TTS_CLARITY_MODE="${TTS_CLARITY_MODE:-1}"
# Speaker gender (pitch on voice reference): GENDER_CONFIDENCE_THRESHOLD (default 0.55),
# GENDER_USE_RAW_FOR_EDGE=1 uses raw male/female for Edge voice when gate → unknown.
# GENDER_F0_MALE_FLOOR_OVERRIDE=1 (default): male speakers mis-read as female when median F0
# is high but pitch floor (p25) is still low — set 0 to disable.
#
# STT: STT_SCRIPT_LANG_FIX=1 (default) — if Whisper says Urdu/Pashto/Sindhi but the transcript
# is almost all Roman letters (no Arabic script), treat source as English (common bug).
#
# CHATTERBOX_ENABLED=1      : allow Chatterbox when TTS_CHATTERBOX_FIRST=1 (or warmup)
#                             set 0 to never load/use Chatterbox
# CHATTERBOX_CFG_WEIGHT=0.0 : KEEP AT 0.0 for cross-lingual cloning
#                             higher values pull output toward reference language phonology
# CHATTERBOX_EXAGGERATION   : 0.0 = flat, ~0.52 = natural, 1.0 = expressive
# CHATTERBOX_TEMPERATURE    : ~0.72 = balanced; lower = more stable, higher = more varied
# CHATTERBOX_ENGLISH_ONLY=1 : legacy — use Chatterbox only when target language is en
# CHATTERBOX_REF_RMS_MATCH  : 1 (default) gently match segment loudness to reference WAV
# CHATTERBOX_CHUNK_GAP_MS   : silence between long-text chunks (default 80)
export CHATTERBOX_ENABLED="${CHATTERBOX_ENABLED:-1}"
export CHATTERBOX_CFG_WEIGHT="${CHATTERBOX_CFG_WEIGHT:-0.0}"
export CHATTERBOX_EXAGGERATION="${CHATTERBOX_EXAGGERATION:-0.52}"
export CHATTERBOX_TEMPERATURE="${CHATTERBOX_TEMPERATURE:-0.72}"

# Voice conversion FALLBACK (used only when OpenVoice is not installed):
# primitive pitch-shift + spectral tilt matching.
export VOICE_CONVERT_ENABLED="${VOICE_CONVERT_ENABLED:-1}"
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


echo "Starting server..."
# Single worker: the pipeline uses in-memory job tracking and mutex locks that
# must not be split across OS processes.  Multi-worker requires an external
# broker (Redis/celery) — not needed for local use.
# --reload is intentionally omitted (development-only; causes double-import and
# model re-downloads on save).
exec python -m uvicorn main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \
  --log-level info \
  --timeout-keep-alive 120 \
  --loop asyncio
