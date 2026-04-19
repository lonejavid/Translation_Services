"""
FastAPI: YouTube audio → mlx-whisper → translate → XTTS/MMS-TTS.

Translation (default): **deep-translator** (Google) using timestamp-free plain text files
then segment-aligned results for TTS. Optional: ``TRANSLATION_BACKEND=local`` for on-device
NLLB / IndicTrans2 (see ``services/translator.py``).

SSE progress polled every ~300ms from in-memory job buffers.
"""
from __future__ import annotations

import os
import signal
import sys

# Ignore SIGPIPE so any write to a closed pipe returns EPIPE / raises
# BrokenPipeError (caught by our handlers) rather than terminating the process.
try:
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
except (AttributeError, OSError):
    pass  # Windows has no SIGPIPE


class _SafeStdout:
    """Wrap sys.stdout so BrokenPipeError from print() is silently swallowed.

    Many service functions call print() for progress logging.  If the Electron
    parent process is slow to drain the pipe buffer, or the pipe is briefly
    broken, an unguarded print() raises BrokenPipeError which propagates out
    of the pipeline and appears as the error message in the UI.
    """
    def __init__(self, wrapped):
        self._w = wrapped

    def write(self, s):
        try:
            return self._w.write(s)
        except (BrokenPipeError, OSError):
            return 0

    def flush(self):
        try:
            self._w.flush()
        except (BrokenPipeError, OSError):
            pass

    def __getattr__(self, name):
        return getattr(self._w, name)


sys.stdout = _SafeStdout(sys.stdout)

from dotenv import load_dotenv

# Load environment variables from .env file automatically
# This means HF_TOKEN is always available without manual export
load_dotenv()

# Set HuggingFace token from env file so models download without auth errors
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

import asyncio
import copy
import hashlib
import json
import logging
import re
import shutil
import threading
import time
import uuid
from functools import partial
from pathlib import Path
from urllib.parse import urlparse

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from services.downloader import download_audio
from services.transcriber import transcribe

# tts_service imports torch + transformers (~GB RAM). Import lazily when dubbing runs
# so uvicorn can start without loading the full TTS stack (avoids macOS OOM SIGKILL).

# ---------------------------------------------------------------------------
# Structured logging — JSON format for production observability
# ---------------------------------------------------------------------------

class _JSONLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        import datetime as _dt
        obj = {
            "ts": _dt.datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "module": record.module,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)
        return json.dumps(obj, ensure_ascii=False)


def _setup_logging() -> logging.Logger:
    handler = logging.StreamHandler()
    handler.setFormatter(_JSONLogFormatter())
    root = logging.getLogger()
    # Only add our handler if none exists yet (avoid duplicate logs in tests)
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    return logging.getLogger(__name__)


logger = _setup_logging()

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = Path(os.environ.get("CACHE_DIR", str(BASE_DIR / "cache")))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TOTAL_STEPS = 5
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "5"))
CACHE_TTL_HOURS = int(os.environ.get("CACHE_TTL_HOURS", "168"))  # 7 days
MIN_FREE_DISK_GB = float(os.environ.get("MIN_FREE_DISK_GB", "0.2"))

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

# ---------------------------------------------------------------------------
# Security headers middleware
# ---------------------------------------------------------------------------

class _SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"
        return response

# ---------------------------------------------------------------------------
# CORS origins (configurable via env)
# ---------------------------------------------------------------------------

_cors_raw = os.environ.get("CORS_ORIGINS", "").strip()
# Default: common Create React App dev ports (3000–3005) on localhost + 127.0.0.1
_default_cors = [
    f"http://localhost:{p}"
    for p in (3000, 3001, 3002, 3003, 3004, 3005)
] + [f"http://127.0.0.1:{p}" for p in (3000, 3001, 3002, 3003, 3004, 3005)]
_cors_origins = (
    [o.strip() for o in _cors_raw.split(",") if o.strip()]
    if _cors_raw
    else _default_cors
)

app = FastAPI(title="YouTube Video Translator API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(_SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
)

# Static cache for /audio/... URLs
app.mount("/audio", StaticFiles(directory=str(CACHE_DIR)), name="audio_cache")


@app.on_event("startup")
def _startup_check_ffmpeg():
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg is not installed or not on PATH. Install: brew install ffmpeg (macOS)"
        )


@app.on_event("startup")
async def startup_check():
    token = os.getenv("HF_TOKEN")
    if token:
        logger.info("HF_TOKEN loaded successfully")
    else:
        logger.warning("HF_TOKEN missing — add it to server/.env for gated models")
    logger.info(
        "Server started",
        extra={
            "cors_origins": _cors_origins,
            "cache_dir": str(CACHE_DIR),
            "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
            "cache_ttl_hours": CACHE_TTL_HOURS,
        },
    )


@app.on_event("startup")
def _startup_openvoice_warmup():
    """Pre-load Chatterbox (primary) and OpenVoice (fallback) in background."""
    def _warm():
        # Each loader acquires ``gpu_exclusive`` (macOS) so this thread never overlaps
        # PyTorch MPS with mlx-whisper / Demucs on the pipeline thread.
        try:
            from services.chatterbox_cloner import warmup as cb_warmup

            ok = cb_warmup()
            if ok:
                logger.info(
                    "[chatterbox] Multilingual TTS model pre-loaded successfully"
                )
            else:
                logger.info(
                    "[chatterbox] Disabled or not installed — using Edge+OpenVoice pipeline"
                )
        except Exception as exc:
            logger.warning(f"[chatterbox] Warmup skipped: {exc!r}")

        try:
            from services.openvoice_cloner import warmup

            ok = warmup()
            if ok:
                logger.info(
                    "[openvoice] ToneColorConverter pre-loaded successfully"
                )
            else:
                logger.info(
                    "[openvoice] Disabled or unavailable — voice cloning will use fallback"
                )
        except Exception as exc:
            logger.warning(f"[openvoice] Warmup skipped: {exc!r}")

    threading.Thread(target=_warm, daemon=True, name="openvoice-warmup").start()


@app.on_event("startup")
def _startup_cache_cleanup_thread():
    """Background thread: auto-delete cache files older than CACHE_TTL_HOURS."""
    def _run():
        while True:
            try:
                ttl_s = CACHE_TTL_HOURS * 3600
                now = time.time()
                deleted = 0
                for f in CACHE_DIR.iterdir():
                    if f.is_file():
                        try:
                            if now - f.stat().st_mtime > ttl_s:
                                f.unlink()
                                deleted += 1
                        except OSError:
                            pass
                if deleted:
                    logger.info(f"Cache TTL cleanup: removed {deleted} file(s) older than {CACHE_TTL_HOURS}h")
            except Exception as exc:
                logger.warning(f"Cache cleanup error: {exc}")
            time.sleep(3600)  # Run every hour

    threading.Thread(target=_run, daemon=True, name="cache-cleanup").start()


_ALLOWED_YT_HOSTS: frozenset[str] = frozenset({
    "youtube.com", "www.youtube.com", "youtu.be",
    "m.youtube.com", "music.youtube.com",
})

VALID_LANGUAGE_CODES: set[str] = set()  # populated after LANGUAGES list below


def _validate_youtube_url(url: str) -> str:
    """Raise ValueError if URL is not a valid YouTube URL."""
    url = url.strip()
    if not url:
        raise ValueError("youtube_url is required")
    if len(url) > 500:
        raise ValueError("URL exceeds maximum length of 500 characters")
    try:
        parsed = urlparse(url)
    except Exception:
        raise ValueError("Malformed URL")
    if parsed.scheme not in ("http", "https"):
        raise ValueError("URL must use http or https scheme")
    host = (parsed.hostname or "").lower()
    if host not in _ALLOWED_YT_HOSTS:
        raise ValueError(
            f"Only YouTube URLs are accepted. Got host: {host!r}"
        )
    return url


def _check_disk_space() -> bool:
    """Return True if enough free disk space is available."""
    try:
        usage = shutil.disk_usage(CACHE_DIR)
        return (usage.free / (1024 ** 3)) >= MIN_FREE_DISK_GB
    except Exception:
        return True  # fail open — don't block on check failure


class ProcessVideoRequest(BaseModel):
    youtube_url: str
    target_language: str
    # If True, response includes transcript download URLs (default). Files are always written to disk.
    include_transcripts: bool = True

    @field_validator("youtube_url")
    @classmethod
    def _chk_url(cls, v: str) -> str:
        _validate_youtube_url(v)
        return v.strip()

    @field_validator("target_language")
    @classmethod
    def _chk_lang(cls, v: str) -> str:
        code = v.strip().lower()
        # VALID_LANGUAGE_CODES is filled after LANGUAGES is defined; we do a
        # deferred check here so the import order doesn't matter.
        if VALID_LANGUAGE_CODES and code not in VALID_LANGUAGE_CODES:
            raise ValueError(f"Unsupported language code: {code!r}")
        return code


class SubtitlesUpdateBody(BaseModel):
    """Segment list matching ``/api/subtitles/{cache_key}.json`` shape."""

    subtitles: list[dict]
    # When True: any segment whose **source** text changed vs the saved JSON gets a fresh
    # Google translation (requires ``TRANSLATION_BACKEND=google``). Unchanged sources keep
    # the subtitle's current translation (so manual Hindi fixes stay put).
    retranslate_source_changes: bool = False


class SaveCorrectionsBody(BaseModel):
    """Payload for POST /api/corrections."""

    corrections: list[dict]
    # When True (default): after applying edits, segments whose **source** changed vs the
    # on-disk JSON get a fresh Google translation (only if TRANSLATION_BACKEND=google).
    # If the user also edited the translation for that segment, their translation wins.
    retranslate_source_changes: bool = True


class ErrorResponse(BaseModel):
    """Standard error envelope used by all endpoints and SSE push errors."""
    error: str
    code: str = "error"  # machine-readable slug: "not_found", "validation", "pipeline", etc.
    detail: str | None = None  # optional extra context


def _error(msg: str, code: str = "error", detail: str | None = None) -> dict:
    """Build a standard error dict for SSE push events and JSON responses."""
    r: dict = {"error": msg, "code": code}
    if detail:
        r["detail"] = detail
    return r


LANGUAGES = [
    {"code": "en", "name": "English", "flag": "🇺🇸"},
    {"code": "hi", "name": "Hindi", "flag": "🇮🇳"},
    {"code": "es", "name": "Spanish", "flag": "🇪🇸"},
    {"code": "fr", "name": "French", "flag": "🇫🇷"},
    {"code": "de", "name": "German", "flag": "🇩🇪"},
    {"code": "ar", "name": "Arabic", "flag": "🇸🇦"},
    {"code": "pt", "name": "Portuguese", "flag": "🇧🇷"},
    {"code": "ja", "name": "Japanese", "flag": "🇯🇵"},
    {"code": "ko", "name": "Korean", "flag": "🇰🇷"},
    {"code": "zh", "name": "Chinese", "flag": "🇨🇳"},
    {"code": "ru", "name": "Russian", "flag": "🇷🇺"},
    {"code": "it", "name": "Italian", "flag": "🇮🇹"},
    {"code": "nl", "name": "Dutch", "flag": "🇳🇱"},
    {"code": "tr", "name": "Turkish", "flag": "🇹🇷"},
    {"code": "pl", "name": "Polish", "flag": "🇵🇱"},
    {"code": "sv", "name": "Swedish", "flag": "🇸🇪"},
]

# Populate after LANGUAGES is defined so the validator can use it
VALID_LANGUAGE_CODES.update(lang["code"] for lang in LANGUAGES)  # type: ignore[union-attr]

# job_id -> buffer with thread-safe event list + sentinel None when finished
job_buffers: dict[str, dict] = {}

# Same URL+language → same files; block concurrent pipelines (React Strict Mode double POST, etc.).
# LRU-bounded to prevent unbounded memory growth: keep at most 256 recent keys.
_pipeline_locks: dict[str, threading.Lock] = {}
_pipeline_locks_meta = threading.Lock()
_PIPELINE_LOCKS_MAX = 256

# cache_key -> job_id while a pipeline is running (so duplicate POSTs share one SSE stream).
_inflight_job_by_cache: dict[str, str] = {}
_inflight_job_lock = threading.Lock()

# cache_key → export-job state for the async "download translated video" feature.
_export_jobs: dict[str, dict] = {}  # {"status": "pending|running|done|error", "error": None, "title": "..."}


def _pipeline_lock(cache_key: str) -> threading.Lock:
    with _pipeline_locks_meta:
        if cache_key not in _pipeline_locks:
            # Evict oldest entry when at capacity (FIFO — dict preserves insertion order in Py3.7+)
            if len(_pipeline_locks) >= _PIPELINE_LOCKS_MAX:
                oldest = next(iter(_pipeline_locks))
                del _pipeline_locks[oldest]
            _pipeline_locks[cache_key] = threading.Lock()
        else:
            # Move to end (most-recently-used) by re-inserting
            _pipeline_locks[cache_key] = _pipeline_locks.pop(cache_key)
        return _pipeline_locks[cache_key]


def _cache_key(youtube_url: str, target_language: str) -> str:
    return hashlib.md5(f"{youtube_url}|{target_language}".encode()).hexdigest()


def _translation_backend() -> str:
    """
    ``google`` (default): deep-translator / Google Translate, timestamp-free .txt then batch translate.
    ``local``: PyTorch NLLB or IndicTrans2 in ``services/translator.py``.
    """
    v = (os.environ.get("TRANSLATION_BACKEND") or "google").strip().lower()
    if v in ("google", "local"):
        return v
    logger.info(f"[pipeline] Unknown TRANSLATION_BACKEND={v!r}; using google")
    return "google"


def _video_disk_cache_enabled() -> bool:
    """Set ENABLE_VIDEO_DISK_CACHE=1 to reuse prior mp3/json for same URL+language (faster)."""
    return os.environ.get("ENABLE_VIDEO_DISK_CACHE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _purge_cache_artifacts(cache_key: str) -> int:
    """
    Delete all server-generated files for this cache key so a new run never serves
    stale audio/subtitles. Safe while no pipeline is running for this key.
    """
    if not _valid_cache_key_hex(cache_key):
        return 0
    removed = 0
    exact_names = (
        f"{cache_key}.mp3",
        f"{cache_key}.wav",
        f"{cache_key}.json",
        f"{cache_key}_meta.json",
        f"{cache_key}_original.txt",
        f"{cache_key}_translated.txt",
        f"{cache_key}_source_plain.txt",
        f"{cache_key}_translated_plain.txt",
        f"{cache_key}_audio.wav",
        f"{cache_key}_xtts_ref.wav",
    )
    for name in exact_names:
        p = CACHE_DIR / name
        if p.is_file():
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
    # Partial yt-dlp outputs: {key}_audio.<ext>
    try:
        for f in CACHE_DIR.iterdir():
            if not f.is_file():
                continue
            if f.name.startswith(f"{cache_key}_audio"):
                try:
                    f.unlink()
                    removed += 1
                except OSError:
                    pass
    except OSError:
        pass
    return removed


def _merge_word_timestamps_into_translated(
    transcribe_segments: list[dict], translated: list[dict]
) -> None:
    """Translator drops extra keys; re-attach word-level timings from STT."""
    for i, tr in enumerate(translated):
        if i < len(transcribe_segments):
            w = transcribe_segments[i].get("words")
            if w:
                tr["words"] = w


def _dub_output_disk_path(cache_key: str) -> Path:
    from services.tts_service import resolve_dub_output_path

    return Path(resolve_dub_output_path(str(CACHE_DIR / f"{cache_key}.mp3")))


def _audio_file_duration_sec(path: Path) -> float:
    try:
        from pydub import AudioSegment

        if path.suffix.lower() == ".wav":
            return len(AudioSegment.from_wav(str(path))) / 1000.0
        return len(AudioSegment.from_mp3(str(path))) / 1000.0
    except Exception:
        return 0.0


def _valid_cache_key_hex(k: str) -> bool:
    k = k.lower().strip()
    return len(k) == 32 and all(c in "0123456789abcdef" for c in k)


def _safe_cache_file(filename: str) -> Path:
    """
    Resolve ``filename`` against CACHE_DIR and verify it stays inside.
    Handles URL-encoded paths (%2e%2e, %2f) by using Path.resolve().
    Raises HTTPException 404 if the resolved path escapes the cache directory.
    """
    from urllib.parse import unquote
    # Decode any URL-encoding first
    decoded = unquote(filename)
    # Reject anything with directory separators or null bytes after decode
    if "/" in decoded or "\\" in decoded or "\x00" in decoded:
        raise HTTPException(status_code=404, detail="Not found")
    resolved = (CACHE_DIR / decoded).resolve()
    try:
        resolved.relative_to(CACHE_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=404, detail="Not found")
    return resolved


_MAX_SUBTITLE_SEGMENTS = int(os.environ.get("MAX_SUBTITLE_SEGMENTS", "5000"))
_MAX_SUBTITLE_TEXT_LEN = int(os.environ.get("MAX_SUBTITLE_TEXT_LEN", "2000"))


def _normalize_editor_subtitle_rows(rows: list) -> list[dict]:
    """Validate and normalize subtitle rows from the transcript editor."""
    if len(rows) > _MAX_SUBTITLE_SEGMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many segments: {len(rows)} exceeds limit {_MAX_SUBTITLE_SEGMENTS}",
        )
    out: list[dict] = []
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise HTTPException(
                status_code=400, detail=f"Segment {i} must be an object"
            )
        try:
            st = float(r["start"])
            en = float(r["end"])
        except (KeyError, TypeError, ValueError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Segment {i} needs numeric start/end: {e}",
            ) from e
        if st > en:
            raise HTTPException(
                status_code=400,
                detail=f"Segment {i}: start ({st}) must not be after end ({en})",
            )
        text = str(r.get("text") or r.get("original") or "").strip()
        tr = str(r.get("translated_text") or r.get("translated") or "").strip()
        if len(text) > _MAX_SUBTITLE_TEXT_LEN or len(tr) > _MAX_SUBTITLE_TEXT_LEN:
            raise HTTPException(
                status_code=400,
                detail=f"Segment {i}: text exceeds max length of {_MAX_SUBTITLE_TEXT_LEN} chars",
            )
        row: dict = {
            "start": st,
            "end": en,
            "text": text,
            "translated_text": tr,
            "original": text,
            "translated": tr,
        }
        if isinstance(r.get("words"), list):
            row["words"] = r["words"]
        sq = r.get("stt_quality")
        if isinstance(sq, dict):
            row["stt_quality"] = sq
        tgen = r.get("tts_gender")
        if isinstance(tgen, str) and tgen.strip():
            row["tts_gender"] = tgen.strip().lower()
        try:
            sid = r.get("speaker_id")
            if sid is not None:
                row["speaker_id"] = int(sid)
        except (TypeError, ValueError):
            pass
        sph = r.get("segment_pitch_hz")
        if sph is not None:
            try:
                row["segment_pitch_hz"] = float(sph)
            except (TypeError, ValueError):
                pass
        out.append(row)
    return out


def _run_redub_sync(ck: str) -> dict:
    """Regenerate dubbed audio from ``{ck}.json`` + cache ref WAV (blocking)."""
    json_path = CACHE_DIR / f"{ck}.json"
    if not json_path.is_file():
        return {"ok": False, "error": "No subtitles for this cache key. Process the video first."}
    meta_path = CACHE_DIR / f"{ck}_meta.json"
    try:
        subs = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return {"ok": False, "error": f"Invalid subtitles JSON: {e}"}
    if not isinstance(subs, list) or not subs:
        return {"ok": False, "error": "Subtitles JSON is empty"}

    meta: dict = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}

    target_language = (meta.get("target_language") or "hi").strip()
    duration = float(meta.get("duration") or 0.0)
    speaker_gender = meta.get("speaker_gender")
    gender_meta = None
    if (
        meta.get("speaker_gender") is not None
        or meta.get("speaker_routing_gender") is not None
    ):
        gender_meta = {
            "gender": speaker_gender,
            "confidence": float(meta.get("speaker_gender_confidence") or 0.0),
            "avg_pitch_hz": meta.get("speaker_avg_pitch_hz"),
        }
        if meta.get("speaker_gender_raw") is not None:
            gender_meta["raw_gender"] = meta.get("speaker_gender_raw")
        if meta.get("speaker_gender_gated"):
            gender_meta["gender_gated"] = True
        if meta.get("speaker_routing_gender") is not None:
            gender_meta["routing_gender"] = meta.get("speaker_routing_gender")

    translated: list[dict] = []
    for s in subs:
        if not isinstance(s, dict):
            continue
        row_t: dict = {
            "start": float(s.get("start", 0)),
            "end": float(s.get("end", 0)),
            "text": str(s.get("text") or s.get("original") or ""),
            "translated_text": str(
                s.get("translated_text") or s.get("translated") or ""
            ),
        }
        tg = s.get("tts_gender")
        if isinstance(tg, str) and tg.strip():
            row_t["tts_gender"] = tg.strip().lower()
        try:
            sid = s.get("speaker_id")
            if sid is not None:
                row_t["speaker_id"] = int(sid)
        except (TypeError, ValueError):
            pass
        sph = s.get("segment_pitch_hz")
        if sph is not None:
            try:
                row_t["segment_pitch_hz"] = float(sph)
            except (TypeError, ValueError):
                pass
        translated.append(row_t)

    dub_path_arg = str((CACHE_DIR / f"{ck}.mp3").resolve())
    xtts_ref = str((CACHE_DIR / f"{ck}_xtts_ref.wav").resolve())
    ref = xtts_ref if Path(xtts_ref).is_file() else None

    wav_src = str(CACHE_DIR / f"{ck}_audio.wav")
    routing_fb = None
    if gender_meta and gender_meta.get("routing_gender"):
        routing_fb = gender_meta.get("routing_gender")
    elif isinstance(speaker_gender, str):
        routing_fb = speaker_gender
    try:
        from services.speaker_segments import enrich_segments_with_speaker_voice

        if os.path.isfile(wav_src):
            enrich_segments_with_speaker_voice(
                wav_src, translated, fallback_routing_gender=routing_fb
            )
    except Exception as ex:
        logger.warning(f"[redub] Speaker-aware enrich skipped: {ex!r}")

    try:
        from services.tts_service import generate_dubbed_audio

        written_dub, dub_sync = generate_dubbed_audio(
            translated,
            dub_path_arg,
            target_language,
            video_duration=duration,
            progress_callback=None,
            speaker_ref_wav=ref,
            speaker_gender=speaker_gender,
            speaker_gender_meta=gender_meta,
        )
    except Exception as e:
        return {"ok": False, "error": repr(e)}

    meta["dub_sync"] = dub_sync
    meta["target_language"] = target_language
    try:
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as e:
        return {"ok": False, "error": f"Dub OK but failed to write meta: {e}"}

    dub_name = Path(written_dub).name
    return {
        "ok": True,
        "audio_url": f"/api/audio/{dub_name}",
        "dub_sync": dub_sync,
        "cache_key": ck,
    }


def _fmt_transcript_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m:d}:{s:05.2f}"


def _transcript_plaintext(subtitles: list[dict], original: bool) -> str:
    """One line per segment: [t0 – t1] text (UTF-8)."""
    lines: list[str] = []
    for s in subtitles:
        t0 = _fmt_transcript_time(s.get("start", 0))
        t1 = _fmt_transcript_time(s.get("end", 0))
        if original:
            text = (s.get("original") or s.get("text") or "").strip()
        else:
            text = (
                s.get("translated")
                or s.get("translated_text")
                or s.get("text")
                or ""
            ).strip()
        lines.append(f"[{t0} – {t1}] {text}")
    return "\n".join(lines) + "\n"


def _write_original_txt_from_stt_segments(cache_key: str, segments: list[dict]) -> None:
    """Right after STT: detected-language text only, same line format as final exports."""
    lines: list[str] = []
    for s in segments:
        t0 = _fmt_transcript_time(s.get("start", 0))
        t1 = _fmt_transcript_time(s.get("end", 0))
        text = (s.get("text") or "").strip()
        lines.append(f"[{t0} – {t1}] {text}")
    body = "\n".join(lines) + "\n"
    (CACHE_DIR / f"{cache_key}_original.txt").write_text(body, encoding="utf-8")
    logger.info(f"[pipeline] Wrote source transcript → {cache_key}_original.txt ({len(segments)} lines)")


def _write_transcript_txt_files(cache_key: str, subtitles: list[dict]) -> None:
    """After translation: source + target lines aligned to final segments (TTS reads this text)."""
    (CACHE_DIR / f"{cache_key}_original.txt").write_text(
        _transcript_plaintext(subtitles, True), encoding="utf-8"
    )
    (CACHE_DIR / f"{cache_key}_translated.txt").write_text(
        _transcript_plaintext(subtitles, False), encoding="utf-8"
    )
    logger.info(
        f"[pipeline] Wrote transcripts → {cache_key}_original.txt + "
        f"{cache_key}_translated.txt ({len(subtitles)} segments)"
    )


def _gender_fields_for_api(meta: dict) -> dict:
    """Expose speaker gender for clients (canonical + Edge routing + diagnostics)."""
    if not meta:
        return {}
    out: dict = {}
    if "gender" in meta and meta.get("gender") is not None:
        out["speaker_gender"] = meta.get("gender")
    if meta.get("confidence") is not None:
        out["speaker_gender_confidence"] = float(meta.get("confidence") or 0.0)
    if meta.get("avg_pitch_hz") is not None:
        out["speaker_avg_pitch_hz"] = meta.get("avg_pitch_hz")
    if meta.get("raw_gender") is not None:
        out["speaker_gender_raw"] = meta.get("raw_gender")
    if meta.get("gender_gated"):
        out["speaker_gender_gated"] = True
    if meta.get("routing_gender") is not None:
        out["speaker_routing_gender"] = meta.get("routing_gender")
    return out


def _enrich_result_payload(
    payload: dict, cache_key: str, include_transcripts: bool
) -> dict:
    out = {**payload, "cache_key": cache_key, "include_transcripts": include_transcripts}
    # .txt files are always written to cache/ for a full pipeline run.
    out["original_txt_url"] = f"/api/transcript/{cache_key}/original.txt"
    out["translated_txt_url"] = f"/api/transcript/{cache_key}/translated.txt"
    # Cached jobs may omit gender in payload — merge from meta when present.
    meta_path = CACHE_DIR / f"{cache_key}_meta.json"
    if meta_path.is_file():
        try:
            disk_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            merged = _gender_fields_for_api(
                {
                    "gender": disk_meta.get("speaker_gender"),
                    "confidence": disk_meta.get("speaker_gender_confidence"),
                    "avg_pitch_hz": disk_meta.get("speaker_avg_pitch_hz"),
                    "raw_gender": disk_meta.get("speaker_gender_raw"),
                    "gender_gated": disk_meta.get("speaker_gender_gated"),
                    "routing_gender": disk_meta.get("speaker_routing_gender"),
                }
            )
            for k, v in merged.items():
                if k not in out or out[k] is None:
                    out[k] = v
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    return out


def _cps_flag(text: str, start: float, end: float) -> str | None:
    """
    Return a readability flag if the subtitle exceeds Netflix/BBC CPS limits.
    Netflix standard: max 17 CPS for non-Latin scripts (Hindi, Arabic, etc.),
    20 CPS for Latin scripts. Under 4 CPS = may be too slow (gap/pause segment).
    """
    dur = end - start
    if dur <= 0:
        return None
    n = len(text.strip())
    if n == 0:
        return None
    cps = n / dur
    # Detect script: if any non-Latin alpha char → non-Latin limit
    has_non_latin = any(
        c.isalpha() and ord(c) > 0x024F for c in text
    )
    limit = 17.0 if has_non_latin else 20.0
    if cps > limit:
        return f"fast:{cps:.1f}cps>{limit}"
    if cps < 2.0 and n > 5:
        return f"slow:{cps:.1f}cps"
    return None


def _subtitles_export(translated: list[dict]) -> list[dict]:
    """JSON / API payload: original + translated labels, optional words, CPS flags."""
    out: list[dict] = []
    cps_warnings = 0
    for s in translated:
        row: dict = {
            "start": s["start"],
            "end": s["end"],
            "text": s.get("text", ""),
            "translated_text": s.get("translated_text", s.get("text", "")),
            "original": s.get("text", ""),
            "translated": s.get(
                "translated_text", s.get("translated", s.get("text", ""))
            ),
        }
        # CPS readability check on translated subtitle
        cps_issue = _cps_flag(
            row["translated_text"], float(s["start"]), float(s["end"])
        )
        if cps_issue:
            row["cps_flag"] = cps_issue
            cps_warnings += 1
        if s.get("words"):
            row["words"] = s["words"]
        q: dict = {}
        for k in ("avg_logprob", "no_speech_prob", "compression_ratio", "temperature"):
            if s.get(k) is not None:
                q[k] = s[k]
        if q:
            row["stt_quality"] = q
        tg = s.get("tts_gender")
        if isinstance(tg, str) and tg.strip():
            row["tts_gender"] = tg.strip().lower()
        try:
            sid = s.get("speaker_id")
            if sid is not None:
                row["speaker_id"] = int(sid)
        except (TypeError, ValueError):
            pass
        sph = s.get("segment_pitch_hz")
        if sph is not None:
            try:
                row["segment_pitch_hz"] = float(sph)
            except (TypeError, ValueError):
                pass
        out.append(row)
    if cps_warnings:
        logger.info(
            f"[subtitles] CPS readability: {cps_warnings}/{len(out)} segments "
            f"outside Netflix limits (flagged in payload as cps_flag)"
        )
    return out


def _register_job(job_id: str) -> None:
    job_buffers[job_id] = {
        "events": [],
        "lock": threading.Lock(),
    }


def _push_event(job_id: str, data: dict | None) -> None:
    buf = job_buffers.get(job_id)
    if not buf:
        return
    with buf["lock"]:
        buf["events"].append(data)


def _run_pipeline(
    job_id: str,
    youtube_url: str,
    target_language: str,
    cache_key: str,
    loop: asyncio.AbstractEventLoop,
    include_transcripts: bool = False,
):
    wav_path = str(CACHE_DIR / f"{cache_key}_audio.wav")
    dub_path_arg = str((CACHE_DIR / f"{cache_key}.mp3").resolve())
    json_path = str(CACHE_DIR / f"{cache_key}.json")
    meta_path = str(CACHE_DIR / f"{cache_key}_meta.json")

    def push(msg: dict | None):
        try:
            loop.call_soon_threadsafe(lambda m=msg: _push_event(job_id, m))
        except Exception:
            pass

    with _pipeline_lock(cache_key):
        try:
            push(
                {
                    "step": 1,
                    "total_steps": TOTAL_STEPS,
                    "message": "Downloading YouTube audio",
                }
            )
            path_wav, title, duration = download_audio(
                youtube_url, str(CACHE_DIR / f"{cache_key}_audio")
            )
            if not os.path.isfile(path_wav):
                push(_error("Failed to download audio", code="download_failed"))
                return

            # Step 0 — RNNoise denoise before STT (cleaner input for mlx-whisper)
            path_wav_clean = path_wav
            if not os.environ.get("DISABLE_RNNOISE", "").strip().lower() in (
                "1", "true", "yes",
            ):
                try:
                    from services.noise_canceller import denoise_file
                    path_wav_clean = denoise_file(path_wav)
                    logger.info("[pipeline] Noise cancellation applied successfully")
                except ImportError as e:
                    path_wav_clean = path_wav
                    logger.info(f"[pipeline] Noise cancellation skipped (not installed): {e}")
                except Exception as e:
                    path_wav_clean = path_wav
                    logger.warning(f"[pipeline] Noise cancellation failed: {e!r}")
                    push({"warning": "Noise cancellation failed — using raw audio", "code": "noise_cancel_failed"})

            # BGM separation: extract background music/ambience BEFORE replacing voice.
            bgm_background_path: str | None = None
            demucs_vocals_path: str | None = None
            try:
                from services.bgm_separator import bgm_separation_enabled, separate_vocals
                if bgm_separation_enabled():
                    push({
                        "step": 1,
                        "total_steps": TOTAL_STEPS,
                        "message": "Separating voice from background music (Demucs)",
                    })
                    v_out, bgm_background_path = separate_vocals(
                        path_wav,
                        output_dir=str(CACHE_DIR),
                    )
                    if v_out and os.path.isfile(v_out):
                        demucs_vocals_path = v_out
                    if not bgm_background_path:
                        push({"warning": "BGM separation produced no output — dubbed audio will not have background music", "code": "bgm_failed"})
            except Exception as e:
                logger.warning(f"[pipeline] BGM separation failed: {e!r}")
                push({"warning": "Background music separation failed — dubbed audio will not have background music", "code": "bgm_failed"})

            # Short speech-only clip for Coqui XTTS speaker embedding (voice clone)
            ref_src = path_wav_clean if os.path.isfile(path_wav_clean) else path_wav
            if demucs_vocals_path and os.path.isfile(demucs_vocals_path):
                ref_src = demucs_vocals_path
                logger.info(
                    f"[pipeline] Voice ref + gender: using demucs vocals "
                    f"({Path(demucs_vocals_path).name})"
                )
            speaker_gender: str | None = None
            gender_meta: dict | None = None
            try:
                from services.voice_extractor import extract_reference_voice
                _ref_written, gender_info = extract_reference_voice(
                    str(ref_src),
                    str((CACHE_DIR / f"{cache_key}_xtts_ref.wav").resolve()),
                )
                if gender_info:
                    gender_meta = dict(gender_info)
                    speaker_gender = gender_info.get("gender")
                    # Edge voice selection: use raw male/female when confidence gate → unknown
                    use_raw_edge = os.environ.get(
                        "GENDER_USE_RAW_FOR_EDGE", "1"
                    ).strip().lower() not in ("0", "false", "no", "off")
                    g_canon = (gender_meta.get("gender") or "unknown").strip().lower()
                    routing = (
                        g_canon if g_canon in ("male", "female", "neutral") else None
                    )
                    if use_raw_edge and gender_meta.get("gender_gated"):
                        raw_g = (gender_meta.get("raw_gender") or "").strip().lower()
                        if raw_g in ("male", "female"):
                            routing = raw_g
                    gender_meta["routing_gender"] = routing
            except Exception as e:
                logger.warning(f"[pipeline] XTTS voice reference extraction failed: {e!r}")
                push({"warning": "Voice reference extraction failed — voice cloning disabled", "code": "voice_ref_failed"})

            push(
                {
                    "step": 2,
                    "total_steps": TOTAL_STEPS,
                    "message": "Detecting language and transcribing speech (mlx-whisper)",
                }
            )
            whisper_model = os.environ.get("WHISPER_MODEL_SIZE", "large-v3").strip() or "large-v3"
            stt_lang = os.environ.get("STT_LANGUAGE", "").strip() or None
            try:
                segments, detected_lang = transcribe(
                    path_wav_clean,
                    model_size=whisper_model,
                    language=stt_lang,
                )
            except Exception as _stt_exc:
                push(_error(f"Speech recognition failed: {_stt_exc}", code="stt_failed"))
                return
            if not segments:
                push({"error": "No speech detected"})
                return

            _write_original_txt_from_stt_segments(cache_key, segments)

            push(
                {
                    "step": 3,
                    "total_steps": TOTAL_STEPS,
                    "message": f"Translating to {target_language}",
                }
            )

            def trans_progress(done: int, total: int):
                push(
                    {
                        "step": 3,
                        "total_steps": TOTAL_STEPS,
                        "message": f"Translating to {target_language}",
                        "progress": f"{done}/{total}",
                    }
                )

            backend = _translation_backend()
            if backend == "local":
                from services.translator import translate_segments

                translated = translate_segments(
                    segments,
                    detected_lang,
                    target_language,
                    progress_callback=trans_progress,
                )
            else:
                from services.google_translate_service import translate_segments_google

                logger.info("[pipeline] Translation: Google via deep-translator (internet required)")
                translated = translate_segments_google(
                    segments,
                    detected_lang,
                    target_language,
                    cache_dir=CACHE_DIR,
                    cache_key=cache_key,
                    progress_callback=trans_progress,
                )

            from services.google_translate_service import translation_validation_report

            translation_validation_report(
                translated, target_language, backend=backend
            )

            # Hindi: align 1st-person verb forms with detected speaker gender (MT often defaults masculine).
            try:
                from services.hindi_gender_grammar import apply_gender_to_translated_segments

                n_hi_g = apply_gender_to_translated_segments(
                    translated, speaker_gender, target_language
                )
                if n_hi_g:
                    logger.info(
                        f"[pipeline] Hindi gender grammar: adjusted {n_hi_g} segment(s) "
                        f"for speaker_gender={speaker_gender!r}"
                    )
            except Exception as e:
                logger.warning(f"[pipeline] Hindi gender grammar skipped: {e!r}")

            # Verify translation actually happened
            failed_count = sum(
                1
                for seg in translated
                if seg.get("translated_text", "").strip()
                == seg.get("text", "").strip()
            )
            total_segs = len(translated)
            if total_segs and failed_count > total_segs * 0.5:
                logger.info(
                    f"[pipeline] WARNING: {failed_count}/{total_segs} segments "
                    f"not translated — translation may have failed"
                )
            elif total_segs:
                logger.info(
                    f"[pipeline] Translation OK: "
                    f"{total_segs - failed_count}/{total_segs} segments translated"
                )

            _merge_word_timestamps_into_translated(segments, translated)

            routing_fb = None
            if gender_meta and gender_meta.get("routing_gender"):
                routing_fb = gender_meta.get("routing_gender")
            elif isinstance(speaker_gender, str):
                routing_fb = speaker_gender
            try:
                from services.speaker_segments import enrich_segments_with_speaker_voice

                _sp_wav = (
                    path_wav_clean
                    if os.path.isfile(path_wav_clean)
                    else path_wav
                )
                enrich_segments_with_speaker_voice(
                    _sp_wav,
                    translated,
                    fallback_routing_gender=routing_fb,
                )
            except Exception as e:
                logger.warning(
                    f"[pipeline] Speaker-aware TTS enrich skipped: {e!r}"
                )

            subtitles_payload = _subtitles_export(translated)

            # Same strings TTS will speak — persisted for inspection / “read from file” workflow.
            _write_transcript_txt_files(cache_key, subtitles_payload)

            push(
                {
                    "step": 4,
                    "total_steps": TOTAL_STEPS,
                    "message": "Generating dubbed voice from translated transcript (XTTS / MMS-TTS)",
                }
            )

            def tts_progress(done: int, total: int):
                push(
                    {
                        "step": 4,
                        "total_steps": TOTAL_STEPS,
                        "message": "Generating dubbed voice (XTTS / MMS-TTS)",
                        "progress": f"{done}/{total}",
                    }
                )

            xtts_ref_path = str((CACHE_DIR / f"{cache_key}_xtts_ref.wav").resolve())
            try:
                from services.tts_service import generate_dubbed_audio

                written_dub, dub_sync = generate_dubbed_audio(
                    translated,
                    dub_path_arg,
                    target_language,
                    video_duration=duration,
                    progress_callback=tts_progress,
                    speaker_ref_wav=xtts_ref_path,
                    speaker_gender=speaker_gender,
                    speaker_gender_meta=gender_meta,
                )
            except Exception as _tts_exc:
                push(_error(f"Audio generation failed: {_tts_exc}", code="tts_failed"))
                return

            # BGM mix: blend dubbed voice with original background music/ambience.
            # This is the Netflix-standard final step — voice is replaced, BGM stays.
            if bgm_background_path and os.path.isfile(bgm_background_path):
                try:
                    push({
                        "step": 5,
                        "total_steps": TOTAL_STEPS,
                        "message": "Mixing dubbed voice with original background music",
                    })
                    from services.bgm_separator import mix_dub_with_background
                    from pydub import AudioSegment as _AS
                    mixed_wav = str(CACHE_DIR / f"{cache_key}_mixed.wav")
                    # Convert dubbed audio to WAV if needed (pipeline outputs MP3)
                    dub_src = written_dub
                    tmp_dub_wav: str | None = None
                    if written_dub.endswith(".mp3") and os.path.isfile(written_dub):
                        tmp_dub_wav = str(CACHE_DIR / f"{cache_key}_dub_tmp.wav")
                        _AS.from_mp3(written_dub).export(tmp_dub_wav, format="wav")
                        dub_src = tmp_dub_wav
                    try:
                        if os.path.isfile(dub_src):
                            mix_result = mix_dub_with_background(dub_src, bgm_background_path, mixed_wav)
                            if mix_result == mixed_wav and os.path.isfile(mixed_wav):
                                if written_dub.endswith(".mp3"):
                                    _AS.from_wav(mixed_wav).export(
                                        written_dub, format="mp3",
                                        bitrate=f"{int(os.environ.get('AUDIO_BITRATE', '320'))}k"
                                    )
                                else:
                                    import shutil as _sh
                                    _sh.copy2(mixed_wav, written_dub)
                                logger.info(f"[pipeline] BGM mix applied → {Path(written_dub).name}")
                    finally:
                        for _p in [tmp_dub_wav, mixed_wav]:
                            if _p and os.path.isfile(_p):
                                try:
                                    os.unlink(_p)
                                except OSError:
                                    pass
                except Exception as e:
                    logger.info(f"[pipeline] BGM mix skipped: {e!r}")

            dub_filename = Path(written_dub).name

            push(
                {
                    "step": 5,
                    "total_steps": TOTAL_STEPS,
                    "message": "Syncing audio timeline",
                }
            )

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(subtitles_payload, f, ensure_ascii=False)
            _gender_meta_for_disk = gender_meta or {}
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "title": title,
                        "duration": duration,
                        "source_language": detected_lang,
                        "target_language": target_language,
                        "dub_sync": dub_sync,
                        "youtube_url": youtube_url,  # stored for video export
                        "speaker_gender": _gender_meta_for_disk.get("gender"),
                        "speaker_gender_confidence": _gender_meta_for_disk.get(
                            "confidence"
                        ),
                        "speaker_avg_pitch_hz": _gender_meta_for_disk.get(
                            "avg_pitch_hz"
                        ),
                        "speaker_gender_raw": _gender_meta_for_disk.get("raw_gender"),
                        "speaker_gender_gated": bool(
                            _gender_meta_for_disk.get("gender_gated")
                        ),
                        "speaker_routing_gender": _gender_meta_for_disk.get(
                            "routing_gender"
                        ),
                    },
                    f,
                    ensure_ascii=False,
                )

            result = _enrich_result_payload(
                {
                    "audio_url": f"/api/audio/{dub_filename}",
                    "subtitles_url": f"/api/subtitles/{cache_key}.json",
                    "subtitles": subtitles_payload,
                    "title": title,
                    "duration": duration,
                    "source_language": detected_lang,
                    "target_language": target_language,
                    "dub_sync": dub_sync,
                    **_gender_fields_for_api(_gender_meta_for_disk),
                },
                cache_key,
                include_transcripts,
            )
            push({"result": result})
            logger.info(f"[pipeline] job {job_id[:8]}… complete → {dub_filename} ({duration:.1f}s)")
        except Exception as e:
            import re as _re
            err = _re.sub(r"\x1b\[[0-9;]*m", "", str(e))
            logger.error(f"[pipeline] job {job_id[:8]}… failed: {err}", exc_info=True)
            push(_error(err, code="pipeline_failed"))
        finally:
            try:
                if os.path.exists(wav_path):
                    os.remove(wav_path)
            except OSError:
                pass
            push(None)
            with _inflight_job_lock:
                if _inflight_job_by_cache.get(cache_key) == job_id:
                    _inflight_job_by_cache.pop(cache_key, None)

            def _cleanup_buffer_later(jid: str, delay_s: float = 120.0) -> None:
                time.sleep(delay_s)
                job_buffers.pop(jid, None)

            threading.Thread(target=_cleanup_buffer_later, args=(job_id,), daemon=True).start()
            logger.info(f"[pipeline] job {job_id[:8]}… finished (progress stream closed)")


@app.get("/health")
def health():
    """Detailed health check for monitoring / load balancers."""
    import urllib.request as _ur
    disk = shutil.disk_usage(CACHE_DIR)
    free_gb = round(disk.free / (1024 ** 3), 2)
    try:
        cache_files = sum(1 for f in CACHE_DIR.iterdir() if f.is_file())
    except OSError:
        cache_files = -1
    with _inflight_job_lock:
        active_jobs = len(_inflight_job_by_cache)

    # Check Ollama / LLM validator status
    ollama_running = False
    llama3_loaded = False
    try:
        with _ur.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            import json as _json
            tags = _json.loads(r.read())
            ollama_running = True
            llama3_loaded = any(
                "llama3" in m.get("name", "") for m in tags.get("models", [])
            )
    except Exception:
        pass

    log_path = CACHE_DIR / "llm_validation_log.txt"
    validation_log_entries = 0
    if log_path.is_file():
        try:
            validation_log_entries = log_path.read_text("utf-8").count("] VALIDATED") + \
                                     log_path.read_text("utf-8").count("] SKIPPED") + \
                                     log_path.read_text("utf-8").count("] CACHE HIT")
        except Exception:
            pass

    return {
        "status": "ok",
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "disk_free_gb": free_gb,
        "disk_ok": free_gb >= MIN_FREE_DISK_GB,
        "cache_files": cache_files,
        "active_jobs": active_jobs,
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
        "llm_validator": {
            "ollama_running": ollama_running,
            "llama3_ready": llama3_loaded,
            "fully_operational": ollama_running and llama3_loaded,
            "validation_log_entries": validation_log_entries,
        },
    }


@app.get("/api/languages")
def get_languages():
    return LANGUAGES


@limiter.limit("10/minute")
async def process_video(request: Request, req: ProcessVideoRequest):
    # Disk space guard
    if not _check_disk_space():
        raise HTTPException(
            status_code=507,
            detail=f"Server is low on disk space (< {MIN_FREE_DISK_GB} GB free). Try again later.",
        )
    # Concurrent job cap — prevent GPU/CPU exhaustion
    with _inflight_job_lock:
        if len(_inflight_job_by_cache) >= MAX_CONCURRENT_JOBS:
            raise HTTPException(
                status_code=429,
                detail=f"Server is busy ({MAX_CONCURRENT_JOBS} jobs running). Please try again shortly.",
            )
    cache_key = _cache_key(req.youtube_url, req.target_language)
    dub_path = _dub_output_disk_path(cache_key)
    json_path = CACHE_DIR / f"{cache_key}.json"

    if _video_disk_cache_enabled() and dub_path.is_file() and json_path.is_file():
        with open(json_path, "r", encoding="utf-8") as f:
            subtitles = json.load(f)
        meta_path = CACHE_DIR / f"{cache_key}_meta.json"
        title = "YouTube Video"
        dur = 0.0
        src = "en"
        tgt = req.target_language
        dub_sync_cached: list | None = None
        if meta_path.is_file():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                title = meta.get("title", title)
                dur = float(meta.get("duration", 0) or 0)
                src = meta.get("source_language", src)
                tgt = meta.get("target_language", tgt)
                dub_sync_cached = meta.get("dub_sync")
            except Exception:
                pass
        if dur <= 0:
            dur = _audio_file_duration_sec(dub_path)
        _write_transcript_txt_files(cache_key, subtitles)
        payload = {
            "cached": True,
            "audio_url": f"/api/audio/{dub_path.name}",
            "subtitles_url": f"/api/subtitles/{cache_key}.json",
            "subtitles": subtitles,
            "title": title,
            "duration": dur,
            "source_language": src,
            "target_language": tgt,
        }
        if dub_sync_cached:
            payload["dub_sync"] = dub_sync_cached
        return _enrich_result_payload(
            payload,
            cache_key,
            req.include_transcripts,
        )

    # One job_id per cache_key until done — avoids UI stuck on "Downloading" when the
    # browser sends two POSTs (e.g. React Strict Mode): the 2nd job would block on the
    # lock and never emit progress while the UI subscribed to the 2nd stream.
    with _inflight_job_lock:
        existing = _inflight_job_by_cache.get(cache_key)
        if existing and existing in job_buffers:
            return {"job_id": existing, "cached": False}
        if not _video_disk_cache_enabled():
            n = _purge_cache_artifacts(cache_key)
            if n:
                logger.info(
                    f"[pipeline] cleared {n} prior file(s) for this URL/language "
                    f"(cache {cache_key[:8]}…); re-running full pipeline"
                )
        job_id = str(uuid.uuid4())
        _register_job(job_id)
        _inflight_job_by_cache[cache_key] = job_id

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None,
        partial(
            _run_pipeline,
            job_id,
            req.youtube_url,
            req.target_language,
            cache_key,
            loop,
            req.include_transcripts,
        ),
    )
    return {"job_id": job_id, "cached": False}


# Fix: `from __future__ import annotations` (PEP 563) makes ALL annotations lazy
# ForwardRefs. When slowapi wraps the function, FastAPI resolves annotations via
# typing.get_type_hints() in the wrapper's globals (slowapi module) — not main's
# globals — so 'ProcessVideoRequest' can't be resolved and FastAPI treats `req` as
# a query param, producing 422.
#
# Fix: set concrete __annotations__ BEFORE registering the route with FastAPI.
# We apply the slowapi decorator first (no FastAPI involvement yet), then patch
# annotations to be the resolved classes, then register with app.post().
# Fix: `from __future__ import annotations` (PEP 563) makes ALL annotations lazy
# strings in the original function. `inspect.signature()` follows __wrapped__ to get
# the ORIGINAL function's params — still strings. FastAPI then resolves them using the
# WRAPPER's __globals__ (slowapi's module), where 'ProcessVideoRequest' doesn't exist.
# Result: FastAPI can't classify `req` as a body param and treats it as a query param (422).
#
# Fix: patch __annotations__ on the INNER (unwrapped) function so inspect.signature()
# sees concrete types, not strings. Then register with app.post().
_inner = getattr(process_video, "__wrapped__", process_video)
_inner.__annotations__ = {
    "request": Request,
    "req": ProcessVideoRequest,
}
app.post("/api/process-video")(process_video)


@app.get("/api/process-video/stream")
async def process_video_stream(job_id: str):
    if job_id not in job_buffers:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_stream():
        idx = 0
        buf = job_buffers[job_id]
        try:
            while True:
                await asyncio.sleep(0.3)
                with buf["lock"]:
                    pending = buf["events"][idx:]
                    idx = len(buf["events"])
                for msg in pending:
                    if msg is None:
                        yield f"data: {json.dumps({'done': True})}\n\n"
                        return
                    if "error" in msg:
                        yield f"data: {json.dumps({'error': msg['error']})}\n\n"
                        return
                    if "result" in msg:
                        yield f"data: {json.dumps({'result': msg['result']})}\n\n"
                        continue
                    yield f"data: {json.dumps(msg)}\n\n"
        finally:
            # Do NOT pop job_buffers here. Closing EventSource (e.g. React Strict Mode
            # remount) would delete the buffer while the pipeline still runs — the UI
            # reconnects to 404 or misses all progress. Buffers are removed after a delay
            # when the pipeline finishes (see _run_pipeline).
            pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/api/audio/{filename}")
def get_audio(filename: str):
    path = _safe_cache_file(filename)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    mt = "audio/wav" if path.suffix.lower() == ".wav" else "audio/mpeg"
    r = FileResponse(path, media_type=mt)
    # Allow same-origin + configured CORS origins only (not wildcard)
    r.headers["Accept-Ranges"] = "bytes"
    return r


@app.get("/api/subtitles/{filename}")
def get_subtitles(filename: str):
    path = _safe_cache_file(filename)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    r = FileResponse(path, media_type="application/json")
    # Editors reload this file after save — avoid stale JSON in the browser cache.
    r.headers["Cache-Control"] = "no-store, max-age=0"
    return r


@app.put("/api/subtitles/cache/{cache_key}")
def put_subtitles_by_cache_key(cache_key: str, body: SubtitlesUpdateBody):
    """
    Save edited segments to ``{cache_key}.json`` and transcript .txt files.
    Records diffs into ``learned_corrections.json`` for future STT / Google translate runs.
    """
    ck = cache_key.lower().strip()
    if not _valid_cache_key_hex(ck):
        raise HTTPException(status_code=400, detail="Invalid cache_key")
    json_path = CACHE_DIR / f"{ck}.json"
    old_rows: list = []
    if json_path.is_file():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                old_rows = loaded
        except (OSError, json.JSONDecodeError):
            old_rows = []

    new_rows = _normalize_editor_subtitle_rows(body.subtitles)

    meta_path = CACHE_DIR / f"{ck}_meta.json"
    tgt = "hi"
    src = "en"
    if meta_path.is_file():
        try:
            meta0 = json.loads(meta_path.read_text(encoding="utf-8"))
            tgt = str(meta0.get("target_language") or "hi").strip().lower()[:8]
            src = str(meta0.get("source_language") or "en").strip().lower()[:8]
        except (OSError, json.JSONDecodeError):
            pass

    n_retranslated = 0
    if body.retranslate_source_changes:
        be = _translation_backend()
        if be != "google":
            raise HTTPException(
                status_code=400,
                detail=(
                    "retranslate_source_changes needs TRANSLATION_BACKEND=google "
                    "(editor auto-retranslate is not implemented for local/NLLB yet). "
                    "Uncheck the option or set TRANSLATION_BACKEND=google in server/.env."
                ),
            )
        from services.google_translate_service import retranslate_editor_segments

        new_rows, n_retranslated = retranslate_editor_segments(
            old_rows, new_rows, src, tgt
        )
        if n_retranslated:
            logger.info(
                f"[editor] Retranslated {n_retranslated} segment(s) after source text edits"
            )

    segs_for_export: list[dict] = []
    for r in new_rows:
        d: dict = {
            "start": r["start"],
            "end": r["end"],
            "text": r["text"],
            "translated_text": r["translated_text"],
        }
        if r.get("words"):
            d["words"] = r["words"]
        sq = r.get("stt_quality")
        if isinstance(sq, dict):
            for k in (
                "avg_logprob",
                "no_speech_prob",
                "compression_ratio",
                "temperature",
            ):
                if sq.get(k) is not None:
                    d[k] = sq[k]
        tge = r.get("tts_gender")
        if isinstance(tge, str) and tge.strip():
            d["tts_gender"] = tge.strip().lower()
        try:
            sid = r.get("speaker_id")
            if sid is not None:
                d["speaker_id"] = int(sid)
        except (TypeError, ValueError):
            pass
        sphz = r.get("segment_pitch_hz")
        if sphz is not None:
            try:
                d["segment_pitch_hz"] = float(sphz)
            except (TypeError, ValueError):
                pass
        segs_for_export.append(d)
    payload = _subtitles_export(segs_for_export)

    from services.learned_corrections import record_edits

    n_asr, n_tm = record_edits(old_rows, new_rows, target_lang=tgt, source_lang=src)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _write_transcript_txt_files(ck, payload)

    logger.info(
        f"[editor] Saved subtitles {ck}.json; learned +{n_asr} ASR phrase(s), "
        f"+{n_tm} translation memory row(s)"
    )
    return {
        "ok": True,
        "cache_key": ck,
        "learned": {"asr_phrases": n_asr, "translation_memory": n_tm},
        "retranslated_segments": n_retranslated,
    }


@app.post("/api/video/cache/{cache_key}/redub")
async def post_redub(cache_key: str):
    """
    Regenerate dubbed audio from the current ``{cache_key}.json`` (after edits).
    Can take several minutes; run asynchronously on the server thread pool.
    """
    ck = cache_key.lower().strip()
    if not _valid_cache_key_hex(ck):
        raise HTTPException(status_code=400, detail="Invalid cache_key")
    result = await asyncio.to_thread(_run_redub_sync, ck)
    if not result.get("ok"):
        raise HTTPException(
            status_code=500,
            detail=result.get("error", "Re-dub failed"),
        )
    return result


# ---------------------------------------------------------------------------
# Translation Correction API  (human-in-the-loop learning)
# ---------------------------------------------------------------------------

@app.post("/api/corrections")
def post_corrections(body: SaveCorrectionsBody):
    """
    Save user-corrected translations from the dual-panel TranslationEditor.

    Each correction is stored in ``data/translation_corrections.json`` and
    also fed into ``learned_corrections.json`` so future pipeline runs on
    videos with the same source text automatically use the corrected version.

    The originating ``.json`` segment file is updated in-place so the Player
    page shows the corrected text on the next load. When ``retranslate_source_changes``
    is true and ``TRANSLATION_BACKEND=google``, any segment whose source text changed
    vs disk gets a fresh machine translation; manual translation edits for that segment
    are preserved. Re-dub is still required to regenerate dubbed MP3.
    """
    from services.correction_store import save_corrections
    from services.learned_corrections import record_edits

    items = body.corrections
    if not items:
        return {"ok": True, "saved": 0}

    saved = save_corrections(items)

    # Also persist into learned_corrections so future translations benefit
    # Group by (cache_key, target_lang, source_lang) to call record_edits once per video
    groups: dict[str, dict] = {}
    for c in saved:
        ck = c.get("cache_key", "")
        key = f"{ck}|{c.get('target_lang', 'en')}|{c.get('source_lang', 'en')}"
        if key not in groups:
            groups[key] = {"cache_key": ck, "target_lang": c.get("target_lang", "en"),
                           "source_lang": c.get("source_lang", "en"), "pairs": []}
        groups[key]["pairs"].append(c)

    total_tm = 0
    total_retranslated = 0
    for g in groups.values():
        # When the user edited source text, use original_source_text as old "text"
        # so record_edits() detects the change and stores an ASR phrase fix globally.
        old_rows = [
            {
                "text": p.get("original_source_text") or p["source_text"],
                "translated_text": p["incorrect_translation"],
            }
            for p in g["pairs"]
        ]
        new_rows = [{"text": p["source_text"], "translated_text": p["corrected_translation"]}
                    for p in g["pairs"]]
        _, n_tm = record_edits(old_rows, new_rows,
                               target_lang=g["target_lang"], source_lang=g["source_lang"])
        total_tm += n_tm

        # Write corrected text back into the segment JSON so Player shows it
        ck = g["cache_key"]
        if _valid_cache_key_hex(ck):
            json_path = CACHE_DIR / f"{ck}.json"
            if json_path.is_file():
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        segs = json.load(f)
                    if isinstance(segs, list):
                        tgt = str(g.get("target_lang") or "hi").strip().lower()[:8]
                        src = str(g.get("source_lang") or "en").strip().lower()[:8]
                        meta_path = CACHE_DIR / f"{ck}_meta.json"
                        if meta_path.is_file():
                            try:
                                meta0 = json.loads(meta_path.read_text(encoding="utf-8"))
                                tgt = str(meta0.get("target_language") or tgt).strip().lower()[:8]
                                src = str(meta0.get("source_language") or src).strip().lower()[:8]
                            except (OSError, json.JSONDecodeError):
                                pass

                        def _row_from_seg(s: dict) -> dict:
                            return {
                                "text": s.get("text") or s.get("original") or "",
                                "translated_text": s.get("translated_text") or s.get("translated") or "",
                            }

                        old_simple = [_row_from_seg(s) for s in segs]
                        new_simple = copy.deepcopy(old_simple)
                        manual_tr_by_idx: dict[int, str] = {}
                        for p in g["pairs"]:
                            si = p.get("segment_index")
                            if si is None or not (0 <= si < len(new_simple)):
                                continue
                            new_simple[si]["text"] = p["source_text"]
                            new_simple[si]["translated_text"] = p["corrected_translation"]
                            if p.get("corrected_translation") != p.get("incorrect_translation"):
                                manual_tr_by_idx[si] = p["corrected_translation"]

                        n_ret = 0
                        if body.retranslate_source_changes and _translation_backend() == "google":
                            from services.google_translate_service import retranslate_editor_segments

                            new_simple, n_ret = retranslate_editor_segments(
                                old_simple, new_simple, src, tgt
                            )
                            total_retranslated += n_ret
                            for si, tr in manual_tr_by_idx.items():
                                if 0 <= si < len(new_simple):
                                    new_simple[si]["translated_text"] = tr

                        for p in g["pairs"]:
                            si = p.get("segment_index")
                            if si is not None and 0 <= si < len(segs):
                                segs[si]["text"] = new_simple[si]["text"]
                                segs[si]["translated_text"] = new_simple[si]["translated_text"]
                                segs[si]["corrected"] = True
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(segs, f, ensure_ascii=False, indent=2)
                        _write_transcript_txt_files(ck, segs)
                except (OSError, json.JSONDecodeError, KeyError):
                    pass

    logger.info(
        f"[corrections] Saved {len(saved)} correction(s); +{total_tm} translation memory rows; "
        f"retranslated_segments={total_retranslated}"
    )
    return {
        "ok": True,
        "saved": len(saved),
        "learned": {"translation_memory": total_tm},
        "retranslated_segments": total_retranslated,
    }


@app.get("/api/corrections/video/{cache_key}")
def get_corrections_for_video(cache_key: str):
    """
    Return all saved corrections for one video, keyed by segment_index.

    Used by the TranslationEditor to show which segments have been corrected
    and to populate the "Reset to AI translation" action with the original text.
    """
    from services.correction_store import get_corrections_for_cache_key

    ck = cache_key.lower().strip()
    if not _valid_cache_key_hex(ck):
        raise HTTPException(status_code=400, detail="Invalid cache_key")
    corrections = get_corrections_for_cache_key(ck)
    # Return as list (JSON keys must be strings, but client indexes by int)
    return {
        "cache_key": ck,
        "count": len(corrections),
        "corrections": {str(k): v for k, v in corrections.items()},
    }


@app.get("/api/corrections")
def get_all_corrections_route(limit: int = 200, offset: int = 0):
    """Return paginated list of all corrections across all videos."""
    from services.correction_store import get_all_corrections
    limit = max(1, min(1000, limit))
    offset = max(0, offset)
    return get_all_corrections(limit=limit, offset=offset)


@app.delete("/api/corrections/{correction_id}")
def delete_correction_route(correction_id: str):
    """
    Hard-delete a correction by ID.

    The caller is responsible for re-saving the segment JSON if the corrected
    text should be reverted — use the ``incorrect_translation`` field returned
    by GET /api/corrections/video/{cache_key} to restore the original.
    """
    from services.correction_store import delete_correction
    found = delete_correction(correction_id.strip())
    if not found:
        raise HTTPException(status_code=404, detail="Correction not found")
    return {"ok": True, "deleted": correction_id}


@app.delete("/api/corrections/video/{cache_key}/segment/{segment_index}")
def reset_segment_correction(cache_key: str, segment_index: int):
    """
    Reset one segment back to its original AI translation.

    1. Removes the correction from the store.
    2. Restores the ``incorrect_translation`` text into the segment JSON.
    """
    from services.correction_store import (
        delete_corrections_for_segment,
        get_corrections_for_cache_key,
    )

    ck = cache_key.lower().strip()
    if not _valid_cache_key_hex(ck):
        raise HTTPException(status_code=400, detail="Invalid cache_key")

    # Fetch before deleting so we have the original text
    corrections = get_corrections_for_cache_key(ck)
    correction = corrections.get(segment_index)
    original_text = correction["incorrect_translation"] if correction else None

    delete_corrections_for_segment(ck, segment_index)

    # Restore original text in segment JSON
    if original_text is not None:
        json_path = CACHE_DIR / f"{ck}.json"
        if json_path.is_file():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    segs = json.load(f)
                if isinstance(segs, list) and 0 <= segment_index < len(segs):
                    segs[segment_index]["translated_text"] = original_text
                    segs[segment_index].pop("corrected", None)
                    segs[segment_index].pop("correction_id", None)
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(segs, f, ensure_ascii=False, indent=2)
                    _write_transcript_txt_files(ck, segs)
            except (OSError, json.JSONDecodeError):
                pass

    return {"ok": True, "segment_index": segment_index, "restored": original_text}


@app.get("/api/transcript/{cache_key}/original.txt")
def get_transcript_original(cache_key: str):
    ck = cache_key.lower().strip()
    if not _valid_cache_key_hex(ck):
        raise HTTPException(status_code=404, detail="Not found")
    disk = CACHE_DIR / f"{ck}_original.txt"
    if disk.is_file():
        return FileResponse(disk, media_type="text/plain; charset=utf-8", filename=f"{ck}_original.txt")
    json_path = CACHE_DIR / f"{ck}.json"
    if not json_path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    with open(json_path, "r", encoding="utf-8") as f:
        subs = json.load(f)
    body = _transcript_plaintext(subs, original=True)
    return Response(
        content=body.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{ck}_original.txt"'},
    )


@app.get("/api/transcript/{cache_key}/translated.txt")
def get_transcript_translated(cache_key: str):
    ck = cache_key.lower().strip()
    if not _valid_cache_key_hex(ck):
        raise HTTPException(status_code=404, detail="Not found")
    disk = CACHE_DIR / f"{ck}_translated.txt"
    if disk.is_file():
        return FileResponse(disk, media_type="text/plain; charset=utf-8", filename=f"{ck}_translated.txt")
    json_path = CACHE_DIR / f"{ck}.json"
    if not json_path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    with open(json_path, "r", encoding="utf-8") as f:
        subs = json.load(f)
    body = _transcript_plaintext(subs, original=False)
    return Response(
        content=body.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{ck}_translated.txt"'},
    )


@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a running pipeline job.
    Signals the SSE stream to close with an error; the background thread
    finishes its current step then exits on the next push() call.
    """
    if job_id not in job_buffers:
        raise HTTPException(status_code=404, detail="Job not found")
    buf = job_buffers.get(job_id)
    if buf:
        with buf["lock"]:
            buf["events"].append({"error": "Job cancelled by user"})
            buf["events"].append(None)
    # Clean up inflight tracking
    with _inflight_job_lock:
        for ck, jid in list(_inflight_job_by_cache.items()):
            if jid == job_id:
                _inflight_job_by_cache.pop(ck, None)
    logger.info(f"Job {job_id[:8]}… cancelled by user")
    return {"ok": True, "job_id": job_id}


@app.get("/api/export/{cache_key}/subtitles.srt")
def export_srt(cache_key: str):
    """
    Export subtitles as SRT (SubRip) format — the universal standard for video players,
    YouTube, Netflix, VLC, Premiere, etc.
    """
    ck = cache_key.lower().strip()
    if not _valid_cache_key_hex(ck):
        raise HTTPException(status_code=404, detail="Not found")
    json_path = CACHE_DIR / f"{ck}.json"
    if not json_path.is_file():
        raise HTTPException(status_code=404, detail="Subtitles not found")
    with open(json_path, "r", encoding="utf-8") as f:
        subs = json.load(f)

    def _srt_time(s: float) -> str:
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = int(s % 60)
        ms = int(round((s - int(s)) * 1000))
        return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

    lines = []
    for i, seg in enumerate(subs, 1):
        start = float(seg.get("start", 0))
        end = float(seg.get("end", 0))
        text = (seg.get("translated_text") or seg.get("translated") or seg.get("text") or "").strip()
        lines.append(str(i))
        lines.append(f"{_srt_time(start)} --> {_srt_time(end)}")
        lines.append(text)
        lines.append("")

    content = "\n".join(lines)
    return Response(
        content=content.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{ck}_dubbed.srt"'},
    )


@app.get("/api/export/{cache_key}/subtitles.vtt")
def export_vtt(cache_key: str):
    """
    Export subtitles as WebVTT format — standard for HTML5 video, browsers, and streaming.
    """
    ck = cache_key.lower().strip()
    if not _valid_cache_key_hex(ck):
        raise HTTPException(status_code=404, detail="Not found")
    json_path = CACHE_DIR / f"{ck}.json"
    if not json_path.is_file():
        raise HTTPException(status_code=404, detail="Subtitles not found")
    with open(json_path, "r", encoding="utf-8") as f:
        subs = json.load(f)

    def _vtt_time(s: float) -> str:
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = int(s % 60)
        ms = int(round((s - int(s)) * 1000))
        return f"{h:02d}:{m:02d}:{sec:02d}.{ms:03d}"

    lines = ["WEBVTT", ""]
    for i, seg in enumerate(subs, 1):
        start = float(seg.get("start", 0))
        end = float(seg.get("end", 0))
        text = (seg.get("translated_text") or seg.get("translated") or seg.get("text") or "").strip()
        lines.append(str(i))
        lines.append(f"{_vtt_time(start)} --> {_vtt_time(end)}")
        lines.append(text)
        lines.append("")

    content = "\n".join(lines)
    return Response(
        content=content.encode("utf-8"),
        media_type="text/vtt; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{ck}_dubbed.vtt"'},
    )


@app.get("/api/export/{cache_key}/video")
async def export_video(cache_key: str, background_tasks=None):
    """
    Export a dubbed MP4 video: original YouTube video + dubbed audio merged via ffmpeg.

    This is the final professional deliverable — a complete dubbed video file ready
    for distribution on YouTube, Netflix, or any streaming platform.

    Requires: ffmpeg installed, original video downloaded (uses yt-dlp with same cache_key).
    """
    ck = cache_key.lower().strip()
    if not _valid_cache_key_hex(ck):
        raise HTTPException(status_code=404, detail="Not found")

    # Check dubbed audio exists
    dub_path = _dub_output_disk_path(ck)
    if not dub_path.is_file():
        raise HTTPException(status_code=404, detail="Dubbed audio not found. Process the video first.")

    meta_path = CACHE_DIR / f"{ck}_meta.json"
    if not meta_path.is_file():
        raise HTTPException(status_code=404, detail="Video metadata not found. Process the video first.")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    youtube_url = meta.get("youtube_url") or meta.get("url")
    if not youtube_url:
        raise HTTPException(
            status_code=422,
            detail="Original YouTube URL not stored. Re-process the video to enable export.",
        )

    output_mp4 = CACHE_DIR / f"{ck}_dubbed.mp4"

    # If already exported and fresh, serve it directly
    if output_mp4.is_file() and output_mp4.stat().st_mtime >= dub_path.stat().st_mtime:
        return FileResponse(
            output_mp4,
            media_type="video/mp4",
            filename=f"{ck}_dubbed.mp4",
            headers={"Content-Disposition": f'attachment; filename="dubbed_{meta.get("title", ck)[:60]}.mp4"'},
        )

    if not shutil.which("ffmpeg"):
        raise HTTPException(status_code=503, detail="ffmpeg not installed on server")

    # Download original video (video-only stream, no audio) via yt-dlp
    import tempfile, subprocess
    tmp_video = CACHE_DIR / f"{ck}_source_video.mp4"

    if not tmp_video.is_file():
        try:
            import yt_dlp
            ydl_opts = {
                "format": "bestvideo[ext=mp4]/bestvideo",
                "outtmpl": str(tmp_video.with_suffix("")) + ".%(ext)s",
                "quiet": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
            # yt-dlp may choose a different extension
            if not tmp_video.is_file():
                candidates = list(CACHE_DIR.glob(f"{ck}_source_video.*"))
                if candidates:
                    tmp_video = candidates[0]
                else:
                    raise HTTPException(status_code=500, detail="Failed to download source video stream")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Video download failed: {e}")

    # Merge: video stream (no audio) + dubbed audio → output MP4
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(tmp_video),
            "-i", str(dub_path),
            "-c:v", "copy",          # copy video stream (no re-encode)
            "-c:a", "aac",           # encode audio as AAC (MP4 standard)
            "-b:a", "192k",
            "-map", "0:v:0",         # video from first input
            "-map", "1:a:0",         # audio from second input (dubbed)
            "-shortest",             # trim to shorter stream
            "-movflags", "+faststart",  # optimize for streaming
            str(output_mp4),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode != 0:
            err = result.stderr.decode("utf-8", errors="replace")[-500:]
            raise HTTPException(status_code=500, detail=f"ffmpeg merge failed: {err}")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Video export timed out (>5 min)")

    logger.info(f"[export] Dubbed video created: {output_mp4.name} ({output_mp4.stat().st_size // 1024 // 1024}MB)")

    return FileResponse(
        output_mp4,
        media_type="video/mp4",
        filename=f"{ck}_dubbed.mp4",
        headers={
            "Content-Disposition": f'attachment; filename="dubbed_{meta.get("title", ck)[:60]}.mp4"',
        },
    )


def _run_export_sync(ck: str) -> None:
    """
    Blocking worker: download source video (video-only stream) via yt-dlp,
    merge with the dubbed MP3 via ffmpeg, write {ck}_dubbed.mp4 to CACHE_DIR.
    Updates _export_jobs[ck] in place.
    """
    import subprocess as _sp
    import yt_dlp as _ytdlp
    from services.downloader import _find_ffmpeg, _QuietLogger, _cookie_opts

    try:
        _export_jobs[ck]["status"] = "running"

        dub_path = _dub_output_disk_path(ck)
        meta_path = CACHE_DIR / f"{ck}_meta.json"
        output_mp4 = CACHE_DIR / f"{ck}_dubbed.mp4"

        meta = json.loads(meta_path.read_text("utf-8"))
        youtube_url = meta.get("youtube_url") or meta.get("url") or ""
        title = meta.get("title") or ck

        if not youtube_url:
            raise RuntimeError("YouTube URL not stored in metadata — re-process the video to fix this.")

        ffmpeg_bin = _find_ffmpeg()
        if not ffmpeg_bin:
            raise RuntimeError("ffmpeg not found. Install with: brew install ffmpeg")

        # ── Step 1: download video-only stream ──────────────────────────────
        tmp_base = str(CACHE_DIR / f"{ck}_source_video")
        raw_video: str | None = None

        # Re-use an already-downloaded source if present
        for _ext in ("mp4", "mkv", "webm"):
            _c = f"{tmp_base}.{_ext}"
            if os.path.isfile(_c) and os.path.getsize(_c) > 10240:
                raw_video = _c
                break

        if not raw_video:
            cookie_opts = _cookie_opts()
            ydl_base: dict = {
                "format": "bestvideo[ext=mp4][height<=1080]/bestvideo[height<=1080]/bestvideo",
                "outtmpl": tmp_base + ".%(ext)s",
                "quiet": True,
                "no_warnings": True,
                "logger": _QuietLogger(),
                "ffmpeg_location": os.path.dirname(ffmpeg_bin),
            }
            for _clients, _cookies in [
                (["web"], True), (["web_embedded"], True), (["ios"], True),
                (None, True), (["ios"], False), (None, False),
            ]:
                _opts = {**ydl_base}
                if _cookies and cookie_opts:
                    _opts.update(cookie_opts)
                if _clients:
                    _opts["extractor_args"] = {"youtube": {"player_client": _clients}}
                try:
                    with _ytdlp.YoutubeDL(_opts) as _ydl:
                        _ydl.extract_info(youtube_url, download=True)
                    for _ext in ("mp4", "mkv", "webm"):
                        _c = f"{tmp_base}.{_ext}"
                        if os.path.isfile(_c) and os.path.getsize(_c) > 10240:
                            raw_video = _c
                            break
                    if raw_video:
                        break
                except Exception as _e:
                    logger.warning(f"[export] video download attempt failed: {_e}")
                    continue

        if not raw_video:
            raise RuntimeError(
                "Could not download the video stream from YouTube.\n"
                "Make sure you are logged into YouTube in Safari and relaunch the app."
            )

        # ── Step 2: merge video + dubbed audio ───────────────────────────────
        cmd = [
            ffmpeg_bin, "-y",
            "-i", raw_video,
            "-i", str(dub_path),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(output_mp4),
        ]
        res = _sp.run(cmd, capture_output=True, timeout=600)

        # Remove raw video to save disk space
        try:
            os.remove(raw_video)
        except OSError:
            pass

        if res.returncode != 0 or not output_mp4.is_file() or output_mp4.stat().st_size < 10240:
            err_snippet = res.stderr.decode("utf-8", errors="replace")[-400:]
            raise RuntimeError(f"ffmpeg merge failed: {err_snippet}")

        size_mb = output_mp4.stat().st_size // 1024 // 1024
        logger.info(f"[export] {ck}: dubbed video ready — {size_mb} MB")
        _export_jobs[ck].update({"status": "done", "title": title})

    except Exception as exc:
        logger.error(f"[export] {ck} failed: {exc}")
        _export_jobs[ck].update({"status": "error", "error": str(exc)})


@app.post("/api/export/{cache_key}/video")
async def start_export_video(cache_key: str):
    """
    Start an async export job: download original video + merge with dubbed audio.
    Returns {"status": "done"} immediately if already cached, otherwise {"status": "pending"}.
    """
    ck = cache_key.lower().strip()
    if not _valid_cache_key_hex(ck):
        raise HTTPException(status_code=404, detail="Not found")

    dub_path = _dub_output_disk_path(ck)
    if not dub_path.is_file():
        raise HTTPException(status_code=404, detail="Dubbed audio not found — process the video first.")

    meta_path = CACHE_DIR / f"{ck}_meta.json"
    if not meta_path.is_file():
        raise HTTPException(status_code=404, detail="Metadata not found — process the video first.")

    output_mp4 = CACHE_DIR / f"{ck}_dubbed.mp4"

    # Already exported and still fresh (not older than the dub file)?
    if output_mp4.is_file() and output_mp4.stat().st_mtime >= dub_path.stat().st_mtime:
        return {"status": "done", "cache_key": ck}

    # Already running?
    existing = _export_jobs.get(ck, {})
    if existing.get("status") in ("pending", "running"):
        return {"status": existing["status"], "cache_key": ck}

    # Kick off background thread
    _export_jobs[ck] = {"status": "pending", "error": None, "title": None}
    asyncio.create_task(asyncio.to_thread(_run_export_sync, ck))
    return {"status": "pending", "cache_key": ck}


@app.get("/api/export/{cache_key}/video-status")
async def get_export_video_status(cache_key: str):
    """Poll the export job status. Returns {status: pending|running|done|error, error?}."""
    ck = cache_key.lower().strip()
    if not _valid_cache_key_hex(ck):
        raise HTTPException(status_code=404, detail="Not found")

    output_mp4 = CACHE_DIR / f"{ck}_dubbed.mp4"
    dub_path = _dub_output_disk_path(ck)
    if (
        output_mp4.is_file()
        and dub_path.is_file()
        and output_mp4.stat().st_mtime >= dub_path.stat().st_mtime
    ):
        return {"status": "done", "cache_key": ck}

    job = _export_jobs.get(ck)
    if not job:
        return {"status": "idle"}
    return {"status": job["status"], "error": job.get("error"), "cache_key": ck}


@app.get("/api/export/{cache_key}/video-file")
async def download_export_video_file(cache_key: str):
    """Serve the exported dubbed MP4 as a browser download."""
    ck = cache_key.lower().strip()
    if not _valid_cache_key_hex(ck):
        raise HTTPException(status_code=404, detail="Not found")

    output_mp4 = CACHE_DIR / f"{ck}_dubbed.mp4"
    if not output_mp4.is_file():
        raise HTTPException(status_code=404, detail="Export not ready — start the export first.")

    meta_path = CACHE_DIR / f"{ck}_meta.json"
    title = ck
    if meta_path.is_file():
        try:
            title = json.loads(meta_path.read_text("utf-8")).get("title") or ck
        except Exception:
            pass

    safe_title = re.sub(r'[^\w\s\-]', '', title).strip()[:60] or "dubbed-video"
    filename = f"{safe_title} (dubbed).mp4"
    return FileResponse(
        str(output_mp4),
        media_type="video/mp4",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/llm-validation-log")
def download_llm_validation_log():
    """Download the LLM validation log file as a .txt attachment."""
    log_path = CACHE_DIR / "llm_validation_log.txt"
    if not log_path.is_file():
        # Return an empty placeholder so the download always works
        empty = (
            "LLM Validation Log\n"
            "==================\n\n"
            "No validation events recorded yet.\n\n"
            "Translate an English video to Hindi (or another Indic language)\n"
            "to see validation entries here.\n"
        )
        return Response(
            content=empty,
            media_type="text/plain",
            headers={"Content-Disposition": 'attachment; filename="llm_validation_log.txt"'},
        )
    return FileResponse(
        str(log_path),
        media_type="text/plain",
        headers={"Content-Disposition": 'attachment; filename="llm_validation_log.txt"'},
    )


@app.post("/api/clear-cache")
def clear_cache():
    # Only allow when no jobs are running to prevent corruption
    with _inflight_job_lock:
        if _inflight_job_by_cache:
            raise HTTPException(
                status_code=409,
                detail="Cannot clear cache while jobs are running. Cancel them first.",
            )
    count = 0
    for f in CACHE_DIR.iterdir():
        if f.is_file():
            try:
                f.unlink()
                count += 1
            except OSError:
                pass
    logger.info(f"Cache cleared: {count} file(s) deleted")
    return {"cleared": count}


@app.get("/")
def root():
    return {"message": "YouTube Video Translator API", "health": "/health", "docs": "/docs"}
