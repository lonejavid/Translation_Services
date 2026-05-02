"""
Per-segment translation correction store.

Users correct AI-generated translations in the dual-panel TranslationEditor.
Those corrections are stored here and applied in two ways:

  1. REPLAY  — corrections are written back to the segment JSON file so the
               Player page immediately shows corrected text on reload.

  2. LEARNING — corrections are also fed into learned_corrections.py so the
               next pipeline run on any video with the same source text will
               use the corrected translation automatically.

Data file: server/data/translation_corrections.json
One JSON array of correction objects (append-only, deduplicated by segment).
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

_DATA_DIR = Path(
    os.environ.get("CACHE_DIR") or (Path(__file__).resolve().parent.parent / "data")
)
_DATA_PATH = _DATA_DIR / "translation_corrections.json"
_lock = threading.RLock()


# ---------------------------------------------------------------------------
# Internal I/O
# ---------------------------------------------------------------------------

def _load() -> list[dict]:
    try:
        raw = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
        return raw if isinstance(raw, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _save(corrections: list[dict]) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _DATA_PATH.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(corrections, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp.replace(_DATA_PATH)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_corrections(items: list[dict[str, Any]]) -> list[dict]:
    """
    Upsert a list of corrections.

    Required keys per item:
      cache_key              — 32-char MD5 hex string
      segment_index          — integer index into the segment array
      source_text            — original source-language text
      incorrect_translation  — AI-generated text before the edit
      corrected_translation  — user-supplied replacement
      source_lang            — e.g. "en"
      target_lang            — e.g. "hi"

    Optional:
      video_id               — YouTube video ID (informational)

    Deduplicates by (cache_key, segment_index) — latest wins.
    Returns the list of saved records (with server-assigned IDs/timestamps).
    """
    with _lock:
        existing = _load()

        # Build index for O(1) upsert
        idx: dict[str, int] = {}
        for i, c in enumerate(existing):
            k = _seg_key(c)
            if k:
                idx[k] = i

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        saved: list[dict] = []

        for item in items:
            cache_key = (item.get("cache_key") or "").strip().lower()
            seg_index = item.get("segment_index")
            corrected = (item.get("corrected_translation") or "").strip()

            if not cache_key or seg_index is None:
                continue

            entry: dict = {
                "id": item.get("id") or str(uuid.uuid4()),
                "cache_key": cache_key,
                "segment_index": int(seg_index),
                "segment_id": item.get("segment_id") or f"segment_{seg_index}",
                "source_text": (item.get("source_text") or "").strip(),
                "incorrect_translation": (
                    item.get("incorrect_translation") or ""
                ).strip(),
                "corrected_translation": corrected,
                "source_lang": (item.get("source_lang") or "en").strip()[:8],
                "target_lang": (item.get("target_lang") or "en").strip()[:8],
                "video_id": (item.get("video_id") or "").strip(),
                "timestamp": now,
            }

            k = _seg_key(entry)
            if k in idx:
                existing[idx[k]] = entry
            else:
                idx[k] = len(existing)
                existing.append(entry)

            saved.append(entry)

        _save(existing)
        return saved


def get_corrections_for_cache_key(cache_key: str) -> dict[int, dict]:
    """
    Return {segment_index: correction_record} for one video.
    Fast path — no lock needed (read-only, file is atomically replaced).
    """
    ck = (cache_key or "").strip().lower()
    result: dict[int, dict] = {}
    for c in _load():
        if (c.get("cache_key") or "").lower() == ck:
            si = c.get("segment_index")
            if si is not None:
                result[int(si)] = c
    return result


def get_all_corrections(limit: int = 500, offset: int = 0) -> dict:
    """Return all corrections, most-recent first, paginated."""
    all_c = sorted(
        _load(), key=lambda x: x.get("timestamp", ""), reverse=True
    )
    return {
        "total": len(all_c),
        "offset": offset,
        "limit": limit,
        "corrections": all_c[offset: offset + limit],
    }


def delete_correction(correction_id: str) -> bool:
    """Hard-delete by ID. Returns True if found."""
    with _lock:
        before = _load()
        after = [c for c in before if c.get("id") != correction_id]
        if len(after) == len(before):
            return False
        _save(after)
        return True


def delete_corrections_for_segment(cache_key: str, segment_index: int) -> bool:
    """Remove a single segment's correction (used for 'Reset to AI' action)."""
    ck = (cache_key or "").strip().lower()
    with _lock:
        before = _load()
        after = [
            c for c in before
            if not (
                (c.get("cache_key") or "").lower() == ck
                and c.get("segment_index") == segment_index
            )
        ]
        if len(after) == len(before):
            return False
        _save(after)
        return True


def apply_corrections_to_segments(
    segments: list[dict], cache_key: str
) -> list[dict]:
    """
    Return a new segment list with user corrections applied.

    Corrected segments gain two extra keys:
      corrected       — True
      correction_id   — ID of the correction record
    """
    corrections = get_corrections_for_cache_key(cache_key)
    if not corrections:
        return segments

    result = []
    for i, seg in enumerate(segments):
        if i in corrections:
            seg = dict(seg)
            c = corrections[i]
            seg["translated_text"] = c["corrected_translation"]
            seg["corrected"] = True
            seg["correction_id"] = c["id"]
        result.append(seg)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg_key(c: dict) -> str | None:
    ck = (c.get("cache_key") or "").strip()
    si = c.get("segment_index")
    if not ck or si is None:
        return None
    return f"{ck}:{si}"
