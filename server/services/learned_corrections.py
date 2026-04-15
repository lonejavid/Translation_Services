"""
Persist user transcript/translation edits and apply them on future pipeline runs.

This is **not** neural fine-tuning: we store phrase-level ASR fixes and exact-line
translation overrides so the next job can correct recurring mistakes automatically.

Data file: ``server/data/learned_corrections.json`` (create ``server/data/`` on first save).
Env:
  ``LEARNED_CORRECTIONS_MAX_ASR`` — max ASR phrase pairs (default 400).
  ``LEARNED_CORRECTIONS_MAX_TM`` — max translation-memory lines (default 400).
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

_DATA_DIR = Path(
    os.environ.get("CACHE_DIR") or (Path(__file__).resolve().parent.parent / "data")
)
_DATA_PATH = _DATA_DIR / "learned_corrections.json"

_mtime_cache: float = 0.0
_cache_payload: dict[str, Any] | None = None


def _max_asr() -> int:
    raw = os.environ.get("LEARNED_CORRECTIONS_MAX_ASR", "2000").strip()
    try:
        return max(10, min(20000, int(raw)))
    except ValueError:
        return 2000


def _max_tm() -> int:
    # Raised from 400 → 2000 so corrections accumulate globally across all videos
    raw = os.environ.get("LEARNED_CORRECTIONS_MAX_TM", "2000").strip()
    try:
        return max(10, min(20000, int(raw)))
    except ValueError:
        return 2000


# Minimum Jaccard word-overlap to count as a fuzzy match (0–1.0)
# 0.82 = "at least 82% of words in common" — catches punctuation variants,
# minor word order differences, and extra/missing articles.
_FUZZY_THRESHOLD = 0.82


def _default_payload() -> dict[str, Any]:
    return {"asr_phrases": [], "translation_memory": []}


def load_payload() -> dict[str, Any]:
    global _mtime_cache, _cache_payload
    if os.environ.get("LEARNED_CORRECTIONS_DISABLE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return _default_payload()
    try:
        st = _DATA_PATH.stat().st_mtime
    except OSError:
        return _default_payload()
    if _cache_payload is not None and st == _mtime_cache:
        return _cache_payload
    try:
        raw = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    ap = raw.get("asr_phrases")
    tm = raw.get("translation_memory")
    if not isinstance(ap, list):
        ap = []
    if not isinstance(tm, list):
        tm = []
    _cache_payload = {"asr_phrases": ap, "translation_memory": tm}
    _mtime_cache = st
    return _cache_payload


def invalidate_cache() -> None:
    global _mtime_cache, _cache_payload
    _mtime_cache = 0.0
    _cache_payload = None


def _save_payload(data: dict[str, Any]) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _DATA_DIR / "learned_corrections.json.tmp"
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_DATA_PATH)
    invalidate_cache()


def _norm_line(s: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation that varies across phrasings."""
    import unicodedata
    # Normalise unicode (é → e), then strip non-alphanumeric/space
    nfkd = unicodedata.normalize("NFKD", (s or ""))
    cleaned = re.sub(r"[^\w\s]", " ", nfkd, flags=re.UNICODE)
    return " ".join(cleaned.strip().split()).lower()


def _jaccard_words(a: str, b: str) -> float:
    """
    Jaccard similarity on word sets.  Fast and effective for short sentences.

    Examples:
      "once upon a time"  vs  "once upon a time,"  → 1.00  (punctuation stripped)
      "good morning sir"  vs  "good morning, sir"  → 1.00
      "the quick fox"     vs  "the quick brown fox" → 0.75
      "hello world"       vs  "goodbye world"       → 0.33
    """
    wa = set(a.split())
    wb = set(b.split())
    if not wa and not wb:
        return 1.0
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def asr_pairs_sorted() -> list[tuple[str, str]]:
    """(wrong, right) longest wrong first."""
    rows = load_payload().get("asr_phrases") or []
    pairs: list[tuple[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        w = (row.get("wrong") or "").strip()
        r = (row.get("right") or "").strip()
        if w and r and w.lower() != r.lower():
            pairs.append((w, r))
    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    return pairs


def apply_learned_phrase_fixes(text: str) -> str:
    """Replace known wrong phrases (e.g. recurring ASR errors) in one segment."""
    if not (text or "").strip():
        return text
    out = text
    for wrong, right in asr_pairs_sorted():
        try:
            pat = re.compile(re.escape(wrong), re.IGNORECASE)
        except re.error:
            continue
        out = pat.sub(right, out)
    return out


def translation_override(source_line: str, current: str, target_lang: str) -> str | None:
    """
    Return a user-corrected translation if this source sentence matches any
    stored correction — **globally across all videos and users**.

    Two-pass matching:

    Pass 1 — Exact match (O(n), hash-equivalent after normalisation).
      Sentences that are identical after lowercasing and stripping punctuation
      get the stored correction immediately.

    Pass 2 — Fuzzy Jaccard word-overlap (O(n×w) where w = avg words/sentence).
      If no exact match, scan all TM entries for this target language and return
      the one with the highest word-overlap score, provided it clears
      ``_FUZZY_THRESHOLD`` (default 0.82 = 82% words in common).

      This catches:
        • Minor punctuation differences  ("hello, world" vs "hello world")
        • Extra/missing articles         ("a great day" vs "great day")
        • Capitalisation variants        ("Good Morning" vs "good morning")
        • Word-order micro-swaps         ("yes, I agree" vs "I agree, yes")

    Why this makes corrections GLOBAL:
      Every correction saved by any user on any video is stored in
      ``data/learned_corrections.json`` under ``translation_memory``.
      This function is called for **every segment of every future video**
      during the translation pipeline step in ``google_translate_service.py``.
      So one correction teaches the system globally — the model does not need
      to see the same video again; it benefits every future translation run.
    """
    key_lang = (target_lang or "").strip().lower()[:8]
    n = _norm_line(source_line)
    if not n:
        return None

    # Filter to the relevant language pair once (avoids repeated checks in loops)
    tm_for_lang = [
        row for row in (load_payload().get("translation_memory") or [])
        if isinstance(row, dict)
        and (row.get("target_lang") or "").strip().lower()[:8] == key_lang
    ]

    # ── Pass 1: exact match ──────────────────────────────────────────────────
    for row in tm_for_lang:
        if _norm_line(row.get("source") or "") == n:
            t = (row.get("translation") or "").strip()
            if t:
                print(
                    f"[learned] Exact match → applying global correction "
                    f"for target_lang={key_lang!r}"
                )
                return t

    # ── Pass 2: fuzzy Jaccard word-overlap ───────────────────────────────────
    best_translation: str | None = None
    best_score: float = 0.0

    for row in tm_for_lang:
        stored_src = _norm_line(row.get("source") or "")
        if not stored_src:
            continue
        t = (row.get("translation") or "").strip()
        if not t:
            continue

        score = _jaccard_words(n, stored_src)
        if score > best_score and score >= _FUZZY_THRESHOLD:
            best_score = score
            best_translation = t

    if best_translation:
        print(
            f"[learned] Fuzzy match (Jaccard={best_score:.2f} ≥ {_FUZZY_THRESHOLD}) "
            f"→ applying global correction for target_lang={key_lang!r}"
        )
        return best_translation

    return None


def record_edits(
    old_rows: list[dict[str, Any]] | None,
    new_rows: list[dict[str, Any]],
    *,
    target_lang: str,
    source_lang: str = "en",
) -> tuple[int, int]:
    """
    Diff subtitle rows by index; store ASR phrase fixes and translation-memory entries
    so every future pipeline run on ANY video benefits from these corrections globally.

    Translation memory entries are stored whenever the corrected translation differs
    from the AI-generated one — old_r is NOT required, so corrections saved directly
    via POST /api/corrections (without a prior "old" state) are still recorded.

    Returns (n_asr_added, n_tm_added).
    """
    pl = load_payload()
    data = {
        "asr_phrases": list(pl.get("asr_phrases") or []),
        "translation_memory": list(pl.get("translation_memory") or []),
    }

    n_asr = 0
    n_tm = 0
    old = old_rows or []
    tgt = (target_lang or "en").strip().lower()[:8]
    src = (source_lang or "en").strip().lower()[:8]
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    for i, new_r in enumerate(new_rows):
        if not isinstance(new_r, dict):
            continue
        new_text = (new_r.get("text") or new_r.get("original") or "").strip()
        new_tr = (
            new_r.get("translated_text")
            or new_r.get("translated")
            or ""
        ).strip()
        old_r = old[i] if i < len(old) and isinstance(old[i], dict) else None
        old_text = (
            (old_r.get("text") or old_r.get("original") or "").strip()
            if old_r
            else ""
        )
        old_tr = (
            (old_r.get("translated_text") or old_r.get("translated") or "").strip()
            if old_r
            else ""
        )

        # ASR phrase fix: source text was manually corrected
        if old_r and old_text and new_text and old_text != new_text:
            data["asr_phrases"].append({
                "wrong": old_text,
                "right": new_text,
                "source_lang": src,
                "ts": now,
            })
            n_asr += 1

        # Translation memory: always store when we have a corrected translation.
        # We no longer require old_r so that corrections from the TranslationEditor
        # (which sends source+corrected without a prior row snapshot) are also saved.
        if new_text and new_tr:
            is_changed = (old_tr != new_tr) if old_r else True
            if is_changed or not old_r:
                data["translation_memory"].append({
                    "source": new_text,
                    "translation": new_tr,
                    "target_lang": tgt,
                    "ts": now,
                })
                n_tm += 1

    def _dedupe_asr(items: list[dict]) -> list[dict]:
        by_wrong: dict[str, dict] = {}
        for it in items:
            if not isinstance(it, dict):
                continue
            w = (it.get("wrong") or "").strip()
            if w:
                by_wrong[w.lower()] = it
        merged = list(by_wrong.values())
        return merged[-_max_asr() :]

    def _dedupe_tm(items: list[dict]) -> list[dict]:
        by_key: dict[str, dict] = {}
        for it in items:
            if not isinstance(it, dict):
                continue
            s = (it.get("source") or "").strip()
            tl = (it.get("target_lang") or "").strip().lower()[:8]
            if not s or not tl:
                continue
            by_key[f"{_norm_line(s)}|{tl}"] = it
        merged = list(by_key.values())
        return merged[-_max_tm() :]

    data["asr_phrases"] = _dedupe_asr(data["asr_phrases"])
    data["translation_memory"] = _dedupe_tm(data["translation_memory"])
    _save_payload(data)
    return n_asr, n_tm


def build_initial_prompt_addon(max_chars: int = 120) -> str:
    """Short suffix for Whisper initial_prompt from recent correct phrases."""
    parts: list[str] = []
    for row in load_payload().get("asr_phrases") or []:
        if not isinstance(row, dict):
            continue
        r = (row.get("right") or "").strip()
        if r and r not in parts:
            parts.append(r)
    if not parts:
        return ""
    tail = parts[-8:]
    s = "Preferred wording: " + "; ".join(tail)
    return s if len(s) <= max_chars else s[: max_chars - 3] + "…"
