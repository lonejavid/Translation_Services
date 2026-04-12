"""
English (or detected) → target language via **deep-translator** (Google Translate).

Flow (matches product spec):
1. Whisper segments already written with timestamps in ``{cache}_original.txt``.
2. Build **timestamp-free** ``{cache}_source_plain.txt`` (one segment per line).
3. Translate with ``GoogleTranslator`` — we use ``translate_batch`` line-by-line chunks so each
   subtitle stays aligned (``translate_file`` reads the whole file and calls ``translate()`` once,
   which is limited to ~5000 chars and does **not** preserve one-line-per-segment boundaries).
4. Write ``{cache}_translated_plain.txt`` as UTF-8 (``Path.write_text(..., encoding="utf-8")``)
   and return segment dicts for TTS (timestamps from STT).

Optional **translation validation** (repetition / length / script) catches corrupted batches;
invalid lines are retranslated (single-line retries) then fall back to source. See env
``TRANSLATION_VALIDATION_*``.

Requires **internet**. See env ``GOOGLE_TRANSLATE_*`` for rate-limit tuning.
"""
from __future__ import annotations

import os
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

# UI language codes → Google Translate ``target`` codes (deep-translator / Google).
TARGET_TO_GOOGLE: dict[str, str] = {
    "en": "en",
    "hi": "hi",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "ar": "ar",
    "pt": "pt",
    "ja": "ja",
    "ko": "ko",
    "zh": "zh-CN",
    "ru": "ru",
    "it": "it",
    "nl": "nl",
    "tr": "tr",
    "pl": "pl",
    "sv": "sv",
}


def translation_validation_enabled() -> bool:
    return os.environ.get("TRANSLATION_VALIDATION_ENABLED", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def translation_validation_strict() -> bool:
    return os.environ.get("TRANSLATION_VALIDATION_STRICT", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _length_ratio_bounds() -> tuple[float, float]:
    if translation_validation_strict():
        return 0.3, 2.0
    return 0.25, 2.5


def _repetition_threshold() -> float:
    return 0.30 if translation_validation_strict() else 0.35


def _word_tokens(text: str) -> list[str]:
    """Whitespace-separated tokens (works reasonably for Hindi + Latin)."""
    return [t for t in re.split(r"\s+", text.strip()) if t]


def _dominant_script_is_latin_only(text: str) -> bool:
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            if "LATIN" not in name.upper():
                return False
    return bool(re.search(r"[a-zA-Z]", text))


def _has_hindi_script(text: str) -> bool:
    return bool(re.search(r"[\u0900-\u097F]", text))


def is_valid_translation(
    source_text: str,
    translated_text: str,
    target_language: str = "hi",
) -> bool:
    """
    Heuristic quality gate: repetition, length vs source, and (for Hindi) Devanagari presence.

    Returns True if the pair looks acceptable; False if likely corrupted (e.g. repeated phrase spam).
    """
    src = (source_text or "").strip()
    tgt = (translated_text or "").strip()
    if not src:
        return True
    if not tgt:
        return False
    if src == tgt:
        return True

    lo, hi = _length_ratio_bounds()
    ls = max(len(src), 1)
    lt = len(tgt)
    ratio = lt / ls
    if len(src) >= 8:
        if ratio < lo or ratio > hi:
            return False

    words = _word_tokens(tgt)
    if len(words) >= 3:
        top = Counter(words).most_common(1)[0][1]
        if top / len(words) > _repetition_threshold():
            return False
    elif len(words) == 2 and words[0] == words[1] and len(src) >= 16:
        return False
    elif len(words) == 1 and len(src) >= 24:
        return False

    tl = (target_language or "en").strip().lower()[:8]
    if tl == "hi":
        if len(tgt) >= 10 and not _has_hindi_script(tgt):
            if _dominant_script_is_latin_only(tgt):
                return False

    if not any(ch.isalnum() for ch in tgt):
        return False

    return True


def _translate_one_line(translator, line: str, delay_s: float) -> str:
    line = (line or "").strip()
    if not line:
        return ""
    try:
        out = translator.translate(line)
        time.sleep(delay_s)
        return (out or "").strip()
    except Exception as e:
        print(f"[google-translate] single-line retry failed: {e!r}")
        time.sleep(delay_s)
        return ""


def retranslate_editor_segments(
    old_rows: list[dict],
    new_rows: list[dict],
    detected_source_lang: str,
    target_language: str,
) -> tuple[list[dict], int]:
    """
    For each segment whose **source** ``text`` changed vs ``old_rows``, replace
    ``translated_text`` with a fresh Google Translate of the new source.

    Segments with unchanged source keep the editor's ``translated_text`` (manual fixes).

    Returns ``(new_rows, n_retranslated)``. Mutates dicts inside ``new_rows`` in place.
    """
    from deep_translator import GoogleTranslator

    tgt = (target_language or "hi").strip().lower()[:8]

    g_tgt = TARGET_TO_GOOGLE.get(tgt, tgt)
    g_src = _google_source_lang(detected_source_lang)
    if g_src != "auto" and g_src == g_tgt:
        for row in new_rows:
            row["translated_text"] = (row.get("text") or "").strip()
        return new_rows, 0

    translator = GoogleTranslator(source=g_src, target=g_tgt)
    delay_s = float(os.environ.get("GOOGLE_TRANSLATE_DELAY_S", "0.25"))

    from services.learned_corrections import apply_learned_phrase_fixes

    n_done = 0
    for i, row in enumerate(new_rows):
        old_t = ""
        if i < len(old_rows):
            old_t = (old_rows[i].get("text") or "").strip()
        new_t = (row.get("text") or "").strip()
        if old_t == new_t:
            continue
        src_line = new_t.replace("\n", " ").replace("\r", " ")
        src_line = apply_learned_phrase_fixes(src_line)
        if not src_line.strip():
            row["translated_text"] = ""
            continue
        tr = _translate_one_line(translator, src_line, delay_s)
        if tr:
            row["translated_text"] = tr
            n_done += 1
            if not is_valid_translation(src_line, tr, tgt):
                print(
                    f"[google-translate] editor segment {i + 1}: note — "
                    f"translation validation soft for this line"
                )
            else:
                print(f"[google-translate] editor segment {i + 1}: retranslated after source edit")
        else:
            print(
                f"[google-translate] editor segment {i + 1}: "
                f"translate failed; keeping previous translation line"
            )

    return new_rows, n_done


def _repair_invalid_lines(
    translator,
    plain_lines: list[str],
    out_lines: list[str],
    target_language: str,
    delay_s: float,
) -> tuple[list[str], int, int]:
    """
    For each invalid (src, tr), up to 2 single-line retries, then source fallback.
    Returns (new_out_lines, n_retried_ok, n_fallback).
    """
    strict = translation_validation_strict()
    repaired = list(out_lines)
    n_ok = 0
    n_fb = 0
    for j, src in enumerate(plain_lines):
        if j >= len(repaired):
            break
        tr = (repaired[j] or "").strip()
        if is_valid_translation(src, tr, target_language):
            continue
        print(
            f"[translation-validation] segment {j + 1}: invalid translation "
            f"(strict={strict}); retrying… src_len={len(src)} tr_preview={tr[:80]!r}"
        )
        new_tr = tr
        for attempt in range(2):
            new_tr = _translate_one_line(translator, src, delay_s)
            if new_tr and is_valid_translation(src, new_tr, target_language):
                repaired[j] = new_tr
                n_ok += 1
                print(
                    f"[translation-validation] segment {j + 1}: recovered on retry {attempt + 1}"
                )
                break
        else:
            repaired[j] = src
            n_fb += 1
            print(
                f"[translation-validation] WARNING: segment {j + 1}: "
                f"fallback to source text after failed validation (TTS will speak source language for this line)"
            )
    return repaired, n_ok, n_fb


def translation_validation_report(
    translated: list[dict],
    target_language: str,
    *,
    backend: str,
) -> None:
    """Post-hoc summary for pipeline logs (all backends)."""
    if not translation_validation_enabled():
        print("[translation-validation] disabled (TRANSLATION_VALIDATION_ENABLED=0)")
        return
    n = len(translated)
    if n == 0:
        return
    bad: list[int] = []
    for i, seg in enumerate(translated):
        src = (seg.get("text") or "").strip()
        tr = (seg.get("translated_text") or "").strip()
        if not is_valid_translation(src, tr, target_language):
            bad.append(i)
    ok = n - len(bad)
    print(
        f"[translation-validation] backend={backend} target={target_language!r} "
        f"strict={translation_validation_strict()} — "
        f"{ok}/{n} segments pass checks"
    )
    if bad:
        preview = ", ".join(str(i + 1) for i in bad[:12])
        more = f" (+{len(bad) - 12} more)" if len(bad) > 12 else ""
        print(
            f"[translation-validation] WARNING: segments still failing checks after repair: "
            f"{preview}{more}"
        )
        for i in bad[:3]:
            seg = translated[i]
            tr = (seg.get("translated_text") or "")[:120]
            print(f"   … seg {i + 1} translated_text preview: {tr!r}")


def _google_source_lang(detected: str) -> str:
    d = (detected or "en").lower().strip()[:8]
    if d.startswith("en"):
        return "en"
    return "auto"


def translate_segments_google(
    segments: list[dict],
    detected_source_lang: str,
    target_language: str,
    *,
    cache_dir: Path,
    cache_key: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> list[dict]:
    from deep_translator import GoogleTranslator

    tgt = target_language.strip().lower()[:8]

    g_tgt = TARGET_TO_GOOGLE.get(tgt, tgt)
    g_src = _google_source_lang(detected_source_lang)
    if g_src != "auto" and g_src == g_tgt:
        out2: list[dict] = []
        for s in segments:
            tx = (s.get("text") or "").strip()
            out2.append({
                "start": s["start"],
                "end": s["end"],
                "text": tx,
                "translated_text": tx,
            })
        return out2

    translator = GoogleTranslator(source=g_src, target=g_tgt)

    from services.learned_corrections import (
        apply_learned_phrase_fixes,
        translation_override,
    )

    plain_lines: list[str] = []
    for s in segments:
        t = (s.get("text") or "").strip().replace("\n", " ").replace("\r", " ")
        t = apply_learned_phrase_fixes(t)
        plain_lines.append(t)

    plain_path = cache_dir / f"{cache_key}_source_plain.txt"
    plain_path.write_text("\n".join(plain_lines) + "\n", encoding="utf-8")
    print(f"[google-translate] Timestamp-free source → {plain_path.name} ({len(plain_lines)} lines)")

    delay_s = float(os.environ.get("GOOGLE_TRANSLATE_DELAY_S", "0.25"))

    # Professional context-aware translation: send the entire numbered script
    # so Google Translate sees the full conversation context — pronouns resolve
    # correctly, idioms translate naturally, terminology stays consistent.
    print(f"[google-translate] Using full-context numbered translation ({len(plain_lines)} lines)")
    try:
        from services.context_translator import translate_lines_professional
        out_lines = translate_lines_professional(
            plain_lines,
            translator,
            tgt,
            delay_s=delay_s,
            progress_callback=progress_callback,
        )
    except Exception as ctx_err:
        print(f"[google-translate] context translator failed ({ctx_err!r}); falling back to chunked")
        # Fallback: original chunked approach
        chunk = max(1, int(os.environ.get("GOOGLE_TRANSLATE_CHUNK", "5")))
        out_lines = []
        n = len(plain_lines)
        done = 0
        for i in range(0, n, chunk):
            batch = plain_lines[i : i + chunk]
            part: list[str] = []
            try:
                raw = translator.translate_batch(batch)
                part = [str(x).strip() if x is not None else "" for x in (raw or [])]
            except Exception as e:
                print(f"[google-translate] translate_batch failed ({e}); falling back per line")
            if len(part) < len(batch):
                for k in range(len(part), len(batch)):
                    line = batch[k]
                    try:
                        r = (translator.translate(line) if line.strip() else "") or ""
                        part.append(r.strip())
                    except Exception:
                        part.append(line)
                    time.sleep(delay_s)
            out_lines.extend(part)
            done = min(i + len(batch), n)
            if progress_callback:
                progress_callback(done, n)
            if i + chunk < n:
                time.sleep(delay_s)

    # Final safety net: guarantees 1-to-1 alignment
    while len(out_lines) < len(plain_lines):
        idx = len(out_lines)
        print(f"[google-translate] WARNING: safety-net filling missing segment {idx + 1} with source")
        out_lines.append(plain_lines[idx])

    for j in range(len(out_lines)):
        if j >= len(plain_lines):
            break
        src_ln = plain_lines[j]
        ov = translation_override(src_ln, out_lines[j], tgt)
        if ov is not None:
            out_lines[j] = ov

    if translation_validation_enabled():
        out_lines, n_rec, n_fb = _repair_invalid_lines(
            translator, plain_lines, out_lines, tgt, delay_s
        )
        if n_rec or n_fb:
            print(
                f"[translation-validation] repair pass: {n_rec} segment(s) fixed by retry, "
                f"{n_fb} segment(s) source fallback"
            )

    translated_plain_path = cache_dir / f"{cache_key}_translated_plain.txt"
    translated_plain_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"[google-translate] Translated plain → {translated_plain_path.name}")

    result: list[dict] = []
    for j, seg in enumerate(segments):
        tr = (out_lines[j] if j < len(out_lines) else "").strip()
        src_t = (seg.get("text") or "").strip()
        result.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg.get("text", ""),
            "translated_text": tr if tr else src_t,
        })
    return result
