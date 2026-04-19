"""
Code-switching handler for mixed-language segments.

When a speaker mixes languages mid-sentence (e.g. English sentence with Hindi
words, or Hindi words written in Devanagari inside an English transcript),
Helsinki-NLP gets confused and produces garbled output.

This module:
  1. Detects mixed-script / mixed-language segments using Unicode script analysis
  2. Protects foreign-script words with placeholders before Helsinki-NLP sees them
  3. Restores the protected words after translation
  4. Uses LLaMA3 (via Ollama) for smart translation when available — it handles
     code-switching naturally since it knows multiple languages

All processing is local — no external APIs.
"""
from __future__ import annotations

import os
import re
import logging
import unicodedata
from typing import Callable

logger = logging.getLogger(__name__)

# ── Unicode script block ranges ──────────────────────────────────────────────
# Each tuple is (start_codepoint, end_codepoint)
_SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    "latin":       [(0x0000, 0x024F), (0x1E00, 0x1EFF)],
    "devanagari":  [(0x0900, 0x097F)],   # Hindi, Marathi, Sanskrit
    "arabic":      [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF)],
    "bengali":     [(0x0980, 0x09FF)],
    "tamil":       [(0x0B80, 0x0BFF)],
    "telugu":      [(0x0C00, 0x0C7F)],
    "kannada":     [(0x0C80, 0x0CFF)],
    "malayalam":   [(0x0D00, 0x0D7F)],
    "gujarati":    [(0x0A80, 0x0AFF)],
    "gurmukhi":    [(0x0A00, 0x0A7F)],   # Punjabi
    "cyrillic":    [(0x0400, 0x04FF)],
    "cjk":         [(0x4E00, 0x9FFF), (0x3040, 0x30FF)],  # Chinese/Japanese
    "hebrew":      [(0x0590, 0x05FF)],
}

# Language code → primary script for that language
_LANG_SCRIPT: dict[str, str] = {
    "en": "latin",   "hi": "devanagari", "ar": "arabic",
    "bn": "bengali", "ta": "tamil",      "te": "telugu",
    "kn": "kannada", "ml": "malayalam",  "gu": "gujarati",
    "pa": "gurmukhi","ru": "cyrillic",   "zh": "cjk",
    "ja": "cjk",     "ko": "cjk",        "ur": "arabic",
    "mr": "devanagari", "ne": "devanagari",
    "es": "latin",   "fr": "latin",      "de": "latin",
    "pt": "latin",   "it": "latin",      "nl": "latin",
    "pl": "latin",   "tr": "latin",      "sv": "latin",
}

_PLACEHOLDER_RE = re.compile(r"\[KEEP_\d+\]")


def _char_script(ch: str) -> str | None:
    """Return the script name for a single character, or None for punctuation/digits/space."""
    cp = ord(ch)
    if cp <= 0x0040:  # ASCII control + punctuation
        return None
    for script, ranges in _SCRIPT_RANGES.items():
        for start, end in ranges:
            if start <= cp <= end:
                return script
    return None


def _dominant_script(text: str) -> str | None:
    """Return the script used by the majority of alphabetic characters."""
    counts: dict[str, int] = {}
    for ch in text:
        s = _char_script(ch)
        if s:
            counts[s] = counts.get(s, 0) + 1
    if not counts:
        return None
    return max(counts, key=lambda k: counts[k])


def detect_code_switching(text: str, source_lang: str, target_lang: str) -> dict:
    """
    Analyse a segment for code-switching.

    Returns a dict:
      is_mixed        : bool   — True when multiple scripts detected
      source_script   : str    — expected script for source_lang
      target_script   : str    — expected script for target_lang
      foreign_spans   : list   — list of (start, end, script) character spans in foreign script
      dominant_script : str    — script used by most characters
      is_already_target: bool  — True when segment is already mostly in target_lang script
    """
    src_script = _LANG_SCRIPT.get(source_lang.split("-")[0].lower(), "latin")
    tgt_script = _LANG_SCRIPT.get(target_lang.split("-")[0].lower(), "devanagari")

    # Build a per-character script map
    char_scripts = [_char_script(ch) for ch in text]

    # Find contiguous spans of each script
    foreign_spans: list[tuple[int, int, str]] = []
    i = 0
    while i < len(text):
        s = char_scripts[i]
        if s and s != src_script and s != "latin":
            # Start of a foreign-script run
            j = i
            while j < len(text) and (char_scripts[j] == s or char_scripts[j] is None):
                j += 1
            span_text = text[i:j].strip()
            if len(span_text) >= 1:
                foreign_spans.append((i, j, s))
            i = j
        else:
            i += 1

    # Count characters by script (ignoring None/punctuation)
    src_count = sum(1 for s in char_scripts if s == src_script)
    tgt_count = sum(1 for s in char_scripts if s == tgt_script)
    other_count = sum(1 for s in char_scripts if s and s != src_script and s != tgt_script)
    total = src_count + tgt_count + other_count or 1

    dom = _dominant_script(text)
    is_mixed = bool(foreign_spans) or (tgt_count / total > 0.15)
    is_already_target = tgt_count / total > 0.70

    return {
        "is_mixed": is_mixed,
        "source_script": src_script,
        "target_script": tgt_script,
        "foreign_spans": foreign_spans,
        "dominant_script": dom,
        "is_already_target": is_already_target,
        "target_script_ratio": round(tgt_count / total, 3),
    }


def protect_and_translate(
    text: str,
    source_lang: str,
    target_lang: str,
    translate_fn: Callable[[str], str],
) -> str:
    """
    Protect foreign-script words with placeholders, run Helsinki-NLP on the rest,
    then restore the original words.

    Example:
      Input:  "I bought चावल and दाल at the market"
      After protect: "I bought [KEEP_0] and [KEEP_1] at the market"
      After translate: "मैंने [KEEP_0] और [KEEP_1] बाज़ार से खरीदे"
      After restore: "मैंने चावल और दाल बाज़ार से खरीदे"
    """
    info = detect_code_switching(text, source_lang, target_lang)

    if info["is_already_target"]:
        # Segment is already in the target language — pass through unchanged
        logger.info(f"[code_switch] Pass-through (already target lang): {text[:60]!r}")
        return text

    if not info["is_mixed"] or not info["foreign_spans"]:
        # No code-switching — translate normally
        return translate_fn(text)

    # Replace each foreign-script span with [KEEP_N]
    protected = text
    placeholders: list[str] = []
    # Process spans in reverse order so offsets stay valid
    for start, end, _script in reversed(info["foreign_spans"]):
        token = text[start:end]
        idx = len(placeholders)
        placeholders.insert(0, token)
        ph = f"[KEEP_{len(placeholders) - 1}]"
        protected = protected[:start] + ph + protected[end:]

    logger.info(
        f"[code_switch] Protected {len(placeholders)} span(s): {protected[:80]!r}"
    )

    # Translate the protected text (Helsinki-NLP sees clean source language)
    try:
        translated = translate_fn(protected)
    except Exception as exc:
        logger.warning(f"[code_switch] Translation of protected text failed: {exc}")
        translated = protected

    # Restore placeholders
    for idx, original in enumerate(placeholders):
        translated = translated.replace(f"[KEEP_{idx}]", original)

    logger.info(f"[code_switch] Restored: {translated[:80]!r}")
    return translated


def llm_translate_mixed(
    text: str,
    source_lang: str,
    target_lang: str,
    source_lang_name: str = "",
    target_lang_name: str = "",
) -> str | None:
    """
    Ask LLaMA3 (via Ollama) to translate a code-switched segment.
    LLaMA3 knows multiple languages and handles mixing naturally.
    Returns None if Ollama is not available.
    """
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
    model = os.environ.get("OLLAMA_MODEL", "llama3")
    src_name = source_lang_name or source_lang
    tgt_name = target_lang_name or target_lang

    prompt = (
        f"Translate the following text to {tgt_name}.\n"
        f"The text is mostly {src_name} but may contain words from other languages.\n"
        f"- Keep proper nouns and names unchanged.\n"
        f"- If a word is already in {tgt_name}, keep it as-is.\n"
        f"- Translate everything else to natural {tgt_name}.\n\n"
        f"Text: {text}\n\n"
        f"Return ONLY the {tgt_name} translation, nothing else."
    )

    try:
        import httpx
        resp = httpx.post(
            ollama_url,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        result = resp.json().get("response", "").strip()
        if result:
            logger.info(f"[code_switch] LLM translation: {result[:80]!r}")
            return result
    except Exception as exc:
        logger.debug(f"[code_switch] LLM unavailable for mixed translation: {exc}")

    return None


# Language code → human-readable name (for LLM prompts)
_LANG_NAMES: dict[str, str] = {
    "en": "English", "hi": "Hindi", "ar": "Arabic", "bn": "Bengali",
    "ta": "Tamil",   "te": "Telugu", "kn": "Kannada", "ml": "Malayalam",
    "gu": "Gujarati","pa": "Punjabi","ru": "Russian", "zh": "Chinese",
    "ja": "Japanese","ko": "Korean", "ur": "Urdu",    "mr": "Marathi",
    "es": "Spanish", "fr": "French", "de": "German",  "pt": "Portuguese",
    "it": "Italian", "nl": "Dutch",  "pl": "Polish",  "tr": "Turkish",
    "sv": "Swedish",
}


def translate_with_code_switch_handling(
    text: str,
    source_lang: str,
    target_lang: str,
    translate_fn: Callable[[str], str],
) -> str:
    """
    Main entry point. Handles a segment that may contain code-switching.

    Priority:
      1. If already in target language → pass through
      2. If Ollama available → use LLaMA3 for natural mixed-language translation
      3. If mixed script → protect foreign words + translate source parts
      4. Otherwise → translate normally (no code-switching detected)
    """
    if not text or not text.strip():
        return text

    info = detect_code_switching(text, source_lang, target_lang)

    if not info["is_mixed"]:
        # Clean segment — no code-switching
        return translate_fn(text)

    if info["is_already_target"]:
        # Already in target language — pass through unchanged
        logger.info(f"[code_switch] Already in target language: {text[:60]!r}")
        return text

    logger.info(
        f"[code_switch] Mixed segment detected "
        f"(target_ratio={info['target_script_ratio']}, "
        f"spans={len(info['foreign_spans'])}): {text[:60]!r}"
    )

    # Try LLaMA3 first — most natural result for mixed segments
    src_name = _LANG_NAMES.get(source_lang.split("-")[0].lower(), source_lang)
    tgt_name = _LANG_NAMES.get(target_lang.split("-")[0].lower(), target_lang)

    llm_result = llm_translate_mixed(text, source_lang, target_lang, src_name, tgt_name)
    if llm_result:
        return llm_result

    # LLaMA3 unavailable — use placeholder protection
    return protect_and_translate(text, source_lang, target_lang, translate_fn)
