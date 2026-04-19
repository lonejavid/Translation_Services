"""
Local LLM translation corrector via Ollama.

For each translated segment the pipeline runs two steps:

  Step 1 — Language competency check (once per source/target language pair per session).
    Ask LLaMA3: "Do you understand [source_lang] and [target_lang] well enough to
    correct translation errors?"
    • YES → proceed to correction
    • NO  → skip LLM correction; return original machine translation unchanged

  Step 2 — Direct correction.
    Pass the original source text and the machine-translated text to LLaMA3.
    Ask it to fix wrong words, grammar errors, and missed meaning using the source
    as the ground truth.  Return ONLY the corrected target-language text.

Advantages over the old YES/NO retry loop:
  • One call instead of 3+ per segment (faster, fewer timeouts)
  • LLaMA3 sees the source meaning while correcting — much more accurate
  • Competency gate prevents low-quality corrections for rare language pairs

Every event is written to llm_validation_log.txt so the user can inspect changes.
Fully offline — no external APIs.  Gracefully skips if Ollama is not running.
"""
from __future__ import annotations

import os
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
_REQUEST_TIMEOUT = 90  # seconds — correction calls generate more tokens than YES/NO

# Set LLM_VALIDATOR_ENABLED=false to disable without code changes
_ENABLED = os.environ.get("LLM_VALIDATOR_ENABLED", "true").strip().lower() not in (
    "0", "false", "no", "off"
)

# Reuse a single httpx client (no per-request connection overhead)
_client: "httpx.Client | None" = None

# Per-segment correction cache: (source_text, translated_text) → corrected
_cache: dict[tuple[str, str], str] = {}

# Language competency cache: (source_lang, target_lang) → True/False/None
# None means competency check failed (Ollama unavailable); treated as skip.
_competency_cache: dict[tuple[str, str], bool | None] = {}
_competency_lock = threading.Lock()

# Human-readable language names for prompts
_LANG_NAMES: dict[str, str] = {
    "en": "English", "hi": "Hindi",  "ar": "Arabic",  "bn": "Bengali",
    "ta": "Tamil",   "te": "Telugu", "kn": "Kannada", "ml": "Malayalam",
    "gu": "Gujarati","pa": "Punjabi","ru": "Russian",  "zh": "Chinese",
    "ja": "Japanese","ko": "Korean", "ur": "Urdu",     "mr": "Marathi",
    "es": "Spanish", "fr": "French", "de": "German",   "pt": "Portuguese",
    "it": "Italian", "nl": "Dutch",  "pl": "Polish",   "tr": "Turkish",
    "sv": "Swedish", "ne": "Nepali",
}

# ── Validation log ───────────────────────────────────────────────────────────

_log_lock = threading.Lock()


def _get_log_path() -> Path:
    cache_dir = Path(
        os.environ.get("CACHE_DIR") or Path(__file__).resolve().parent.parent / "data"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "llm_validation_log.txt"


def _write_log(lines: list[str]) -> None:
    try:
        with _log_lock:
            with open(_get_log_path(), "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
    except Exception as exc:
        logger.debug(f"[llm_validator] Could not write log: {exc}")


def _log_skipped(reason: str, source_text: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _write_log([f"[{ts}] SKIPPED — {reason}", f"  Source: {source_text[:120]}", ""])


def _log_correction(
    source_lang: str,
    target_lang: str,
    source_text: str,
    original_translation: str,
    corrected: str,
    cache_hit: bool,
    competency_skipped: bool,
) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    src_name = _LANG_NAMES.get(source_lang, source_lang)
    tgt_name = _LANG_NAMES.get(target_lang, target_lang)
    lines = [
        f"[{ts}] {'CACHE HIT' if cache_hit else ('COMPETENCY_SKIP' if competency_skipped else 'CORRECTED')}  "
        f"({src_name} → {tgt_name})",
        f"  Source ({src_name})      : {source_text}",
        f"  Machine translation     : {original_translation}",
    ]
    if cache_hit:
        lines.append(f"  Returned from cache     : {corrected}")
    elif competency_skipped:
        lines.append(f"  LLM skipped (competency gate) — original kept")
    else:
        changed = corrected.strip() != original_translation.strip()
        lines.append(f"  LLM corrected           : {corrected}")
        lines.append(
            f"  Changed                 : {'YES — LLM improved the translation' if changed else 'NO — original was already correct'}"
        )
    lines.append("")
    _write_log(lines)


# ── HTTP client ───────────────────────────────────────────────────────────────

def _get_client() -> "httpx.Client":
    global _client
    if _client is None:
        import httpx
        _client = httpx.Client(timeout=_REQUEST_TIMEOUT)
    return _client


def _ollama_generate(prompt: str, *, num_predict: int | None = None) -> str | None:
    """Send prompt to Ollama. Returns response text or None on any error."""
    try:
        payload: dict = {"model": _MODEL, "prompt": prompt, "stream": False}
        if num_predict is not None:
            payload["options"] = {"num_predict": num_predict}
        resp = _get_client().post(_OLLAMA_URL, json=payload)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as exc:
        import httpx as _httpx
        if isinstance(exc, (_httpx.TimeoutException, _httpx.ConnectTimeout, _httpx.ReadTimeout)):
            logger.warning("[llm_validator] LLM timeout")
        else:
            logger.debug(f"[llm_validator] Ollama unavailable: {exc}")
        return None


# ── Step 1: Language competency gate ─────────────────────────────────────────

def _check_language_competency(source_lang: str, target_lang: str) -> bool | None:
    """
    Ask LLaMA3 if it understands both source and target languages well enough
    to correct translation errors between them.

    Returns:
        True  → LLM is confident in both languages → proceed with correction
        False → LLM not confident → skip correction, use original translation
        None  → Ollama unavailable → skip correction
    """
    src_name = _LANG_NAMES.get(source_lang, source_lang)
    tgt_name = _LANG_NAMES.get(target_lang, target_lang)

    prompt = (
        f"Do you understand {src_name} and {tgt_name} well enough to identify and "
        f"correct errors in a translation from {src_name} to {tgt_name}?\n\n"
        f"Reply ONLY with YES or NO."
    )
    response = _ollama_generate(prompt, num_predict=5)
    if response is None:
        return None  # Ollama unavailable
    return "YES" in response.upper()


def _get_competency(source_lang: str, target_lang: str) -> bool | None:
    """Return cached competency result (checked once per language pair per session)."""
    key = (source_lang, target_lang)
    with _competency_lock:
        if key in _competency_cache:
            return _competency_cache[key]
    # Outside lock: do the actual LLM call (may take seconds on cold model)
    result = _check_language_competency(source_lang, target_lang)
    with _competency_lock:
        _competency_cache[key] = result
    src_name = _LANG_NAMES.get(source_lang, source_lang)
    tgt_name = _LANG_NAMES.get(target_lang, target_lang)
    if result is True:
        logger.info(
            f"[llm_validator] Competency check PASSED: LLM knows {src_name} + {tgt_name} — corrections enabled"
        )
    elif result is False:
        logger.info(
            f"[llm_validator] Competency check FAILED: LLM does not know {src_name} + {tgt_name} — falling back to machine translation"
        )
    else:
        logger.info("[llm_validator] Competency check skipped (Ollama unavailable)")
    return result


# ── Step 2: Direct correction ─────────────────────────────────────────────────

def _correct_translation(
    source_text: str,
    translated_text: str,
    source_lang: str,
    target_lang: str,
) -> str | None:
    """
    Ask LLaMA3 to correct the machine translation using the source as ground truth.
    Returns the corrected text, or None if the call fails.
    """
    src_name = _LANG_NAMES.get(source_lang, source_lang)
    tgt_name = _LANG_NAMES.get(target_lang, target_lang)

    prompt = (
        f"You are a professional translator fluent in both {src_name} and {tgt_name}.\n\n"
        f"A speech recognition system captured speech in {src_name} and a machine translation "
        f"model translated it to {tgt_name}. Both may have errors.\n\n"
        f"Your task:\n"
        f"  • Use the {src_name} source text as the ground truth for MEANING.\n"
        f"  • Correct any words in the {tgt_name} translation that do not match the source meaning.\n"
        f"  • Fix grammar and naturalness in {tgt_name}.\n"
        f"  • Do NOT add or remove information.\n"
        f"  • Keep proper nouns, names, and numbers unchanged.\n"
        f"  • If the translation is already correct, return it as-is.\n\n"
        f"{src_name} source:\n{source_text}\n\n"
        f"{tgt_name} machine translation:\n{translated_text}\n\n"
        f"Return ONLY the corrected {tgt_name} text. No explanation. No preamble."
    )
    return _ollama_generate(prompt)


# ── Main entry point ──────────────────────────────────────────────────────────

def validate_and_improve(
    source_text: str,
    translated_text: str,
    retranslate_fn: Callable[[str], str] | None = None,
    source_lang: str = "en",
    target_lang: str = "hi",
) -> str:
    """
    Correct `translated_text` using the local LLM, with `source_text` as ground truth.

    Flow:
      1. Skip if disabled / Ollama down / segment too short
      2. Check competency (once per language pair, cached for session)
         → if LLM doesn't know both languages, return original unchanged
      3. Ask LLM to correct the translation
         → if correction is empty or identical to source, return original
      4. Log everything to llm_validation_log.txt
      5. Cache result for identical (source, translation) pairs

    `retranslate_fn` is accepted for API compatibility but not used in this approach
    (direct correction is more reliable than retranslation + YES/NO voting).
    """
    if not _ENABLED:
        return translated_text
    if not source_text or not translated_text:
        return translated_text

    # Skip very short segments — not enough context for meaningful correction
    if len(source_text.split()) <= 2:
        _log_skipped("segment too short (≤2 words)", source_text)
        return translated_text

    src = source_lang.strip().lower()[:8]
    tgt = target_lang.strip().lower()[:8]

    # Cache check — identical (source, translation) pair seen before
    cache_key = (source_text.strip(), translated_text.strip())
    if cache_key in _cache:
        cached = _cache[cache_key]
        logger.debug(f"[llm_validator] Cache hit for src={source_text[:40]!r}")
        _log_correction(src, tgt, source_text, translated_text, cached,
                        cache_hit=True, competency_skipped=False)
        return cached

    # Step 1: Language competency gate
    competent = _get_competency(src, tgt)

    if competent is None:
        # Ollama unavailable — skip silently, no log spam
        return translated_text

    if not competent:
        # LLM does not know these languages — do not attempt correction
        _log_correction(src, tgt, source_text, translated_text, translated_text,
                        cache_hit=False, competency_skipped=True)
        return translated_text

    # Step 2: Direct correction
    logger.info(
        f"[llm_validator] Correcting | src={source_text[:50]!r} | trans={translated_text[:50]!r}"
    )
    corrected = _correct_translation(source_text, translated_text, src, tgt)

    if not corrected or not corrected.strip():
        logger.warning("[llm_validator] LLM returned empty correction — keeping original")
        corrected = translated_text

    corrected = corrected.strip()

    # Safety guard: if correction looks like the source (not target) language, discard it
    if corrected == source_text.strip():
        logger.warning("[llm_validator] LLM returned source text unchanged — keeping original translation")
        corrected = translated_text

    _cache[cache_key] = corrected
    _log_correction(src, tgt, source_text, translated_text, corrected,
                    cache_hit=False, competency_skipped=False)

    changed = corrected != translated_text.strip()
    logger.info(
        f"[llm_validator] Done | changed={'YES' if changed else 'NO'} | {corrected[:60]!r}"
    )
    return corrected
