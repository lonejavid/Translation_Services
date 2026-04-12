"""
Post-processing for mlx-whisper transcripts: known ASR confusions → canonical names.

Also builds an optional initial_prompt string from a configurable term list
(see server/config/stt_entity_map.json) to bias Whisper toward correct entities.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

_CONFIG_CACHE: dict[str, Any] | None = None


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "stt_entity_map.json"


def _load_raw_config() -> dict[str, Any]:
    global _CONFIG_CACHE
    override = os.environ.get("STT_ENTITY_MAP_PATH", "").strip()
    path = Path(override) if override else _default_config_path()
    if not path.is_file():
        return {"initial_prompt_terms": [], "asr_corrections": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"initial_prompt_terms": [], "asr_corrections": []}
    if not isinstance(data, dict):
        return {"initial_prompt_terms": [], "asr_corrections": []}
    return data


def get_stt_entity_config() -> dict[str, Any]:
    """Merged config (cached per process). Set STT_ENTITY_MAP_PATH to override file."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = _load_raw_config()
    return _CONFIG_CACHE


def reset_stt_entity_config_cache() -> None:
    """For tests: force reload on next access."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None


def build_initial_prompt(max_chars: int = 220) -> str | None:
    """
    Short context string for Whisper `initial_prompt` (proper nouns / domain terms).
    Whisper uses only a prefix of tokens; keep under ~220 chars.
    Appends user-learned wording hints from ``learned_corrections`` when enabled.
    """
    if os.environ.get("STT_DISABLE_INITIAL_PROMPT", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return None
    terms = get_stt_entity_config().get("initial_prompt_terms") or []
    parts: list[str] = []
    if isinstance(terms, list):
        for t in terms:
            if not isinstance(t, str):
                continue
            s = t.strip()
            if s:
                parts.append(s)

    base: str | None = None
    if parts:
        base = "Key terms: " + ", ".join(parts)
        if len(base) > max_chars:
            truncated: list[str] = []
            prefix = "Key terms: "
            for s in parts:
                trial = prefix + ", ".join(truncated + [s])
                if len(trial) <= max_chars:
                    truncated.append(s)
                else:
                    break
            if not truncated:
                base = prefix + parts[0][: max(0, max_chars - len(prefix))]
            else:
                base = prefix + ", ".join(truncated)

    try:
        from services.learned_corrections import build_initial_prompt_addon

        addon = build_initial_prompt_addon(max_chars=min(100, max_chars // 3))
    except Exception:
        addon = ""

    if base and addon:
        sep = " "
        room = max_chars - len(base) - len(sep)
        if room > 20:
            add = addon if len(addon) <= room else addon[: max(0, room - 1)] + "…"
            return (base + sep + add)[:max_chars]
        return base[:max_chars]
    if base:
        return base[:max_chars]
    if addon:
        return addon[:max_chars]
    return None


def _compile_correction_pairs() -> list[tuple[re.Pattern[str], str]]:
    raw = get_stt_entity_config().get("asr_corrections") or []
    pairs: list[tuple[str, str]] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            hear = (item.get("hear_as") or "").strip()
            corr = (item.get("correct") or "").strip()
            if hear and corr:
                pairs.append((hear, corr))
    # Longest phrases first so multi-word fixes win over single-token ones
    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    compiled: list[tuple[re.Pattern[str], str]] = []
    for hear, corr in pairs:
        try:
            pat = re.compile(re.escape(hear), re.IGNORECASE)
        except re.error:
            continue
        compiled.append((pat, corr))
    return compiled


_PAIRS: list[tuple[re.Pattern[str], str]] | None = None


def _get_pairs() -> list[tuple[re.Pattern[str], str]]:
    global _PAIRS
    if _PAIRS is None:
        _PAIRS = _compile_correction_pairs()
    return _PAIRS


def apply_asr_entity_corrections(text: str) -> tuple[str, bool]:
    """
    Replace known mis-hearings with canonical entity strings.
    Returns (new_text, changed_flag).
    """
    if os.environ.get("STT_DISABLE_ENTITY_CORRECTIONS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return text, False
    out = text
    changed = False
    for pat, corr in _get_pairs():
        new_out, n = pat.subn(corr, out)
        if n:
            changed = True
            out = new_out
    return out, changed
