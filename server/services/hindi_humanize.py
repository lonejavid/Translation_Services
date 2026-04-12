"""
Lightweight Hindi post-edit for dubbing: conversational tone via regex only.
"""
from __future__ import annotations

import re

# Shield chunk / context markers during humanization.
_SEG_MARKER_RE = re.compile(r"<<<SEG_\d+>>>", re.IGNORECASE)
DUB_CONTEXT_MARK = "<<<__DUB_CUR__>>>"
_ALL_SHIELD_RE = re.compile(
    r"<<<SEG_\d+>>>|" + re.escape(DUB_CONTEXT_MARK),
    re.IGNORECASE,
)
# Pause helper: prev_clause + "\x1f" + next_clause (optional)
_PAUSE_SEP = "\x1f"

_SPLIT_MARK = "\x1e"

_SERIOUS_EN = re.compile(
    r"\b(terrible|wrong|horrible|awful|disaster|tragic|hate|crime|evil|danger|stop)\b",
    re.IGNORECASE,
)
_EXCITED_EN = re.compile(
    r"\b(amazing|great|awesome|fantastic|wonderful|love|best|incredible|excellent|perfect|yes)\b",
    re.IGNORECASE,
)


def detect_emotion(text: str) -> str:
    """Lightweight cue from English (or any Latin) source text."""
    if not (text or "").strip():
        return "casual"
    if _SERIOUS_EN.search(text):
        return "serious"
    if _EXCITED_EN.search(text):
        return "excited"
    return "casual"


def split_into_speech_clauses(text: str) -> list[str]:
    """
    Natural spoken chunks: commas, कि / और / लेकिन / तो boundaries, line breaks.
    """
    raw = (text or "").strip()
    if not raw:
        return []
    t = re.sub(r"\s+", " ", raw)
    t = re.sub(r"[\r\n]+", _SPLIT_MARK, t)
    t = re.sub(r",\s*", _SPLIT_MARK, t)
    t = re.sub(r"(?<=\S)\s+(?=कि\s)", _SPLIT_MARK, t)
    t = re.sub(r"(?<=\S)\s+(?=और\s)", _SPLIT_MARK, t)
    t = re.sub(r"(?<=\S)\s+(?=लेकिन\s)", _SPLIT_MARK, t)
    t = re.sub(r"(?<=\S)\s+(?=तो\s)", _SPLIT_MARK, t)
    parts = [p.strip() for p in t.split(_SPLIT_MARK) if p.strip()]
    return parts if parts else [t]


def strip_seg_markers_for_emotion(en_text: str) -> str:
    """English blob with <<<SEG_n>>> → plain text for detect_emotion."""
    t = _SEG_MARKER_RE.sub(" ", en_text or "")
    return re.sub(r"\s+", " ", t).strip()


def _pick_ms(lo: int, hi: int, key: str) -> int:
    if hi <= lo:
        return lo
    h = sum(ord(c) for c in key) % (hi - lo + 1)
    return lo + h


def get_clause_pause_ms(text: str, emotion: str) -> int:
    """
    Dynamic pause after a clause. Pass previous clause in ``text``, or
    ``prev_clause + "\\x1f" + next_clause`` to use following-clause cues (लेकिन, तो).
    """
    em = (emotion or "casual").strip().lower()
    if em not in ("serious", "casual", "excited"):
        em = "casual"

    if _PAUSE_SEP in text:
        prev, nxt = text.split(_PAUSE_SEP, 1)
        prev, nxt = prev.strip(), nxt.strip()
    else:
        prev, nxt = (text or "").strip(), ""

    pause = _pick_ms(190, 210, prev + em)

    if prev.rstrip().endswith((",", "،")):
        pause = _pick_ms(150, 200, prev + "comma")

    nxt_st = nxt.lstrip()
    if nxt_st.startswith("लेकिन") or nxt_st.startswith("तो"):
        pause = max(pause, _pick_ms(250, 300, nxt + em))

    if em == "serious":
        pause = max(pause, _pick_ms(300, 450, prev + "sev"))
    elif em == "excited":
        pause = min(pause, _pick_ms(120, 180, prev + "ex"))

    return max(80, min(500, pause))


def emphasize_keywords(text: str, emotion: str) -> str:
    """
    Previously wrapped words in *...* for “emphasis”; TTS engines often read
    asterisks aloud or pause oddly, so we no longer add them. Callers kept
    for API stability.
    """
    return (text or "").strip()


def format_pause_lookup(prev_clause: str, next_clause: str | None = None) -> str:
    """Build ``text`` argument for :func:`get_clause_pause_ms`."""
    p, n = (prev_clause or "").strip(), (next_clause or "").strip()
    if n:
        return f"{p}{_PAUSE_SEP}{n}"
    return p


def _shield_seg_markers(text: str) -> tuple[str, list[str]]:
    markers: list[str] = []

    def _sub(m: re.Match) -> str:
        markers.append(m.group(0))
        return f"\x00MK{len(markers) - 1}\x00"

    return _ALL_SHIELD_RE.sub(_sub, text), markers


def _unshield_seg_markers(text: str, markers: list[str]) -> str:
    t = text
    for i, mk in enumerate(markers):
        t = t.replace(f"\x00MK{i}\x00", mk)
    return t


# (pattern, replacement) — longer / more specific patterns first.
_HUMANIZE_RULES: list[tuple[str, str]] = [
    (r"आप\s+जानते\s+हैं\s+कि\b", "आप जानते हैं"),
    (r"हम\s+जानते\s+हैं\s+कि\b", "हम जानते हैं"),
    (r"तुम\s+जानते\s+हो\s+कि\b", "तुम जानते हो"),
    (r"मैं\s+सोचता\s+हूँ\s+कि\b", "मैं सोचता हूँ"),
    (r"मैं\s+सोचती\s+हूँ\s+कि\b", "मैं सोचती हूँ"),
    (r"इसका\s+मतलब\s+है\s+कि\b", "मतलब"),
    (r"मैं\s+चाहता\s+हूँ\s+कि\b", "मैं चाहता हूँ"),
    (r"मैं\s+चाहती\s+हूँ\s+कि\b", "मैं चाहती हूँ"),
    (r"यह\s+एक\s+भयानक\s+बात\s+है\b", "ये बहुत गलत है"),
    (r"यह\s+बहुत\s+भयानक\s+है\b", "ये बहुत गलत है"),
    (r"एक\s+\*?भयानक\*?\s+बात\s+है\b", "ये बहुत गलत है"),
    (r"\*भयानक\*", "बहुत गलत"),
    (r"डरावना\s+आदमी\b", "बेहद बुरा आदमी"),
    # Spoken flow: model loves ASCII dashes; commas sound more natural in Hindi dub.
    (r"\s+-\s+", ", "),
    (r"यह\s+एक\s+बुरी\s+बात\s+है\b", "ये बुरा है"),
    (r"कृपया\s+", ""),
    (r"आप\s+कृपया\s+", "आप "),
    (r"\bइस\s+प्रकार\b", "ऐसे"),
    (r"\bउस\s+प्रकार\b", "वैसे"),
    (r"\bअतः\s*", "तो "),
    (r"\bनिश्चित\s+रूप\s+से\b", "ज़रूर"),
    (r"\bनिश्चित\s+तौर\s+पर\b", "ज़रूर"),
    (r"\bवास्तव\s+में\b", "सच में"),
    (r"\bनिश्चित\s+ही\b", "ज़रूर"),
]

# Casual-only softening (after base rules; emotion == "casual").
_CASUAL_EXTRA: list[tuple[str, str]] = [
    (r"मुझे\s+लगता\s+है\s+कि\b", "मुझे लगता है"),
]


def _apply_emotion_tone(t: str, emotion: str) -> str:
    if emotion == "serious":
        t = re.sub(
            r"(?<!बहुत )गलत(?=\s|[।,.!?]|$)",
            "बहुत गलत",
            t,
        )
        return t
    if emotion == "casual":
        for pat, repl in _CASUAL_EXTRA:
            t = re.sub(pat, repl, t)
        return t
    if emotion == "excited":
        t = re.sub(r"बहुत\s+अच्छा\s+है\b", "कमाल का है!", t)
        t = re.sub(r"बहुत\s+अच्छी\s+है\b", "कमाल की है!", t)
        return t
    return t


def humanize_hindi_text(text: str, emotion: str = "casual") -> str:
    """
    Spoken-style Hindi; optional emotion adjusts tone after base rules.
    Preserves <<<SEG_n>>> markers.
    """
    if not text or not text.strip():
        return text or ""

    shielded, markers = _shield_seg_markers(text)
    t = shielded
    for pat, repl in _HUMANIZE_RULES:
        t = re.sub(pat, repl, t)
    t = _unshield_seg_markers(t, markers)
    t = re.sub(r"\s+", " ", t).strip()
    em = (emotion or "casual").strip().lower()
    if em not in ("serious", "casual", "excited"):
        em = "casual"
    t2, m2 = _shield_seg_markers(t)
    t2 = _apply_emotion_tone(t2, em)
    t2 = _unshield_seg_markers(t2, m2)
    return re.sub(r"\s+", " ", t2).strip()


def finalize_hindi_spoken_segment(text: str) -> str:
    """
    Per subtitle segment: natural closing punctuation for dubbing (timestamps unchanged).
    Adds Hindi danda if Devanagari text lacks a strong sentence end.
    """
    t = (text or "").strip()
    if not t:
        return t
    # Strip markdown-style emphasis if anything upstream added *...*
    t = re.sub(r"\*+([^*]+)\*+", r"\1", t)
    t = re.sub(r"\*+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not re.search(r"[\u0900-\u097F]", t):
        return t
    last = t[-1]
    if last in "।?!…":
        return t
    # Trailing Latin ellipsis or comma → close with danda for TTS/subtitle flow
    if last in "…":
        return t[:-1].rstrip() + "।"
    if last in ",,;":
        return t[:-1].rstrip() + "।"
    return t + "।"
