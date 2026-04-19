"""
English → Indian languages (local, CPU).

**Default: Meta NLLB-200-distilled** (`facebook/nllb-200-distilled-600M`) — large multilingual
parallel training (FLORES-style ``eng_Latn`` → ``hin_Deva``, etc.), often **cleaner standard Hindi**
than specialized en→Indic models on some content.

**Optional:** ``TRANSLATION_ENGINE=indictrans2`` → AI4Bharat IndicTrans2-1B (legacy).

IndicTrans2 tokenizer expects: ``eng_Latn {tgt_indic} {body}``. NLLB uses ``src_lang=eng_Latn``
and ``forced_bos_token_id`` for the target FLORES code.

Chunked translation: non-linguistic markers ``<<<SEG_0>>>``, ``<<<SEG_1>>>``, … with spaces;
after decode, segments are recovered by regex (ordered marker positions). Any failure →
per-segment fallback (no length-based splitting).
"""
from __future__ import annotations

import difflib
import os
import re
from typing import Callable, Optional

from services.llm_validator import validate_and_improve as _llm_validate

from dotenv import load_dotenv

load_dotenv()

from huggingface_hub import login

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from services.hindi_humanize import (
    DUB_CONTEXT_MARK,
    detect_emotion,
    emphasize_keywords,
    finalize_hindi_spoken_segment,
    humanize_hindi_text,
    strip_seg_markers_for_emotion,
)

INDICTRANS2_MODEL_NAME = "ai4bharat/indictrans2-en-indic-1B"
NLLB_MODEL_DEFAULT = "facebook/nllb-200-distilled-600M"


def _translation_engine_from_env() -> str:
    v = (os.environ.get("TRANSLATION_ENGINE") or "nllb").strip().lower()
    if v in ("nllb", "indictrans2"):
        return v
    print(f"[translate] Unknown TRANSLATION_ENGINE={v!r}; using nllb")
    return "nllb"

# Regex to split model output on numbered segment markers (Unicode-safe: no code-unit slicing)
SEG_MARKER_SPLIT_RE = re.compile(r"\s*<<<\s*SEG_\d+\s*>>>\s*", re.IGNORECASE)

# Compact markers for the model (no <<< >>>); restored after decode so chunk split still works.
_SEG_TO_MODEL_RE = re.compile(r"<<<SEG_(\d+)>>>", re.IGNORECASE)
_SEG_FROM_MODEL_RE = re.compile(r"SEG_(\d+)\s*>>>", re.IGNORECASE)


def _seg_markers_for_model(text: str) -> str:
    return _SEG_TO_MODEL_RE.sub(r"SEG_\1>>>", text)


def _seg_markers_from_model(text: str) -> str:
    return _SEG_FROM_MODEL_RE.sub(r"<<<SEG_\1>>>", text)


# Former natural-language prompt fragments (must never appear in model I/O).
_INSTRUCTION_LEAK_PHRASES: tuple[str, ...] = (
    "Translate the following text into natural, fluent Hindi.",
    "Do not use dialects like Bhojpuri or Maithili.",
    "Maintain conversational tone:",
    "Translate the following text into natural, fluent Hindi",
    "Do not use dialects like Bhojpuri or Maithili",
    "Translate the following text",
    "natural, fluent Hindi",
    "Bhojpuri or Maithili",
    "Maintain conversational tone",
)

DEFAULT_CHUNK_SIZE = 10
# Hindi: long batches often merge into one paragraph; word-split recovery then cuts mid-phrase.
# Cap batch length for ``hi`` (env ``TRANSLATE_CHUNK_SIZE`` still applies up to this cap).
HI_MAX_CHUNK_SEGMENTS = max(3, int(os.environ.get("TRANSLATE_CHUNK_SIZE_HI", "5")))
# Proportional Hindi recovery only when segment count is small enough to align safely.
_HI_WEIGHT_SPLIT_MAX_SEGS = 5

# Exact English → standard Hindi (conversational); skips model for these spans.
_SHORT_EN_HI_EXACT: dict[str, str] = {
    "no.": "नहीं।",
    "no": "नहीं।",
    "is that common?": "क्या यह आम बात है?",
    "is that common": "क्या यह आम बात है?",
    "yes.": "हाँ।",
    "yes": "हाँ।",
    "ok.": "ठीक है।",
    "okay.": "ठीक है।",
    "okay": "ठीक है।",
    "ok": "ठीक है।",
}

# If any substring remains after clean_translation, treat as dialect bleed → retry / fallback.
_DIALECT_RETRY_SUBSTRINGS: tuple[str, ...] = (
    "बरखक",
    "ओतऽ",
    "छलहुँ",
    "छलाह",
    "लोगोकेँ",
    "देखलहुँ",
    "नेतृत्वक",
    "डग्गनक",
    "जा रहल",
    "उपयोग कयल",
    "एकटा",
    "पसिन्न",
    "छेकिन",
    "कतेको",
    "गेल आहि",
    "अछि।",
    "नहि।",
    " भऽ ",
    "अहाँ",
    "किछु",
    "छलहँ",
    "दृष्टि सँ",
    "रहल ",
    " गेल",
    "कयल ",
    " नहि",
    " अछि",
    # Maithili / mixed Eastern forms often seen on per-segment IndicTrans2 decode
    "जनैत छ",
    "जनैत ",
    "कखन ",
    " पहिल ",
    " बेर ",
    " करैत ",
    "हमर ",
    "हमरा",
    "जकाँ",
    "कोनो ",
    "एकरूपता",
    "नौकरीमे",
    "आपकेँ",
    " आपक ",
    "ओकर ",
    "सोचबाक",
    " चाही ",
    "छेत्रीस",
    "आबि रहल",
    "रूपसे ",
    " अगिला ",
)

# Carried from mlx-whisper segments through translation for API / QA (optional).
_STT_PASSTHROUGH_KEYS = frozenset({
    "words",
    "avg_logprob",
    "no_speech_prob",
    "compression_ratio",
    "temperature",
})


def _passthrough_stt_fields(seg: dict) -> dict:
    return {k: seg[k] for k in _STT_PASSTHROUGH_KEYS if k in seg}


def _format_chunk_for_translation(texts: list[str]) -> str:
    """``<<<SEG_0>>> t0 <<<SEG_1>>> t1 ...`` — single string for one model call."""
    return " ".join(f"<<<SEG_{i}>>> {t}" for i, t in enumerate(texts))


def _split_at_seg_markers(translated: str, expected: int) -> Optional[list[str]]:
    """
    Split on ``<<<SEG_n>>>`` boundaries. Returns ``expected`` content parts or None if mismatch.
    """
    parts = [
        p.strip()
        for p in SEG_MARKER_SPLIT_RE.split((translated or "").strip())
        if p.strip()
    ]
    if len(parts) != expected:
        return None
    return parts


def _extract_chunk_segments(text: str, expected: int) -> Optional[list[str]]:
    """Split chunk decode into segment strings; None if marker count does not match."""
    try:
        if expected < 1 or not isinstance(text, str):
            return None
        return _split_at_seg_markers(text, expected)
    except Exception:
        return None


def _word_ends_sentence_pause(w: str) -> bool:
    x = w.rstrip(')"\'’»,.')
    return bool(x) and x[-1] in ("।", "?", "!", "؟", "॥", "…")


def _snap_hindi_cut_to_pause(words: list[str], cut: int, lo: int, hi_max: int) -> int:
    """Move ``cut`` backward to just after a danda / ? / ! if one exists in (lo, cut)."""
    cut = max(lo + 1, min(cut, hi_max))
    for j in range(min(cut, len(words)) - 1, lo - 1, -1):
        if j < 0:
            break
        if _word_ends_sentence_pause(words[j]):
            return max(lo + 1, j + 1)
    return cut


def _split_hindi_by_en_weights(hi_full: str, en_segs: list[str]) -> Optional[list[str]]:
    """
    When the model drops <<<SEG_n>>> markers but returns one fluent Hindi block,
    split using English length weights **snapped to Hindi sentence pauses** (। ? !).
    Falls back to None if the paragraph is too run-on (unsafe to split) or too many segments.
    """
    hi_full = (hi_full or "").strip()
    n = len(en_segs)
    if n < 2 or not hi_full:
        return None
    if n > _HI_WEIGHT_SPLIT_MAX_SEGS:
        return None
    if re.search(r"<<<\s*SEG_", hi_full, re.I):
        return None
    if _has_dialect_retry_substrings(hi_full):
        return None
    t = re.sub(r"\s+", " ", hi_full)
    words = [w for w in re.split(r"\s+", t) if w]
    if len(words) < n:
        return None
    pause_words = sum(1 for w in words if _word_ends_sentence_pause(w))
    # Need enough natural breaks; else word-split will slice clauses (seen in production logs).
    if pause_words < max(2, (n + 2) // 2):
        return None
    weights = [max(12, len(re.sub(r"\s+", " ", (s or "").strip()))) for s in en_segs]
    tw = sum(weights)
    if tw <= 0:
        return None
    cuts: list[int] = [0]
    acc = 0
    for i in range(n - 1):
        acc += weights[i]
        ideal = round(acc / tw * len(words))
        prev = cuts[-1]
        ideal = max(ideal, prev + 1)
        max_cut = len(words) - (n - i - 1)
        ideal = min(ideal, max_cut)
        cuts.append(ideal)
    cuts.append(len(words))
    # Snap internal boundaries backward to danda / ? / ! (monotonic).
    for i in range(1, n):
        lo = cuts[i - 1]
        hi_lim = (cuts[i + 1] - 1) if i + 1 < len(cuts) else len(words) - 1
        snapped = _snap_hindi_cut_to_pause(words, cuts[i], lo, hi_lim)
        cuts[i] = max(snapped, lo + 1)
        if i + 1 < len(cuts):
            cuts[i] = min(cuts[i], cuts[i + 1] - 1)
    for i in range(1, n):
        cuts[i] = max(cuts[i], cuts[i - 1] + 1)
        if i + 1 < len(cuts):
            cuts[i] = min(cuts[i], cuts[i + 1] - 1)
    out: list[str] = []
    for i in range(n):
        chunk = " ".join(words[cuts[i] : cuts[i + 1]]).strip()
        if not chunk:
            return None
        if len(en_segs[i]) > 25 and _devanagari_letter_count(chunk) < 2:
            return None
        out.append(chunk)
    return out


def _canonicalize_seg_markers(t: str) -> str:
    """Normalize flexible ``<<< SEG_n >>>`` / ``<<<seg_0>>>`` to ``<<<SEG_n>>>``."""

    def repl(m) -> str:
        return f"<<<SEG_{int(m.group(1))}>>>"

    return re.sub(
        r"<<<\s*SEG\s*_?\s*(\d+)\s*>>>",
        repl,
        t,
        flags=re.IGNORECASE,
    )


def _short_hi_gloss(en_clean: str) -> str | None:
    s = (en_clean or "").strip()
    if not s:
        return None
    low = s.lower()
    return _SHORT_EN_HI_EXACT.get(low) or _SHORT_EN_HI_EXACT.get(s)


# ASR one-word garbage / fillers — no meaningful speech (skip model + TTS-friendly silence).
_EN_FILLER_ONLY: frozenset[str] = frozenset({
    "be",
    "uh",
    "um",
    "umm",
    "uhh",
    "erm",
    "er",
    "hmm",
    "mmm",
    "mm",
    "mhm",
    "hm",
    "ah",
    "oh",
    "ooh",
    "huh",
    "ha",
    "heh",
    "pfft",
    "psst",
})


def _is_english_filler_only(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t or len(t) > 16:
        return False
    return t in _EN_FILLER_ONLY


def _has_dialect_retry_substrings(hi_text: str) -> bool:
    t = hi_text or ""
    return any(sub in t for sub in _DIALECT_RETRY_SUBSTRINGS)


def _devanagari_letter_count(s: str) -> int:
    return len(re.findall(r"[\u0900-\u097F]", s or ""))


def _has_substantial_hindi_script(s: str, *, min_letters: int = 2) -> bool:
    return _devanagari_letter_count(s) >= min_letters


def _huggingface_token() -> str | None:
    t = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    return t if t else None


LANGUAGE_CODE_MAP: dict[str, str] = {
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "bn": "ben_Beng",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "pa": "pan_Guru",
    "ur": "urd_Arab",
}

PROTECTED_PROPER_NOUNS: list[str] = [
    "Trump",
    "Biden",
    "Obama",
    "Netanyahu",
    "Modi",
    "Putin",
    "Macron",
    "Zelensky",
    "Scott Bessent",
    "Elon Musk",
    "Eleni Giokos",
    "Jeremy Diamond",
    "Anderson Cooper",
    "CNN",
    "BBC",
    "Reuters",
    "Bloomberg",
    "Forbes",
    "IRGC",
    "Pentagon",
    "NATO",
    "Haifa",
    "Gaza",
    "Tel Aviv",
    "Jerusalem",
    "Rafah",
    "Kuwait",
    "Bahrain",
    "UAE",
    "Qatar",
    "Saudi Arabia",
    "Israel",
    "Iran",
    "Iraq",
    "Syria",
    "Ukraine",
    "Russia",
    "Hormuz",
    "Ras Laffan",
    "Riyadh",
    "Tehran",
    "Dubai",
    "Washington",
    "New York",
    "London",
    "Beijing",
    "Moscow",
    "YouTube",
    "Google",
    "Apple",
    "Microsoft",
    "OpenAI",
    "Doug Clinton",
    "Clinton",
    "Jeffrey Epstein",
    "Epstein",
    "Kentucky",
    "Massey",
    "Democrats",
    "Democrat",
]


def clean_translation(text: str, dialect_map: dict[str, str]) -> str:
    """
    Post-process model output: artifacts, tags, dialect → Hindi, grammar heuristics, spacing.
    Intended to run on Devanagari text; safe for restored proper nouns (Latin).
    """
    if not text or not text.strip():
        return ""

    t = text.strip()

    # --- Markdown / model “emphasis” markers (TTS reads * aloud) ---
    t = re.sub(r"\*+([^*]{1,48})\*+", r"\1", t)
    t = re.sub(r"\*+", "", t)

    # --- Chunk artifacts: transliterated ``SEG``, stray quotes, dashed glue ---
    t = re.sub(r"(?i)\bसेग\b\s*[।\-–,]*\s*", " ", t)
    t = re.sub(r"\s*[-–]\s*[\"']+\s*", " ", t)
    t = re.sub(r"^[\"'\s\u200b\u200c\u200d]+|[\"'\s\u200b\u200c\u200d]+$", "", t)
    t = re.sub(r"\s*[\"']{2,}\s*", " ", t)
    # Spurious Latin letter spills from marker noise (e.g. ``एस. इ. जि.``)
    t = re.sub(r"\s*एस\.\s*इ\.\s*जि\.\s*", " ", t)
    t = re.sub(r"\s*एस\.\s*।\s*", " ", t)
    t = re.sub(r"\s*ई\.\s*जी\.\s*\.\s*", " ", t)

    # --- Accidental English instruction leakage (legacy prompt / model echo) ---
    for phrase in sorted(_INSTRUCTION_LEAK_PHRASES, key=len, reverse=True):
        t = re.sub(re.escape(phrase), "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()

    # --- Leading garbage (छ*, छेकिन, …) — short tokens only, avoid eating real Hindi ---
    _garbage_exact = frozenset(
        ["छ", "छे", "छें", "छेकिन", "छेकिन,", "छे,", "छें,"]
    )
    for _ in range(8):
        m = re.match(r"^(\S+)\s*", t)
        if not m:
            break
        w = m.group(1).rstrip(",।॥")
        if w in _garbage_exact or (
            w.startswith("छ")
            and len(w) <= 6
            and not re.search(r"[\u0900-\u097F]{6,}", w)
        ):
            t = t[len(m.group(0)) :].strip()
            continue
        break

    # --- Language tag leaks ---
    for tag in [
        "eng_Latn",
        "hin_Deva",
        "hin_Dev",
        "eng_Lat",
        "tam_Taml",
        "tel_Telu",
        "ben_Beng",
        "mar_Deva",
        "guj_Gujr",
        "kan_Knda",
        "mal_Mlym",
        "pan_Guru",
        "urd_Arab",
    ]:
        t = t.replace(tag, "")

    # --- Malformed placeholder glue (drop glued Latin after PROPNnEND) ---
    t = re.sub(r"PROPN\d+END[A-Za-z]+", "", t)

    # --- Dialect / Maithili–Bhojpuri → standard Hindi (long keys first) ---
    for src, dst in sorted(dialect_map.items(), key=lambda kv: len(kv[0]), reverse=True):
        if src in t:
            t = t.replace(src, dst)

    # --- Grammar heuristics (Hindi) ---
    # "में नहीं" meaning "I don't" at utterance / clause start → "मैं नहीं"
    t = re.sub(r"(^|[.!?।]\s*)में नहीं", r"\1मैं नहीं", t)
    t = re.sub(r"(\s)में नहीं(\s)", r"\1मैं नहीं\2", t)
    # isolated मे → में (word-ish)
    t = re.sub(r"(^|\s)मे(\s|$)", r"\1में\2", t)

    # --- Chunk markers: canonicalize flexible ``<<< SEG_n >>>`` forms ---
    if "<<" in t and "SEG" in t:
        t = _canonicalize_seg_markers(t)
        t = re.sub(
            r"\s*(<<<\s*SEG_\d+\s*>>>)\s*",
            r" \1 ",
            t,
            flags=re.IGNORECASE,
        )

    # --- Repeated token collapse (simple) ---
    prev = None
    for _ in range(4):
        t2 = re.sub(r"(\S+)(\s+\1){2,}", r"\1", t)
        if t2 == t:
            break
        t = t2

    # --- Punctuation / spacing ---
    t = re.sub(r"\s*,\s*", ", ", t)
    t = re.sub(r"\s*\.\s*", ". ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


class Translator:
    def __init__(self) -> None:
        self._chunk_size = max(
            1,
            int(os.environ.get("TRANSLATE_CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE))),
        )

        self._engine = _translation_engine_from_env()
        self.device = torch.device("cpu")
        hf_token = _huggingface_token()
        kw: dict = {}
        if hf_token:
            kw["token"] = hf_token

        self._lang_token_ids: dict[str, int] = {}
        self.src_lang_id: int | None = None

        if self._engine == "nllb":
            nllb_name = (
                os.environ.get("NLLB_MODEL_NAME") or NLLB_MODEL_DEFAULT
            ).strip() or NLLB_MODEL_DEFAULT
            print(f"[translate] Loading NLLB: {nllb_name} (Meta NLLB-200, multilingual MT)…")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(nllb_name, **kw)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    nllb_name,
                    **kw,
                    torch_dtype=torch.float32,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load NLLB model {nllb_name}. "
                    f"Try: pip install -U transformers sentencepiece. Error: {e}"
                ) from e
            self.model = self.model.to(self.device).eval()
            lcd = getattr(self.tokenizer, "lang_code_to_id", None) or {}
            for lang_code, flores in LANGUAGE_CODE_MAP.items():
                if flores in lcd:
                    tid = int(lcd[flores])
                else:
                    tid = self.tokenizer.convert_tokens_to_ids(flores)
                    if tid == self.tokenizer.unk_token_id:
                        raise RuntimeError(
                            f"NLLB tokenizer has no language id for {flores} ({lang_code})"
                        )
                self._lang_token_ids[lang_code] = tid
                print(f"[translate] {lang_code} ({flores}) → forced_bos id {tid}")
            print("[translate] NLLB → CPU (eng_Latn → target FLORES code). Ready.")
        else:
            model_name = INDICTRANS2_MODEL_NAME
            print(f"[translate] Loading IndicTrans2: {model_name}…")
            kw_it = {**kw, "trust_remote_code": True}
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kw_it)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    **kw_it,
                    torch_dtype=torch.float32,
                )
            except Exception as e:
                msg = str(e)
                low = msg.lower()
                if (
                    "401" in msg
                    or "403" in msg
                    or "gated" in low
                    or "repository not found" in low
                    or "invalid username or password" in low
                ):
                    raise RuntimeError(
                        f"Cannot download gated model {model_name}.\n"
                        "1) Open https://huggingface.co/ai4bharat/indictrans2-en-indic-1B "
                        "and accept the terms.\n"
                        "2) Create a read token: https://huggingface.co/settings/tokens\n"
                        "3) Export it before starting the server:\n"
                        "   export HF_TOKEN=hf_...\n"
                        "   Or run: huggingface-cli login\n"
                        f"Original error: {e}"
                    ) from e
                raise
            self.model = self.model.to(self.device).eval()
            for lang_code, indic_code in LANGUAGE_CODE_MAP.items():
                token_id = self.tokenizer.convert_tokens_to_ids(indic_code)
                if token_id == self.tokenizer.unk_token_id:
                    token_id = self.tokenizer.convert_tokens_to_ids(f"__{indic_code}__")
                self._lang_token_ids[lang_code] = token_id
                print(f"[translate] {lang_code} ({indic_code}) → id {token_id}")
            self.src_lang_id = self.tokenizer.convert_tokens_to_ids("eng_Latn")
            if self.src_lang_id == self.tokenizer.unk_token_id:
                self.src_lang_id = self.tokenizer.convert_tokens_to_ids("__eng_Latn__")
            print(f"[translate] eng_Latn token id: {self.src_lang_id}")
            print("[translate] IndicTrans2 ready.")

        self._recent_context: list[str] = []

        self.DIALECT_TO_HINDI: dict[str, str] = {
            # Long phrases first
            "बरखक": "बहुत",
            "ओतऽ": "वह",
            "छलहुँ": "था",
            "छलहँ": "था",
            "छलाह": "थे",
            "उपयोग कयल जा रहल": "उपयोग किया जा रहा",
            "भऽ गेल": "हो गया",
            "भऽ गई": "हो गई",
            "भऽ गए": "हो गए",
            "उपयोग कयल": "उपयोग किया",
            "लोगोकेँ": "लोगों को",
            "देखलहुँ": "देखा",
            "नेतृत्वक": "नेतृत्व के",
            "डग्गनक": "डगन के",
            "दृष्टि सँ": "दृष्टि से",
            "जा रहल": "जा रहा",
            "देखाओल": "दिखाया",
            "देखायल": "दिखाई",
            "कतेको": "कई",
            "गेल आहि": "गया है",
            "गेल": "गया",
            "अछि।": "है।",
            "नहि।": "नहीं।",
            "सङ्गीत": "संगीत",
            "पसिन्न": "पसंद",
            "एकटा": "एक",
            "भटकय": "भटकाने",
            "छलाह": "थे",
            "किछु": "कुछ",
            "लोक": "लोग",
            "भऽ": "हो",
            "अहाँ": "आप",
            "अछि": "है",
            "नहि": "नहीं",
            "ओहि": "उस",
            "हमरा": "हमारा",
            "हमर नौकरीमे": "मेरी नौकरी में",
            "हमर नौकरी": "मेरी नौकरी",
            "हमर ": "मेरे ",
            "जनैत छी": "जानते हैं",
            "जनैत": "जानते",
            "नहीं जनैत": "नहीं जानते",
            "कखन": "कब",
            "पहिल बेर": "पहली बार",
            "पहिल": "पहली",
            " बेर ": " बार ",
            "करैत छी": "करते हैं",
            "करैत": "करते",
            "अपन करियर": "अपना करियर",
            "अपन ": "अपना ",
            " ई ": " और ",
            "विशेष रूपसे": "विशेष रूप से",
            "रूपसे": "रूप से",
            "नौकरी जकाँ है": "नौकरी जैसी है",
            "नौकरीमे": "नौकरी में",
            "कोनो एकरूपता": "कोई नियमितता",
            "कोनो ": "कोई ",
            "एकरूपता": "नियमितता",
            "आपकेँ": "आपको",
            " आपक ": " आपका ",
            "अगिला चेक": "अगला चेक",
            "अगिला": "अगला",
            "आबि रहल है": "आ रहा है",
            "आबि रहल": "आ रहा",
            "ओकर": "उसके",
            "विषयमे": "विषय में",
            "सोचबाक चाही": "सोचना चाहिए",
            "सोचबाक": "सोचना",
            "छेत्रीस": "सात",
            "चाही ।": "चाहिए।",
            "चाही।": "चाहिए।",
            "कयल": "किया",
            "सँ": "से",
            " जे ": " जो ",
            " जे,": " जो,",
            " जे।": " जो।",
        }

    def clean_translation(self, text: str) -> str:
        """Instance wrapper using ``DIALECT_TO_HINDI`` (same rules as module ``clean_translation``)."""
        return clean_translation(text, self.DIALECT_TO_HINDI)

    def _batch_flush_threshold(self, target_lang: str) -> int:
        """Smaller Hindi batches reduce one-paragraph merges and bad marker recovery."""
        t = target_lang.strip().lower()[:8]
        if t == "hi":
            return max(1, min(self._chunk_size, HI_MAX_CHUNK_SEGMENTS))
        return self._chunk_size

    def _protect_proper_nouns(self, text: str) -> tuple[str, dict[str, str]]:
        placeholders: dict[str, str] = {}
        protected = text
        pid = 0
        for noun in sorted(PROTECTED_PROPER_NOUNS, key=len, reverse=True):
            if len(noun.strip()) < 2:
                continue
            if " " in noun:
                pattern = re.compile(re.escape(noun), re.IGNORECASE)
            else:
                pattern = re.compile(
                    r"\b" + re.escape(noun) + r"\b", re.IGNORECASE
                )
            if not pattern.search(protected):
                continue
            key = f"PROPN{pid}END"
            placeholders[key] = noun
            protected = pattern.sub(key, protected)
            pid += 1
        return protected, placeholders

    def _restore_proper_nouns(self, text: str, placeholders: dict[str, str]) -> str:
        for key, value in placeholders.items():
            text = text.replace(key, value)
        return text

    def _clean_input(self, text: str) -> str:
        text = re.sub(r"\[[\d:\.]+\s*[–\-]\s*[\d:\.]+\]", "", text)
        text = re.sub(r"_[A-Z]+_[\d_]+", "", text)
        text = re.sub(r"\[.*?\]|\(.*?\)", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _build_input_ids(
        self, text: str, tgt_indic_code: str, *, max_input_tokens: int = 256
    ) -> torch.Tensor:
        body = " ".join(text.split())
        prefixed = f"eng_Latn {tgt_indic_code} {body}"
        token_ids = self.tokenizer.encode(prefixed, add_special_tokens=False)
        token_ids = token_ids[:max_input_tokens]
        eos_id = self.tokenizer.eos_token_id or 2
        if not token_ids or token_ids[-1] != eos_id:
            input_ids = token_ids + [eos_id]
        else:
            input_ids = token_ids
        return torch.tensor([input_ids], dtype=torch.long)

    def _translation_looks_bad(
        self, source_en: str, out: str, target_lang: str
    ) -> bool:
        if not (out or "").strip():
            return True
        s = source_en.strip()
        o = out.strip()
        if s == o:
            return True
        if len(s) > 12 and difflib.SequenceMatcher(None, s.lower(), o.lower()).ratio() > 0.88:
            return True
        if target_lang == "hi" and len(s) > 8:
            if not re.search(r"[\u0900-\u097F]", o) and re.search(r"[a-zA-Z]{4,}", s):
                return True
            if _has_dialect_retry_substrings(o):
                return True
        return False

    def _generate_once(
        self,
        protected_text: str,
        indic_tgt: str,
        forced_bos_token_id: int,
        *,
        max_input_tokens: int,
        max_new_tokens: int,
        num_beams: int,
    ) -> str:
        for_model = _seg_markers_for_model(protected_text)
        if self._engine == "nllb":
            enc = self.tokenizer(
                for_model,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens,
                src_lang="eng_Latn",
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
        else:
            input_ids = self._build_input_ids(
                for_model, indic_tgt, max_input_tokens=max_input_tokens
            ).to(self.device)
            attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                forced_bos_token_id=forced_bos_token_id,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
            )
        raw = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return _seg_markers_from_model(raw)

    def _hindi_emotion_from_english(self, en_text: str, *, is_chunk: bool) -> str:
        src = strip_seg_markers_for_emotion(en_text) if is_chunk else (en_text or "").strip()
        return detect_emotion(src)

    def _decode_and_postprocess(self, raw: str, placeholders: dict[str, str]) -> str:
        """clean → restore PROPN → clean (standard pipeline on model string)."""
        t = self.clean_translation(raw)
        t = self._restore_proper_nouns(t, placeholders)
        t = self.clean_translation(t)
        return t

    def _hi_output_quality_rank(
        self, source_en: str, out: str, *, is_chunk: bool
    ) -> tuple[int, int]:
        """
        Lower is better. (score, dialect_hits).
        """
        if not (out or "").strip():
            return (100, 999)
        dialect = 1 if _has_dialect_retry_substrings(out) else 0
        bad = 1 if (not is_chunk and self._translation_looks_bad(source_en, out, "hi")) else 0
        deva = _devanagari_letter_count(out)
        script_pen = 0 if deva >= 2 or len(source_en.strip()) <= 12 else 5
        return (bad * 20 + dialect * 10 + script_pen, dialect)

    def _apply_hindi_dub_post(self, translated: str, emotion: str) -> str:
        """Context window → humanize → emphasize → finalize (current segment only)."""
        if not (translated or "").strip():
            return translated or ""
        cur = translated.strip()
        if self._recent_context:
            blob = (
                "\n".join(self._recent_context[-2:])
                + "\n"
                + DUB_CONTEXT_MARK
                + "\n"
                + cur
            )
        else:
            blob = cur
        out = humanize_hindi_text(blob, emotion)
        if DUB_CONTEXT_MARK in out:
            out = out.split(DUB_CONTEXT_MARK, 1)[-1].strip()
        out = emphasize_keywords(out, emotion)
        out = finalize_hindi_spoken_segment(out)
        return out

    def translate(self, text: str, target_language: str = "hi") -> str:
        if not text or not text.strip():
            return ""

        text = self._clean_input(text)
        tgt = target_language.strip().lower()[:8]

        if _is_english_filler_only(text):
            return ""

        if tgt == "hi":
            gloss = _short_hi_gloss(text)
            if gloss is not None:
                return gloss

        protected_text, placeholders = self._protect_proper_nouns(text)

        forced_bos_token_id = self._lang_token_ids.get(tgt)
        if forced_bos_token_id is None or forced_bos_token_id == self.tokenizer.unk_token_id:
            indic = LANGUAGE_CODE_MAP.get(tgt, "hin_Deva")
            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(indic)
            if forced_bos_token_id == self.tokenizer.unk_token_id:
                forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(f"__{indic}__")

        indic_tgt = LANGUAGE_CODE_MAP.get(tgt, "hin_Deva")
        # IndicTrans2: force hin_Deva BOS the way the checkpoint expects.
        if self._engine == "indictrans2" and tgt == "hi":
            indic_tgt = "hin_Deva"
            hid = self.tokenizer.convert_tokens_to_ids("hin_Deva")
            if hid != self.tokenizer.unk_token_id:
                forced_bos_token_id = hid
            else:
                hid = self.tokenizer.convert_tokens_to_ids("__hin_Deva__")
                if hid != self.tokenizer.unk_token_id:
                    forced_bos_token_id = hid

        is_chunk = "<<<SEG_" in protected_text
        max_in = 1024 if is_chunk else 256
        max_new = 512 if is_chunk else 256
        hi_emotion = (
            self._hindi_emotion_from_english(text, is_chunk=is_chunk)
            if tgt == "hi"
            else "casual"
        )

        print(f"[translate] before model (preview): {protected_text[:160]!r}…")

        def run_pass(num_beams: int) -> str:
            raw = self._generate_once(
                protected_text,
                indic_tgt,
                forced_bos_token_id,
                max_input_tokens=max_in,
                max_new_tokens=max_new,
                num_beams=num_beams,
            )
            return self._decode_and_postprocess(raw, placeholders)

        try:
            translated = run_pass(4)
            print(f"[translate] after decode (post-clean preview): {translated[:120]!r}…")

            if tgt == "hi":
                src_for_rank = strip_seg_markers_for_emotion(text) if is_chunk else text
                best = translated
                best_rank, _ = self._hi_output_quality_rank(
                    src_for_rank, best, is_chunk=is_chunk
                )

                if is_chunk:
                    if _has_dialect_retry_substrings(translated):
                        print("[translate] dialect artifacts in chunk; retry beams=6")
                        t2 = run_pass(6)
                        r2, _ = self._hi_output_quality_rank(
                            src_for_rank, t2, is_chunk=is_chunk
                        )
                        if r2 < best_rank:
                            translated, best_rank = t2, r2
                else:
                    if best_rank > 0 or _has_dialect_retry_substrings(translated):
                        print("[translate] Hindi quality retry #1 (beams=6)")
                        t2 = run_pass(6)
                        r2, _ = self._hi_output_quality_rank(
                            src_for_rank, t2, is_chunk=is_chunk
                        )
                        if r2 < best_rank:
                            translated, best_rank = t2, r2
                    if best_rank > 0 or _has_dialect_retry_substrings(translated):
                        print("[translate] Hindi cleaner re-decode (greedy beams=1)")
                        t3 = run_pass(1)
                        r3, _ = self._hi_output_quality_rank(
                            src_for_rank, t3, is_chunk=is_chunk
                        )
                        if r3 < best_rank:
                            translated, best_rank = t3, r3

                    if (
                        self._translation_looks_bad(text, translated, "hi")
                        or _has_dialect_retry_substrings(translated)
                        or not _has_substantial_hindi_script(translated)
                    ):
                        print(
                            "[translate] Hindi unusable after retries; "
                            "falling back to English source"
                        )
                        return text

        except Exception as e:
            print(f"[translate] Failed ({e}), returning original")
            return text

        # Non-Hindi: legacy bad-output retry (unchanged idea).
        if tgt != "hi" and not is_chunk:
            if self._translation_looks_bad(text, translated, tgt):
                print("[translate] WARNING: retrying once (non-Hindi)")
                try:
                    t2 = run_pass(3)
                    if not self._translation_looks_bad(text, t2, tgt):
                        translated = t2
                except Exception as e2:
                    print(f"[translate] WARNING: retry failed ({e2})")
            if self._translation_looks_bad(text, translated, tgt):
                print("[translate] WARNING: falling back to English source")
                return text

        translated = re.sub(r"\s+", " ", translated).strip()
        if tgt == "hi" and not is_chunk:
            translated = self._apply_hindi_dub_post(translated, hi_emotion)
            if _is_english_filler_only(translated):
                translated = ""
        print(f"[translate] Result: {translated[:80]}")
        return translated

    def translate_segments(
        self,
        segments: list[dict],
        source_lang: str,
        target_lang: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[dict]:
        self._recent_context = []
        src = source_lang.strip().lower()[:8]
        tgt = target_lang.strip().lower()[:8]
        total = len(segments)
        out: list[dict] = []

        use_indic = src.startswith("en") and tgt in LANGUAGE_CODE_MAP
        if not use_indic and src != tgt:
            print(
                f"[translate] English→Indic only ({self._engine}); "
                f"passthrough for {src!r}→{tgt!r} (no translation)."
            )

        pending: list[tuple[int, dict, str]] = []
        i = 0

        def flush_batch() -> None:
            nonlocal pending
            if not pending:
                return
            indices_batch, segs_batch, texts_batch = zip(*pending)
            texts_list = list(texts_batch)
            combined = _format_chunk_for_translation(texts_list)
            print(
                f"[translate] chunk {indices_batch[0]+1}-{indices_batch[-1]+1}/{total} "
                f"({len(texts_list)} segs)…"
            )
            mapped: list[str] | None = None
            try:
                big = self.translate(combined, tgt)
                mapped = _extract_chunk_segments(big, len(texts_list))
                if mapped is None and tgt == "hi":
                    alt = _split_hindi_by_en_weights(big, texts_list)
                    if alt is not None:
                        print(
                            "[translate] Recovered chunk via English-length weighting "
                            "(model dropped SEG markers; avoiding per-segment dialect drift)"
                        )
                        mapped = alt
                if mapped is None:
                    print(
                        f"[translate] chunk split mismatch (markers not preserved); "
                        f"falling back per-segment for this batch"
                    )
            except Exception as e:
                print(f"[translate] chunk failed ({e}); falling back per-segment")
                mapped = None

            if mapped is None:
                mapped = []
                for src_t in texts_list:
                    try:
                        mapped.append(self.translate(src_t, tgt))
                    except Exception as e:
                        print(f"[translate] per-segment fallback failed: {e}")
                        mapped.append(src_t)
            elif tgt == "hi" and mapped and len(mapped) == len(texts_list):
                # Batch humanize: join → humanize+emotion → split on <<<SEG_n>>> (timestamps unchanged).
                try:
                    emo_batch = detect_emotion(" ".join(texts_list))
                    reconstructed = " ".join(
                        f"<<<SEG_{k}>>> {mapped[k]}" for k in range(len(mapped))
                    )
                    full_h = humanize_hindi_text(reconstructed, emo_batch)
                    remapped = _extract_chunk_segments(full_h, len(texts_list))
                    if remapped is not None:
                        mapped = remapped
                    else:
                        mapped = [
                            humanize_hindi_text(p, detect_emotion(st))
                            for p, st in zip(mapped, texts_list)
                        ]
                except Exception:
                    mapped = [
                        humanize_hindi_text(p, detect_emotion(st))
                        for p, st in zip(mapped, texts_list)
                    ]

            for j, idx in enumerate(indices_batch):
                seg = segs_batch[j]
                src_t = texts_list[j]
                tt = (mapped[j] if j < len(mapped) else "").strip()
                tt = re.sub(
                    r"\s*<<<\s*SEG_\d+\s*>>>\s*",
                    " ",
                    tt,
                    flags=re.IGNORECASE,
                ).strip()
                if tgt == "hi" and tt and _is_english_filler_only(tt):
                    tt = ""
                emo_r = detect_emotion(src_t)
                if not tt or self._translation_looks_bad(src_t, tt, tgt):
                    print(
                        f"[translate] WARNING: bad/empty chunk piece seg {idx + 1}; "
                        f"retry single-segment"
                    )
                    try:
                        tt = self.translate(src_t, tgt)
                    except Exception as e:
                        print(f"[translate] seg {idx + 1} fallback failed: {e}")
                        tt = src_t
                elif tgt == "hi" and tt:
                    tt = emphasize_keywords(tt, emo_r)
                    tt = finalize_hindi_spoken_segment(tt)
                if tgt == "hi" and tt:
                    self._recent_context.append(tt.strip())
                    self._recent_context = self._recent_context[-2:]

                # LLM correction — uses source as ground truth to fix machine translation
                if tt and src_t:
                    tt = _llm_validate(
                        src_t,
                        tt,
                        retranslate_fn=lambda t: self.translate(t, tgt),
                        source_lang=src,
                        target_lang=tgt,
                    )

                out.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": src_t,
                    "translated_text": tt.strip(),
                    **({"dub_emotion": emo_r} if tgt == "hi" else {}),
                    **_passthrough_stt_fields(seg),
                })
                print(f"[translate] Done {idx + 1}: {(tt or '')[:60]!r}…")
                if progress_callback:
                    progress_callback(idx + 1, total)
            pending = []

        while i < total:
            seg = segments[i]
            raw = seg.get("text") or ""
            preview = raw.strip()[:60]
            print(f"[translate] Segment {i + 1}/{total}: {preview!r}…")
            text = raw.strip()

            if not text:
                flush_batch()
                out.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": "",
                    "translated_text": "",
                    **_passthrough_stt_fields(seg),
                })
                if progress_callback:
                    progress_callback(i + 1, total)
                i += 1
                continue

            if use_indic and _is_english_filler_only(text):
                flush_batch()
                out.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": text,
                    "translated_text": "",
                    **({"dub_emotion": detect_emotion(text)} if tgt == "hi" else {}),
                    **_passthrough_stt_fields(seg),
                })
                if progress_callback:
                    progress_callback(i + 1, total)
                i += 1
                continue

            if src == tgt:
                flush_batch()
                out.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": text,
                    "translated_text": text,
                    **_passthrough_stt_fields(seg),
                })
                if progress_callback:
                    progress_callback(i + 1, total)
                i += 1
                continue

            if not use_indic:
                flush_batch()
                out.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": text,
                    "translated_text": text,
                    **_passthrough_stt_fields(seg),
                })
                if progress_callback:
                    progress_callback(i + 1, total)
                i += 1
                continue

            # Code-switching check: segment may mix source language with target-script words
            if use_indic:
                try:
                    from services.code_switch_handler import (
                        detect_code_switching,
                        translate_with_code_switch_handling,
                    )
                    cs_info = detect_code_switching(text, src, tgt)
                    if cs_info["is_already_target"]:
                        # Already in the target language — flush batch, emit as-is
                        flush_batch()
                        emo_cs = detect_emotion(text) if tgt == "hi" else "casual"
                        out.append({
                            "start": seg["start"],
                            "end": seg["end"],
                            "text": text,
                            "translated_text": text,
                            **({"dub_emotion": emo_cs} if tgt == "hi" else {}),
                            **_passthrough_stt_fields(seg),
                        })
                        if progress_callback:
                            progress_callback(i + 1, total)
                        i += 1
                        continue
                    elif cs_info["is_mixed"]:
                        # Mixed-language segment — handle individually with code-switch handler
                        flush_batch()
                        print(
                            f"[translate] Code-switching detected in seg {i + 1} "
                            f"(target_ratio={cs_info['target_script_ratio']}, "
                            f"spans={len(cs_info['foreign_spans'])})"
                        )
                        cs_translated = translate_with_code_switch_handling(
                            text, src, tgt, lambda t: self.translate(t, tgt)
                        )
                        if not cs_translated or cs_translated == text:
                            # Fallback: translate normally
                            cs_translated = self.translate(text, tgt)
                        cs_translated = re.sub(r"\s+", " ", cs_translated).strip()
                        emo_cs = detect_emotion(text) if tgt == "hi" else "casual"
                        if tgt == "hi" and cs_translated:
                            cs_translated = emphasize_keywords(cs_translated, emo_cs)
                            cs_translated = finalize_hindi_spoken_segment(cs_translated)
                            self._recent_context.append(cs_translated.strip())
                            self._recent_context = self._recent_context[-2:]
                        # LLM correction on code-switched segment
                        if cs_translated and text:
                            cs_translated = _llm_validate(
                                text,
                                cs_translated,
                                retranslate_fn=lambda t: self.translate(t, tgt),
                                source_lang=src,
                                target_lang=tgt,
                            )
                        out.append({
                            "start": seg["start"],
                            "end": seg["end"],
                            "text": text,
                            "translated_text": cs_translated.strip(),
                            **({"dub_emotion": emo_cs} if tgt == "hi" else {}),
                            **_passthrough_stt_fields(seg),
                        })
                        if progress_callback:
                            progress_callback(i + 1, total)
                        i += 1
                        continue
                except Exception as cs_exc:
                    print(f"[translate] Code-switch check failed (seg {i + 1}): {cs_exc}; proceeding normally")

            pending.append((i, seg, text))
            if len(pending) >= self._batch_flush_threshold(tgt):
                flush_batch()
            i += 1

        flush_batch()

        return out


_translator_instance: Translator | None = None


def get_translator() -> Translator:
    global _translator_instance
    if _translator_instance is None:
        _translator_instance = Translator()
    return _translator_instance


def translate(text: str, target_language: str = "hi") -> str:
    return get_translator().translate(text, target_language)


def translate_segments(
    segments: list[dict],
    source_lang: str,
    target_lang: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> list[dict]:
    """Same signature as before for main.py."""
    return get_translator().translate_segments(
        segments, source_lang, target_lang, progress_callback
    )
