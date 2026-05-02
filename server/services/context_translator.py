"""
Context-aware batch translation for professional dubbing quality.

The core insight: translating 5 lines at a time (the old approach) loses context.
Pronouns resolve incorrectly, idioms are translated literally, terminology drifts.

This module sends the ENTIRE script to Google Translate numbered, so the model
sees the full conversation — exactly like a human translator reading the whole
script before starting.

                                    ─────────────────────────────────────
  Old approach:          [1] Hello, how are you?   →  [1] नमस्ते, कैसे हैं?
                         (context blind)

  Context approach:      [1] Hello, how are you?
                         [2] I heard you quit.      →  Full context translation
                         [3] That's not quite right.    with correct pronouns,
                         [N] ...                        idioms, register
                                    ─────────────────────────────────────

Additional passes:
  - Entity preservation: named entities (people, places, brands) are detected
    before translation and restored afterward to prevent Google mangling them.
    **Critical:** Title-case *function words* (There, They, When, Try, …) and
    contraction stems (``Don`` + ``'t``) must **never** be masked — otherwise
    ``_restore_entities`` puts raw English back into every target language.
    See ``_NE_STOPWORDS``, ``_NE_NEVER_MASK``, and ``_NE_CONTRACTION_PREFIX``.
  - Hindi natural speech rewrite: after translation, apply conversational
    tone rules to make TTS output sound human, not robotic.
  - Consistency pass: terminology used in segment 1 matches segment 40.
"""
from __future__ import annotations

import re
import time
import unicodedata
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Named entity preservation
# ---------------------------------------------------------------------------

# Patterns for tokens we do NOT want translated:
#   - Proper nouns (Title Case in the middle of a sentence)
#   - All-caps words (acronyms, brands)
#   - Numbers + units
_NE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,3}\b"),  # Proper nouns
    re.compile(r"\b[A-Z]{2,}\b"),                                    # Acronyms / all-caps brands
]

# Placeholder format that survives translation (looks like XML, Google leaves it alone)
_NE_PLACEHOLDER = "XENTX{idx}X"
_NE_PLACEHOLDER_RE = re.compile(r"XENTX(\d+)X")

# Common words that look like proper nouns but shouldn't be protected
_NE_STOPWORDS: frozenset[str] = frozenset({
    "I", "A", "The", "In", "On", "At", "To", "Of", "And", "But", "Or",
    "So", "It", "He", "She", "We", "You", "Is", "Are", "Was", "Were",
    "Be", "Do", "Did", "Can", "Will", "May", "Has", "Have", "Had",
    "That", "This", "These", "Those", "For", "With", "From", "By",
    "Not", "No", "Yes", "My", "His", "Her", "Our", "Your", "Its",
    "Mr", "Mrs", "Dr", "Ms", "Prof",
    # Sentence-initial function words the NE regex mistakes for proper nouns
    # (otherwise masked → restored as raw English in every target language).
    "There", "They", "Them", "Their", "When", "Where", "What", "Which",
    "Who", "Whom", "Whose", "Why", "How", "Then", "Than", "Here",
    "Every", "Each", "Both", "Few", "Such", "Same", "Other", "Some",
    "Many", "Much", "More", "Most", "Less", "Very", "Just", "Even",
    "Still", "Also", "Only", "Well", "Back", "Away", "Down", "Into",
    "Onto", "Upon", "Over", "Once", "Twice", "Today", "Tonight",
})

# Words that match ``[A-Z][a-z]{2,}`` but are normal English sentence words — never
# mask as "named entities" or they survive translation (e.g. "Sometimes، …" in Arabic).
_NE_NEVER_PROTECT: frozenset[str] = frozenset({
    "sometimes", "perhaps", "maybe", "however", "therefore", "usually", "often",
    "because", "although", "though", "unless", "until", "while", "whenever",
    "whatever", "wherever", "somewhere", "everywhere", "nowhere", "anybody",
    "everybody", "somebody", "nobody", "everyone", "someone", "anyone",
    "today", "tomorrow", "yesterday", "tonight", "already", "finally", "actually",
    "basically", "especially", "generally", "obviously", "clearly", "probably",
    "definitely", "certainly", "hopefully", "luckily", "sadly", "fortunately",
    "unfortunately", "instead", "otherwise", "similarly", "indeed", "anyway",
    "eventually", "currently", "recently", "naturally", "originally", "traditionally",
    "first", "second", "third", "fourth", "fifth", "last", "next", "soon", "later",
    "early", "late", "always", "never", "again", "almost", "nearly", "quite", "rather",
    "even", "still", "yet", "also", "only", "just", "very", "really", "truly",
    "simply", "together", "apart", "alone", "little", "much", "more", "less", "least",
    "most", "many", "few", "another", "other", "such", "each", "every", "both",
    "neither", "either", "whether", "since", "before", "after", "during", "within",
    "without", "inside", "outside", "above", "below", "between", "among", "across",
    "towards", "against", "beyond", "besides", "beside", "about", "around",
    "throughout", "nothing", "everything", "something", "anything", "everywhere",
    "somewhere", "nowhere", "anytime", "sometime", "anywhere", "overall", "otherwise",
    "otherwise", "meanwhile", "hence", "thus", "accordingly", "furthermore", "moreover",
    "nevertheless", "nonetheless", "regardless", "according", "considering",
    "regarding", "concerning", "including", "excluding", "following", "leading",
    "remaining", "pending", "depending", "supposing", "assuming", "given",
    # Short imperatives / auxiliaries (Title Case → wrongly treated as proper nouns by NE regex)
    "try", "ask", "run", "get", "let", "put", "set", "add", "use", "pay", "buy",
    "say", "see", "sit", "eat", "lie", "lay", "win", "die", "fly", "cry", "beg",
    "act", "aim", "end", "fit", "fix", "mix", "cut", "rid", "tap", "rub", "nod",
    "bow", "owe", "dry", "fry", "pry", "shy", "spy", "got", "did", "was", "has",
    "had", "led", "met", "sat", "won", "lost", "keep", "held", "read", "made",
    "took", "gave", "went", "came", "stop", "wait", "walk", "talk", "call", "hold",
    "turn", "open", "push", "pull", "pick", "drop", "work", "play", "live", "love",
    "hope", "wish", "plan", "move", "face", "mind", "note", "join", "stay",
    "leave", "bring", "build", "break", "choose", "dream", "drive", "fight", "follow",
    "forget", "learn", "think", "speak", "write", "watch", "listen", "remember",
})

# Lowercase stems that are the first part of English contractions (Don't → Don+'t).
# Without this, ``\b[A-Z][a-z]{2,}\b`` matches "Don" and leaves "Don" untranslated.
_NE_CONTRACTION_PREFIX: frozenset[str] = frozenset({
    "don", "won", "can", "isn", "aren", "wasn", "weren", "haven", "hadn",
    "doesn", "didn", "wouldn", "couldn", "shouldn", "mightn", "mustn",
    "shan", "ain", "let",  # Let's, don't, … (pronouns covered by _NE_STOPWORDS)
})

# Extra common English function words (lowercase) not covered above — never mask.
_NE_FUNCTION_WORDS: frozenset[str] = frozenset({
    "there", "they", "them", "their", "theirs", "when", "where", "what",
    "which", "who", "whom", "whose", "why", "how", "than", "then",
    "here", "therein", "thereof", "thereby", "therefore", "though",
    "although", "because", "unless", "until", "while", "during", "within",
    "without", "through", "across", "toward", "towards", "against",
    "beyond", "behind", "beside", "besides", "except", "inside", "outside",
    "around", "among", "upon", "onto", "into", "from", "about", "above",
    "below", "under", "ever", "never", "always", "often", "sometimes",
    "already", "perhaps", "maybe", "rather", "quite", "almost", "nearly",
    "again", "once", "twice", "today", "tonight", "tomorrow", "yesterday",
    "every", "each", "both", "few", "such", "same", "other", "some",
    "many", "much", "more", "most", "less", "least", "very", "just",
    "even", "still", "also", "only", "well", "back", "away", "down",
    "own", "same", "sure", "like", "used", "using", "being",
    "having", "doing", "going", "coming", "making", "taking", "getting",
    "giving", "looking", "working", "seeming", "seems", "seemed",
    "would", "could", "should", "might", "must", "shall", "ought",
})

_NE_NEVER_MASK: frozenset[str] = _NE_NEVER_PROTECT | _NE_FUNCTION_WORDS


def _is_contraction_prefix_token(text: str, end: int, token: str) -> bool:
    """True if ``token`` at ``text[:end]`` is the stem before an English contraction."""
    if token.lower() not in _NE_CONTRACTION_PREFIX:
        return False
    if end >= len(text):
        return False
    if text[end] not in "'\u2019":  # ASCII or Unicode apostrophe
        return False
    if end + 1 >= len(text):
        return False
    return text[end + 1].isalpha()


def _extract_entities(text: str) -> tuple[str, dict[str, str]]:
    """
    Replace named entities with XENTX{N}X placeholders.
    Returns (masked_text, {placeholder: original}).
    """
    # Collect matches first, sort by position descending, replace from end → start.
    entities2: dict[str, str] = {}
    result2 = text
    all_matches = []
    for pattern in _NE_PATTERNS:
        for m in pattern.finditer(text):
            token = m.group(0)
            if token in _NE_STOPWORDS or len(token) < 3:
                continue
            if token.lower() in _NE_NEVER_MASK:
                continue
            if _is_contraction_prefix_token(text, m.end(), token):
                continue
            all_matches.append((m.start(), m.end(), token))

    # Sort by start position descending to replace from end → start (no offset shift)
    all_matches.sort(key=lambda x: -x[0])
    counter = 0
    for start, end, token in all_matches:
        ph = _NE_PLACEHOLDER.format(idx=counter)
        entities2[ph] = token
        result2 = result2[:start] + ph + result2[end:]
        counter += 1

    return result2, entities2


def _restore_entities(text: str, entities: dict[str, str]) -> str:
    """Restore XENTX{N}X placeholders with original entity strings."""
    result = text
    for ph, original in entities.items():
        result = result.replace(ph, original)
    # Clean up any un-matched placeholders (translation added extra ones)
    result = _NE_PLACEHOLDER_RE.sub("", result)
    return result.strip()


# ---------------------------------------------------------------------------
# Numbered-line batch translation
# ---------------------------------------------------------------------------

def _build_numbered_block(lines: list[str], start_idx: int = 0) -> str:
    """
    Format lines as:
      [1] First sentence
      [2] Second sentence
      ...
    """
    return "\n".join(
        f"[{start_idx + i + 1}] {line}" if line.strip() else f"[{start_idx + i + 1}]"
        for i, line in enumerate(lines)
    )


def _parse_numbered_block(
    text: str,
    count: int,
    global_offset: int,
    fallback_lines: list[str],
) -> list[str]:
    """
    Parse a translated numbered block back to a list of strings.

    Handles Google's various ways of formatting the numbers:
      [1], (1), 1., 1:, [1]:, etc.
    Also handles Google occasionally merging or splitting lines.
    """
    results: list[str | None] = [None] * count

    # Primary pattern: optional bracket/paren + digits + optional bracket/paren/dot/colon
    # followed by the translated content up to the next numbered marker or end
    primary = re.compile(
        r"[\[\(]?\s*(\d{1,4})\s*[\]\)\.:\-]?\s*(.+?)(?=\n[\[\(]?\s*\d{1,4}[\]\)\.:\-]|\Z)",
        re.DOTALL,
    )

    for m in primary.finditer(text):
        num = int(m.group(1))
        content = m.group(2).strip().replace("\n", " ")
        # Collapse multiple spaces
        content = re.sub(r"\s{2,}", " ", content)
        # Convert to batch-local index
        idx = num - global_offset - 1
        if 0 <= idx < count and results[idx] is None:
            results[idx] = content

    # If no numbered markers found, split by newlines and align positionally
    found = sum(1 for r in results if r is not None)
    if found == 0:
        raw_lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
        for i in range(min(len(raw_lines), count)):
            results[i] = raw_lines[i]

    # Fill still-missing slots
    for i in range(count):
        if not results[i]:
            results[i] = fallback_lines[i] if i < len(fallback_lines) else ""

    return results  # type: ignore[return-value]


def translate_full_context(
    lines: list[str],
    translator,
    *,
    batch_size: int = 35,
    delay_s: float = 0.3,
    max_chars_per_batch: int = 4500,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> list[str]:
    """
    Translate ``lines`` with full script context using numbered batches.

    Each batch contains up to ``batch_size`` lines formatted as:
      [1] sentence one
      [2] sentence two
      ...
    so Google Translate sees the surrounding context for each line, producing
    dramatically better pronoun resolution, idiom translation, and register
    consistency.

    Parameters
    ----------
    lines        : Source text lines (one per subtitle segment).
    translator   : A ``deep_translator.GoogleTranslator`` instance (already configured).
    batch_size   : Max lines per API call (default 35).
    delay_s      : Sleep between batches (rate-limit courtesy).
    max_chars_per_batch : Hard char limit; batches are split further if exceeded.
    progress_callback   : Called as ``(done_lines, total_lines)``.

    Returns
    -------
    Translated strings, 1-to-1 aligned with ``lines``.
    """
    if not lines:
        return []

    results: list[str] = [""] * len(lines)
    n = len(lines)
    done = 0

    i = 0
    while i < n:
        # Determine batch end (respect both batch_size and char limit)
        j = min(i + batch_size, n)
        batch = lines[i:j]

        # Further split if char count would exceed Google's limit
        while j > i + 1:
            block = _build_numbered_block(batch, i)
            if len(block) <= max_chars_per_batch:
                break
            j -= 1
            batch = lines[i:j]

        block = _build_numbered_block(batch, i)
        translated_block: str | None = None

        for attempt in range(3):
            try:
                translated_block = translator.translate(block)
                if translated_block:
                    break
            except Exception as exc:
                wait = delay_s * (2 ** attempt)
                print(
                    f"[ctx-translate] batch [{i+1}–{j}] attempt {attempt+1} failed "
                    f"({exc!r}); retrying in {wait:.1f}s"
                )
                time.sleep(wait)

        if translated_block:
            parsed = _parse_numbered_block(translated_block, j - i, i, batch)
            for k, tr in enumerate(parsed):
                if i + k < n:
                    results[i + k] = tr or lines[i + k]
        else:
            # Full per-line fallback for this batch
            print(f"[ctx-translate] batch [{i+1}–{j}] all attempts failed; translating per line")
            for k, line in enumerate(batch):
                tr = ""
                for _att in range(2):
                    try:
                        tr = (translator.translate(line) if line.strip() else "") or ""
                        break
                    except Exception as exc2:
                        print(f"[ctx-translate] line {i+k+1} fallback failed: {exc2!r}")
                        time.sleep(delay_s)
                results[i + k] = tr.strip() if tr else line

        done = j
        if progress_callback:
            progress_callback(done, n)
        if j < n:
            time.sleep(delay_s)
        i = j

    return results


# ---------------------------------------------------------------------------
# Hindi natural-speech post-processor
# ---------------------------------------------------------------------------

# Words/phrases Google Translate renders in formal/literary Hindi that sound
# unnatural when spoken aloud by a TTS voice. Each tuple is (pattern, replacement).
# These are applied AFTER translation, before TTS.
# NOTE: Python's \b word boundary does NOT work with Devanagari combining marks
# (vowel signs like ि U+093F are category Mc, which is \W in re).
# We use (?<!\S) / (?!\S) instead: "not preceded/followed by non-whitespace"
# which is equivalent to start/end-of-string or whitespace boundary.
def _hw(word: str) -> str:
    """Wrap a Hindi word/phrase pattern with Unicode-safe word boundaries."""
    return r"(?<!\S)" + word + r"(?!\S)"


_HINDI_NATURALISE: list[tuple[str, str]] = [
    # Overly formal conjunctions → conversational
    (_hw("तथापि"),      "फिर भी"),
    (_hw("परंतु"),       "लेकिन"),
    (_hw("किंतु"),       "लेकिन"),
    (_hw("अतएव"),        "इसलिए"),
    (_hw("अतः"),         "तो"),
    (_hw("एवं"),         "और"),
    (_hw("तदनुसार"),     "उसके अनुसार"),
    (_hw(r"इसलिए\s+कि"),  "क्योंकि"),
    # Formal 'है' phrases → spoken contractions
    (_hw(r"यह\s+है\s+कि"),  "ये है कि"),
    (_hw(r"वह\s+है\s+कि"),  "वो है कि"),
    (_hw(r"ऐसा\s+है\s+कि"), "ऐसे है"),
    # Robotic 'करना' constructs → natural
    (_hw(r"करने\s+की\s+आवश्यकता\s+है"),  "करना ज़रूरी है"),
    (_hw(r"करने\s+की\s+जरूरत\s+है"),     "करना ज़रूरी है"),
    (_hw(r"करना\s+होगा"),                  "करना पड़ेगा"),
    # Literal question constructs
    (_hw(r"क्या\s+आप\s+जानते\s+हैं\s+कि"), "क्या आप जानते हैं"),
    # Formal first-person → natural
    (_hw(r"मैं\s+आपको\s+बता\s+दूं\s+कि"),  "मैं बताता/बताती हूं"),
    # Excessively long honorifics in rapid speech
    (_hw("महोदय"),    "जी"),
    (_hw("श्रीमान"),   "जी"),
    # Overly formal / textbook → spoken (natural dubbing)
    (_hw("परिपूर्ण"),   "परफ़ेक्ट"),
    (_hw("व्यक्ति"),    "इंसान"),
    (_hw("मूल्यवान"),   "कीमती"),
    # Number words rendered as digits → spoken form hints
    (r"\b100\b",  "सौ"),
    (r"\b1000\b", "हज़ार"),
    # Remove ellipsis that confuses TTS
    (r"\.\.\.",   " "),
    (r"…",        " "),
    # Dash in middle of Hindi sentence → comma pause
    (r"\s+[-–—]\s+", ", "),
]

_HINDI_NATURALISE_COMPILED = [
    (re.compile(pat), repl) for pat, repl in _HINDI_NATURALISE
]

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def _strip_spurious_leading_latin_before_devanagari(text: str) -> str:
    """
    Remove a stuck English word before Hindi (bad NE mask or rare MT glitch).

    Examples: ``Try सफल…`` → ``सफल…`` ; ``Sometimes، आप…`` with Devanagari tail.
    """
    if not text or not _DEVANAGARI_RE.search(text):
        return text
    t = text.strip()
    for _ in range(3):
        m = re.match(r"^([A-Za-z]+)\s*[،,]\s*(.+)$", t, re.DOTALL)
        if not m:
            m = re.match(r"^([A-Za-z]+)\s+(.+)$", t, re.DOTALL)
        if not m:
            break
        first, rest = m.group(1), m.group(2).strip()
        if not rest or not _DEVANAGARI_RE.search(rest):
            break
        if first.lower() in _NE_NEVER_MASK:
            t = rest
            continue
        break
    return t


def naturalise_hindi_for_speech(text: str) -> str:
    """
    Post-process a Google-translated Hindi string to sound natural when
    spoken aloud by a TTS engine.

    Applies a curated set of formal→conversational substitutions, then
    normalises unicode (NFC) and whitespace.
    """
    if not text or not text.strip():
        return text or ""

    # NFC normalise — prevents rendering differences between precomposed and
    # combining Devanagari characters from causing silent mis-matches
    t = unicodedata.normalize("NFC", text)
    t = _strip_spurious_leading_latin_before_devanagari(t)

    for pattern, replacement in _HINDI_NATURALISE_COMPILED:
        t = pattern.sub(replacement, t)

    # Collapse runs of whitespace / punctuation
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"[।!?]{2,}", "।", t)  # deduplicate sentence-final markers

    return t.strip()


# Arabic / Hebrew / Persian script (broad) — for post-translation cleanup
_ARABIC_SCRIPT_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)

_RTL_POST_LANGS = frozenset({"ar", "fa", "ur", "he", "ps"})


def _strip_spurious_leading_latin_before_rtl(text: str) -> str:
    """
    Remove a stuck English first word before RTL text (legacy bad entity masks).

    Example: ``Sometimes، المشاكل…`` → ``المشاكل…``
    """
    if not text or not _ARABIC_SCRIPT_RE.search(text):
        return text
    m = re.match(r"^\s*([A-Za-z]+)\s*[،,]\s*(.+)$", text, re.DOTALL)
    if not m:
        return text
    first, rest = m.group(1), m.group(2).strip()
    if not rest or not _ARABIC_SCRIPT_RE.search(rest):
        return text
    if first.lower() in _NE_NEVER_MASK:
        return rest
    if len(first) <= 3 and first.isalpha():
        return rest
    return text


def naturalise_rtl_target_for_speech(text: str) -> str:
    """Light cleanup so Edge / MMS TTS gets clean Arabic (etc.) without Latin glue."""
    if not text or not text.strip():
        return text or ""
    t = unicodedata.normalize("NFC", text)
    t = _strip_spurious_leading_latin_before_rtl(t)
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"\.{3,}|…", " ", t)
    return t.strip()


# ---------------------------------------------------------------------------
# Universal entity-safe translation pipeline
# ---------------------------------------------------------------------------

def translate_lines_professional(
    lines: list[str],
    translator,
    target_language: str,
    *,
    delay_s: float = 0.3,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> list[str]:
    """
    Full professional translation pipeline:

    1. Extract named entities → replace with placeholders (protects proper nouns)
    2. Translate with full-context numbered batching
    3. Restore named entities
    4. If target is Hindi: apply natural-speech post-processing
    5. Unicode NFC normalise all output

    Returns 1-to-1 aligned list of translated strings.
    """
    if not lines:
        return []

    tgt = (target_language or "en").lower().strip()

    # --- Step 1: Entity extraction ---
    masked_lines: list[str] = []
    all_entities: list[dict[str, str]] = []
    for line in lines:
        masked, ents = _extract_entities(line)
        masked_lines.append(masked)
        all_entities.append(ents)

    # --- Step 2: Context-aware translation ---
    translated = translate_full_context(
        masked_lines,
        translator,
        delay_s=delay_s,
        progress_callback=progress_callback,
    )

    # --- Step 3: Restore entities + post-process ---
    results: list[str] = []
    for i, (tr, ents) in enumerate(zip(translated, all_entities)):
        # Restore named entities
        tr = _restore_entities(tr, ents)
        # NFC normalise
        tr = unicodedata.normalize("NFC", tr).strip()
        # Language-specific naturalisation
        if tgt == "hi":
            tr = naturalise_hindi_for_speech(tr)
        elif tgt in _RTL_POST_LANGS:
            tr = naturalise_rtl_target_for_speech(tr)
        if not tr:
            tr = lines[i]  # last-resort: use source
        results.append(tr)

    return results
