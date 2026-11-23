"""Read the raw task before any model call: its parts, its task type and the expected output format.

Nothing here depends on one prompt template. A prompt is split into the document to read (if any) and the
ask about it, using labelled sections when the prompt has them ("Passage:", "CONTEXT:", "Q:") and block
structure when it does not. The task type is then inferred from what the ask requests, or taken from the
caller's hint. A prompt that fits no type is "unknown", and routers treat unknown as risky.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from triage.schemas import TASK_TYPES, Profile, Task

# Section labels people put in front of the material to read and in front of the ask.
_DOC_LABELS = (
    "passage|context|document|article|text|excerpt|policy(?: text)?|contract(?: text)?|agreement|"
    "source text|reference|background"
)
_ASK_LABELS = "question|query|task|instructions?|problem|request|details|ticket|message|q"
_LABEL = re.compile(
    rf"(?:^|(?<=[\n'\".?!)]))[ \t]*(?P<label>(?i:{_DOC_LABELS}|{_ASK_LABELS}))[ \t]*:", re.MULTILINE
)
_IS_DOC_LABEL = re.compile(rf"(?i:{_DOC_LABELS})$")

_DOC_MIN_WORDS = 40  # a shorter block is part of the ask, not a document
_ASK_TAIL_CHARS = 2000  # an ask that follows a document sits in the prompt's last stretch
_SQL = r"SELECT\b.+|INSERT INTO\b|UPDATE \w+ SET\b|DELETE FROM\b|CREATE TABLE\b|WITH \w+ AS\b"
_CODE_LINE = re.compile(
    rf"^[ \t]*(def \w+\(|class \w+[(:]|import \w+|from [\w.]+ import\b|assert\b|>>>|{_SQL})", re.MULTILINE
)
_QUOTED = re.compile(r"\"\"\"(.+?)\"\"\"|'''(.+?)'''|(?<!\w)'(.{120,}?)'(?!\w)|\"(.{120,}?)\"", re.DOTALL)


@dataclass(frozen=True)
class Parts:
    document: str  # the material to read; "" when the prompt has none
    ask: str  # the question or instruction about it
    frame: str  # everything outside the document: instructions plus the ask

    @property
    def question(self) -> str:
        """The last question in the ask, without a lead-in ("Based on the text above: ..."), or the whole
        ask when it holds no question."""
        found = re.findall(r"[^.?!\n]*\?", self.ask)
        if not found:
            return self.ask.strip()
        return found[-1].rsplit(":", 1)[-1].strip()


def _words(text: str) -> int:
    return len(text.split())


def _is_list(block: str) -> bool:
    """A list of options (bullets, or short comma-separated items), not running text."""
    lines = [ln for ln in block.splitlines() if ln.strip()]
    bullets = sum(bool(re.match(r"\s*(?:[-*\u2022]|\d+[.)])\s+\S", ln)) for ln in lines)
    if len(lines) >= 3 and bullets >= 0.6 * len(lines) and _words(block) <= 8 * len(lines):
        return True  # short bulleted items; numbered clauses of running text are not a list
    return block.count(",") >= 5 and block.count(",") >= 0.5 * _words(block)


def _is_prose(block: str) -> bool:
    return _words(block) >= _DOC_MIN_WORDS and not _CODE_LINE.search(block) and not _is_list(block)


def _labelled(text: str) -> Parts | None:
    labels = [m for m in _LABEL.finditer(text) if m.group("label")[0].isupper()]
    docs = [m for m in labels if _IS_DOC_LABEL.match(m.group("label"))]
    asks = [m for m in labels if not _IS_DOC_LABEL.match(m.group("label"))]
    if not docs:
        if not asks:
            return None
        before, ask = text[: asks[0].start()], text[asks[-1].end() :]
        blocks = [b for b in re.split(r"\n\s*\n", before) if b.strip()]
        prose = [b for b in blocks if not _is_ask(b)]
        document = "\n\n".join(prose) if prose and _is_prose("\n\n".join(prose)) else ""
        frame = "\n\n".join(b for b in blocks if b not in prose) + "\n" + text[asks[0].start() :]
        return Parts(document.strip(), ask.strip(), frame if document else text)
    doc = docs[0]
    before = [m for m in asks if m.start() < doc.start()]
    if not before and _words(text[: doc.start()]) > _ASK_MAX_WORDS:
        return None  # a label deep inside running text is part of the document, not its heading
    tail = [m for m in asks if m.start() > doc.end() and m.start() >= len(text) - _ASK_TAIL_CHARS]
    if tail:  # document, then the ask
        document, ask = text[doc.end() : tail[0].start()], text[tail[-1].end() :]
        frame = text[: doc.start()] + " " + text[tail[0].start() :]
    elif before:  # the ask, then the document
        document, ask = text[doc.end() :], text[before[-1].end() : doc.start()]
        frame = text[: doc.start()]
    else:  # a labelled document with unlabelled instructions around it
        document = text[doc.end() :]
        blocks = [b for b in re.split(r"\n\s*\n", document.strip()) if b.strip(" \n-=*_#")]
        short = len(blocks) > 1 and _words(blocks[-1]) <= 60 and _words(blocks[-1]) * 4 <= _words(document)
        closing = blocks[-1] if len(blocks) > 1 and (_is_ask(blocks[-1]) or short) else ""
        opening = text[: doc.start()]
        if closing:
            document = document[: document.rindex(closing)]
            if not _REQUEST.search(opening):  # text before the label with no request in it is document too
                document, opening = opening + document, ""
        ask = closing or opening
        frame = opening + " " + closing
    return Parts(document.strip(" \n'\""), ask.strip(" \n'\""), frame.strip())


_REQUEST = re.compile(
    r"\?|\b(return|reply|respond|answer|list|extract|give|find|summari[sz]e|classif\w+|categori[sz]e|write|"
    r"highlight|identify|describe|explain|provide|output|determine|state|name|show|compare|decide|"
    r"quote|pick|choose|select|complete|solve|compute|calculate|translate|rewrite|implement|"
    r"tell me|based on|according to|using)\b",
    re.IGNORECASE,
)
_ASK_MAX_WORDS = 150


def _is_ask(block: str) -> bool:
    return _words(block) <= _ASK_MAX_WORDS and bool(_REQUEST.search(block)) and not _CODE_LINE.search(block)


def _unlabelled(text: str) -> Parts:
    blocks = [b for b in re.split(r"\n\s*\n", text.strip()) if b.strip(" \n-=*_#")]
    if len(blocks) > 1:
        # The ask is the request at either end; the document is everything between.
        head = 1 if _is_ask(blocks[0]) else 0
        tail = len(blocks) - 1 if _is_ask(blocks[-1]) else len(blocks)
        if not head and tail == len(blocks):
            # No end reads like a request: a short closing (or opening) block next to much more text is it.
            rest_words = _words(text)
            if _words(blocks[-1]) * 4 <= rest_words and _words(blocks[-1]) <= 60:
                tail = len(blocks) - 1
            elif _words(blocks[0]) * 4 <= rest_words and _words(blocks[0]) <= 60:
                head = 1
        if head and tail < len(blocks) and tail - head < 1:
            # Both ends read like a request and nothing lies between: the shorter one is the ask, unless
            # the longer one ends in its own question (a word problem), which makes it the ask too.
            longer = max(blocks[0], blocks[-1], key=_words)
            if longer.rstrip(" \n'\"").endswith("?"):
                return Parts("", text.strip(), text)
            if _words(blocks[0]) <= _words(blocks[-1]):
                tail = len(blocks)
            else:
                head = 0
        middle = blocks[head:tail]
        if (head or tail < len(blocks)) and middle and _is_prose("\n\n".join(middle)):
            asks = blocks[tail:] or blocks[:head]
            frame = blocks[:head] + blocks[tail:]
            return Parts("\n\n".join(middle).strip(" \n'\"`"), "\n\n".join(asks).strip(), "\n\n".join(frame))
    quoted = [next(g for g in m.groups() if g) for m in _QUOTED.finditer(text)]
    quoted = [q for q in quoted if _is_prose(q)]
    if quoted:
        document = max(quoted, key=_words)
        rest = text.replace(document, " ")
        return Parts(document.strip(), rest.strip(" \n'\":"), rest)
    return Parts("", text.strip(), text)


def split_prompt(text: str) -> Parts:
    """The document to read and the ask about it, for any prompt layout."""
    return _labelled(text) or _unlabelled(text)


# --- Task type ------------------------------------------------------------------------------------

_CODE_REQUEST = re.compile(
    r"\b(write|rewrite|implement|complete|create|finish|fix|debug|refactor|optimi[sz]e|code)\b[^.?!\n]{0,80}"
    r"\b(function|program|script|query|code|class|method|python|sql|javascript)\b",
    re.IGNORECASE,
)
_CODE_BLOCK = re.compile(rf"```|^[ \t]*(def \w+\(|class \w+[(:]|{_SQL})", re.MULTILINE)
_CODE_LANGUAGE = re.compile(r"\b(SQL|Python|JavaScript|TypeScript)\b|\b(the|only|just) code\b")
_SENTIMENT = re.compile(r"\bsentiment\b", re.IGNORECASE)
_OPTION = r"(?:labels?|categor(?:y|ies)|class(?:es)?|options?|intents?|topics?)"
_CLASSIFY = re.compile(
    rf"\b(classif\w+|categori[sz]e\w*|intent labels?)\b|"
    rf"\bwhich (one )?(of the (following |listed )?)?{_OPTION}\b|"
    rf"\bone of the (following |listed )?{_OPTION}\b|"
    rf"\b(pick|choose|select|assign)\b[^.?!\n]{{0,60}}\b{_OPTION}\b",
    re.IGNORECASE,
)
_EXTRACT = re.compile(
    r"\bJSON\b|\b(extract\w*|highlight\w*|verbatim)\b|"
    r"\b(all|every|each|list of)\b[^.?!\n]{0,40}\b(spans|passages|clauses|quotes)\b",
    re.IGNORECASE,
)
_QUANTITY = re.compile(
    r"\bhow (many|much|old|long|far|fast|tall|heavy)\b|"
    r"\bwhat (is|are|was|were|will|would)\b[^?]{0,80}\b(total|sum|difference|average|cost|price|percent\w*|"
    r"number|amount|earnings|profit|area|speed|distance|age|change)\b",
    re.IGNORECASE,
)
_NUMBER_ONLY = re.compile(
    r"\b(numeric answer|number only|(only|just) the (final |numeric )?(number|answer)|final number)\b",
    re.IGNORECASE,
)
_SOLVE = re.compile(
    r"\b(solve|calculate|compute|numeric answer|number only|only the number)\b", re.IGNORECASE
)


def infer_task_type(text: str, parts: Parts | None = None) -> str | None:
    parts = parts or split_prompt(text)
    frame = parts.frame
    if _CODE_BLOCK.search(text) and not parts.document:
        return "coding"
    if _CODE_BLOCK.search(frame) or _CODE_REQUEST.search(frame) or _CODE_LANGUAGE.search(frame):
        return "coding"
    if _SENTIMENT.search(frame) and not parts.document:
        return "misc_sentiment"
    if _CLASSIFY.search(frame):
        return "misc_classification"
    numbers = len(re.findall(r"\d+(?:[.,/]\d+)?", parts.document + " " + parts.ask))
    if parts.document:
        if _EXTRACT.search(frame):
            return "contract_extraction"
        # A short story with numbers and a request for one number back is a word problem, not a lookup.
        word_problem = (
            numbers >= 2 and _words(parts.document) <= _ASK_MAX_WORDS and _NUMBER_ONLY.search(frame)
        )
        return "misc_math" if word_problem else "document_qa"
    if numbers >= 2 and (_SOLVE.search(frame) or _QUANTITY.search(parts.question)):
        return "misc_math"
    return None


_OPTIONS_HEAD = re.compile(
    r"\b(labels?|categories|classes|options|intents|choices)\b[^:\n]{0,40}:[ \t]*(.*)$",
    re.IGNORECASE | re.MULTILINE,
)
_OPTIONS_AS = re.compile(r"\bas ((?:[\w-]+, )+[\w-]+,? or [\w-]+)")


def option_list(text: str) -> list[str]:
    """The labels a classification prompt lets the model choose from, however they are laid out: one per
    line (bare or bulleted) under a heading, comma-separated after it, or "as A, B or C" in a sentence."""
    if m := _OPTIONS_HEAD.search(text):
        inline = [x.strip(" .\"'") for x in m.group(2).split(",") if x.strip(" .\"'")]
        if len(inline) >= 2:
            return inline
        lines = []
        for line in text[m.end() :].lstrip("\n").splitlines():
            item = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", "", line).strip()
            if not item or len(item.split()) > 4:
                break
            lines.append(item)
        if len(lines) >= 2:
            return lines
    if m := _OPTIONS_AS.search(text):
        return [x.strip() for x in re.split(r",\s*(?:or\s+)?|\s+or\s+", m.group(1)) if x.strip()]
    best: list[str] = []  # no heading: the longest run of short bulleted lines
    run: list[str] = []
    for line in [*text.splitlines(), ""]:
        m = re.match(r"\s*(?:[-*\u2022]|\d+[.)])\s+(\S.*)$", line)
        if m and len(m.group(1).split()) <= 4:
            run.append(m.group(1).strip())
            continue
        best, run = (run if len(run) > len(best) else best), []
    return best if len(best) >= 2 else []


def expected_output(text: str, task_type: str) -> str:
    if re.search(r"\bJSON\b", text):
        return "json"
    if task_type == "coding":
        return "code"
    if re.search(
        r"\bnumeric answer\b|\bnumber only\b|\b(only|just) the (final )?number\b", text, re.IGNORECASE
    ):
        return "number"
    if re.search(
        r"\b(only|just) the (label|category|intent|class)\b|\b(label|category|intent|class)( name)? only\b",
        text,
        re.IGNORECASE,
    ):
        return "label"
    return "text"


def build_profile(task: Task) -> Profile:
    text = "\n".join(p for p in (task.system, task.user) if p)
    if task.task_type_hint in TASK_TYPES:
        task_type, source = task.task_type_hint, "provided"
    elif inferred := infer_task_type(text):
        task_type, source = inferred, "inferred"
    else:
        task_type, source = "unknown", "unknown"
    return Profile(
        task_type=task_type, task_type_source=source, expected_output=expected_output(text, task_type)
    )
