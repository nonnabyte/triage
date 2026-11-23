"""Grade a model's answer against the dataset's expected answer, one grader per check_type.

Code graders (deterministic, free):
  exact_match  SQuAD-style: normalise, then match any accepted answer; also passes when the answer quotes an
               accepted answer with a word or two around it, or most of it. Score = best token F1.
  numeric      Compare the answer's final number with the accepted values; a bolded number wins over a later
               aside.
  json_match   Parse the JSON and compare with an accepted value: key-exact, string case/space-insensitive,
               numbers by value, lists order-insensitive. Score = share of top-level keys that match.
  span_set     JSON array of verbatim spans vs the gold spans, matched word for word (one contains the other).
               Passes when every gold span is found and no extra span is returned. Score = F1. One returned
               span may cover several gold spans; repeating a found span or a duplicated gold span is fine.
               Not CUAD's token-Jaccard >= 0.5 rule: that scores a mirrored clause ("The Franchisor shall be
               entitled to assign" vs "The Franchisee shall not be entitled to assign") 0.84, i.e. correct.
  sql_exec     Run both queries on a fixture SQLite database; compare the result rows.
  unit_tests   Run the code plus the tests in a sandboxed subprocess; passes when every assert holds.
LLM grader:
  rubric       A judge model compares the answer with the reference answer and returns pass / fail.

Every grader returns a Grade. `needs_review` marks answers the code grader failed but that look close
(e.g. a correct number that isn't the last one), so a person or the judge can take a second look.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import string
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Literal

from pydantic import BaseModel, PrivateAttr
from pydantic_ai import Agent

from triage.config import ModelEntry
from triage.eval.sandbox import run_python
from triage.providers import build_model

# span_set: a returned span quoting the gold span may be up to ~2x as long (headings, clause numbers);
# a returned span inside the gold span must cover at least 80% of it.
SPAN_MIN_GOLD_SHARE = 0.5
SPAN_MIN_PRED_COVERAGE = 0.8
SPAN_NEAR_MISS = 0.8
REVIEW_SCORE = 0.5


@dataclass
class Grade:
    passed: bool
    score: float
    grader: str
    detail: str = ""
    needs_review: bool = False
    # Judge spend for rubric grades; None for code graders (free) or an unpriced judge.
    cost_usd: float | None = None


# --- Shared helpers ----------------------------------------------------------------------------------

_FENCE = re.compile(r"```[\w+-]*[ \t]*\n(.*?)```", re.DOTALL)
_ANSWER_PREFIX = re.compile(r"^\s*(final\s+)?answer\s*[:\-]\s*", re.IGNORECASE)
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', " ": " "})


def strip_fences(text: str) -> str:
    """Return the first fenced code block's content if there is one, else the text, trimmed."""
    m = _FENCE.search(text)
    return (m.group(1) if m else text).strip()


def normalize_text(s: str) -> str:
    """SQuAD normalisation: lower case, drop punctuation and articles, collapse whitespace."""
    s = unicodedata.normalize("NFKC", s).translate(_QUOTES).lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_text(pred).split(), normalize_text(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def extract_json(text: str) -> Any:
    """Parse JSON from a model answer: fenced or bare, allowing prose around it."""
    body = strip_fences(text)
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (body.find("["), body.find("{")) if i != -1]
    if not starts:
        raise ValueError("no JSON found")
    value, _ = json.JSONDecoder().raw_decode(body[min(starts) :])
    return value


# --- exact_match --------------------------------------------------------------------------------------


def _quotes_answer(pred: str, gold: str) -> bool:
    """The answer quotes an accepted answer with a word or two around it ("nearly 86 km" for "86 km"), or
    most of it. Same limits as span_set; NOT_IN_DOCUMENT must match exactly."""
    p, g = normalize_text(pred), normalize_text(gold)
    if not p or not g or gold == "NOT_IN_DOCUMENT":
        return False
    if _contains(p, g):
        return len(g.split()) / len(p.split()) >= SPAN_MIN_GOLD_SHARE
    if _contains(g, p):
        return len(p.split()) / len(g.split()) >= SPAN_MIN_PRED_COVERAGE
    return False


def grade_exact_match(output: str, expected: str) -> Grade:
    golds: list[str] = json.loads(expected)
    pred = _ANSWER_PREFIX.sub("", strip_fences(output))
    passed = any(normalize_text(pred) == normalize_text(g) or _quotes_answer(pred, g) for g in golds)
    score = 1.0 if passed else max(token_f1(pred, g) for g in golds)
    review = not passed and score >= REVIEW_SCORE
    return Grade(passed, score, "exact_match", "" if passed else f"best token F1 {score:.2f}", review)


# --- numeric ------------------------------------------------------------------------------------------

_NUMBER = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?|-?\.\d+")
_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)


def extract_numbers(text: str) -> list[float]:
    return [float(n.replace(",", "")) for n in _NUMBER.findall(text)]


def _same_number(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)


def grade_numeric(output: str, expected: str) -> Grade:
    golds = [float(g) for g in json.loads(expected)]
    text = strip_fences(output)
    numbers = extract_numbers(text)
    if not numbers:
        return Grade(False, 0.0, "numeric", "no number in the answer")
    # A bolded number is the answer, over a number in a later aside ("**3 days** ... (exceptions: 30 days)").
    bolded = [n for b in _BOLD.findall(text) for n in extract_numbers(b)]
    final = (bolded or numbers)[-1]
    if any(_same_number(final, g) for g in golds):
        return Grade(True, 1.0, "numeric")
    seen = any(_same_number(n, g) for n in numbers for g in golds)
    detail = f"final number {final:g}" + (" (expected value appears earlier)" if seen else "")
    return Grade(False, 0.0, "numeric", detail, needs_review=seen)


# --- json_match ---------------------------------------------------------------------------------------


def _canon(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        s = unicodedata.normalize("NFKC", value).translate(_QUOTES).casefold()
        return "".join(s.split()).rstrip(".")
    if isinstance(value, dict):
        return {str(k).strip(): _canon(v) for k, v in value.items()}
    if isinstance(value, list):
        return sorted((_canon(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))
    return value


def grade_json_match(output: str, expected: str) -> Grade:
    golds: list[Any] = json.loads(expected)
    try:
        pred = extract_json(output)
    except ValueError as e:
        return Grade(False, 0.0, "json_match", f"answer is not JSON ({e})")
    cp = _canon(pred)
    if any(cp == _canon(g) for g in golds):
        return Grade(True, 1.0, "json_match")
    score = 0.0
    for g in golds:
        cg = _canon(g)
        if isinstance(cg, dict) and isinstance(cp, dict) and cg:
            score = max(score, sum(cp.get(k) == v for k, v in cg.items()) / len(cg))
    review = score >= 0.75
    return Grade(False, score, "json_match", f"{score:.0%} of keys match", review)


# --- span_set -----------------------------------------------------------------------------------------


def _span_words(s: str) -> str:
    """Words only, lower case, single-spaced: ignores case, punctuation and line breaks."""
    s = unicodedata.normalize("NFKC", s).translate(_QUOTES).casefold()
    return " ".join(re.findall(r"\w+", s))


def _contains(outer: str, inner: str) -> bool:
    return f" {inner} " in f" {outer} "


def spans_match(gold: str, pred: str) -> bool:
    """Word-for-word containment, so a mirrored clause ("Franchisor shall" vs "Franchisee shall not") never
    matches, however many words the two share."""
    g, p = _span_words(gold), _span_words(pred)
    if not g or not p:
        return False
    if _contains(p, g):  # the answer quotes the gold span, maybe with a heading or a little extra
        return len(g.split()) / len(p.split()) >= SPAN_MIN_GOLD_SHARE
    if _contains(g, p):  # the answer quotes most of the gold span
        return len(p.split()) / len(g.split()) >= SPAN_MIN_PRED_COVERAGE
    return False


def grade_span_set(output: str, expected: str) -> Grade:
    golds: list[str] = json.loads(expected)
    try:
        pred = extract_json(output)
    except ValueError as e:
        return Grade(False, 0.0, "span_set", f"answer is not a JSON array ({e})")
    if not isinstance(pred, list) or not all(isinstance(p, str) for p in pred):
        return Grade(False, 0.0, "span_set", "answer is not a JSON array of strings")
    pred = [p for p in pred if p.strip()]
    if not golds:
        ok = not pred
        return Grade(ok, float(ok), "span_set", "" if ok else f"expected no spans, got {len(pred)}")
    golds = list({_span_words(g): g for g in golds}.values())  # CUAD lists some spans twice
    unmatched_pred = list(pred)
    unmatched_gold = []
    for gold in golds:
        hit = next((p for p in unmatched_pred if spans_match(gold, p)), None)
        if hit is None:
            unmatched_gold.append(gold)
        else:
            unmatched_pred.remove(hit)
    # One returned span may quote several gold spans that CUAD split, e.g. a sentence and its sub-clause.
    # Checks every returned span: the one-to-one pass above may already have given it to the first of them.
    for p in pred:
        words = _span_words(p)
        inside = [g for g in golds if _contains(words, _span_words(g))]
        if (
            set(inside) & set(unmatched_gold)
            and sum(len(_span_words(g).split()) for g in inside) / len(words.split()) >= SPAN_MIN_GOLD_SHARE
        ):
            if p in unmatched_pred:
                unmatched_pred.remove(p)
            unmatched_gold = [g for g in unmatched_gold if g not in inside]
    # Quoting a found span twice (e.g. with and without its heading) is a repeat, not an extra span.
    found = [g for g in golds if g not in unmatched_gold]
    unmatched_pred = [p for p in unmatched_pred if not any(spans_match(g, p) for g in found)]
    matched = len(golds) - len(unmatched_gold)
    recall = matched / len(golds)
    precision = matched / (matched + len(unmatched_pred)) if matched else 0.0
    f1 = 2 * precision * recall / (precision + recall) if matched else 0.0
    passed = not unmatched_gold and not unmatched_pred
    pairs = [(_span_words(g), _span_words(p)) for g in unmatched_gold for p in unmatched_pred]
    # Right clause, wrong length: quotes the gold span inside much more text, or only a small part of it.
    wrong_length = any(_contains(p, g) or _contains(g, p) for g, p in pairs)
    # A near-verbatim miss is often a mirrored or neighbouring clause.
    near_miss = any(SequenceMatcher(None, g, p).ratio() >= SPAN_NEAR_MISS for g, p in pairs)
    detail = ""
    if not passed:
        detail = f"found {matched}/{len(golds)} gold spans, {len(unmatched_pred)} extra"
        if wrong_length:
            detail += "; a returned span contains or is part of a gold span, but is much longer or shorter"
        if near_miss:
            detail += "; a returned span is close to a gold span but not the same text"
    review = not passed and (wrong_length or near_miss or f1 >= REVIEW_SCORE)
    return Grade(passed, 1.0 if passed else f1, "span_set", detail, review)


# --- sql_exec -----------------------------------------------------------------------------------------

# Fixture databases, chosen by the table the expected query reads. Values avoid ties, so ORDER BY
# results are deterministic, and include the boundary case (price exactly 50).
SQL_FIXTURES: dict[str, str] = {
    "products": """
        CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL);
        INSERT INTO products VALUES
            (1, 'pen', 2.5), (2, 'desk', 120), (3, 'lamp', 50), (4, 'chair', 75.25),
            (5, 'monitor', 210), (6, 'mug', 50.01), (7, 'cable', 9.99);
    """,
}


def _run_sql(script: str, query: str) -> list[tuple]:
    con = sqlite3.connect(":memory:")
    try:
        con.executescript(script)
        con.execute("PRAGMA query_only = ON")
        return con.execute(query.strip().rstrip(";")).fetchall()
    finally:
        con.close()


def grade_sql_exec(output: str, expected: str) -> Grade:
    table = next((t for t in SQL_FIXTURES if re.search(rf"\b{t}\b", expected, re.IGNORECASE)), None)
    if table is None:
        raise ValueError("no SQL fixture for this query; add one to SQL_FIXTURES")
    query = strip_fences(output)
    try:
        got = _run_sql(SQL_FIXTURES[table], query)
    except sqlite3.Error as e:
        return Grade(False, 0.0, "sql_exec", f"query failed: {e}")
    want = _run_sql(SQL_FIXTURES[table], expected)
    if not re.search(r"\border\s+by\b", expected, re.IGNORECASE):
        got, want = sorted(got), sorted(want)
    passed = got == want
    return Grade(
        passed, float(passed), "sql_exec", "" if passed else f"{len(got)} rows vs {len(want)} expected"
    )


# --- unit_tests ---------------------------------------------------------------------------------------


def grade_unit_tests(output: str, expected: str) -> Grade:
    code = strip_fences(output)
    result = run_python(f"{code}\n\n\n{expected}\n")
    detail = "" if result.ok else result.detail
    return Grade(result.ok, float(result.ok), "unit_tests", detail)


# --- rubric: LLM judge --------------------------------------------------------------------------------

JUDGE_INSTRUCTIONS = """\
You grade an answer to a task against a reference answer written by the task's author.

PASS if the answer reaches the same conclusions as the reference and covers every key fact, step or
requirement in it. Extra correct detail is fine. Wording, format and order do not matter.
FAIL if the answer contradicts the reference, misses a key point, or makes a materially wrong claim.
For code, judge whether the code would do what the reference describes.

Judge correctness only, not style or length. Reply with the verdict and a one-sentence reason."""

MAX_TASK_CHARS = 8000


class JudgeVerdict(BaseModel):
    verdict: Literal["pass", "fail"]
    reason: str
    # Set by LLMJudge after the call; not part of the schema the judge model fills in.
    _cost_usd: float | None = PrivateAttr(default=None)


class LLMJudge:
    """Grades rubric items with a model from the catalog.

    Use a model from a different family than the tiers being evaluated: models tend to favour answers
    written by their own family.
    """

    def __init__(self, entry: ModelEntry):
        self.entry = entry
        self.agent = Agent(
            build_model(entry), name="grade-rubric", output_type=JudgeVerdict, instructions=JUDGE_INSTRUCTIONS
        )

    async def __call__(self, task: str, reference: str, answer: str) -> JudgeVerdict:
        if len(task) > MAX_TASK_CHARS:
            task = task[:MAX_TASK_CHARS] + "\n[... task truncated ...]"
        prompt = f"TASK:\n{task}\n\nREFERENCE ANSWER:\n{reference}\n\nANSWER TO GRADE:\n{answer}"
        result = await self.agent.run(prompt, model_settings={"temperature": 0, "max_tokens": 1024})
        verdict = result.output
        usage = result.usage
        verdict._cost_usd = self.entry.price.cost(usage.input_tokens or 0, usage.output_tokens or 0)
        return verdict


async def grade_rubric(output: str, expected: str, task: str, judge: LLMJudge) -> Grade:
    v = await judge(task, expected, output)
    passed = v.verdict == "pass"
    return Grade(passed, float(passed), "rubric", f"{judge.entry.key}: {v.reason}", cost_usd=v._cost_usd)


# --- Dispatch -----------------------------------------------------------------------------------------

CODE_GRADERS: dict[str, Callable[[str, str], Grade]] = {
    "exact_match": grade_exact_match,
    "numeric": grade_numeric,
    "json_match": grade_json_match,
    "span_set": grade_span_set,
    "sql_exec": grade_sql_exec,
    "unit_tests": grade_unit_tests,
}
CHECK_TYPES = (*CODE_GRADERS, "rubric")


async def grade(item: Mapping[str, Any], output: str, judge: LLMJudge | None = None) -> Grade:
    """Grade `output` for a dataset row (needs check_type, answer and prompt)."""
    check_type = item["check_type"]
    if check_type == "rubric":
        if judge is None:
            raise ValueError("rubric items need an LLMJudge")
        return await grade_rubric(output, item["answer"], item["prompt"], judge)
    if check_type not in CODE_GRADERS:
        raise ValueError(f"no grader for check_type '{check_type}'")
    return CODE_GRADERS[check_type](output, item["answer"])
