"""Failure checks: judge an answer without an answer key. A failed check escalates (v2, v3 only).

A check is `async def check(ctx: CheckContext) -> CheckResult`. Register it in CHECKS and list it under a
version's `checks:` in configs/strategies.yaml, for every task or per task type. A check that needs a model
calls `ctx.ask_model(...)`; that call is logged as purpose="judge" and counted as routing overhead.

A check that does not apply to the task (e.g. `examples` on a prompt with no examples) passes.
The checks, cheapest first. Each was measured offline on saved answers before it was built
(uv run python -m triage.eval.checks_report):
  non_empty        the answer is not empty
  refusal          the answer is not a refusal ("I'm sorry, I can't ...")
  json_format      the answer has the requested shape: JSON, a listed label, one number, Python that
                   parses, SQL that parses
  grounding        every extracted span appears in the document
  span_count       a span-list extraction returns at most one span; the cheap model is rarely right when
                   it returns several
  examples         the code passes the prompt's own worked examples
  agreement        a second answer from the same model must match (self-consistency)
  cross_agreement  a second answer from a different cheap model must match; two models repeat each
                   other's mistakes less often than one model repeats its own
  escalation_model v3: a classifier weighs all of the above against the cost of escalating
                   (triage.escalation)
A QA "answer appears in the passage" check was dropped: it escalated 11% of correct answers.
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
import sqlite3
import string
import unicodedata
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from triage.config import ClassifierConfig
from triage.features import wants_span_list
from triage.profile import option_list
from triage.sandbox import run_python
from triage.schemas import Call, CheckOutcome, CheckResult, Profile, Task, Tier

if TYPE_CHECKING:
    from triage.models import ModelClient


@dataclass
class CheckContext:
    task: Task
    profile: Profile
    output: str
    client: ModelClient
    next_seq: Callable[[], int]
    # The tier that produced `output`, so a check can ask the same model again.
    tier: Tier = Tier.cheap
    # v3's classifier settings (configs/strategies.yaml `classifier`), for the escalation_model check.
    classifier: ClassifierConfig | None = None
    # Model calls made by checks, in order. The cascade adds them to the response's call log.
    extra_calls: list[Call] = field(default_factory=list)

    async def ask_model(self, prompt: str, system: str | None = None, tier: Tier = Tier.cheap) -> Call:
        call = await self.client.complete(
            tier, Task(user=prompt, system=system), seq=self.next_seq(), purpose="judge"
        )
        self.extra_calls.append(call)
        return call

    async def ask_verifier(self, prompt: str, system: str | None = None) -> Call:
        """Ask the second-opinion model (configs/models.yaml `verifier`); the same tier when none is set."""
        call = await self.client.complete(
            self.tier, Task(user=prompt, system=system), seq=self.next_seq(), purpose="judge", verifier=True
        )
        self.extra_calls.append(call)
        return call


CheckFn = Callable[[CheckContext], Awaitable[CheckResult]]


# --- Reading the prompt and the answer ----------------------------------------------------------------

_FENCE = re.compile(r"```[\w+-]*[ \t]*\n(.*?)```", re.DOTALL)
_NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def _unfence(text: str) -> str:
    m = _FENCE.search(text)
    return (m.group(1) if m else text).strip()


def _words(text: str) -> str:
    """Lower-case words only, space-padded, so `in` tests whole-word containment."""
    return " " + " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())) + " "


def _normal(text: str) -> str:
    """SQuAD-style normalisation, for comparing two short answers."""
    s = unicodedata.normalize("NFKC", text).lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _json(text: str) -> Any:
    body = _unfence(text)
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        starts = [i for i in (body.find("["), body.find("{")) if i != -1]
        if not starts:
            raise ValueError("no JSON found") from None
        return json.JSONDecoder().raw_decode(body[min(starts) :])[0]


def _labels(prompt: str) -> list[str]:
    return option_list(prompt)


def _label(text: str) -> str:
    return _unfence(text).strip(" \n\"'.`").casefold()


def _wants_span_list(prompt: str) -> bool:
    return wants_span_list(prompt)


def _function_name(prompt: str) -> str | None:
    names = re.findall(r"def (\w+)\(", prompt)
    return names[-1] if names else None


def _examples(prompt: str) -> list[str]:
    """The prompt's own worked examples as assert statements: `f(x) == y` lines and `>>> f(x)` + result."""
    lines = prompt.splitlines()
    out = []
    for k, line in enumerate(lines):
        if m := re.match(r"\s*(\w+\(.*\))\s*==\s*(.+?)\s*$", line):
            out.append(f"assert ({m.group(1)}) == ({m.group(2)}), {m.group(1)!r}")
        elif (m := re.match(r"\s*>>>\s*(.+)$", line)) and k + 1 < len(lines):
            result = lines[k + 1].strip()
            if result and not result.startswith((">>>", '"""', "'''")):
                out.append(f"assert ({m.group(1)}) == ({result}), {m.group(1)!r}")
    return out


def _is_sql(prompt: str) -> bool:
    return bool(re.search(r"\bSQL\b", prompt))


# --- Checks -------------------------------------------------------------------------------------------


async def non_empty(ctx: CheckContext) -> CheckResult:
    ok = bool(ctx.output.strip())
    return CheckResult(name="non_empty", passed=ok, detail="" if ok else "empty answer")


async def json_format(ctx: CheckContext) -> CheckResult:
    """The answer has the shape the task asks for: JSON (a span list or the requested keys), one of the
    listed labels, a single number, or Python that parses and defines the requested function."""
    name, prompt, out, kind = "json_format", ctx.task.user, ctx.output, ctx.profile.expected_output
    if kind == "json":
        try:
            value = _json(out)
        except ValueError:
            return CheckResult(name=name, passed=False, detail="answer is not JSON")
        if _wants_span_list(prompt):
            ok = isinstance(value, list) and all(isinstance(v, str) for v in value)
            return CheckResult(name=name, passed=ok, detail="" if ok else "not a JSON array of strings")
        keys = list(dict.fromkeys(re.findall(r'"(\w+)"\s*:', prompt)))
        if not isinstance(value, dict):
            return CheckResult(name=name, passed=False, detail="not a JSON object")
        missing = [k for k in keys if k not in value]
        return CheckResult(name=name, passed=not missing, detail=f"missing keys {missing}" if missing else "")
    if kind == "label" and (labels := _labels(prompt)):
        ok = _label(out) in {_label(x) for x in labels}
        return CheckResult(name=name, passed=ok, detail="" if ok else "not one of the listed labels")
    if kind == "number":
        ok = bool(re.fullmatch(r"\s*" + _NUMBER.pattern + r"\s*", _unfence(out)))
        return CheckResult(name=name, passed=ok, detail="" if ok else "not a single number")
    if kind == "code" and _is_sql(prompt):
        error = _sql_syntax_error(_unfence(out))
        return CheckResult(
            name=name, passed=error is None, detail=f"SQL does not parse: {error}" if error else ""
        )
    if kind == "code":
        try:
            tree = ast.parse(_unfence(out))
        except SyntaxError as e:
            return CheckResult(name=name, passed=False, detail=f"code does not parse: {e.msg}")
        fn = _function_name(prompt)
        defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)}
        if fn and fn not in defined:
            return CheckResult(name=name, passed=False, detail=f"function {fn} is not defined")
    return CheckResult(name=name, passed=True)


def _sql_syntax_error(query: str) -> str | None:
    """A syntax error in the SQL, or None. Unknown tables and columns are not errors: the schema is not
    known here, only the grammar (SQLite's, which rejects little that other dialects accept)."""
    statements = [q.strip() for q in query.split(";") if q.strip()]
    if not statements:
        return "no statement"
    conn = sqlite3.connect(":memory:")
    try:
        for statement in statements:
            try:
                conn.execute(f"EXPLAIN {statement}")
            except sqlite3.Error as e:
                if "syntax error" in str(e) or "incomplete input" in str(e):
                    return str(e)
    finally:
        conn.close()
    return None


_REFUSAL = re.compile(
    r"^\W*(i['\u2019]?m sorry|i am sorry|sorry,|i apologi[sz]e|as an ai\b|"
    r"i can(?:no|['\u2019])t (?:help|assist|answer|provide|comply|do that|fulfil)|"
    r"i(?:['\u2019]m| am) (?:unable|not able) to|i (?:won['\u2019]t|will not) be able to)",
    re.IGNORECASE,
)


async def refusal(ctx: CheckContext) -> CheckResult:
    """The model declined to answer. Only an answer that opens with a refusal counts, so an answer that
    merely contains "cannot" (a contract clause, a passage quote) passes."""
    refused = bool(_REFUSAL.match(_unfence(ctx.output)))
    return CheckResult(name="refusal", passed=not refused, detail="the model refused" if refused else "")


MAX_SPANS = 1


async def span_count(ctx: CheckContext) -> CheckResult:
    """A span-list extraction that returns several spans is escalated: the cheap model over-collects, and
    on saved answers it was right on 58% of answers with at most one span and 11% of those with more."""
    if ctx.profile.expected_output != "json" or not _wants_span_list(ctx.task.user):
        return CheckResult(name="span_count", passed=True)
    try:
        spans = _json(ctx.output)
    except ValueError:
        return CheckResult(name="span_count", passed=False, detail="answer is not JSON")
    n = len(spans) if isinstance(spans, list) else 0
    return CheckResult(
        name="span_count", passed=n <= MAX_SPANS, detail="" if n <= MAX_SPANS else f"{n} spans returned"
    )


async def grounding(ctx: CheckContext) -> CheckResult:
    """Every span in a contract-extraction answer must appear word for word in the contract."""
    if ctx.profile.expected_output != "json" or not _wants_span_list(ctx.task.user):
        return CheckResult(name="grounding", passed=True)
    try:
        spans = _json(ctx.output)
    except ValueError:
        return CheckResult(name="grounding", passed=False, detail="answer is not JSON")
    document = _words(ctx.task.user)
    made_up = [s for s in spans if isinstance(s, str) and _words(s).strip() and _words(s) not in document]
    return CheckResult(
        name="grounding",
        passed=not made_up,
        detail=f"{len(made_up)} span(s) not in the contract" if made_up else "",
    )


async def examples(ctx: CheckContext) -> CheckResult:
    """Run the code against the worked examples in the prompt (sandboxed, with a timeout)."""
    tests = _examples(ctx.task.user) if ctx.profile.expected_output == "code" else []
    if not tests:
        return CheckResult(name="examples", passed=True)
    result = await asyncio.to_thread(run_python, _unfence(ctx.output) + "\n\n" + "\n".join(tests))
    return CheckResult(name="examples", passed=result.ok, detail="" if result.ok else result.detail)


_PROBES = [0, 1, 2, -3, 10, 1.5, "", "abc", "Hello World", [], [1, 2, 3], [3, 1, 2, 3], (1, 2), {"a": 1}]

_DIFF_HARNESS = """
def _run(src, name, probes, calls):
    ns = {{}}
    try:
        exec(src, ns)
    except Exception:
        return None
    f = ns.get(name)
    if f is None:
        return None
    out = []
    for args in probes:
        try:
            out.append(repr(f(*args)))
        except Exception:
            out.append("ERR")
    for call in calls:
        try:
            out.append(repr(eval(call, ns)))
        except Exception:
            out.append("ERR")
    return out

a = _run({a!r}, {name!r}, {probes!r}, {calls!r})
b = _run({b!r}, {name!r}, {probes!r}, {calls!r})
assert a is not None and b is not None, "one answer does not run"
both = [(x, y) for x, y in zip(a, b) if x != "ERR" and y != "ERR"]
assert both, "no input both answers accept"
assert all(x == y for x, y in both), "the two answers behave differently"
"""


def _same_code(prompt: str, first: str, second: str) -> tuple[bool, str]:
    """Run both programs on the same probe inputs (plus the prompt's example calls) and compare outputs.
    Inputs that make either one raise are ignored, so a type mismatch is not a disagreement."""
    name = _function_name(prompt)
    if not name:
        return True, ""
    m = re.search(r"def " + name + r"\(([^)]*)\)", prompt)
    argc = len([a for a in m.group(1).split(",") if a.strip() and "=" not in a]) if m else 1
    probes = [[p] * argc for p in _PROBES]
    calls = [re.match(r"assert \((.*?)\) ==", t).group(1) for t in _examples(prompt)]
    code = _DIFF_HARNESS.format(a=_unfence(first), b=_unfence(second), name=name, probes=probes, calls=calls)
    result = run_python(code)
    return result.ok, result.detail


def _span_covers(a: str, b: str) -> bool:
    """One span quotes the other: the shorter is inside the longer and is at least half of it."""
    x, y = sorted((_words(a).strip(), _words(b).strip()), key=len)
    return bool(x) and f" {x} " in f" {y} " and len(x.split()) * 2 >= len(y.split())


def _same_spans(a: list[Any], b: list[Any]) -> bool:
    """Two span lists point at the same text: each span on one side is matched on the other, allowing
    one answer to quote a little more or less of the same clause."""
    a, b = [str(x) for x in a if str(x).strip()], [str(x) for x in b if str(x).strip()]
    if not a or not b:
        return not a and not b
    return all(any(_span_covers(x, y) for y in b) for x in a) and all(
        any(_span_covers(x, y) for x in a) for y in b
    )


def _same_answer(ctx: CheckContext, other: str) -> tuple[bool, str]:
    kind, prompt = ctx.profile.expected_output, ctx.task.user
    if kind == "code":
        return (True, "") if _is_sql(prompt) else _same_code(prompt, ctx.output, other)
    if kind == "json":
        try:
            a, b = _json(ctx.output), _json(other)
        except ValueError:
            return False, "an answer is not JSON"
        if isinstance(a, list) and isinstance(b, list):
            same = _same_spans(a, b)
        else:
            same = json.dumps(a, sort_keys=True).lower() == json.dumps(b, sort_keys=True).lower()
        return same, "" if same else "the two answers extract different text"
    if kind == "number":
        a, b = _NUMBER.findall(ctx.output), _NUMBER.findall(other)
        same = bool(a and b) and a[-1].replace(",", "").replace("$", "") == b[-1].replace(",", "").replace(
            "$", ""
        )
        return same, "" if same else "the two answers give different numbers"
    same = _normal(_unfence(ctx.output)) == _normal(_unfence(other))
    return same, "" if same else "the two answers differ"


async def _second_opinion(ctx: CheckContext, name: str, second: Call) -> CheckResult:
    if second.error is not None or not (second.output or "").strip():
        # Not compared, so not a pass on the merits: the detail lets the run count how often this happens.
        return CheckResult(name=name, passed=True, detail="second answer unavailable; not compared")
    same, detail = await asyncio.to_thread(_same_answer, ctx, second.output)
    return CheckResult(name=name, passed=same, detail=detail)


def _open_ended(ctx: CheckContext) -> bool:
    return ctx.profile.expected_output == "text" and ctx.profile.task_type == "unknown"


async def agreement(ctx: CheckContext) -> CheckResult:
    """Self-consistency: ask the same model again; escalate when the two answers disagree.
    Open-ended text (no fixed format) is not compared."""
    if _open_ended(ctx):
        return CheckResult(name="agreement", passed=True)
    second = await ctx.ask_model(ctx.task.user, system=ctx.task.system, tier=ctx.tier)
    return await _second_opinion(ctx, "agreement", second)


async def cross_agreement(ctx: CheckContext) -> CheckResult:
    """Ask a different cheap model the same question; escalate when the two answers disagree. Two models
    share fewer mistakes than one model asked twice. Open-ended text is not compared."""
    if _open_ended(ctx):
        return CheckResult(name="cross_agreement", passed=True)
    second = await ctx.ask_verifier(ctx.task.user, system=ctx.task.system)
    return await _second_opinion(ctx, "cross_agreement", second)


async def self_report(ctx: CheckContext) -> CheckResult:
    """The model says it cannot answer, or gives a confidence below a threshold.

    This changes the answering prompt (it asks for a confidence), so evaluate it as its own variant.
    """
    raise NotImplementedError("self_report check is not implemented yet (src/triage/checks.py)")


async def judge(ctx: CheckContext) -> CheckResult:
    """A second cheap call grades the answer against the question, via ctx.ask_model(...)."""
    raise NotImplementedError("judge check is not implemented yet (src/triage/checks.py)")


CHECKS: dict[str, CheckFn] = {
    "non_empty": non_empty,
    "refusal": refusal,
    "json_format": json_format,
    "grounding": grounding,
    "span_count": span_count,
    "examples": examples,
    "agreement": agreement,
    "cross_agreement": cross_agreement,
    "self_report": self_report,
    "judge": judge,
}


async def run_checks(names: list[str], ctx: CheckContext) -> CheckOutcome:
    """Run checks in order and stop at the first failure, so a model-calling check is skipped when a free
    one has already failed."""
    results = []
    for name in names:
        results.append(await CHECKS[name](ctx))
        if not results[-1].passed:
            break
    return CheckOutcome(passed=all(r.passed for r in results), results=results)
