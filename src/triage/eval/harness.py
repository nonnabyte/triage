"""Run the dataset through the service, grade every answer, and save it all under results/<run-id>/.

  manifest.json  what ran: version, each tier's model + settings + price, judge, dataset hash, git commit
  calls.jsonl    one line per (item, run): HTTP status, final answer, the full triage block
  grades.jsonl   one line per (item, run): the grade
  summary.json   accuracy, cost, tokens and latency, overall and per use case

Latency comes from run 1 only: runs 2+ repeat the same prompts minutes later, and providers that cache
repeated prompts (Gemini does, automatically) answer those faster than real traffic would get.

Requests go through the HTTP app in-process with eval mode on, so the numbers are the service's own. Only
the prompt is sent; expected answers stay here. Rerunning into the same directory resumes: answered
(item, run) pairs are skipped and failed ones retried. --max-cost stops sending once spend passes it.
"""

from __future__ import annotations

import asyncio
import csv
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from triage import tracing
from triage.api import EVAL_ITEM_HEADER, EVAL_REPEAT_HEADER, EVAL_RUN_HEADER, STRATEGY_HEADER, create_app
from triage.config import REPO_ROOT, Settings, load_models
from triage.eval.graders import CODE_GRADERS, Grade, LLMJudge, grade
from triage.models import ModelClient

# The 300 public prompts plus the synthetic (unseen) ones; see triage.eval.synthetic.
DATASET = REPO_ROOT / "data" / "triage_eval_v3.csv"
RESULTS = REPO_ROOT / "results"
# Not a Gemini model, so it doesn't favour the frontier's answers.
DEFAULT_JUDGE = "bedrock-qwen3-235b"
# A resumed run must match these, or its rows wouldn't be comparable.
FIXED_FIELDS = ("strategy", "backend", "reasoning", "tiers", "judge", "dataset_sha256", "reuse_frontier")
LATENCY_RUN = 1

Key = tuple[str, int]


@dataclass
class RunConfig:
    strategy: str
    out_dir: Path
    settings: Settings
    runs: int = 3
    limit: int | None = None
    max_cost: float | None = None
    concurrency: int = 8
    judge_key: str | None = None
    dataset: Path = DATASET


# --- Files ------------------------------------------------------------------------------------------------


def load_items(path: Path = DATASET) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def sample(items: list[dict[str, str]], limit: int | None) -> list[dict[str, str]]:
    """The first `limit` items, round-robin across use cases and public/synthetic, so a pilot covers all."""
    if limit is None or limit >= len(items):
        return items
    groups: dict[tuple[str, bool], list[dict[str, str]]] = defaultdict(list)
    for item in items:
        groups[(item["use_case"], item.get("source") == "synthetic")].append(item)
    picked: list[dict[str, str]] = []
    while len(picked) < limit:
        for group in groups.values():
            if group and len(picked) < limit:
                picked.append(group.pop(0))
    return picked


def read_jsonl(path: Path) -> dict[Key, dict]:
    """Rows keyed by (id, run); a later line for the same key replaces the earlier one."""
    rows: dict[Key, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[(row["id"], row["run"])] = row
    return rows


def _append(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except OSError:
        return {"commit": None, "dirty": None}


def build_manifest(cfg: RunConfig, n_items: int) -> dict[str, Any]:
    client = ModelClient(cfg.settings, load_models(cfg.settings.models_config))
    return {
        "run_id": cfg.out_dir.name,
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "strategy": cfg.strategy,
        "backend": cfg.settings.backend,
        "reasoning": cfg.settings.reasoning,
        "model": cfg.settings.model,
        "tiers": {
            str(tier): entry.model_dump(
                mode="json", include={"key", "provider", "model", "settings", "price", "reasoning"}
            )
            for tier, entry in client.entries.items()
        },
        "verifier": client.verifier.model_dump(
            mode="json", include={"key", "provider", "model", "settings", "price", "reasoning"}
        )
        if client.verifier
        else None,
        "reuse_frontier": cfg.settings.reuse_frontier.name if cfg.settings.reuse_frontier else None,
        "judge": cfg.judge_key,
        "runs": cfg.runs,
        "items": n_items,
        "dataset": str(cfg.dataset.relative_to(REPO_ROOT)) if cfg.dataset.is_relative_to(REPO_ROOT) else "",
        "dataset_sha256": _sha256(cfg.dataset),
        "git": _git(),
    }


def _check_resume(old: dict, new: dict) -> None:
    changed = [f for f in FIXED_FIELDS if f in old and old.get(f) != new.get(f)]
    if changed:
        raise SystemExit(
            f"{new['run_id']} was run with different {', '.join(changed)}; "
            "use a new --name instead of resuming"
        )


# --- Running ----------------------------------------------------------------------------------------------


def _cost(call_row: dict) -> float:
    triage = call_row.get("triage") or {}
    return (triage.get("totals") or {}).get("cost_usd") or 0.0


def _judge_cost(grade_row: dict) -> float:
    return grade_row.get("cost_usd") or 0.0


async def ask(
    http: httpx.AsyncClient, item: dict[str, str], strategy: str, run_id: str = "", run: int = 1
) -> dict[str, Any]:
    """Send one prompt to the service; return the call row (answer or error, plus the triage block)."""
    body = {"model": "triage", "messages": [{"role": "user", "content": item["prompt"]}]}
    headers = {
        STRATEGY_HEADER: strategy,
        EVAL_RUN_HEADER: run_id,
        EVAL_ITEM_HEADER: item["id"],
        EVAL_REPEAT_HEADER: str(run),
    }
    start = time.perf_counter()
    try:
        r = await http.post("/v1/chat/completions", json=body, headers=headers)
        status, data = r.status_code, r.json()
    except Exception as e:  # the app crashed or the response wasn't JSON
        status, data = 0, {"error": {"message": f"{type(e).__name__}: {e}"[:300]}}
    ok = status == 200
    return {
        "status": status,
        "answer": data["choices"][0]["message"]["content"] if ok else None,
        "error": None if ok else (data.get("error") or {}).get("message", f"HTTP {status}"),
        "triage": data.get("triage"),
        "wall_ms": round((time.perf_counter() - start) * 1000, 1),
    }


async def grade_answer(item: dict[str, str], call_row: dict, judge: LLMJudge | None) -> dict[str, Any]:
    """The grade row. A request that got no answer is graded wrong; a grader crash is recorded, not graded."""
    if call_row["status"] != 200:
        g = Grade(False, 0.0, "no_answer", f"no answer: {call_row['error']}")
        return asdict(g)
    answer = call_row["answer"] or ""
    try:
        if item["check_type"] == "rubric":
            if judge is None:
                return {"error": "rubric item and no judge configured"}
            g = await grade(item, answer, judge)
        else:  # code graders can run a subprocess; keep them off the event loop
            g = await asyncio.to_thread(CODE_GRADERS[item["check_type"]], answer, item["answer"])
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:300]}
    return asdict(g)


def _row_key(item: dict[str, str], run: int) -> dict[str, Any]:
    return {
        "id": item["id"],
        "run": run,
        "use_case": item["use_case"],
        "check_type": item["check_type"],
        "set": "synthetic" if item.get("source") == "synthetic" else "public",
    }


async def run_eval(
    cfg: RunConfig, judge: LLMJudge | None = None, transport: httpx.AsyncBaseTransport | None = None
) -> Path:
    """Run (or resume) the evaluation. Returns the run directory."""
    items = sample(load_items(cfg.dataset), cfg.limit)
    by_id = {item["id"]: item for item in items}
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    calls_path, grades_path, manifest_path = out / "calls.jsonl", out / "grades.jsonl", out / "manifest.json"

    manifest = build_manifest(cfg, len(items))
    if reuse := cfg.settings.reuse_frontier:
        reused = json.loads((reuse / "manifest.json").read_text())
        if reused.get("dataset_sha256") != manifest["dataset_sha256"]:
            raise SystemExit(f"{reuse.name} was run on a different dataset: its answers cannot be reused")
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        _check_resume(old, manifest)
        manifest = {**old, "runs": max(old["runs"], cfg.runs), "items": max(old["items"], len(items))}
        manifest.setdefault("resumed", []).append({"at": manifest["created"], "git": _git()})
    _write_json(manifest_path, manifest)

    calls, grades = read_jsonl(calls_path), read_jsonl(grades_path)
    spent = sum(_cost(r) for r in calls.values()) + sum(_judge_cost(r) for r in grades.values())
    todo = [
        (item, run)
        for run in range(1, cfg.runs + 1)  # run-major: an interrupted run still covers every item once
        for item in items
        if calls.get((item["id"], run), {}).get("status") != 200
    ]
    # Answered but never graded (e.g. the judge was down): grade them without asking the model again.
    ungraded = [
        (by_id[i], run, row)
        for (i, run), row in calls.items()
        if i in by_id and row["status"] == 200 and "error" in grades.get((i, run), {"error": None})
    ]

    progress = {"done": 0, "failed": 0, "spent": spent}
    total = len(todo) + len(ungraded)
    sem = asyncio.Semaphore(cfg.concurrency)
    transport = transport or httpx.ASGITransport(app=create_app(cfg.settings))

    def report() -> None:
        print(
            f"\r[{cfg.strategy}] {progress['done']}/{total} | ${progress['spent']:.4f} spent"
            f" | {progress['failed']} failed",
            end="",
            file=sys.stderr,
            flush=True,
        )

    async def one(http: httpx.AsyncClient, item: dict[str, str], run: int, call_row: dict | None) -> None:
        async with sem:
            if call_row is None:
                if cfg.max_cost is not None and progress["spent"] >= cfg.max_cost:
                    return
                call_row = {**_row_key(item, run), **await ask(http, item, cfg.strategy, out.name, run)}
                _append(calls_path, call_row)
                progress["spent"] += _cost(call_row)
            grade_row = {**_row_key(item, run), **await grade_answer(item, call_row, judge)}
            _append(grades_path, grade_row)
            progress["spent"] += _judge_cost(grade_row)
            progress["done"] += 1
            progress["failed"] += call_row["status"] != 200
            report()

    async with httpx.AsyncClient(transport=transport, base_url="http://triage", timeout=None) as http:
        report()
        await asyncio.gather(
            *(one(http, item, run, None) for item, run in todo),
            *(one(http, item, run, row) for item, run, row in ungraded),
        )
    print(file=sys.stderr)
    tracing.flush()
    if cfg.max_cost is not None and progress["spent"] >= cfg.max_cost:
        print(f"Stopped at --max-cost ${cfg.max_cost}; rerun the same command to continue.", file=sys.stderr)

    summary = summarize(out)
    print(format_summary(summary))
    return out


async def regrade(out: Path, judge: LLMJudge | None, dataset: Path = DATASET) -> dict[str, Any]:
    """Grade every saved answer again (e.g. after a grader fix). No model calls except the judge."""
    by_id = {item["id"]: item for item in load_items(dataset)}
    calls = read_jsonl(out / "calls.jsonl")
    rows = await asyncio.gather(*(grade_answer(by_id[i], row, judge) for (i, _), row in calls.items()))
    tmp = out / "grades.jsonl.tmp"
    tmp.write_text(
        "".join(
            json.dumps({**_row_key(by_id[i], run), **g}) + "\n"
            for ((i, run), _), g in zip(calls.items(), rows, strict=True)
        )
    )
    tmp.replace(out / "grades.jsonl")
    return summarize(out)


# --- Summary ----------------------------------------------------------------------------------------------


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 1)


def _metrics(pairs: Iterable[tuple[dict, dict]]) -> dict[str, Any]:
    pairs = list(pairs)
    graded = [g for _, g in pairs if "error" not in g]
    answered = [c for c, _ in pairs if c["status"] == 200]
    triage = [c["triage"] for c, _ in pairs if c.get("triage")]
    calls = [call for t in triage for call in t["calls"]]
    latency = [
        c["triage"]["totals"]["latency_ms"] for c, _ in pairs if c["run"] == LATENCY_RUN and c.get("triage")
    ]
    n_passed = sum(g["passed"] for g in graded)
    return {
        "n": len(pairs),
        "accuracy": round(n_passed / len(graded), 4) if graded else None,
        "passed": n_passed,
        "graded": len(graded),
        "grading_errors": len(pairs) - len(graded),
        "needs_review": sum(g.get("needs_review", False) for g in graded),
        "failed_requests": len(pairs) - len(answered),
        # Final answers cut off at max_tokens; at high reasoning, thinking can use up the whole budget.
        "truncated": sum(
            1 for t in triage if any(c.get("truncated") for c in t["calls"] if c["purpose"] == "answer")
        ),
        "cost_usd": round(sum(t["totals"]["cost_usd"] for t in triage), 6),
        "cost_per_item_usd": round(sum(t["totals"]["cost_usd"] for t in triage) / len(pairs), 6)
        if pairs
        else 0,
        "cost_complete": all(t["totals"]["cost_complete"] for t in triage),
        "judge_cost_usd": round(sum(_judge_cost(g) for _, g in pairs), 6),
        "input_tokens": sum(c["input_tokens"] for c in calls),
        "output_tokens": sum(c["output_tokens"] for c in calls),
        "latency_p50_ms": _pct(latency, 0.5),
        "latency_p95_ms": _pct(latency, 0.95),
        "final_tier": dict(Counter(t["final_tier"] for t in triage)),
    }


def summarize(out: Path) -> dict[str, Any]:
    calls, grades = read_jsonl(out / "calls.jsonl"), read_jsonl(out / "grades.jsonl")
    pairs = [(c, grades.get(k, {"error": "not graded"})) for k, c in calls.items()]
    by_case: dict[str, list] = defaultdict(list)
    by_set: dict[str, list] = defaultdict(list)
    by_run: dict[int, list] = defaultdict(list)
    for c, g in pairs:
        by_case[c["use_case"]].append((c, g))
        by_set[c.get("set", "public")].append((c, g))
        by_run[c["run"]].append((c, g))
    summary = {
        "run_id": out.name,
        "overall": _metrics(pairs),
        "by_use_case": {case: _metrics(p) for case, p in sorted(by_case.items())},
        # Public (online datasets a model may have trained on) vs synthetic (written for this project).
        "by_set": {name: _metrics(p) for name, p in sorted(by_set.items())},
        "accuracy_by_run": {run: _metrics(p)["accuracy"] for run, p in sorted(by_run.items())},
    }
    _write_json(out / "summary.json", summary)
    return summary


def format_summary(summary: dict[str, Any]) -> str:
    def acc(m: dict) -> str:
        return "  n/a " if m["accuracy"] is None else f"{m['accuracy']:6.1%}"

    rows = [
        ("overall", summary["overall"]),
        *((f"set: {name}", m) for name, m in summary.get("by_set", {}).items()),
        *summary["by_use_case"].items(),
    ]
    width = max(len(name) for name, _ in rows)
    lines = [
        f"{summary['run_id']}",
        f"{'':<{width}}  {'n':>4}  {'acc':>6}  {'cost $':>9}  {'in tok':>9}  {'out tok':>9}  "
        f"{'p50 ms*':>7}  {'p95 ms*':>7}  {'review':>6}  {'failed':>6}",
    ]
    for name, m in rows:
        lines.append(
            f"{name:<{width}}  {m['n']:>4}  {acc(m)}  {m['cost_usd']:>9.4f}  {m['input_tokens']:>9}  "
            f"{m['output_tokens']:>9}  {m['latency_p50_ms'] or 0:>7.0f}  {m['latency_p95_ms'] or 0:>7.0f}  "
            f"{m['needs_review']:>6}  {m['failed_requests']:>6}"
        )
    o = summary["overall"]
    lines.append("* latency from run 1 only (later runs repeat prompts, which providers may cache)")
    lines.append(f"judge cost ${o['judge_cost_usd']:.4f} | accuracy by run {summary['accuracy_by_run']}")
    if o["truncated"]:
        lines.append(f"{o['truncated']} answers cut off at max_tokens (see calls.jsonl: truncated)")
    if not o["cost_complete"]:
        lines.append("some models have no price: cost is a lower bound")
    if o["grading_errors"]:
        lines.append(
            f"{o['grading_errors']} answers not graded; run `triage-eval regrade` after fixing the cause"
        )
    return "\n".join(lines)
