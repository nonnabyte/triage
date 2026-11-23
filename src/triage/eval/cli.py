"""triage-eval: run a version over the dataset, regrade saved answers, or print a run's summary.

triage-eval run --strategy baseline --runs 3 [--limit 20] [--max-cost 25] [--name NAME]
triage-eval regrade results/<run-id>
triage-eval summary results/<run-id>
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
from pathlib import Path

from triage.config import Settings, load_env, load_models
from triage.eval.graders import LLMJudge
from triage.eval.harness import (
    DATASET,
    DEFAULT_JUDGE,
    RESULTS,
    RunConfig,
    format_summary,
    regrade,
    run_eval,
    summarize,
)
from triage.schemas import STRATEGIES

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")


def _judge(key: str | None, settings: Settings) -> LLMJudge | None:
    if key in (None, "none"):
        return None
    catalog = load_models(settings.models_config).catalog
    if key not in catalog:
        raise SystemExit(f"--judge: '{key}' is not in configs/models.yaml")
    return LLMJudge(catalog[key])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="triage-eval", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="send the dataset through the service, grade, save under results/")
    run.add_argument("--strategy", choices=STRATEGIES, help="required unless --model is given")
    run.add_argument(
        "--model", help="catalog key that answers every prompt directly, no routing (implies baseline, real)"
    )
    run.add_argument("--runs", type=int, default=3, help="answers per prompt (default 3)")
    run.add_argument("--limit", type=int, help="only this many prompts, spread across use cases (pilot)")
    run.add_argument("--max-cost", type=float, help="stop sending once spend (USD) passes this")
    run.add_argument(
        "--reuse-frontier",
        metavar="RUN",
        help="results/<RUN> of the frontier baseline: when a router sends a prompt to frontier, reuse that "
        "run's saved answer (same prompt and repeat, original cost and latency) instead of calling again",
    )
    run.add_argument("--concurrency", type=int, default=8)
    run.add_argument("--name", help="run directory name; reuse a name to resume (default <strategy>-<time>)")
    run.add_argument("--backend", choices=["fake", "real"], help="default: TRIAGE_BACKEND")
    run.add_argument(
        "--reasoning",
        choices=["low", "medium", "high"],
        help="effort for every reasoning model (TRIAGE_REASONING)",
    )
    run.add_argument(
        "--dataset", type=Path, default=DATASET, help="CSV of prompts (default: public + synthetic)"
    )
    run.add_argument(
        "--judge", help=f"catalog key for rubric items, or none (default {DEFAULT_JUDGE} on real)"
    )

    rg = sub.add_parser("regrade", help="grade a run's saved answers again; no model calls except the judge")
    rg.add_argument("run_dir", type=Path)
    rg.add_argument("--judge", help="default: the judge in the run's manifest")

    sm = sub.add_parser("summary", help="recompute and print a run's summary")
    sm.add_argument("run_dir", type=Path)

    args = parser.parse_args(argv)
    load_env()

    if args.command == "run":
        if args.model:
            if args.strategy not in (None, "baseline") or args.backend == "fake":
                parser.error(
                    "--model runs one real model on every prompt: use it without --strategy or --backend"
                )
            args.strategy, args.backend = "baseline", "real"
        elif not args.strategy:
            parser.error("give --strategy or --model")
        overrides = {"eval_mode": True, "strategy": args.strategy, "model": args.model}
        if args.backend:
            overrides["backend"] = args.backend
        if args.reasoning:
            overrides["reasoning"] = args.reasoning
        if args.reuse_frontier:
            reuse = RESULTS / args.reuse_frontier
            if not (reuse / "calls.jsonl").exists():
                parser.error(f"--reuse-frontier: {reuse} has no calls.jsonl")
            overrides["reuse_frontier"] = reuse
        settings = Settings(**overrides)
        judge_key = args.judge or (DEFAULT_JUDGE if settings.backend == "real" else "none")
        level = f"-{settings.reasoning}" if settings.reasoning else ""
        name = args.name or f"{args.model or args.strategy}{level}-{dt.datetime.now():%Y%m%d-%H%M%S}"
        cfg = RunConfig(
            strategy=args.strategy,
            out_dir=RESULTS / name,
            settings=settings,
            runs=args.runs,
            limit=args.limit,
            max_cost=args.max_cost,
            concurrency=args.concurrency,
            judge_key=None if judge_key == "none" else judge_key,
            dataset=args.dataset.resolve(),
        )
        out = asyncio.run(run_eval(cfg, judge=_judge(judge_key, settings)))
        print(f"\nSaved to {out}")
    elif args.command == "regrade":
        manifest = json.loads((args.run_dir / "manifest.json").read_text())
        judge = _judge(args.judge or manifest.get("judge"), Settings())
        print(format_summary(asyncio.run(regrade(args.run_dir, judge))))
    else:
        print(format_summary(summarize(args.run_dir)))


if __name__ == "__main__":
    main()
