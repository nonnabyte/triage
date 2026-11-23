# Triage

Routes each task to a cheap model or a frontier model, to cut cost without losing quality.
Four routing versions are compared against always using the frontier model:

| Version | First model | Checks answer + escalates |
|---|---|---|
| `baseline` | always frontier | no |
| `v1` | task-type rules pick cheap or frontier | no |
| `v2` | always cheap | yes |
| `v3` | always cheap | yes: a classifier weighs the answer's signals against the cost of escalating |

Architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Diagrams:
`docs/diagrams/triage-architecture.excalidraw` (open at excalidraw.com).

## Results

300 prompts (document QA, contract extraction, coding, classification, math), 3 runs each. Frontier is
Gemini 4 Argon; cheap is gpt-oss-20b.

| Version | Accuracy | Cost vs frontier | p50 | p90 |
|---|---|---|---|---|
| Frontier only | 86.9% | 1.00 | 3.5 s | 12.4 s |
| Cheap only | 71.2% | 0.04 | 0.7 s | 2.1 s |
| `v1` rules | 84.2% | 0.51 | 3.2 s | 8.1 s |
| `v2` checks | 80.9% | 0.47 | 2.2 s | 8.8 s |
| `v3` classifier | 84.3% | 0.45 | 4.3 s | 11.4 s |

What was measured, how, and the caveats: [`docs/FINDINGS.md`](docs/FINDINGS.md). The tests and every
offline report run without API keys (`TRIAGE_BACKEND=fake`); the saved runs are in `results/`.

## Setup

Needs [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/nonnabyte/triage.git
cd triage
uv sync
cp .env.example .env
```

Put your own keys into `.env` (`GOOGLE_API_KEY`, `GOOGLE_CLOUD_PROJECT`,
`AWS_BEARER_TOKEN_BEDROCK`, `AWS_REGION`, and optionally `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`,
`LANGFUSE_BASE_URL` for tracing). Never commit `.env`.

## Check the cloud models work

```bash
uv run triage-smoke              # every model in configs/models.yaml
uv run triage-smoke bedrock      # only entries whose name contains "bedrock"
```

## Run the service

```bash
uv run uvicorn triage.api:create_app --factory --reload
```

```bash
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model": "triage", "messages": [{"role": "user", "content": "What is 2+2?"}]}'
```

`TRIAGE_BACKEND=fake` (the default) uses scripted models and needs no keys. `TRIAGE_BACKEND=real` calls the
models in the `tiers:` slots of `configs/models.yaml`: frontier is Gemini 4 Argon, cheap is gpt-oss-20b (chosen from
10 candidates, `results/cheap-*`); smart is unset. All four versions run on real models.
`TRIAGE_STRATEGY` picks the version (`baseline`, `v1`, `v2`, `v3`). `TRIAGE_REASONING` (`low`, `medium`,
`high`) fixes the effort of reasoning models such as Argon; it is required when one is in a tier.

## Run an evaluation

```bash
uv run triage-eval run --strategy baseline --reasoning low --limit 30 --runs 1 --name pilot-low  # pilot
uv run triage-eval run --strategy baseline --reasoning high --runs 3 --max-cost 40              # full run
uv run triage-eval run --model bedrock-gpt-oss-20b --reasoning low --runs 3  # one model answers every prompt
uv run triage-eval summary results/<run-id>
uv run triage-eval regrade results/<run-id>     # after a grader change; no model calls
```

Runs `data/triage_eval_v3.csv` by default (`--dataset` for another file). Results go to
`results/<run-id>/`: `manifest.json`, `calls.jsonl`, `grades.jsonl`, `summary.json`. The summary reports
accuracy, cost, tokens and latency overall, per use case and for public vs synthetic prompts. Latency comes
from run 1 only, since later runs repeat prompts that providers may cache. Cost is tokens × the prices in
`configs/models.yaml`, ignoring provider cache discounts. `--model <catalog key>` skips routing and has that one model answer every prompt (real models,
baseline strategy): use it to measure cheap and smart candidates. Rerun with the same `--name` to resume.
`results/` is git-ignored; commit the baseline with `git add -f`.

## Tracing (Langfuse)

Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_BASE_URL` in `.env` and every request is
traced: a `route-request` trace (user message, final answer, routing decision and cost in metadata) with each
model call as a generation (model, tokens, cost) and each check as a `check-answer` guardrail. Eval runs are
tagged environment `eval` and grouped into one session per run. `TRIAGE_TRACE_CONTENT=false` keeps prompts
and answers out of Langfuse; `TRIAGE_TRACING=false` turns it off. Tests never trace.

## Tests and lint

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Tests use the fake models, so they need no keys. CI runs the same commands on every PR.

## Layout

```
docs/FINDINGS.md          results and what we learned from them
docs/ARCHITECTURE.md      how the service is built
configs/models.yaml       model catalog + which model fills cheap / smart / frontier
configs/strategies.yaml   the four versions
data/triage_eval_v3.csv   the eval set: 260 public + 40 synthetic prompts (build_dataset.py)
data/triage_eval_v2.csv   the original 300 public prompts
data/triage_synthetic_v1.csv       the 40 synthetic prompts (synthetic.py)
data/triage_eval_v3_dropped.txt    the 40 public prompts left out of v3
src/triage/
  api.py                  POST /v1/chat/completions (OpenAI format), GET /healthz
  schemas.py              request / response models
  config.py               settings and YAML loading
  providers.py            Vertex Gemini, Vertex open models, Bedrock, fake
  models.py               calls a tier: timeout, retries (temporary errors only), tokens, cost
  profile.py              reads the task from any prompt layout: document, ask, type, expected output
  cascade.py              decide → call → check → escalate
  policies.py             first-tier choice; v1 task-type rules and its cost rule
  checks.py               failure checks: non_empty, refusal, json_format, grounding, span_count, examples,
                          agreement, cross_agreement
  escalation.py           v3: logistic model over the cheap answer's signals, escalates when worth its cost
  features.py             prompt signals for the v1 rules
  sandbox.py              runs model-written code with limits (graders and checks)
  tracing.py              Langfuse tracing
  smoke.py                triage-smoke command
  eval/                   evaluation only; the service never imports it
    graders.py            one grader per check_type in the data
    train_classifier.py   trains the v3 classifier: uv run python -m triage.eval.train_classifier
    offline.py            replays a routing decision on saved answers (no model calls)
    rules_report.py       scores v1: eval set, shuffled control, re-worded prompts, external set
    checks_report.py      scores candidate v2 check lists on saved answers
    retemplate.py         re-words eval prompts into other layouts (template-independence test)
    external.py           builds data/triage_external_v1.csv from five other public datasets
    harness.py, cli.py    triage-eval: run a version over the dataset, grade, save results/
    synthetic.py          generates the synthetic (unseen) prompts
    build_dataset.py      builds triage_eval_v3.csv: public minus 40, plus synthetic
tests/
```

Writing a check: `async def my_check(ctx: CheckContext) -> CheckResult`, add it to `CHECKS` in
`checks.py`, then list it under a version's `checks:` in `configs/strategies.yaml`. A check that needs a
model calls `await ctx.ask_model(prompt)`; that call is logged as a judge call and counted as overhead.

## Graders

`triage.eval.graders.grade(item, output, judge)` grades one answer for a dataset row:

| check_type | How |
|---|---|
| `exact_match` | SQuAD normalisation, match any accepted answer, or quote one with a word or two around it |
| `numeric` | the answer's final number (or last bolded number) vs the accepted values |
| `json_match` | parsed JSON vs the accepted value: exact keys, case/space-insensitive strings, unordered lists |
| `span_set` | every gold span quoted word for word (one quote may cover several), no extra spans |
| `sql_exec` | both queries run on a fixture SQLite database, rows compared |
| `unit_tests` | code + tests run in a sandboxed subprocess (no network block: use a container for untrusted code) |
| `rubric` | an `LLMJudge` compares the answer with the reference; use a model family not being evaluated |

Each returns `passed`, a 0–1 `score` and `needs_review` for close misses worth a human look.

## Not built yet

the self_report / judge checks, a smart tier, the comparison report.

## Data sources

The evaluation prompts in `data/` are drawn from public datasets, each under its own licence: SQuAD 2.0,
CUAD, HumanEval, MBPP, Banking77 and GSM8K (eval set), and HotpotQA, ContractNLI, LeetCodeDataset, SVAMP and
CLINC150 (external test set). The eval set also has hand-written and synthetic prompts made for this
project.
