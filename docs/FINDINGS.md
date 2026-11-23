# Findings

What we measured, what we learned, and what is still open. Every number comes from `results/` (each
run's `summary.json`, `calls.jsonl`, `grades.jsonl`) unless marked *offline*, which means it was estimated
by routing on saved answers instead of a live run.

Setup for every number below: `data/triage_eval_v3.csv` (300 prompts: 90 coding, 90 contract extraction,
90 document QA, 14 classification, 14 math, 1 planning, 1 sentiment), each run 3 times (900 answers),
graded by `triage.eval.graders`. Frontier is Gemini 4 Argon on Vertex AI; cheap is gpt-oss-20b on Bedrock.

## Key findings

1. **v1 and v3 both reach 84% accuracy (Argon 86.9%) at about half of Argon's cost.** v1 does it with
   task-type rules and no model call; v3 with a classifier over the cheap answer. v2 costs about the same
   and reaches 80.9%.
2. **The saving is a dial, and contracts set it.** Contracts are 78% of Argon's bill and the cheap model
   gets about half of them wrong. Every router's saving and loss comes from how many contracts it leaves
   on cheap. The first versions left most of them there (74 to 81% saved, 7 to 9 points lost).
3. **No signal available before the call separates easy contracts from hard ones.** Surface features,
   embeddings and nearest neighbours all reach an AUC of about 0.62 on held-out documents. Signals from the
   cheap answer (agreement, span count, check results) reach 0.78.
4. **Rules finer than task type do not generalise.** Question-word and overlap rules beat a shuffled control
   on the eval set and did not on an external test set, so v1 routes on task type alone.
5. **The checks catch broken answers, not plausible wrong ones.** On 900 cheap answers there were 0 empty
   answers, 0 refusals and 3 unparseable JSON answers against 259 wrong answers.
6. **High reasoning is not worth it.** Argon at high reasoning gains 1.7 points (88.6%) for 3 times the
   cost ($10.85 vs $3.67) and 4 times the median latency.
7. **Even a perfect router has little room.** Choosing the right model for every prompt with hindsight
   reaches 88.4% at 62% saved (*offline*).

## 1. Baseline: Argon only

| Reasoning | Accuracy | Cost | Median latency | p95 latency |
|---|---|---|---|---|
| low | 86.9% | $3.67 | 3.5 s | 24.7 s |
| high | 88.6% | $10.85 | 14.9 s | 55.0 s |

Low reasoning is the baseline for everything below. By task type (low): coding 97.0%, contracts 66.7%,
QA 94.1%, classification 92.9%, math 100%. Contracts cost $2.87 of the $3.67.

## 2. Choosing the cheap model

Eleven open models on Bedrock, each on all 900 answers (`triage-eval run --model KEY`, `results/cheap-*`):

| Model | Accuracy | Cost | p95 latency | Cut off |
|---|---|---|---|---|
| gpt-oss-120b (high) | 74.2% | $0.95 | 24.5 s | 9 |
| gpt-oss-20b (high) | 73.4% | $0.65 | 30.2 s | 51 |
| **gpt-oss-20b (low)** | **71.2%** | **$0.15** | **2.6 s** | **0** |
| gpt-oss-120b (low) | 70.8% | $0.32 | 4.4 s | 0 |
| Gemma 3 27B | 64.8% | $0.37 | 6.8 s | 0 |
| Gemma 3 12B | 59.2% | $0.16 | 7.3 s | 1 |
| Qwen3 32B | 62.2% | $0.27 | 3.1 s | 0 |
| Qwen3 Coder 30B | 61.8% | $0.27 | 2.4 s | 0 |
| Ministral 3 14B | 55.3% | $0.32 | 4.8 s | 6 |
| Nemotron Nano 3 30B | 52.9% | $0.12 | 3.7 s | 1 |
| GLM 4.7 Flash | 52.0% | $0.14 | 2.7 s | 1 |
| Gemma 3 4B | 46.2% | $0.07 | 4.6 s | 4 |
| Ministral 3 3B | 37.9% | $0.17 | 4.2 s | 25 |

**gpt-oss-20b at low reasoning** is the cheap tier: within 3 points of the best for a quarter of its cost,
the fastest, and it never truncates. The 120b is no more accurate at low reasoning and costs twice as much.

## 3. Results

Live runs `v1-low-2`, `v2-low-2`, `v3-low-2`, made from commit `8d12335`. When a router sends a prompt to
Argon it reuses the baseline's saved answer for that prompt and repeat, with its original cost and latency
(`--reuse-frontier baseline-low`), so each router is compared with Argon on the same Argon answers.

| Version | Accuracy | vs Argon | Cost vs Argon | Cost per correct vs Argon | p50 | p90 | Escalated | Hard sent to cheap | Easy sent to frontier |
|---|---|---|---|---|---|---|---|---|---|
| Argon only | 86.9% | — | 1.00 | 1.00 | 3.5 s | 12.4 s | | | |
| Cheap only | 71.2% | −15.7 | 0.04 | 0.05 | 0.7 s | 2.1 s | | | |
| v1 rules | 84.2% | −2.7 | 0.51 | 0.53 | 3.2 s | 8.1 s | none | 102 of 360 | 423 of 540 |
| v2 checks | 80.9% | −6.0 | 0.47 | 0.51 | 2.2 s | 8.8 s | 291 of 900 | 170 of 360 | 101 of 540 |
| v3 classifier | 84.3% | −2.6 | 0.45 | 0.46 | 4.3 s | 11.4 s | 692 of 900 | 104 of 360 | 436 of 540 |

A prompt is hard when the cheap model did not pass all 3 of its runs (120 of 300). Latency is run 1.

By task type and by source (accuracy):

| | Coding | Contracts | QA | Classification | Math | Public | Synthetic |
|---|---|---|---|---|---|---|---|
| Argon | 97.0 | 66.7 | 94.1 | 92.9 | 100 | 85.0 | 99.2 |
| Cheap | 74.8 | 48.9 | 82.2 | 88.1 | 100 | 68.2 | 90.8 |
| v1 | 97.0 | 59.3 | 93.7 | 85.7 | 97.6 | 81.9 | 99.2 |
| v2 | 86.3 | 60.0 | 91.8 | 88.1 | 100 | 78.2 | 98.3 |
| v3 | 97.0 | 57.8 | 94.1 | 92.9 | 100 | 82.0 | 99.2 |

The first versions of the routers (`v1-low`, `v2-low`, `v3-low`) scored 79.7% at 81% saved, 77.7% at 74% and
84.1% at 55%.

External test set (`data/triage_external_v1.csv`: 145 prompts from HotpotQA, ContractNLI, LeetCodeDataset,
SVAMP and CLINC150, in other wording; *offline*, one saved answer per model):

| | Accuracy | Saved |
|---|---|---|
| Argon | 76.6% | 0% |
| Cheap | 60.7% | 95% |
| v1 | 72.4% | 14% |
| v2 | 74.5% | 36% |
| v3 | 76.6% | 23% |

## 4. v1: rules

**What it does.** Routes on the kind of task, before any call (`policies.py`). Math (when the cheap model
reasons), classification and named-field extraction go to cheap; code and document QA go to frontier; a
long extraction goes to cheap when the estimated saving on its input is at least `rules.min_saving_usd`
($0.0125). Anything unrecognised goes to frontier.

**How it was chosen.** `uv run python -m triage.eval.rules_report`:
- The threshold is the largest saving within 3 points of Argon, picked on training documents and scored on
  held-out ones (5 folds grouped by source document, 3 seeds): 84.3% at 50% saved.
- Re-worded prompts: every eval prompt rewritten into 2 or 3 other layouts (775 variants) keeps its task
  type and its decision.
- External set: 72.4% against Argon's 76.6%. Math on cheap costs 13 points there (SVAMP), which the eval
  set did not show.

**What we learned.**
- The first v1 had 10 stay-cheap and 9 send-to-frontier rules, several of which detected the source
  dataset through its prompt template, and it sent unmatched prompts to cheap. That default produced most
  of its saving and all of its loss.
- Question-word, overlap and worked-example rules were removed: on the external set they did no better
  than sending the same number of prompts per task type to cheap at random.

## 5. v2: checks

**What it does.** Every request starts on cheap; the answer is checked without an answer key and
escalated to Argon if a check fails (`checks.py`, lists per task type in `configs/strategies.yaml`).

**Which checks fired** (900 answers, 291 escalated): `cross_agreement` 112, `agreement` 68, `span_count` 50,
`grounding` 32, `json_format` 20, `examples` 8, `non_empty` 1.

**How often a kept answer was still wrong:** contracts 55 of 179, coding 34 of 186, QA 6 of 157,
classification 4 of 39.

**What we learned.** `uv run python -m triage.eval.checks_report`:
- `span_count` is the best free check: when the cheap model returns two or more spans for a contract it
  is right 11% of the time, against 58% for none or one.
- A second opinion from a different model catches 96% of wrong contract answers (the same model asked
  twice: 67%) but disagrees with 43% of correct ones, so two thirds of contracts escalate and the saving
  goes. Contracts therefore get free checks only.
- A second model only helps where it is at least as accurate as the first. Gemma 3 12B is used for
  document QA; elsewhere the same model is asked twice.
- Agreement on code breaks down on hard problems: on the external LeetCode prompts no two answers behaved
  identically, so every coding answer escalated.

## 6. v3: classifier

**What it does.** Answers on cheap and asks once more. A logistic model (`escalation.py`) estimates the
probability that escalating would fix the answer, from the task type and length, the free check results,
the span count and whether the two cheap answers agree. It escalates when the estimated Argon cost per
fixed answer is at most `max_usd_per_fix` ($0.03), so a short question escalates on a small probability and
a long contract needs a large one. Training: `uv run python -m triage.eval.train_classifier`; the model is
`artifacts/escalation.json`.

- **Held-out documents:** AUC 0.78, 84.4% at 59% saved (*offline*); live 84.3% at 55% saved.
- **External set:** AUC 0.76 and 76.6%, equal to Argon, with nothing fitted there.
- **Data:** 900 answers from 300 prompts, 159 positive. Held-out AUC levels off at about 120 training
  prompts, so the signals are the limit, not the sample size.

**What we learned.**
- The first v3 was a nearest-neighbour vote over Titan embeddings of the prompt. Its ranking was near
  chance within contracts and a contract-length rule matched it. Seeing the cheap answer is what raised the
  AUC, for nearest neighbours and for the logistic model alike; embeddings added nothing after that.
- The rebuilt v3 lands where the first one did on accuracy and cost. It no longer needs an embedding call,
  falls back to frontier when no model is trained, and has an external result.
- v1 ties v3 on this workload with no model in the loop.

## 7. The grader fixes

Reviewing flagged answers found graders failing correct answers. Fixed (`graders.py`, PR #6):
exact match now accepts an answer that quotes the gold span; numeric reads the bolded or last number;
span sets de-duplicate gold, credit one predicted span that covers several gold spans, and do not count a
repeated span as extra. Extra clauses still fail. The gold answers were not changed. Two prompts (MS01,
MS02) gained "Return only the label." (PR #5).

## 8. Known problems

- **Contract grading is strict.** An answer that finds every gold span and adds one extra fails. The
  graders flag these near-misses (64 of Argon's failed answers, 97 of the cheap model's), and counting
  them as correct would raise every contract number. Nobody has reviewed them.
- **Reused Argon answers.** Router runs reuse the baseline's Argon answers, so they do not measure Argon's
  own run-to-run variation, and escalated requests report the latency Argon had in the baseline run.
- **The external set is small:** 25 to 30 prompts per task type, one answer per model, and the prompt
  parser was fixed after its first pass over it (113 of 145 task types recognised before, 145 after).
- **Tuned on the eval set.** Check lists, the v1 threshold and the v3 limit were chosen on these 300
  prompts; the held-out and external numbers are the guard against that.
- **Judge on regrade.** The rubric grader asks a judge model again on every regrade, so planning results
  can change between regrades.
- **Unfinished checks.** `self_report` and `judge` in `checks.py` are stubs.

## 9. What to try next

1. **A stronger model for contracts only.** None of the eleven open models reaches Argon there (best:
   gpt-oss-120b at 51.1%), so this needs a mid-priced commercial model.
2. **Review the flagged contract answers** and report strict and lenient grading side by side.
3. **Generated-test agreement for code** (as in CodeT), in place of comparing two answers' behaviour.
4. **Run the external set live,** with 3 runs, for all three routers.
5. **Math on cheap:** check why the cheap model drops to 83% on SVAMP when it scores 100% on our math.
