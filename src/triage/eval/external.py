"""Build an external test set: public prompts from datasets the eval set does not use, in other wording.

  uv run python -m triage.eval.external --contract-nli PATH/contract-nli/test.json

The eval set is where the routers were designed, so it cannot show that they generalise. This set can: none
of its sources is in the eval set, and no prompt uses the eval set's template. It is for testing only.
Nothing is tuned on it.

  document QA       HotpotQA (distractor), multi-paragraph context          hotpotqa/hotpot_qa
  extraction        ContractNLI, evidence spans in NDAs                     stanfordnlp.github.io/contract-nli
  coding            LeetCodeDataset, problems published after mid-2024      newfacade/LeetCodeDataset
  math              SVAMP word problems                                     ChilleD/SVAMP
  classification    CLINC150 intents                                        clinc/clinc_oos

HotpotQA, SVAMP, CLINC150 and LeetCodeDataset are read through the Hugging Face datasets server. ContractNLI
is a download (CC BY 4.0): unzip it and pass the path to test.json. Writes data/triage_external_v1.csv with
the same columns as the eval set; `label` is left empty because difficulty is measured, not judged.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import httpx

from triage.config import REPO_ROOT
from triage.sandbox import run_python

SEED = 2701
PER_TASK = 30
OUT = REPO_ROOT / "data" / "triage_external_v1.csv"
ROWS_API = "https://datasets-server.huggingface.co/rows"
COLUMNS = [
    "id", "use_case", "source", "source_id", "check_type", "prompt", "answer", "override_reason", "label",
    "label_reason",
]  # fmt: skip


def fetch(dataset: str, config: str, split: str, offset: int, length: int) -> tuple[list[dict], dict]:
    params = {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length}
    r = httpx.get(ROWS_API, params=params, timeout=60)
    r.raise_for_status()
    body = r.json()
    return [x["row"] for x in body["rows"]], {f["name"]: f["type"] for f in body["features"]}


def row(n: int, use_case: str, source: str, source_id: Any, check: str, prompt: str, answer: Any) -> dict:
    prefix = {
        "document_qa": "qa",
        "contract_extraction": "ct",
        "coding": "cd",
        "misc_math": "mm",
        "misc_classification": "cl",
    }[use_case]
    return {
        "id": f"ext-{prefix}-{n:03d}",
        "use_case": use_case,
        "source": source,
        "source_id": str(source_id),
        "check_type": check,
        "prompt": prompt,
        "answer": answer if isinstance(answer, str) else json.dumps(answer),
        "override_reason": "",
        "label": "",
        "label_reason": "",
    }


def document_qa(rng: random.Random) -> list[dict]:
    rows, _ = fetch("hotpotqa/hotpot_qa", "distractor", "validation", 0, 100)
    rows = [r for r in rows if r["answer"].lower() not in ("yes", "no")]
    out = []
    for n, r in enumerate(rng.sample(rows, PER_TASK)):
        context = "\n\n".join(
            f"{title}: {''.join(sentences).strip()}"
            for title, sentences in zip(r["context"]["title"], r["context"]["sentences"], strict=True)
        )
        q = r["question"].strip()
        prompt = [
            f"{context}\n\nUsing only the notes above, answer briefly: {q}",
            f"Q: {q}\n\nBackground:\n{context}\n\nAnswer with a short phrase only.",
            f"Here are some reference paragraphs.\n\n{context}\n\n{q} Reply with just the answer, "
            "no sentence.",
        ][n % 3]
        out.append(row(n + 1, "document_qa", "hotpotqa", r["id"], "exact_match", prompt, [r["answer"]]))
    return out


def extraction(rng: random.Random, path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    hypotheses = {k: v["hypothesis"].rstrip(".") for k, v in data["labels"].items()}
    candidates = []
    for doc in data["documents"]:
        if not 4000 <= len(doc["text"]) <= 30000:
            continue
        for key, a in doc["annotation_sets"][0]["annotations"].items():
            if len(a["spans"]) <= 3:
                spans = [doc["text"][s:e].strip() for s, e in (doc["spans"][k] for k in a["spans"])]
                candidates.append((doc, key, spans))
    rng.shuffle(candidates)
    picked, seen_docs = [], set()
    want_empty = PER_TASK // 4
    for doc, key, spans in candidates:  # one question per NDA; a quarter with nothing to find
        if doc["id"] in seen_docs or (not spans and want_empty == 0):
            continue
        if len(picked) - (PER_TASK // 4 - want_empty) >= PER_TASK - PER_TASK // 4 and spans:
            continue
        want_empty -= not spans
        seen_docs.add(doc["id"])
        picked.append((doc, key, spans))
        if len(picked) == PER_TASK:
            break
    out = []
    for n, (doc, key, spans) in enumerate(picked):
        nda, hyp = doc["text"].strip(), hypotheses[key]
        prompt = [
            f"{nda}\n\n======\n\nQuote every passage of this NDA that bears on the following statement, as a "
            f"JSON array of exact quotes. Use [] if the NDA is silent on it.\nStatement: {hyp}.",
            f'Statement to check: "{hyp}."\n\nReturn a JSON array with the verbatim sentences from the '
            "agreement below that support or contradict the statement ([] when it is not addressed).\n\n"
            f"Agreement:\n{nda}",
            f"Document:\n{nda}\n\nRequest: list, as a JSON array of verbatim quotes, the parts of the "
            f'document that decide whether "{hyp}". Return [] if nothing is relevant.',
        ][n % 3]
        out.append(
            row(n + 1, "contract_extraction", "contract_nli", f"{doc['id']}:{key}", "span_set", prompt, spans)
        )
    return out


def coding(rng: random.Random) -> list[dict]:
    rows = []
    for offset in (0, 100, 200):
        rows += fetch("newfacade/LeetCodeDataset", "default", "test", offset, 100)[0]
    quota = {"Easy": 12, "Medium": 12, "Hard": 6}
    rng.shuffle(rows)
    out = []
    for r in rows:
        if quota.get(r["difficulty"], 0) == 0 or len(r["test"]) > 6000:
            continue
        tests = f"{r['test']}\n\ncheck({r['entry_point']})"
        # Keep a problem only if its reference solution passes its own tests in our sandbox.
        if not run_python(f"{r['prompt']}\n\n{r['completion']}\n\n\n{tests}\n").ok:
            continue
        quota[r["difficulty"]] -= 1
        n = len(out)
        desc, starter = r["problem_description"].strip(), r["starter_code"].rstrip()
        prompt = [
            f"{desc}\n\nComplete this starter code and return the whole solution in Python, with every "
            f"import it needs (for example from typing import List):\n\n{starter}",
            "Solve the following in Python 3. Reply with code only, including imports such as "
            f"from typing import List.\n\n### Problem\n{desc}\n\n### Starter\n```python\n{starter}\n```",
        ][n % 2]
        source_id = f"{r['question_id']}:{r['difficulty']}"
        out.append(row(n + 1, "coding", "leetcode", source_id, "unit_tests", prompt, tests))
        if len(out) == PER_TASK:
            break
    return out


def math(rng: random.Random) -> list[dict]:
    rows, _ = fetch("ChilleD/SVAMP", "default", "test", 0, 100)
    out = []
    for n, r in enumerate(rng.sample(rows, PER_TASK)):
        body, q = r["Body"].strip(), r["Question"].strip()
        prompt = [
            f"{body} {q} Reply with the number only.",
            f"Word problem: {body} {q}\nGive only the final number.",
            f"{body}\n\n{q}\n\n(Just the numeric answer, please.)",
        ][n % 3]
        out.append(row(n + 1, "misc_math", "svamp", r["ID"], "numeric", prompt, [float(r["Answer"])]))
    return out


def classification(rng: random.Random) -> list[dict]:
    rows, features = [], {}
    for offset in (0, 1500, 3000):
        got, features = fetch("clinc/clinc_oos", "plus", "test", offset, 100)
        rows += got
    names = features["intent"]["names"]
    rows = [r for r in rows if names[int(r["intent"])] != "oos"]
    intents = [x for x in names if x != "oos"]
    out = []
    for n, r in enumerate(rng.sample(rows, PER_TASK)):
        text, gold = r["text"].strip(), names[int(r["intent"])]
        prompt = [
            "Pick the intent that best matches the user's utterance. Reply with the intent name only, "
            f'exactly as listed.\n\nIntents: {", ".join(intents)}\n\nUtterance: "{text}"',
            f"Utterance: {text}\n\nWhich one of the following intents is it? Answer with just the intent.\n"
            + "\n".join(f"- {x}" for x in intents),
        ][n % 2]
        out.append(row(n + 1, "misc_classification", "clinc150", n, "exact_match", prompt, [gold]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--contract-nli", type=Path, required=True, help="path to ContractNLI's test.json")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    rng = random.Random(SEED)
    rows = (
        document_qa(rng) + extraction(rng, args.contract_nli) + coding(rng) + math(rng) + classification(rng)
    )
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    by = {}
    for r in rows:
        by[r["use_case"]] = by.get(r["use_case"], 0) + 1
    print(f"wrote {len(rows)} prompts to {args.out}: {by}")


if __name__ == "__main__":
    main()
