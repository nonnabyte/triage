"""triage-smoke: send a one-word prompt to every catalog model and report which answer.

uv run triage-smoke              # all models in configs/models.yaml
uv run triage-smoke bedrock      # only entries whose name contains "bedrock"
uv run triage-smoke --tiers      # only the models assigned to cheap / smart / frontier
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from pydantic_ai import Agent  # noqa: E402

from triage.config import ModelEntry, get_settings, load_env, load_models  # noqa: E402
from triage.providers import build_model  # noqa: E402

PROMPT = "Reply with exactly one word: pong"
# Reasoning models (gpt-oss, Gemini Pro) spend tokens thinking before answering; too few leaves no text.
MAX_TOKENS = 1024


async def ping(sem: asyncio.Semaphore, entry: ModelEntry) -> tuple[ModelEntry, bool, str]:
    async with sem:
        start = time.perf_counter()
        try:
            agent = Agent(build_model(entry))
            settings = {**entry.settings, "max_tokens": MAX_TOKENS}
            result = await asyncio.wait_for(
                agent.run(PROMPT, model_settings=settings), timeout=entry.timeout_s
            )
        except Exception as e:  # report every failure, keep going
            return entry, False, f"{type(e).__name__}: {' '.join(str(e).split())}"[:160]
        ms = (time.perf_counter() - start) * 1000
        u = result.usage
        text = " ".join((result.output or "").split())[:30]
        return entry, True, f"{ms:6.0f} ms  in={u.input_tokens} out={u.output_tokens}  -> {text!r}"


async def main_async(filter_text: str, tiers_only: bool) -> int:
    models = load_models(get_settings().models_config)
    if tiers_only:
        entries = [models.catalog[k] for k in models.tiers.values() if k]
        if not entries:
            print("No tiers assigned in configs/models.yaml.")
            return 1
    else:
        entries = [e for k, e in models.catalog.items() if filter_text in k.lower()]
    print(f"Pinging {len(entries)} models...\n")
    sem = asyncio.Semaphore(8)
    results = await asyncio.gather(*(ping(sem, e) for e in entries))
    width = max(len(e.key) for e in entries)
    ok = 0
    for entry, passed, detail in results:
        ok += passed
        print(f"[{' OK ' if passed else 'FAIL'}] {entry.key:<{width}}  {detail}")
    print(f"\n{ok}/{len(results)} models answered.")
    return 0 if ok == len(results) else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("filter", nargs="?", default="", help="only catalog entries whose name contains this")
    ap.add_argument("--tiers", action="store_true", help="only the models assigned to tiers")
    args = ap.parse_args()
    load_env()
    sys.exit(asyncio.run(main_async(args.filter.lower(), args.tiers)))


if __name__ == "__main__":
    main()
