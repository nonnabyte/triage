"""v1 routing: task-type rules plus a cost rule, with frontier as the default."""

from triage import policies
from triage.profile import build_profile
from triage.schemas import Task, Tier


def v1(text, cheap_reasons=True, saving_usd=None, min_saving_usd=float("inf")):
    task = Task(user=text)
    return policies.rules(task, build_profile(task), cheap_reasons, saving_usd, min_saving_usd)


MATH = (
    "Solve the problem. Return only the final numeric answer.\n\n"
    "PROBLEM:\nTom has 3 apples and buys 4 more. How many apples does he have?"
)
NO_EXAMPLES = (
    "Write Python 3 code satisfying the task below. Return only Python code, without Markdown fences.\n\n"
    "TASK:\nWrite a function to find the longest common subsequence.\n\n"
    "FUNCTION INTERFACES:\ndef lcs(a, b): ..."
)
PASSAGE = (
    "The treaty was signed in Paris in 1783 after long talks between the delegations. Benjamin Franklin led "
    "the American side, and the British side was led by David Hartley, a member of parliament. The talks "
    "had started the year before and were slowed by arguments over fishing rights and debts."
)
CONTRACT = (
    "This Supply Agreement is made between Acme Ltd and Bolt Inc. " + "The parties agree as follows. " * 400
)


def test_v1_goes_cheap_only_for_task_types_the_cheap_model_handles():
    assert v1(MATH) == (Tier.cheap, ["S1 math word problem, cheap model reasons"])
    tier, why = v1(MATH, cheap_reasons=False)
    assert tier == Tier.frontier and why[0].startswith("M0")
    tier, why = v1("Classify this ticket as Billing or Other: refund my order")
    assert tier == Tier.cheap and why[0].startswith("S2")
    tier, why = v1(NO_EXAMPLES)
    assert tier == Tier.frontier and why[0].startswith("K0")
    tier, why = v1(f"PASSAGE:\n{PASSAGE}\n\nQUESTION:\nWho led the American side?")
    assert tier == Tier.frontier and why[0].startswith("Q0")
    fields = f"Agreement:\n{PASSAGE}\n\nRequest: return JSON with keys city and year."
    tier, why = v1(fields)
    assert tier == Tier.cheap and why[0].startswith("S3")


def test_v1_sends_what_it_does_not_recognise_to_frontier():
    assert v1("hi") == (Tier.frontier, ["U0 unknown task type"])
    tier, _ = v1("Draft a go-to-market plan for our new product line.")
    assert tier == Tier.frontier


def test_v1_does_not_depend_on_the_prompt_template():
    layouts = [
        f"PASSAGE:\n{PASSAGE}\n\nQUESTION:\nWho led the American side?",
        f"Context: {PASSAGE}\n\nQ: Who led the American side?",
        f"{PASSAGE}\n\nBased only on the text above: who led the American side?",
        f"Who led the American side? Use only the article below.\n\n{PASSAGE}",
    ]
    assert all(build_profile(Task(user=p)).task_type == "document_qa" for p in layouts)
    assert len({tuple(v1(p)[1]) for p in layouts}) == 1


def test_v1_cost_rule_sends_a_long_extraction_to_cheap_only_above_the_threshold():
    prompt = f"Document:\n{CONTRACT}\n\nInstruction: return a JSON array of every clause about termination."
    tier, why = v1(prompt, saving_usd=0.02, min_saving_usd=0.01)
    assert tier == Tier.cheap and why[0].startswith("S4")
    tier, why = v1(prompt, saving_usd=0.005, min_saving_usd=0.01)
    assert tier == Tier.frontier and why[0].startswith("X1")
    tier, _ = v1(prompt, saving_usd=None, min_saving_usd=0.01)  # prices unknown: the cost rule is off
    assert tier == Tier.frontier
