import pandas as pd
import pytest

from triage.config import REPO_ROOT
from triage.profile import build_profile
from triage.schemas import Task

DATA = REPO_ROOT / "data" / "triage_eval_v2.csv"


def test_hint_wins():
    p = build_profile(Task(user="anything", task_type_hint="coding"))
    assert (p.task_type, p.task_type_source) == ("coding", "provided")


def test_unknown_hint_is_ignored():
    p = build_profile(Task(user="hello there", task_type_hint="not_a_type"))
    assert p.task_type_source == "unknown"


def test_json_expected_output():
    p = build_profile(Task(user="CONTRACT TEXT: '...' TASK: Return JSON with keys a and b."))
    assert (p.task_type, p.expected_output) == ("contract_extraction", "json")


@pytest.mark.parametrize("use_case", ["document_qa", "contract_extraction", "coding"])
def test_infers_main_use_cases_on_dataset(use_case):
    """Baseline for the profiler: most dataset prompts of the three main use cases are recognised."""
    df = pd.read_csv(DATA)
    prompts = df.loc[df.use_case == use_case, "prompt"]
    hits = sum(build_profile(Task(user=p)).task_type == use_case for p in prompts)
    assert hits / len(prompts) >= 0.9
