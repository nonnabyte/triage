import pandas as pd
import pytest

from triage.cascade import validate_strategies
from triage.config import REPO_ROOT, Settings, load_models, load_strategies
from triage.schemas import STRATEGIES, TASK_TYPES


def test_models_config_loads():
    models = load_models(Settings().models_config)
    assert models.catalog
    for entry in models.catalog.values():
        assert entry.provider in ("vertex_gemini", "gemini_api", "vertex_maas", "bedrock")
        assert entry.settings.get("max_tokens")


def test_strategies_config_loads_with_known_checks():
    strategies = load_strategies(Settings().strategies_config)
    assert set(strategies.versions) == set(STRATEGIES)
    validate_strategies(strategies)
    assert strategies.versions["baseline"].checks == []
    assert strategies.versions["v1"].checks == []
    assert strategies.versions["v2"].checks and strategies.versions["v3"].checks


def test_dataset_shape():
    df = pd.read_csv(REPO_ROOT / "data" / "triage_eval_v2.csv")
    assert len(df) == 300
    assert df["id"].is_unique
    assert set(df["label"]) == {"easy", "hard"}
    assert set(df["use_case"]) <= set(TASK_TYPES)
    for col in ("prompt", "answer", "check_type"):
        assert df[col].notna().all()


def test_reasoning_level_is_fixed_per_entry():
    catalog = load_models(Settings().models_config).catalog
    argon = catalog["vertex-gemini-4-argon"].at_level("high")
    assert argon.settings["thinking"] == "high" and argon.reasoning == "high"
    assert argon.settings["max_tokens"] == 8192  # other settings kept
    qwen = catalog["bedrock-qwen3-235b"]
    assert qwen.at_level("high") is qwen  # not a reasoning model: unchanged
    with pytest.raises(ValueError, match="TRIAGE_REASONING"):
        catalog["bedrock-gpt-oss-20b"].at_level(None)


def test_one_model_can_answer_everything():
    """triage-eval run --model: a cheap candidate fills the only (frontier) slot, at the run's level."""
    from triage.models import ModelClient
    from triage.schemas import Tier

    models = load_models(Settings().models_config)
    client = ModelClient(Settings(backend="real", model="bedrock-gpt-oss-20b", reasoning="low"), models)
    assert client.ladder() == [Tier.frontier]
    assert client.entries[Tier.frontier].key == "bedrock-gpt-oss-20b"
    assert client.entries[Tier.frontier].reasoning == "low"
    with pytest.raises(ValueError, match="not in configs/models.yaml"):
        ModelClient(Settings(backend="real", model="no-such-model"), models)
