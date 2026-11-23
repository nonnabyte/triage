"""Settings from the environment (.env) plus the YAML configs under configs/."""

from __future__ import annotations

import datetime as dt
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from triage.schemas import STRATEGIES, ReasoningLevel, Strategy, Tier

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_env() -> None:
    """Make provider keys in .env (GOOGLE_API_KEY, AWS_BEARER_TOKEN_BEDROCK, ...) visible via os.environ.

    Real environment variables win over .env. Called by the service and the smoke command, not on import.
    """
    load_dotenv(REPO_ROOT / ".env", override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRIAGE_", env_file=REPO_ROOT / ".env", extra="ignore")

    # fake: scripted models, no keys. real: the models named in configs/models.yaml `tiers:`.
    backend: Literal["fake", "real"] = "fake"
    strategy: Strategy = "v2"
    log_level: str = "INFO"
    # Lets the eval harness pick a version per request (X-Triage-Strategy) and see every call's output.
    eval_mode: bool = False
    # Reasoning effort for every reasoning model in the tiers (low | medium | high). Required when a tier uses
    # a reasoning model, so no experiment runs with the model choosing its own effort.
    reasoning: ReasoningLevel | None = None
    # A catalog key that answers every request directly, in place of the tiers (triage-eval run --model).
    # Used to measure one model on its own, e.g. a cheap-tier candidate. Real backend, baseline strategy only.
    model: str | None = None
    # Langfuse tracing, when LANGFUSE_* keys are set. trace_content=false sends no prompts or answers.
    # Eval runs: a results/<run> directory whose saved frontier answers are reused whenever a router sends a
    # prompt to frontier, instead of calling the model again (triage-eval run --reuse-frontier).
    reuse_frontier: Path | None = None
    tracing: bool = True
    trace_content: bool = True
    models_config: Path = REPO_ROOT / "configs" / "models.yaml"
    strategies_config: Path = REPO_ROOT / "configs" / "strategies.yaml"


# --- Models ------------------------------------------------------------------------

Provider = Literal["vertex_gemini", "gemini_api", "vertex_maas", "bedrock", "fake"]


class Price(BaseModel):
    """USD per 1M tokens."""

    input: float | None = None
    output: float | None = None

    def cost(self, input_tokens: int, output_tokens: int) -> float | None:
        if self.input is None or self.output is None:
            return None
        return (input_tokens * self.input + output_tokens * self.output) / 1e6


class ModelEntry(BaseModel):
    key: str
    provider: Provider
    model: str
    region: str | None = None
    price: Price = Price()
    settings: dict[str, Any] = {}
    timeout_s: float = 120
    max_retries: int = 2
    tested: dt.date | None = None
    # Reasoning efforts this model honours (checked live). Empty = not a reasoning model, or not controllable.
    reasoning_levels: list[ReasoningLevel] = []
    # The level this entry runs at, once resolved by at_level().
    reasoning: ReasoningLevel | None = None

    def at_level(self, level: ReasoningLevel | None) -> ModelEntry:
        """This entry with its reasoning effort fixed. Non-reasoning models are returned unchanged."""
        if not self.reasoning_levels:
            return self
        if level is None:
            raise ValueError(
                f"{self.key} is a reasoning model: set TRIAGE_REASONING (or --reasoning) to "
                f"{' / '.join(self.reasoning_levels)} so its effort is fixed"
            )
        if level not in self.reasoning_levels:
            raise ValueError(f"{self.key} does not support reasoning level '{level}'")
        return self.model_copy(update={"settings": {**self.settings, "thinking": level}, "reasoning": level})


class ModelsConfig(BaseModel):
    tiers: dict[Tier, str | None]
    catalog: dict[str, ModelEntry]
    # Catalog key of the second-opinion model for the cross_agreement check; None = ask the same tier again.
    verifier: str | None = None

    def tier_entry(self, tier: Tier) -> ModelEntry | None:
        key = self.tiers.get(tier)
        return self.catalog[key] if key else None


def load_models(path: Path) -> ModelsConfig:
    raw = yaml.safe_load(path.read_text())
    defaults = raw.get("defaults", {})
    catalog = {}
    for key, entry in (raw.get("catalog") or {}).items():
        settings = {**defaults.get("settings", {}), **entry.get("settings", {})}
        merged = {
            "timeout_s": defaults.get("timeout_s", 120),
            "max_retries": defaults.get("max_retries", 2),
            **entry,
            "settings": settings,
        }
        catalog[key] = ModelEntry(key=key, **merged)
    tiers = {Tier(t): k for t, k in (raw.get("tiers") or {}).items()}
    for tier, key in tiers.items():
        if key is not None and key not in catalog:
            raise ValueError(f"tiers.{tier} points to '{key}', which is not in the catalog")
    verifier = raw.get("verifier")
    if verifier is not None and verifier not in catalog:
        raise ValueError(f"verifier points to '{verifier}', which is not in the catalog")
    return ModelsConfig(tiers=tiers, catalog=catalog, verifier=verifier)


# --- Strategies --------------------------------------------------------------------


class StrategyConfig(BaseModel):
    start: Literal["frontier", "cheap", "rules"]
    # One list for every task, or a list per task type plus `default`, e.g. {coding: [...], default: [...]}.
    checks: list[str] | dict[str, list[str]] = []

    def checks_for(self, task_type: str) -> list[str]:
        if isinstance(self.checks, list):
            return self.checks
        return self.checks.get(task_type, self.checks.get("default", []))

    def all_checks(self) -> set[str]:
        lists = [self.checks] if isinstance(self.checks, list) else self.checks.values()
        return {name for names in lists for name in names}


class ClassifierConfig(BaseModel):
    artifact: Path
    # v3 escalates when the estimated frontier cost per fixed answer is at most this many dollars. Higher =
    # more escalation (more accurate, less saving). Unset: the trainer's choice, stored in the artifact.
    max_usd_per_fix: float | None = None


class RulesConfig(BaseModel):
    # v1's cost rule: a long extraction goes to cheap when the estimated saving on its input is at least
    # this many dollars. Lower = more to cheap (cheaper, less accurate). inf switches the rule off.
    min_saving_usd: float = float("inf")


class StrategiesConfig(BaseModel):
    versions: dict[str, StrategyConfig]
    classifier: ClassifierConfig
    rules: RulesConfig = RulesConfig()


def load_strategies(path: Path) -> StrategiesConfig:
    raw = yaml.safe_load(path.read_text())
    versions = {name: StrategyConfig(**raw[name]) for name in STRATEGIES}
    clf = raw.get("classifier", {"artifact": "artifacts/escalation.json"})
    clf_cfg = ClassifierConfig(**clf)
    if not clf_cfg.artifact.is_absolute():
        clf_cfg.artifact = REPO_ROOT / clf_cfg.artifact
    return StrategiesConfig(versions=versions, classifier=clf_cfg, rules=RulesConfig(**raw.get("rules", {})))


@lru_cache
def get_settings() -> Settings:
    return Settings()
