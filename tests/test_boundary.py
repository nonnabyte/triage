"""The service must never import the evaluation package: it holds the expected answers."""

import re

from triage.config import REPO_ROOT

SERVICE = REPO_ROOT / "src" / "triage"


def test_service_does_not_import_eval():
    offenders = [
        p.name
        for p in SERVICE.glob("*.py")
        if re.search(r"^\s*(from|import)\s+triage\.eval\b", p.read_text(), re.MULTILINE)
    ]
    assert not offenders, f"service modules import triage.eval: {offenders}"
