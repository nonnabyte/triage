"""Run model-written Python in a separate, limited process (graders and the v2 checks).

Limits: a fresh temp directory, isolated interpreter (-I), no inherited environment (so no API keys), a
wall-clock timeout, and on POSIX CPU / memory / file-size limits. Network access is NOT blocked; this is
enough for benchmark code, not for hostile code. Run the evaluation in a container if that matters.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

TIMEOUT_S = 10
CPU_S = 10
MEMORY_BYTES = 1024 * 1024 * 1024
FILE_BYTES = 10 * 1024 * 1024


@dataclass
class RunResult:
    ok: bool
    timed_out: bool
    detail: str


# Sets the limits inside the new interpreter, then runs the candidate file as __main__. Not preexec_fn: that
# runs in a fork of the (multi-threaded) harness and can deadlock before exec, which the timeout never sees.
_PRELUDE = f"""
import runpy, sys
try:
    import resource
except ImportError:  # Windows
    resource = None
if resource:
    for limit, value in ((resource.RLIMIT_CPU, {CPU_S}), (resource.RLIMIT_AS, {MEMORY_BYTES}),
                         (resource.RLIMIT_FSIZE, {FILE_BYTES})):
        try:
            resource.setrlimit(limit, (value, value))
        except (ValueError, OSError):  # e.g. RLIMIT_AS is not enforceable on macOS
            pass
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def run_python(code: str, timeout_s: float = TIMEOUT_S) -> RunResult:
    """Run `code` in a fresh interpreter. ok means exit code 0 (every assert passed)."""
    with tempfile.TemporaryDirectory(prefix="triage-grade-") as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(code)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", _PRELUDE, str(path)],
                cwd=tmp,
                env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
                stdin=subprocess.DEVNULL,  # code that calls input() fails instead of hanging
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return RunResult(ok=False, timed_out=True, detail=f"timed out after {timeout_s}s")
    if proc.returncode == 0:
        return RunResult(ok=True, timed_out=False, detail="")
    lines = [ln for ln in proc.stderr.strip().splitlines() if ln.strip()]
    return RunResult(
        ok=False, timed_out=False, detail=(lines[-1] if lines else f"exit {proc.returncode}")[:300]
    )
