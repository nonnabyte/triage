"""Evaluation harness: graders, and later the response matrix, runs and metrics.

Nothing in the service (triage.api, cascade, models, ...) may import this package: it reads the dataset's
expected answers, which must never reach the router. tests/test_boundary.py enforces that.
"""
