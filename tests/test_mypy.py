"""Tests for test_mypy."""

from __future__ import annotations


def test_mypy() -> None:
    """Verify static typing via mypy across src/ml_switcheroo_compiler/, scripts/, and tests/."""
    import shutil
    import subprocess

    import pytest

    if not shutil.which("mypy"):
        pytest.skip("mypy executable not found")

    res = subprocess.run(
        ["mypy", "--config-file", "pyproject.toml", "src/ml_switcheroo_compiler/", "scripts/", "tests/"],
        capture_output=True,
        text=True,
    )
    msg = f"mypy failure: {res.stdout} {res.stderr}"
    assert res.returncode == 0, msg
