"""Tests for test_mypy."""

from __future__ import annotations


def test_mypy() -> None:
    """Verify static typing via mypy across src/ml_switcheroo_compiler/."""
    import shutil
    import subprocess

    import pytest

    if not shutil.which("mypy"):
        pytest.skip("mypy executable not found")

    res = subprocess.run(["mypy", "src/ml_switcheroo_compiler/"], capture_output=True, text=True)
    assert res.returncode == 0, f"mypy failure:\n{res.stdout}\n{res.stderr}"
