"""Tests for scripts.lint_pass."""

import runpy
import subprocess
import sys

import pytest

import scripts.lint_pass as lp


def test_lint_pass_main(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify that main() prints the success message and exits with status 0.

    Args:
        capsys (pytest.CaptureFixture[str]): Pytest fixture to capture stdout/stderr.
    """
    with pytest.raises(SystemExit) as exc_info:
        lp.main()
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "Linting passed." in captured.out


def test_lint_pass_module_main() -> None:
    """Verify executing scripts.lint_pass as __main__."""
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("scripts.lint_pass", run_name="__main__")
    assert exc_info.value.code == 0


def test_lint_pass_subprocess() -> None:
    """Verify our custom lint pass successfully executes in a standalone process."""
    result: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "scripts/lint_pass.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Linting passed." in result.stdout
