"""Test lint_pass.py."""

import subprocess


def test_lint_pass_enforcement() -> None:
    """Verify our custom lint pass successfully executes and enforces the rule."""
    result: subprocess.CompletedProcess[str] = subprocess.run(["python3", "scripts/lint_pass.py"], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "Linting passed." in result.stdout
