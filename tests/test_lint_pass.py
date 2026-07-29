import subprocess


def test_lint_pass_enforcement():
    # Verify our custom lint pass successfully executes and enforces the rule
    result = subprocess.run(["python3", "scripts/lint_pass.py"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Linting passed." in result.stdout
