import subprocess


def test_no_numpy_leak():
    """Verify that the codebase does not leak numpy into non-numpy backends."""
    result = subprocess.run(["python3", "scripts/lint_numpy_leak.py"], capture_output=True, text=True)
    assert result.returncode == 0
