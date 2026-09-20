def test_mypy():
    """Verify static typing via mypy across src/ml_switcheroo_compiler/."""
    import shutil
    import subprocess

    import pytest

    if not shutil.which("mypy"):
        pytest.skip("mypy executable not found")

    res = subprocess.run(["mypy", "src/ml_switcheroo_compiler/"], capture_output=True, text=True)
    assert res.returncode == 0
