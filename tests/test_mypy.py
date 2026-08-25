def test_mypy():
    import subprocess

    res = subprocess.run(["mypy", "src/ml_switcheroo_compiler/"], capture_output=True, text=True)
    assert res.returncode == 0
