"""Tests for lint_numpy_leak.py."""

import os
import tempfile
from unittest.mock import patch

from pytest import CaptureFixture

from scripts.lint_numpy_leak import check_for_architectural_imports, check_for_numpy_leaks, main


def test_check_for_numpy_leaks() -> None:
    """Test finding numpy leaks in backends."""
    with tempfile.TemporaryDirectory() as d:
        backends_dir: str = os.path.join(d, "backends")
        mlx_dir: str = os.path.join(backends_dir, "mlx")
        numpy_dir: str = os.path.join(backends_dir, "numpy")
        eager_dir: str = os.path.join(backends_dir, "eager")
        grad_dir: str = os.path.join(d, "grad")

        os.makedirs(mlx_dir)
        os.makedirs(numpy_dir)
        os.makedirs(eager_dir)
        os.makedirs(grad_dir)

        # This one should leak
        with open(os.path.join(mlx_dir, "bad.py"), "w") as f:
            f.write("import numpy as np\n")
            f.write("res = np.add(1, 2)\n")
            # This triggers has_np and the numpy() exception
            f.write("np.foo(); foo.numpy()\n")
            f.write("# import numpy\n")  # Comment ignored
            f.write("np.gradient(...)  # torch.gradient  # Ignored exception\n")
            f.write("from numpy import array\n")
            f.write("a = 1 + 1\n")

        # These should be ignored
        with open(os.path.join(numpy_dir, "good.py"), "w") as f:
            f.write("import numpy\n")

        with open(os.path.join(eager_dir, "good.py"), "w") as f:
            f.write("import numpy as np\n")

        with open(os.path.join(backends_dir, "generator_mixins.py"), "w") as f:
            f.write("import numpy as np\n")

        with open(os.path.join(grad_dir, "bad.py"), "w") as f:
            f.write("import numpy\n")

        violations: list[str] = check_for_numpy_leaks(d)

        assert len(violations) == 4
        # check if grad and mlx violations are recorded
        # The exact text match depends on order, but we can verify substrings
        violation_texts: str = " ".join(violations)
        assert "import numpy as np" in violation_texts
        assert "res = np.add(1, 2)" in violation_texts
        assert "from numpy import array" in violation_texts
        # one from grad/bad.py
        assert "import numpy" in violation_texts


def test_check_for_architectural_imports() -> None:
    """Test checking for architectural imports."""
    with tempfile.TemporaryDirectory() as d:
        core_dir: str = os.path.join(d, "core")
        ir_dir: str = os.path.join(d, "ir")
        transforms_dir: str = os.path.join(d, "transforms")

        os.makedirs(core_dir)
        os.makedirs(ir_dir)
        os.makedirs(transforms_dir)

        with open(os.path.join(core_dir, "bad.py"), "w") as f:
            f.write("from ml_switcheroo_compiler.backends import mlx\n")
            f.write("from ml_switcheroo_compiler.backends.registry import BackendRegistry\n")  # Allowed
            f.write("import ml_switcheroo_compiler.backends.linker\n")  # Allowed
            f.write("# from ml_switcheroo_compiler.backends import oops\n")  # Comment
            f.write("import something_else\n")  # Normal line

        violations: list[str] = check_for_architectural_imports(d)

        assert len(violations) == 1
        assert "from ml_switcheroo_compiler.backends import mlx" in violations[0]


def test_main_success(capsys: CaptureFixture[str]) -> None:
    """Test main block with no violations.

    Args:
        capsys: Pytest fixture.
    """
    with patch("scripts.lint_numpy_leak.check_for_numpy_leaks", return_value=[]), patch("scripts.lint_numpy_leak.check_for_architectural_imports", return_value=[]):
        assert main() == 0
        captured = capsys.readouterr()
        assert "Linting passed" in captured.out


def test_main_failure_numpy(capsys: CaptureFixture[str]) -> None:
    """Test main block with numpy violations.

    Args:
        capsys: Pytest fixture.
    """
    with patch("scripts.lint_numpy_leak.check_for_numpy_leaks", return_value=["leak1"]), patch("scripts.lint_numpy_leak.check_for_architectural_imports", return_value=[]):
        assert main() == 1
        captured = capsys.readouterr()
        assert "NumPy Leak Linting failed" in captured.out
        assert "leak1" in captured.out


def test_main_failure_arch(capsys: CaptureFixture[str]) -> None:
    """Test main block with architectural violations.

    Args:
        capsys: Pytest fixture.
    """
    with patch("scripts.lint_numpy_leak.check_for_numpy_leaks", return_value=[]), patch("scripts.lint_numpy_leak.check_for_architectural_imports", return_value=["arch1"]):
        assert main() == 1
        captured = capsys.readouterr()
        assert "Architectural Boundaries failed" in captured.out
        assert "arch1" in captured.out


def test_main_block(capsys: CaptureFixture[str]) -> None:
    """Test the __main__ execution block."""
    import runpy
    import sys

    with patch.object(sys, "argv", ["lint_numpy_leak.py"]):
        with patch("glob.glob", return_value=[]):
            try:
                runpy.run_path("scripts/lint_numpy_leak.py", run_name="__main__")
            except SystemExit as e:
                assert e.code == 0
    captured = capsys.readouterr()
    assert "Linting passed" in captured.out
