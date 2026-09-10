"""Tests for scripts/lint_dependencies.py."""

import os
import runpy
import sys
import tempfile
from unittest.mock import patch

from scripts.lint_dependencies import _check_module_name, check_dependencies, main


def test_check_dependencies_clean() -> None:
    """Test checking dependencies on clean directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        core_dir: str = os.path.join(temp_dir, "core")
        os.makedirs(core_dir)
        clean_file: str = os.path.join(core_dir, "clean.py")
        with open(clean_file, "w", encoding="utf-8") as f:
            f.write("import os\nimport sys\nimport yaml\nimport numpy as np\n")

        # non-python file skipped
        with open(os.path.join(core_dir, "notes.txt"), "w", encoding="utf-8") as f:
            f.write("Some notes")

        violations: list[str] = check_dependencies(temp_dir)
        assert len(violations) == 0


def test_check_dependencies_violations() -> None:
    """Test checking dependencies detects disallowed 3rd party import."""
    with tempfile.TemporaryDirectory() as temp_dir:
        core_dir: str = os.path.join(temp_dir, "core")
        os.makedirs(core_dir)
        bad_file: str = os.path.join(core_dir, "bad.py")
        with open(bad_file, "w", encoding="utf-8") as f:
            f.write("import sklearn\nfrom scipy.optimize import minimize\n")

        violations: list[str] = check_dependencies(temp_dir)
        assert len(violations) >= 2
        assert any("sklearn" in v for v in violations)
        assert any("scipy" in v for v in violations)


def test_check_dependencies_backend_allowed() -> None:
    """Test that allowed backend imports are accepted in their respective folders."""
    with tempfile.TemporaryDirectory() as temp_dir:
        pytorch_dir: str = os.path.join(temp_dir, "backends", "pytorch")
        os.makedirs(pytorch_dir)
        torch_file: str = os.path.join(pytorch_dir, "runner.py")
        with open(torch_file, "w", encoding="utf-8") as f:
            f.write("import torch\nfrom torch import nn\n")

        violations: list[str] = check_dependencies(temp_dir)
        assert len(violations) == 0


def test_check_dependencies_local_module_ignored() -> None:
    """Test that local module imports inside the same directory are ignored."""
    with tempfile.TemporaryDirectory() as temp_dir:
        mod_dir: str = os.path.join(temp_dir, "my_module")
        os.makedirs(mod_dir)
        helper_file: str = os.path.join(mod_dir, "helper.py")
        with open(helper_file, "w", encoding="utf-8") as f:
            f.write("X = 1\n")

        consumer_file: str = os.path.join(mod_dir, "consumer.py")
        with open(consumer_file, "w", encoding="utf-8") as f:
            f.write("import helper\n")

        violations: list[str] = check_dependencies(temp_dir)
        assert len(violations) == 0


def test_check_dependencies_syntax_error_handled() -> None:
    """Test that files with syntax errors are skipped gracefully."""
    with tempfile.TemporaryDirectory() as temp_dir:
        bad_syntax: str = os.path.join(temp_dir, "invalid.py")
        with open(bad_syntax, "w", encoding="utf-8") as f:
            f.write("def invalid_syntax(\n")

        violations: list[str] = check_dependencies(temp_dir)
        assert len(violations) == 0


def test_check_module_name_empty() -> None:
    """Test empty module name in _check_module_name."""
    violations: list[str] = []
    _check_module_name("", "file.py", 1, set(), ".", violations)
    assert len(violations) == 0


def test_main_clean() -> None:
    """Test main function when no violations are found."""
    with patch("scripts.lint_dependencies.check_dependencies", return_value=[]):
        exit_code: int = main()
        assert exit_code == 0


def test_main_with_violations() -> None:
    """Test main function when violations are detected."""
    with patch("scripts.lint_dependencies.check_dependencies", return_value=["test.py:1: error"]):
        exit_code: int = main()
        assert exit_code == 1


def test_entrypoint_call() -> None:
    """Test direct module invocation entrypoint."""
    with patch.object(sys, "argv", ["lint_dependencies.py"]):
        with patch("scripts.lint_dependencies.check_dependencies", return_value=[]):
            try:
                runpy.run_path("scripts/lint_dependencies.py", run_name="__main__")
            except SystemExit as e:
                assert e.code == 0
