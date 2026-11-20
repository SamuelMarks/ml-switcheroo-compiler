"""Unit tests for scripts/lint_dependencies.py enforcing dependency and dynamic-import isolation."""

from __future__ import annotations

import os
import runpy
import sys
import tempfile
import textwrap
from unittest.mock import patch

import pytest

from scripts.lint_dependencies import (
    BACKEND_ALLOWED,
    GLOBAL_ALLOWED_THIRD_PARTY,
    _check_module_name,
    _validate_single_backend_dependency,
    check_dependencies,
    main,
)


def test_global_allowed_third_party_exact_whitelist() -> None:
    """Verify GLOBAL_ALLOWED_THIRD_PARTY contains strictly Tier-1 IR and approved global packages."""
    assert GLOBAL_ALLOWED_THIRD_PARTY == {
        "ml_switcheroo_ir",
        "yaml",
        "PIL",
        "pillow",
        "pydantic",
        "cdd",
        "libcst",
        "h5py",
    }
    assert "ml_switcheroo_ir" in GLOBAL_ALLOWED_THIRD_PARTY


def test_check_dependencies_clean_and_syntax_error() -> None:
    """Test clean allowed global/stdlib/relative imports and graceful SyntaxError skip."""
    with tempfile.TemporaryDirectory() as temp_dir:
        good_file: str = os.path.join(temp_dir, "good.py")
        local_helper: str = os.path.join(temp_dir, "local_helper.py")
        bad_syntax: str = os.path.join(temp_dir, "broken.py")
        txt_file: str = os.path.join(temp_dir, "notes.txt")
        with open(local_helper, "w", encoding="utf-8") as f:
            f.write("X = 1")
        with open(good_file, "w", encoding="utf-8") as f:
            f.write(
                textwrap.dedent(
                    """
                    import os
                    import importlib
                    import ml_switcheroo_ir
                    import yaml
                    import pydantic
                    import libcst
                    import h5py
                    import local_helper
                    from . import sibling
                    m = importlib.import_module("math")
                    rel = importlib.import_module("...")
                    """
                )
            )
        with open(bad_syntax, "w", encoding="utf-8") as f:
            f.write("def broken(:")
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write("ignored")

        reg_dir: str = os.path.join(temp_dir, "backends")
        os.makedirs(reg_dir)
        reg_file: str = os.path.join(reg_dir, "registry.py")
        with open(reg_file, "w", encoding="utf-8") as f:
            f.write(
                textwrap.dedent(
                    """
                    import importlib
                    mod = importlib.import_module(dynamic_var)
                    """
                )
            )

        assert check_dependencies(temp_dir) == []


def test_check_dependencies_disallowed_dynamic_eval_and_external() -> None:
    """Test disallowed 3rd-party packages, upward zero_* imports, dynamic imports, and eval()."""
    with tempfile.TemporaryDirectory() as temp_dir:
        core_dir: str = os.path.join(temp_dir, "core")
        os.makedirs(core_dir)
        bad_file: str = os.path.join(core_dir, "bad.py")
        with open(bad_file, "w", encoding="utf-8") as f:
            f.write(
                textwrap.dedent(
                    """
                    import importlib
                    import sklearn
                    import zero_pytorch
                    import numpy as np
                    from scipy.optimize import minimize
                    p = __import__("psutil")
                    sub = "special"
                    s = importlib.import_module(f"scipy.{sub}")
                    dyn = importlib.import_module(sub)
                    other = __import__()
                    x = eval("1 + 1")
                    exec("y = 2")
                    """
                )
            )
        violations: list[str] = check_dependencies(temp_dir)
        assert any("sklearn" in v for v in violations)
        assert any("zero_pytorch" in v for v in violations)
        assert any("numpy" in v for v in violations)
        assert any("scipy" in v for v in violations)
        assert any("psutil" in v for v in violations)
        assert any("Non-literal dynamic import" in v for v in violations)
        assert any("Prohibited builtin 'eval()'" in v for v in violations)
        assert any("Prohibited builtin 'exec()'" in v for v in violations)


def test_check_dependencies_backend_allowed_and_single_root_enforcement() -> None:
    """Test backend-scoped imports and multi-root backend directory rejection."""
    with tempfile.TemporaryDirectory() as temp_dir:
        pytorch_dir: str = os.path.join(temp_dir, "backends", "pytorch")
        os.makedirs(pytorch_dir)
        with open(os.path.join(pytorch_dir, "runner.py"), "w", encoding="utf-8") as f:
            f.write(
                textwrap.dedent(
                    """
                    import torch
                    from torch import nn
                    """
                )
            )

        numpy_dir: str = os.path.join(temp_dir, "backends", "numpy")
        os.makedirs(numpy_dir)
        with open(os.path.join(numpy_dir, "runner.py"), "w", encoding="utf-8") as f:
            f.write(
                textwrap.dedent(
                    """
                    import numpy as np
                    import scipy.special
                    """
                )
            )

        assert check_dependencies(temp_dir) == []

        with patch.dict(BACKEND_ALLOWED, {"pytorch": {"torch", "torchvision"}}):
            with open(os.path.join(pytorch_dir, "vision.py"), "w", encoding="utf-8") as f:
                f.write("import torchvision")
            violations: list[str] = check_dependencies(temp_dir)
            assert any("imports multiple third-party roots" in v for v in violations)

        direct_violations: list[str] = []
        _validate_single_backend_dependency(temp_dir, {"custom": {"pkg_a", "pkg_b"}}, direct_violations)
        assert len(direct_violations) == 1


def test_check_module_name_empty_and_upward_imports() -> None:
    """Test empty module name handling and forbidden upward zero* imports."""
    violations: list[str] = []
    _check_module_name("", "file.py", 1, set(), ".", violations)
    assert violations == []

    with tempfile.TemporaryDirectory() as temp_dir:
        core_dir: str = os.path.join(temp_dir, "core")
        os.makedirs(core_dir)
        bad_file: str = os.path.join(core_dir, "bad_upward.py")
        with open(bad_file, "w", encoding="utf-8") as f:
            f.write(
                textwrap.dedent(
                    """
                    import zero_pytorch
                    import zero_zoo
                    """
                )
            )
        upward_violations: list[str] = check_dependencies(temp_dir)
        assert len(upward_violations) == 2
        assert any("zero_pytorch" in v for v in upward_violations)
        assert any("zero_zoo" in v for v in upward_violations)


def test_main_and_entrypoint() -> None:
    """Test main() return codes and __main__ CLI execution."""
    with patch("scripts.lint_dependencies.check_dependencies", return_value=[]):
        assert main() == 0

    with patch("scripts.lint_dependencies.check_dependencies", return_value=["test.py:1: error"]):
        assert main() == 1

    with patch.object(sys, "argv", ["lint_dependencies.py"]):
        with patch("scripts.lint_dependencies.check_dependencies", return_value=[]):
            with pytest.raises(SystemExit) as exc_info:
                runpy.run_path("scripts/lint_dependencies.py", run_name="__main__")
            assert exc_info.value.code == 0
