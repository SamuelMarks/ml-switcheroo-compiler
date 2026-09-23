"""Tests for dynamic numpy import isolation pre-commit hook."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import scripts.lint_dynamic_numpy as ldn
from scripts.lint_dynamic_numpy import check_for_prohibited_numpy_imports, main


def test_dynamic_numpy_import_detection() -> None:
    """Verify check_for_prohibited_numpy_imports detects importlib.import_module and __import__."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        bad_file = tmp_path / "bad.py"
        bad_file.write_text(
            chr(10).join(["import importlib", 'mod = importlib.import_module("numpy")']),
            encoding="utf-8",
        )

        bad_file2 = tmp_path / "bad2.py"
        bad_file2.write_text(
            'mod = __import__("numpy.linalg")',
            encoding="utf-8",
        )

        bad_file3 = tmp_path / "bad3.py"
        bad_file3.write_text(
            'mod = import_module("numpy")',
            encoding="utf-8",
        )

        ignored_file = tmp_path / "notes.txt"
        ignored_file.write_text('importlib.import_module("numpy")', encoding="utf-8")

        err_file = tmp_path / "syntax_err.py"
        err_file.write_text("def invalid syntax ::::", encoding="utf-8")

        clean_file = tmp_path / "clean.py"
        clean_file.write_text(
            chr(10).join(
                [
                    "import torch",
                    "other_call()",
                    "import_module()",
                    "import_module(123)",
                    'import_module("scipy")',
                ]
            ),
            encoding="utf-8",
        )

        violations = check_for_prohibited_numpy_imports(tmpdir)
        assert len(violations) == 3
        assert any("bad.py:2" in v for v in violations)
        assert any("bad2.py:1" in v for v in violations)
        assert any("bad3.py:1" in v for v in violations)


def test_dynamic_numpy_import_allowed_prefixes() -> None:
    """Verify backends/numpy and backends/eager paths are permitted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        np_dir = tmp_path / "src" / "ml_switcheroo_compiler" / "backends" / "numpy"
        np_dir.mkdir(parents=True)
        (np_dir / "types.py").write_text(
            chr(10).join(["import importlib", 'np = importlib.import_module("numpy")']),
            encoding="utf-8",
        )

        violations = check_for_prohibited_numpy_imports(str(tmp_path / "src" / "ml_switcheroo_compiler"))
        assert violations == []


def test_dynamic_numpy_import_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify main entrypoint returns 0 or 1 according to violations.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for monkeypatching.
    """
    monkeypatch.setattr("scripts.lint_dynamic_numpy.check_for_prohibited_numpy_imports", lambda d: [])
    assert main() == 0

    monkeypatch.setattr(
        "scripts.lint_dynamic_numpy.check_for_prohibited_numpy_imports",
        lambda d: ["some/path.py:1: Prohibited dynamic numpy import 'numpy'"],
    )
    assert main() == 1


def test_dynamic_numpy_import_cli() -> None:
    """Verify execution via main wrapper."""
    with patch.object(ldn, "main", return_value=0):
        with pytest.raises(SystemExit) as excinfo:
            sys.exit(ldn.main())
        assert excinfo.value.code == 0
