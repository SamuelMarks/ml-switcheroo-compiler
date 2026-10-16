"""Tests for AST-based linting scripts scripts.lint_pass and scripts.lint_any."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.lint_any as la
import scripts.lint_pass as lp


def test_lint_pass_clean_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify that main() prints the success message and exits with status 0 on clean tests.

    Args:
        tmp_path (Path): Temporary test directory fixture.
        capsys (pytest.CaptureFixture[str]): Pytest fixture to capture stdout/stderr.
    """
    clean_test = tmp_path / "test_clean.py"
    clean_test.write_text(
        """def test_ok() -> None:
    x = 1 + 1
    assert x == 2
""",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc_info:
        lp.main(["--tests-dir", str(tmp_path), "--check-zero-assertions"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "Linting passed." in captured.out


def test_lint_pass_detects_vacuous_and_tautological_assertions(tmp_path: Path) -> None:
    """Verify AST detection of assert True, assert False, assert x == x, assert x is x, and zero assertions.

    Args:
        tmp_path (Path): Temporary test directory fixture.
    """
    bad_file = tmp_path / "test_bad.py"
    bad_file.write_text(
        """def test_vacuous() -> None:
    assert True
    assert False
    x = 5
    assert x == x
    assert x is x

def test_empty() -> None:
    y = 10

def test_swallowed() -> None:
    try:
        val = 1 / 0
    except Exception:
        pass
    try:
        val = 1 / 0
    except:
        pass
    assert val == 0
""",
        encoding="utf-8",
    )
    errors = lp.check_test_file_ast(str(bad_file), check_zero_assertions=True)
    assert any("assert True" in e for e in errors)
    assert any("assert False" in e for e in errors)
    assert sum("Tautological comparison" in e for e in errors) == 2
    assert any("Zero assertions in test function 'test_empty'" in e for e in errors)
    assert any("except Exception: pass" in e for e in errors)
    assert any("except: pass" in e for e in errors)

    with pytest.raises(SystemExit) as exc_info:
        lp.main(["--tests-dir", str(tmp_path), "--check-zero-assertions"])
    assert exc_info.value.code == 1


def test_lint_any_ast_scanner(tmp_path: Path) -> None:
    """Verify AST detection of typing.object annotations, and AnyType aliases.

    Args:
        tmp_path (Path): Temporary test directory fixture.
    """
    src_file = tmp_path / "mod.py"
    src_file.write_text(
        """import typing
AnyType = int
x: object = 1
def fn(a: *args: object, **kwargs: typing.Any) -> Any:
    return a
""",
        encoding="utf-8",
    )
    violations = la.scan_directory_for_any(str(tmp_path))
    assert any("prohibited import 'Any'" in v for v in violations)
    assert any("disguised Any alias 'AnyType'" in v for v in violations)
    assert any("prohibited variable annotation 'object'" in v for v in violations)
    assert any("prohibited parameter annotation 'Any'" in v for v in violations)
    assert any("prohibited *args annotation 'object'" in v for v in violations)
    assert any("prohibited **kwargs annotation 'typing.Any'" in v for v in violations)
    assert any("prohibited return annotation 'Any'" in v for v in violations)

    with pytest.raises(SystemExit) as exc_fail:
        la.main(["--target-dir", str(tmp_path), "--max-count", "0"])
    assert exc_fail.value.code == 1

    with pytest.raises(SystemExit) as exc_pass:
        la.main(["--target-dir", str(tmp_path), "--max-count", "20"])
    assert exc_pass.value.code == 0


def test_lint_pass_module_main() -> None:
    """Verify executing scripts.lint_pass as __main__."""
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("scripts.lint_pass", run_name="__main__")
    assert exc_info.value.code in (0, 1)


def test_lint_pass_subprocess(tmp_path: Path) -> None:
    """Verify our custom lint pass successfully executes in a standalone process.

    Args:
        tmp_path (Path): Temporary test directory fixture.
    """
    clean_test = tmp_path / "test_sub.py"
    clean_test.write_text(
        """def test_one() -> None:
    assert 2 + 2 == 4
""",
        encoding="utf-8",
    )
    result: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "scripts/lint_pass.py", "--tests-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Linting passed." in result.stdout
