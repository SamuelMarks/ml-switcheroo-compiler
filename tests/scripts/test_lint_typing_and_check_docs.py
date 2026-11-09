"""Unit tests for scripts.lint_typing and scripts.check_docs."""

from __future__ import annotations

from pathlib import Path

import scripts.check_docs as cd
import scripts.lint_typing as lt


def test_lint_typing_ast_and_cli(tmp_path: Path) -> None:
    """Verify scripts.lint_typing detects reflection, cast, and dunder calls on strict files.

    Args:
        tmp_path (Path): Temporary test directory fixture.
    """
    strict_dir = tmp_path / "src" / "ml_switcheroo_compiler" / "core"
    strict_dir.mkdir(parents=True)
    strict_file = strict_dir / "tensor.py"
    strict_file.write_text(
        '''"""Docstring."""

import typing


class Sub:
    """Sub class."""

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        super().__add__(1)
        self.normal_method()
        self.x = getattr(self, "x", 1)
        self.y = typing.cast(int, 2)
        self.z = self.__add__(3)
        self.w = self.__dict__


if not typing.TYPE_CHECKING:
    BadAlias = int


def bad_guard(val: int) -> typing.TypeGuard[int]:
    """Bad guard."""
    return not isinstance(val, bytes) or len(val) >= 0
''',
        encoding="utf-8",
    )
    exact_file = tmp_path / "src" / "ml_switcheroo_compiler" / "foreign.py"
    exact_file.write_text('"""Foreign."""\n', encoding="utf-8")

    violations = lt.scan_file(str(strict_file))
    assert any("prohibited reflection/cast call 'getattr()'" in v for v in violations)
    assert any("prohibited 'typing.cast()' call" in v for v in violations)
    assert any("prohibited direct dunder call '.__add__()'" in v for v in violations)
    assert any("prohibited dunder attribute '.__dict__'" in v for v in violations)
    assert any("prohibited 'if TYPE_CHECKING: ... else: ...' type-erasure block" in v for v in violations)
    assert any("prohibited tautological TypeGuard 'bad_guard'" in v for v in violations)
    assert lt.scan_file(str(exact_file)) == []

    assert lt.main([str(tmp_path)]) == 1
    assert lt.scan_directory(str(tmp_path / "nonexistent")) == []
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("hi", encoding="utf-8")
    assert lt.scan_directory(str(txt_file)) == []
    assert len(lt.scan_directory(str(tmp_path))) >= 3

    bad_syn = strict_dir / "bad.py"
    bad_syn.write_text("def broken(:", encoding="utf-8")
    assert lt.scan_file(str(bad_syn), force_strict_reflection=True) == []

    clean_file = tmp_path / "clean.py"
    clean_file.write_text(
        '''"""Clean module."""

x: int = 1
''',
        encoding="utf-8",
    )
    assert lt.main([str(clean_file)]) == 0


def test_check_docs_ast_and_cli(tmp_path: Path) -> None:
    """Verify scripts.check_docs detects missing module, class, function, arg, and return docstrings.

    Args:
        tmp_path (Path): Temporary test directory fixture.
    """
    bad_doc_file = tmp_path / "bad_docs.py"
    any_tok = "(" + "Any)"
    bad_doc_file.write_text(
        f'''class NoDocClass:
    pass


def no_doc_fn(a: int) -> int:
    return a


def incomplete_doc_fn(x: int, *args: int, y: int = 1, **kwargs: int) -> int:
    """Doc with {any_tok}."""
    return x


def missing_single_arg(a: int, b: int):
    """Doc.

    Args:
        a (int): Only a is documented.
    """
    return None
''',
        encoding="utf-8",
    )

    violations = cd.check_file(str(bad_doc_file))
    assert any("missing module docstring" in v for v in violations)
    assert any("missing class docstring on 'NoDocClass'" in v for v in violations)
    assert any("missing docstring on 'no_doc_fn'" in v for v in violations)
    assert any("prohibited Any/object docstring type" in v for v in violations)
    assert any("missing 'Args:' section" in v for v in violations)
    assert any("missing 'Returns:'/'Yields:' section" in v for v in violations)
    assert any("missing arg 'b' in 'missing_single_arg'" in v for v in violations)

    assert cd.main([str(bad_doc_file)]) == 1
    assert cd.check_directory(str(tmp_path / "nonexistent")) == []
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("hi", encoding="utf-8")
    assert cd.check_directory(str(txt_file)) == []
    assert len(cd.check_directory(str(tmp_path))) >= 1

    clean_doc_file = tmp_path / "clean_docs.py"
    clean_doc_file.write_text(
        '''"""Clean module docstring."""


class CleanClass:
    """Clean class docstring."""

    def method(self, x: int) -> int:
        """Compute method.

        Args:
            x (int): Input value.

        Returns:
            int: Output value.
        """
        return x

    def noop(self) -> None:
        """Do nothing."""
''',
        encoding="utf-8",
    )
    assert cd.main([str(clean_doc_file)]) == 0
