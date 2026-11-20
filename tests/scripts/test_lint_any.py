"""Unit tests for kitchen-sink union, mega-alias, and Any/object rules in `scripts.lint_any`."""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.lint_any as la


def test_lint_any_kitchen_sink_unions_and_mega_aliases(tmp_path: Path) -> None:
    """Verify `scripts.lint_any` flags >8-member unions, primitive+meta unions, and legacy mega-union names.

    Args:
        tmp_path (Path): Temporary directory fixture.
    """
    src_core = tmp_path / "src" / "ml_switcheroo_compiler" / "core"
    src_core.mkdir(parents=True)
    bad_file = src_core / "mega_unions.py"
    bad_file.write_text(
        '''"""Fixture with prohibited mega-unions and kitchen-sink aliases."""

import types
import typing
from pydantic import BaseModel

from ml_switcheroo_compiler.backends.eager_registry import BackendArg, EagerValue

TensorOrScalar = int | float
GradValue: typing.TypeAlias = int | float
EvalValue: TypeAlias = int | str
AttrValue = str | bytes

SingleMemberUnion = typing.Union[int]
AnnotatedWideUnion = typing.Annotated[
    typing.Union[int, float, str, bool, bytes, complex, list[int], dict[str, int], tuple[int, ...]],
    "meta",
]
MetaKitchenSink = int | types.ModuleType | types.FunctionType | BaseModel | "list[int]" | "not valid syntax!"
SoundSmallUnion: typing.TypeAlias = int | float | str | bool
RegularAnnotatedVar: int = 5
''',
        encoding="utf-8",
    )

    violations = la.scan_file_for_any(str(bad_file))
    assert any("prohibited import 'BackendArg' outside backends/" in v for v in violations)
    assert any("prohibited import 'EagerValue' outside backends/" in v for v in violations)
    assert any("disguised Any alias 'TensorOrScalar'" in v for v in violations)
    assert any("disguised Any alias 'GradValue'" in v for v in violations)
    assert any("disguised Any alias 'EvalValue'" in v for v in violations)
    assert any("disguised Any alias 'AttrValue'" in v for v in violations)
    assert any("prohibited kitchen-sink union alias 'AnnotatedWideUnion' (9 members > 8)" in v for v in violations)
    assert any(
        "prohibited kitchen-sink union alias 'MetaKitchenSink' combining primitives and runtime meta-types" in v
        for v in violations
    )
    assert not any("SoundSmallUnion" in v for v in violations)
    assert not any("SingleMemberUnion" in v for v in violations)

    with pytest.raises(SystemExit) as exc_fail:
        la.main(["--target-dir", str(tmp_path), "--max-count", "0"])
    assert exc_fail.value.code == 1

    clean_dir = tmp_path / "clean_pkg"
    clean_dir.mkdir()
    (clean_dir / "ok.py").write_text(
        '''"""Clean."""

ValidUnion = int | float | str
''',
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc_ok:
        la.main(["--target-dir", str(clean_dir), "--max-count", "0"])
    assert exc_ok.value.code == 0
