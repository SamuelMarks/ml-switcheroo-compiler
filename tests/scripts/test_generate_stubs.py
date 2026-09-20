"""Tests for the generate_stubs script."""

from __future__ import annotations

import inspect
import json
import runpy
import sys
import types
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

from pytest import CaptureFixture

import scripts.generate_stubs as gs


def test_clean_type_annotation() -> None:
    """Verify _clean_type_annotation across safe types, None, arrays, and syntax errors."""
    assert gs._clean_type_annotation(None) == "Tensor"
    assert gs._clean_type_annotation("") == "Tensor"
    assert gs._clean_type_annotation("None") == "Tensor"
    assert gs._clean_type_annotation("array_like") == "Tensor"
    assert gs._clean_type_annotation("`np._NoValue`") == "None"
    assert gs._clean_type_annotation("int | float") == "Tensor"
    assert gs._clean_type_annotation("{a, b}") == "Tensor"
    assert gs._clean_type_annotation("invalid-syntax") == "Tensor"
    assert gs._clean_type_annotation("UnknownType") == "Tensor"
    assert gs._clean_type_annotation("int") == "int"
    assert gs._clean_type_annotation("float") == "float"
    assert gs._clean_type_annotation("bool") == "bool"
    assert gs._clean_type_annotation("str") == "str"
    assert gs._clean_type_annotation("Optional[int]") == "Optional[int]"
    assert gs._clean_type_annotation("123 syntax error !@#") == "Tensor"


def test_format_param_stub() -> None:
    """Verify _format_param_stub for positional, var-positional, var-keyword, and invalid names."""
    assert gs._format_param_stub({}) is None
    assert gs._format_param_stub({"name": ""}) is None
    assert gs._format_param_stub({"name": "**kwargs"}) is None
    assert gs._format_param_stub({"name": "a, b"}) is None
    assert gs._format_param_stub({"name": "a b"}) is None
    assert gs._format_param_stub({"name": "123invalid"}) is None
    assert gs._format_param_stub({"name": "x", "kind": "VAR_POSITIONAL"}) == "*x: Tensor"
    assert gs._format_param_stub({"name": "kw", "kind": "VAR_KEYWORD"}) == "**kw: Tensor"
    assert gs._format_param_stub({"name": "x", "default": 0}) == "x: Tensor = ..."
    assert gs._format_param_stub({"name": "x"}) == "x: Tensor"


def test_generate_stubs_from_snapshot_exhaustive(tmp_path: Path) -> None:
    """Verify _generate_stubs_from_snapshot with complex parameters, returns, and fallbacks.

    Args:
        tmp_path (Path): Pytest temporary path fixture.
    """
    out_file = tmp_path / "snapshot_stubs.pyi"
    snap_data = {
        "categories": {
            "math": [
                {
                    "name": "f_full",
                    "kind": "function",
                    "params": [
                        {"name": "x", "kind": "POSITIONAL_OR_KEYWORD"},
                        {"name": "y", "kind": "POSITIONAL_OR_KEYWORD", "default": 0},
                        {"name": "args", "kind": "VAR_POSITIONAL"},
                        {"name": "kw1", "kind": "KEYWORD_ONLY"},
                        {"name": "kwargs", "kind": "VAR_KEYWORD"},
                        "not_dict_param",
                        {"name": "invalid,name"},
                    ],
                    "returns": "int",
                },
                {
                    "name": "f_kw_only_star",
                    "kind": "function",
                    "params": [{"name": "kw1", "kind": "KEYWORD_ONLY"}],
                    "returns": "UnknownReturn",
                },
                {
                    "name": "f_empty",
                    "kind": "function",
                    "params": [],
                },
                {
                    "name": "f_non_list_params",
                    "kind": "function",
                    "params": "not_a_list",
                },
                {"name": "Tensor", "kind": "function"},
                {"name": "_private", "kind": "function"},
                {"name": "123invalid", "kind": "function"},
                {"name": "f_class", "kind": "class"},
                "not_dict_item",
            ],
            "invalid_category": "not_a_list",
        }
    }
    count = gs._generate_stubs_from_snapshot(snap_data, "numpy", str(out_file))
    assert count >= 3
    content = out_file.read_text(encoding="utf-8")
    assert "def f_full(x: Tensor, y: Tensor = ..., *args: Tensor, kw1: Tensor, **kwargs: Tensor) -> int: ..." in content
    assert "def f_kw_only_star(*, kw1: Tensor) -> Tensor: ..." in content
    assert "def f_empty(*args: Tensor, **kwargs: Tensor) -> Tensor: ..." in content
    assert "def f_non_list_params(*args: Tensor, **kwargs: Tensor) -> Tensor: ..." in content


def test_generate_stubs_from_live_module_exhaustive(tmp_path: Path) -> None:
    """Verify _generate_stubs_from_live_module with imports, signatures, and fallback handlers.

    Args:
        tmp_path (Path): Pytest temporary path fixture.
    """
    out_file = tmp_path / "live_stubs.pyi"

    # 1. Unimportable module branch
    count_unimportable = gs._generate_stubs_from_live_module("nonexistent_module_xyz", "dummy", str(out_file))
    assert count_unimportable == 0
    assert "def __getattr__(name: str) -> Tensor: ..." in out_file.read_text(encoding="utf-8")

    # 2. Fake module with inspectable functions and signature failure
    fake_mod = types.ModuleType("fake_fw")
    fake_mod._priv = 1  # type: ignore[attr-defined]
    fake_mod.Tensor = 2  # type: ignore[attr-defined]
    fake_mod.non_callable = 42  # type: ignore[attr-defined]

    def f_normal(x: object, y: object = 10) -> None:
        pass

    def f_varargs(*args: object, **kwargs: object) -> None:
        pass

    def f_kwonly(*, k1: object, k2: object = 20) -> None:
        pass

    def f_varpos_kw(*args: object, k1: object) -> None:
        pass

    class BadSig:
        """Callable that throws TypeError when inspected."""

        fail_sig = True

        def __call__(self) -> None:
            pass

    bad_obj = BadSig()
    fake_mod.f_normal = f_normal  # type: ignore[attr-defined]
    fake_mod.f_varargs = f_varargs  # type: ignore[attr-defined]
    fake_mod.f_kwonly = f_kwonly  # type: ignore[attr-defined]
    fake_mod.f_varpos_kw = f_varpos_kw  # type: ignore[attr-defined]
    fake_mod.bad_sig = bad_obj  # type: ignore[attr-defined]

    real_sig = inspect.signature

    def smart_sig(obj: Callable[..., object]) -> inspect.Signature:
        if getattr(obj, "fail_sig", False):
            raise ValueError("Signature failed")
        return real_sig(obj)

    with patch("scripts.generate_stubs.importlib.import_module", return_value=fake_mod):
        with patch("inspect.signature", side_effect=smart_sig):
            count_live = gs._generate_stubs_from_live_module("fake_fw", "dummy", str(out_file))
            assert count_live >= 5

    content = out_file.read_text(encoding="utf-8")
    assert "def f_normal(x: Tensor, y: Tensor = ...) -> Tensor: ..." in content
    assert "def f_varargs(*args: Tensor, **kwargs: Tensor) -> Tensor: ..." in content
    assert "def f_kwonly(*, k1: Tensor, k2: Tensor = ...) -> Tensor: ..." in content
    assert "def f_varpos_kw(*args: Tensor, k1: Tensor) -> Tensor: ..." in content
    assert "def bad_sig(*args: Tensor, **kwargs: Tensor) -> Tensor: ..." in content


def test_generate_stubs_corrupt_json_fallback(tmp_path: Path) -> None:
    """Verify fallback from corrupt JSON to live module inspection.

    Args:
        tmp_path (Path): Pytest temporary path fixture.
    """
    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir()
    # Write invalid JSON
    (snap_dir / "numpy_v1.json").write_text("corrupted json {", encoding="utf-8")

    out_base = tmp_path / "backends"
    numpy_dir = out_base / "numpy"
    numpy_dir.mkdir(parents=True)

    with patch("scripts.generate_stubs._generate_stubs_from_live_module", return_value=10) as mock_live:
        gs.generate_stubs(snapshot_dir=str(snap_dir), out_base_dir=str(out_base))
        mock_live.assert_called()


def test_generate_stubs_no_snapshot_dir(capsys: CaptureFixture[str]) -> None:
    """Test generate_stubs when snapshot directory does not exist.

    Args:
        capsys (CaptureFixture[str]): Pytest fixture for capturing stdout/stderr.
    """
    with patch("os.path.exists", return_value=False):
        gs.generate_stubs()

    captured = capsys.readouterr()
    assert "Snapshot directory not found" in captured.out


def test_generate_stubs_success(capsys: CaptureFixture[str], tmp_path: Path) -> None:
    """Test generate_stubs successfully.

    Args:
        capsys (CaptureFixture[str]): Pytest fixture for capturing stdout/stderr.
        tmp_path (Path): Pytest fixture for temp path.
    """
    mock_data = {
        "categories": {
            "math": [
                {"name": "add", "kind": "function"},
                {"name": "sub", "kind": "function"},
                {"name": "invalid-name", "kind": "function"},
                {"name": "__private", "kind": "function"},
                {"name": "Constant", "kind": "class"},
                "not a dict",
            ],
            "not a list": "string value",
        }
    }

    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir()
    (snap_dir / "numpy_v1.json").write_text(json.dumps(mock_data), encoding="utf-8")

    out_base = tmp_path / "backends"
    numpy_dir = out_base / "numpy"
    numpy_dir.mkdir(parents=True)

    gs.generate_stubs(snapshot_dir=str(snap_dir), out_base_dir=str(out_base))

    captured = capsys.readouterr()
    assert "Generated" in captured.out

    stub_path = numpy_dir / "snapshot_stubs.pyi"
    assert stub_path.exists()
    written = stub_path.read_text(encoding="utf-8")

    assert "def add(*args: Tensor, **kwargs: Tensor) -> Tensor: ..." in written
    assert "def sub(*args: Tensor, **kwargs: Tensor) -> Tensor: ..." in written
    assert "invalid" not in written
    assert "__private" not in written
    assert "Constant" not in written


def test_generate_stubs_malformed_categories(tmp_path: Path) -> None:
    """Test when categories is not a dict.

    Args:
        tmp_path (Path): Pytest fixture for temp path.
    """
    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir()
    (snap_dir / "numpy_v1.json").write_text(json.dumps({"categories": ["not", "a", "dict"]}), encoding="utf-8")

    out_base = tmp_path / "backends"
    numpy_dir = out_base / "numpy"
    numpy_dir.mkdir(parents=True)

    gs.generate_stubs(snapshot_dir=str(snap_dir), out_base_dir=str(out_base))
    assert (numpy_dir / "snapshot_stubs.pyi").exists()


def test_generate_stubs_no_files(tmp_path: Path) -> None:
    """Test generate_stubs when there are no snapshot files for a framework.

    Args:
        tmp_path (Path): Pytest fixture for temp path.
    """
    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir()

    out_base = tmp_path / "backends"
    numpy_dir = out_base / "numpy"
    numpy_dir.mkdir(parents=True)

    with patch("scripts.generate_stubs._generate_stubs_from_live_module", return_value=5) as mock_live:
        gs.generate_stubs(snapshot_dir=str(snap_dir), out_base_dir=str(out_base))
        mock_live.assert_called_once()


def test_generate_stubs_file_not_dir(tmp_path: Path) -> None:
    """Test generate_stubs when snapshot_dir is a file rather than a directory.

    Args:
        tmp_path (Path): Pytest fixture for temp path.
    """
    snap_file = tmp_path / "snapshot_file"
    snap_file.write_text("not a directory", encoding="utf-8")

    out_base = tmp_path / "backends"
    numpy_dir = out_base / "numpy"
    numpy_dir.mkdir(parents=True)

    with patch("scripts.generate_stubs._generate_stubs_from_live_module", return_value=1) as mock_live:
        gs.generate_stubs(snapshot_dir=str(snap_file), out_base_dir=str(out_base))
        mock_live.assert_called_once()


def test_generate_stubs_missing_backend_dir(tmp_path: Path) -> None:
    """Test generate_stubs when backend dir does not exist.

    Args:
        tmp_path (Path): Pytest fixture for temp path.
    """
    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir()
    (snap_dir / "numpy_v1.json").write_text(json.dumps({}), encoding="utf-8")

    out_base = tmp_path / "backends"

    gs.generate_stubs(snapshot_dir=str(snap_dir), out_base_dir=str(out_base))
    assert not (out_base / "numpy" / "snapshot_stubs.pyi").exists()


def test_main_and_runpy(capsys: CaptureFixture[str]) -> None:
    """Test main entry point and __main__ block.

    Args:
        capsys (CaptureFixture[str]): Pytest fixture for capturing stdout/stderr.
    """
    with patch("scripts.generate_stubs.generate_stubs") as mock_gen:
        gs.main()
        mock_gen.assert_called_once()

    with patch.object(sys, "argv", ["generate_stubs.py"]):
        with patch("os.path.exists", return_value=False):
            runpy.run_module("scripts.generate_stubs", run_name="__main__")

    captured = capsys.readouterr()
    assert "Snapshot directory not found" in captured.out
