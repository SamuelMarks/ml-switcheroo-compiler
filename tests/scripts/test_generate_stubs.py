"""Tests for the generate_stubs script."""

import json
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

from pytest import CaptureFixture

import scripts.generate_stubs as gs


def test_generate_stubs_no_snapshot_dir(capsys: CaptureFixture[str]) -> None:
    """Test generate_stubs when snapshot directory does not exist.

    Args:
        capsys: Pytest fixture for capturing stdout/stderr.
    """
    with patch("os.path.exists", return_value=False):
        gs.generate_stubs()

    captured = capsys.readouterr()
    assert "Snapshot directory not found" in captured.out


def test_generate_stubs_success(capsys: CaptureFixture[str], tmp_path: Path) -> None:
    """Test generate_stubs successfully.

    Args:
        capsys: Pytest fixture for capturing stdout/stderr.
        tmp_path: Pytest fixture for temp path.
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

    # We create a real file structure to test easily
    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir()
    (snap_dir / "numpy_v1.json").write_text(json.dumps(mock_data))

    out_base = tmp_path / "backends"
    numpy_dir = out_base / "numpy"
    numpy_dir.mkdir(parents=True)

    gs.generate_stubs(snapshot_dir=str(snap_dir), out_base_dir=str(out_base))

    captured = capsys.readouterr()
    assert "Generated" in captured.out

    stub_path = numpy_dir / "stub.pyi"
    assert stub_path.exists()
    written = stub_path.read_text()

    assert "def add(*args: Tensor, **kwargs: Tensor) -> Tensor: ..." in written
    assert "def sub(*args: Tensor, **kwargs: Tensor) -> Tensor: ..." in written
    assert "invalid" not in written
    assert "__private" not in written
    assert "Constant" not in written


def test_generate_stubs_malformed_categories(tmp_path: Path) -> None:
    """Test when categories is not a dict."""
    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir()
    (snap_dir / "numpy_v1.json").write_text(json.dumps({"categories": ["not", "a", "dict"]}))

    out_base = tmp_path / "backends"
    numpy_dir = out_base / "numpy"
    numpy_dir.mkdir(parents=True)

    gs.generate_stubs(snapshot_dir=str(snap_dir), out_base_dir=str(out_base))
    assert (numpy_dir / "stub.pyi").exists()


def test_generate_stubs_no_files(tmp_path: Path) -> None:
    """Test generate_stubs when there are no snapshot files for a framework."""
    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir()
    # No json files inside

    out_base = tmp_path / "backends"
    out_base.mkdir()

    gs.generate_stubs(snapshot_dir=str(snap_dir), out_base_dir=str(out_base))


def test_generate_stubs_missing_backend_dir(tmp_path: Path) -> None:
    """Test generate_stubs when backend dir does not exist."""
    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir()
    (snap_dir / "numpy_v1.json").write_text(json.dumps({}))

    out_base = tmp_path / "backends"
    # we DO NOT create numpy directory inside out_base

    gs.generate_stubs(snapshot_dir=str(snap_dir), out_base_dir=str(out_base))
    assert not (out_base / "numpy" / "stub.pyi").exists()


def test_main(capsys: CaptureFixture[str]) -> None:
    """Test main entry point and __main__ block."""
    with patch("scripts.generate_stubs.generate_stubs") as mock_gen:
        gs.main()
        mock_gen.assert_called_once()

    with patch.object(sys, "argv", ["generate_stubs.py"]):
        with patch("os.path.exists", return_value=False):
            runpy.run_path("scripts/generate_stubs.py", run_name="__main__")

    captured = capsys.readouterr()
    assert "Snapshot directory not found" in captured.out
