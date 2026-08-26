"""Tests for the generate_stubs script."""

import json
from unittest.mock import mock_open, patch

from pytest import CaptureFixture

import scripts.generate_stubs as gs


def test_generate_stubs_no_snapshot_dir(capsys: CaptureFixture[str]) -> None:
    """Test generate_stubs when snapshot directory does not exist.

    Args:
        capsys: Pytest fixture for capturing stdout/stderr.
    """
    with patch("os.path.exists", return_value=False):
        gs.main()

    captured = capsys.readouterr()
    assert "Snapshot directory not found" in captured.out


def test_generate_stubs_success(capsys: CaptureFixture[str]) -> None:
    """Test generate_stubs successfully.

    Args:
        capsys: Pytest fixture for capturing stdout/stderr.
    """
    mock_data = {"categories": {"math": [{"name": "add", "kind": "function"}, {"name": "sub", "kind": "function"}, {"name": "invalid-name", "kind": "function"}, {"name": "__private", "kind": "function"}, {"name": "Constant", "kind": "class"}]}}
    mock_json_str = json.dumps(mock_data)

    def mock_exists(path: str) -> bool:
        if "ml-framework-snapshots" in path:
            return True
        if "src/ml_switcheroo_compiler/backends/" in path:
            return True
        return False

    def mock_listdir(path: str) -> list[str]:
        if "ml-framework-snapshots" in path:
            return ["numpy_v1.0.json", "numpy_v2.0.json", "torch_v1.json"]
        return []

    m_open = mock_open(read_data=mock_json_str)

    with patch("os.path.exists", side_effect=mock_exists), patch("os.listdir", side_effect=mock_listdir), patch("builtins.open", m_open):
        gs.main()

    captured = capsys.readouterr()
    assert "Generated src/ml_switcheroo_compiler/backends/numpy/stub.pyi" in captured.out

    # Check that open was called for reading JSON and writing pyi
    m_open.assert_any_call("src/ml_switcheroo_compiler/backends/numpy/stub.pyi", "w")

    # Check the written content
    written = "".join([str(call.args[0]) for call in m_open().write.call_args_list])
    assert "def add(*args: Tensor, **kwargs: Tensor) -> Tensor: ..." in written
    assert "def sub(*args: Tensor, **kwargs: Tensor) -> Tensor: ..." in written
    assert "invalid" not in written
    assert "__private" not in written
    assert "Constant" not in written


def test_generate_stubs_no_files() -> None:
    """Test generate_stubs when there are no snapshot files for a framework."""

    def mock_exists(path: str) -> bool:
        return True

    def mock_listdir(path: str) -> list[str]:
        return []

    with patch("os.path.exists", side_effect=mock_exists), patch("os.listdir", side_effect=mock_listdir):
        gs.main()


def test_generate_stubs_missing_backend_dir() -> None:
    """Test generate_stubs when backend dir does not exist."""
    mock_data = {"categories": {}}

    def mock_exists(path: str) -> bool:
        if "ml-framework-snapshots" in path:
            return True
        # Backend dir doesn't exist
        return False

    def mock_listdir(path: str) -> list[str]:
        return ["numpy_v1.0.json"]

    m_open = mock_open(read_data=json.dumps(mock_data))

    with patch("os.path.exists", side_effect=mock_exists), patch("os.listdir", side_effect=mock_listdir), patch("builtins.open", m_open):
        gs.main()

    # write should not be called since backend dir is missing
    m_open().write.assert_not_called()
