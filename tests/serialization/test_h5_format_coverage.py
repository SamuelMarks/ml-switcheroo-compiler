"""Tests for test_h5_format_coverage."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import pytest

import ml_switcheroo_compiler.backends.registry as registry
import ml_switcheroo_compiler.serialization.formats.h5 as h5_mod
from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat


class _DummyGenerator:
    """Mock generator providing fallback prefix."""

    def get_fallback_prefix(self) -> str:
        """Return prefix for fallback ops.

        Returns:
            str: Prefix string.
        """
        return "np_eager"


class _MockArrayWithNumpy:
    """Mock object with numpy() method."""

    def __init__(self, arr: np.ndarray) -> None:
        """Initialize with numpy array.

        Args:
            arr (np.ndarray): Underlying array.
        """
        self.arr = arr

    def numpy(self) -> np.ndarray:
        """Return underlying numpy array.

        Returns:
            np.ndarray: Array.
        """
        return self.arr


class _MockDataWithNumpy:
    """Mock object with data.numpy() method."""

    def __init__(self, arr: np.ndarray) -> None:
        """Initialize with nested data object.

        Args:
            arr (np.ndarray): Underlying array.
        """
        self.data = _MockArrayWithNumpy(arr)


class _MockArrayWithToList:
    """Mock object with tolist() method."""

    def __init__(self, lst: list[float]) -> None:
        """Initialize with list.

        Args:
            lst (list[float]): List elements.
        """
        self.lst = lst

    def tolist(self) -> list[float]:
        """Return list representation.

        Returns:
            list[float]: The list.
        """
        return self.lst


class _NoShapeObject:
    """Mock object with dtype and tobytes, but missing shape."""

    dtype: str = "float32"

    def tobytes(self) -> bytes:
        """Return empty bytes.

        Returns:
            bytes: Empty bytes.
        """
        return b""


class _NoTobytesObject:
    """Mock object with dtype and shape, but missing tobytes."""

    dtype: str = "float32"
    shape: tuple[int, ...] = (2,)


class _MockBackendH5:
    """Mock backend implementing custom h5 save and load methods."""

    def __init__(self) -> None:
        """Initialize mock backend."""
        self.saved: tuple[dict[str, np.ndarray], str] | None = None

    def load_h5(self, path: str) -> dict[str, np.ndarray]:
        """Load h5 mock weights.

        Args:
            path (str): Filepath.

        Returns:
            dict[str, np.ndarray]: Dict of weights.
        """
        return {"backend_key": np.array([42.0])}

    def save_h5(self, weights: dict[str, np.ndarray], path: str) -> None:
        """Save h5 mock weights.

        Args:
            weights (dict[str, np.ndarray]): Weights dict.
            path (str): Filepath.
        """
        self.saved = (weights, path)


class _MockBackendNPZ:
    """Mock backend implementing custom npz save and load methods."""

    def __init__(self) -> None:
        """Initialize mock backend."""
        self.saved: tuple[dict[str, np.ndarray], str] | None = None

    def save_npz(self, weights: dict[str, np.ndarray], path: str) -> None:
        """Save npz mock weights.

        Args:
            weights (dict[str, np.ndarray]): Weights dict.
            path (str): Filepath.
        """
        self.saved = (weights, path)

    def load_npz(self, path: str) -> dict[str, np.ndarray]:
        """Load npz mock weights.

        Args:
            path (str): Filepath.

        Returns:
            dict[str, np.ndarray]: Dict of weights.
        """
        return {"backend_key": np.array([100.0])}


class _MockBackendSafetensors:
    """Mock backend implementing custom from_buffer method."""

    def from_buffer(self, buffer: bytes, dtype: str, shape: Sequence[int]) -> np.ndarray:
        """Reconstruct array from buffer.

        Args:
            buffer (bytes): Raw bytes.
            dtype (str): Dtype identifier.
            shape (Sequence[int]): Shape tuple or list.

        Returns:
            np.ndarray: Reconstructed array.
        """
        return np.frombuffer(buffer, dtype=np.float32).reshape(tuple(shape))


class _PlainBackend:
    """Plain backend without custom format hooks."""

    pass


def test_h5_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify H5WeightFormat with all data conversions and backend hooks.

    Args:
        tmp_path (Path): Pytest temporary path.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture.

    Returns:
        None
    """
    # 1. Missing h5py module import branch via reload
    monkeypatch.setitem(sys.modules, "h5py", None)
    importlib.reload(h5_mod)
    assert h5_mod.h5py is None

    fmt_missing = H5WeightFormat()
    dummy_path = str(tmp_path / "missing.h5")
    with pytest.raises(ImportError, match="h5py is required"):
        fmt_missing.load(dummy_path)
    with pytest.raises(ImportError, match="h5py is required"):
        fmt_missing.save({}, dummy_path)

    # Restore h5py
    monkeypatch.undo()
    importlib.reload(h5_mod)
    assert h5_mod.h5py is not None

    fmt = H5WeightFormat()

    # 2. Backend hooks load_h5 and save_h5
    mock_b = _MockBackendH5()
    monkeypatch.setattr(registry, "get_active_backend", lambda: mock_b)
    res = fmt.load(dummy_path)
    assert "backend_key" in res
    fmt.save({"w": np.array([1.0])}, dummy_path)
    assert mock_b.saved is not None
    assert mock_b.saved[1] == dummy_path

    # 3. Plain backend without custom hooks
    monkeypatch.setattr(registry, "get_active_backend", lambda: _PlainBackend())

    filepath = str(tmp_path / "actual.h5")
    data = {
        "v_arr": np.array([1.0, 2.0]),
        "v_numpy": _MockArrayWithNumpy(np.array([3.0, 4.0])),
        "v_data_numpy": _MockDataWithNumpy(np.array([5.0, 6.0])),
        "v_tolist": _MockArrayWithToList([7.0, 8.0]),
        "v_raw": 100,
    }
    fmt.save(data, filepath)

    # Insert a group to exercise both branches of dataset visiting
    with h5py.File(filepath, "a") as f:
        grp = f.create_group("subgroup")
        grp.create_dataset("nested", data=np.array([9.0]))

    loaded = fmt.load(filepath)
    assert "v_arr" in loaded
    assert "v_numpy" in loaded
    assert "v_data_numpy" in loaded
    assert "v_tolist" in loaded
    assert "v_raw" in loaded
    assert "subgroup/nested" in loaded
