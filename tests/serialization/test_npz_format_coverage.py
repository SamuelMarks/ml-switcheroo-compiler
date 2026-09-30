"""Tests for test_npz_format_coverage."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.registry as registry
from ml_switcheroo_compiler.serialization.formats.npz import NpzWeightFormat


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


def test_npz_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify NpzWeightFormat default, backend, and fallback behavior.

    Args:
        tmp_path (Path): Pytest temporary path.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture.

    Returns:
        None
    """
    fmt = NpzWeightFormat()
    filepath = str(tmp_path / "model.npz")
    data = {"w1": np.array([1, 2]), "w2": np.array([3, 4])}

    # Default save and load
    fmt.save(data, filepath)
    loaded = fmt.load(filepath)
    assert "w1" in loaded
    np.testing.assert_array_equal(loaded["w1"], data["w1"])

    # Backend with save_npz and load_npz
    mock_b = _MockBackendNPZ()
    monkeypatch.setattr(registry, "get_active_backend", lambda: mock_b)
    fmt.save(data, filepath)
    assert mock_b.saved == (data, filepath)
    loaded_mock = fmt.load(filepath)
    assert "backend_key" in loaded_mock

    # Backend load_npz error with fallback warning
    class _FailingBackend:
        """Mock backend that raises on load_npz."""

        def load_npz(self, path: str) -> None:
            """Raise error on load.

            Args:
                path (str): Filepath.

            Raises:
                RuntimeError: Always raised.
            """
            raise RuntimeError("load error")

    monkeypatch.setattr(registry, "get_active_backend", lambda: _FailingBackend())
    with pytest.warns(UserWarning, match="Backend load_npz failed"):
        fallback_loaded = fmt.load(filepath)
    assert "w1" in fallback_loaded
