"""Tests for test_backend_package_inits."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

import ml_switcheroo_compiler.backends.awkward as awk_mod
import ml_switcheroo_compiler.backends.eager.core_math_ops as cmo_mod
import ml_switcheroo_compiler.backends.numpy as bnp_mod
import ml_switcheroo_compiler.backends.pyarrow_compute as pa_mod
import ml_switcheroo_compiler.ops.info_and_histograms as iah_mod
import ml_switcheroo_compiler.ops.text as text_mod
from ml_switcheroo_compiler.backends.awkward.generator import AwkwardGenerator
from ml_switcheroo_compiler.backends.numpy import distributed as bnp_dist_mod
from ml_switcheroo_compiler.backends.pyarrow_compute.generator import PyArrowComputeGenerator


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


def test_package_inits() -> None:
    """Verify package __init__.py modules and bound classmethods.

    Returns:
        None
    """
    assert awk_mod.AwkwardGenerator is AwkwardGenerator
    assert hasattr(awk_mod.AwkwardGenerator, "zeros")
    assert hasattr(awk_mod.AwkwardGenerator, "array")
    assert hasattr(awk_mod.AwkwardGenerator, "get_layout")
    assert "AwkwardGenerator" in awk_mod.__all__

    assert pa_mod.PyArrowComputeGenerator is PyArrowComputeGenerator
    assert hasattr(pa_mod.PyArrowComputeGenerator, "zeros")
    assert hasattr(pa_mod.PyArrowComputeGenerator, "to_table")
    assert hasattr(pa_mod.PyArrowComputeGenerator, "from_arrow")
    assert "PyArrowComputeGenerator" in pa_mod.__all__

    assert cmo_mod._accumulate_n is not None
    assert bnp_mod.NumpyGenerator is not None
    assert bnp_dist_mod.__doc__ is not None
    assert iah_mod.histogram is not None
    assert text_mod.AsStringConfig is not None
