"""Tests for test_safetensors_format_coverage."""

from __future__ import annotations

import json
import struct
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.registry as registry
from ml_switcheroo_compiler.serialization.formats.safetensors import SafetensorsWeightFormat


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


def test_safetensors_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify SafetensorsWeightFormat save, load, and header parsing branches.

    Args:
        tmp_path (Path): Pytest temporary path.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture.

    Returns:
        None
    """
    fmt = SafetensorsWeightFormat()
    filepath = str(tmp_path / "model.safetensors")

    # 1. Save with skipped values exercising each conditional branch of hasattr checks
    data = {
        "no_dtype": "string_not_array",
        "no_shape": _NoShapeObject(),
        "no_tobytes": _NoTobytesObject(),
        "w1": np.array([1.0, 2.0], dtype=np.float32),
        "w2": np.array([10, 20], dtype=np.int32),
    }
    fmt.save(data, filepath)

    # 2. Load without backend from_buffer hook (raw buffer dict returned)
    monkeypatch.setattr(registry, "get_active_backend", lambda: _PlainBackend())
    loaded_plain = fmt.load(filepath)
    assert "w1" in loaded_plain
    assert "buffer" in loaded_plain["w1"]

    # 3. Load with backend from_buffer hook
    monkeypatch.setattr(registry, "get_active_backend", lambda: _MockBackendSafetensors())
    loaded_hook = fmt.load(filepath)
    assert "w1" in loaded_hook
    np.testing.assert_allclose(loaded_hook["w1"], data["w1"])

    # 4. Empty / corrupted file (< 8 bytes)
    empty_file = str(tmp_path / "empty.safetensors")
    with open(empty_file, "wb") as f:
        f.write(b"123")
    assert fmt.load(empty_file) == {}

    # 5. Header containing __metadata__ entry
    meta_file = str(tmp_path / "meta.safetensors")
    header_dict = {
        "__metadata__": {"framework": "test"},
        "w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
    }
    header_bytes = json.dumps(header_dict).encode("utf-8")
    header_len = len(header_bytes)
    data_bytes = np.array([1.0, 2.0], dtype=np.float32).tobytes()

    with open(meta_file, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_bytes)
        f.write(data_bytes)

    loaded_meta = fmt.load(meta_file)
    assert "__metadata__" not in loaded_meta
    assert "w" in loaded_meta
