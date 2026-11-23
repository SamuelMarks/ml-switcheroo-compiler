"""Behavioral unit tests for safetensors."""

from __future__ import annotations

import json
import os
import struct
import tempfile
import typing
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from pytest_mock import MockerFixture

import ml_switcheroo_compiler.backends.registry as registry
from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.serialization.formats.safetensors import SafetensorsWeightFormat
from tests.testing_utils import GenericNDArray

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_safetensors_format() -> None:
    """Test function test_safetensors_format."""
    from ml_switcheroo_compiler.core.shape import OpValue

    fmt = SafetensorsWeightFormat()
    data: dict[str, OpValue] = {
        "a": np.array([1.0, 2.0], dtype=np.float32),
        "b": np.array([3, 4], dtype=np.int32),
        "c": "not_a_tensor",  # Should be ignored because no dtype
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "weights.safetensors")
        fmt.save(data, filepath)

        loaded = fmt.load(filepath)
        assert "a" in loaded
        assert "b" in loaded
        assert "c" not in loaded

        # In our implementation, since we might not have a backend, it returns dicts
        # with 'buffer', 'dtype', 'shape' or directly backends.
        if isinstance(loaded["a"], dict):
            assert loaded["a"]["dtype"] == "F32"
            assert loaded["b"]["dtype"] == "I32"


def test_safetensors_invalid_header() -> None:
    """Test function test_safetensors_invalid_header."""
    fmt = SafetensorsWeightFormat()
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "invalid.safetensors")
        with open(filepath, "wb") as f:
            f.write(b"short")  # Less than 8 bytes
        loaded = fmt.load(filepath)
        assert loaded == {}


def test_safetensors_metadata() -> None:
    """Test function test_safetensors_metadata."""
    fmt = SafetensorsWeightFormat()
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "metadata.safetensors")
        # Create a safetensors file manually with __metadata__
        header = {"__metadata__": {"foo": "bar"}, "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
        header_bytes = json.dumps(header).encode("utf-8")
        padding = (8 - (len(header_bytes) % 8)) % 8
        header_bytes += b" " * padding
        with open(filepath, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            f.write(np.array([1.0, 2.0], dtype=np.float32).tobytes())

        loaded = fmt.load(filepath)
        assert "a" in loaded
        assert "__metadata__" not in loaded


def test_safetensors_with_backend_mock(mocker: MockerFixture) -> None:
    """Test function test_safetensors_with_backend_mock.

    Args:
        mocker: Test parameter `mocker`.
    """
    fmt = SafetensorsWeightFormat()

    class MockBackend:
        """Test class MockBackend."""

        def from_buffer(self, buffer: BackendArg, dtype: BackendArg, shape: BackendArg) -> BackendArg:
            """Test function from_buffer.

            Args:
                buffer: Test parameter `buffer`.
                dtype: Test parameter `dtype`.
                shape: Test parameter `shape`.

            Returns:
                BackendArg: Computed output value from `from_buffer`.
            """
            return "mocked_tensor"

    mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=MockBackend())

    data = {
        "a": np.array([1.0, 2.0], dtype=np.float32),
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "weights_backend.safetensors")
        fmt.save(data, filepath)

        loaded = fmt.load(filepath)
        assert loaded["a"] == "mocked_tensor"


def test_safetensors_missing_attrs() -> None:
    """Test function test_safetensors_missing_attrs."""
    fmt = SafetensorsWeightFormat()

    from ml_switcheroo_compiler.core.shape import OpValue

    class MissingDtype:
        """Test class MissingDtype."""

        data = None

    class MissingShape:
        """Test class MissingShape."""

        dtype = "float32"

    class MissingTobytes:
        """Test class MissingTobytes."""

        dtype = "float32"
        shape = (1,)

    data: dict[str, OpValue] = {
        "a": MissingDtype(),
        "b": MissingShape(),
        "c": MissingTobytes(),
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "weights_missing.safetensors")
        fmt.save(data, filepath)
        assert os.path.exists(filepath)


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

    def __init__(
        self,
        arr: GenericNDArray,
    ) -> None:
        """Initialize with numpy array.

        Args:
            arr (GenericNDArray): Underlying array.
        """
        self.arr = arr

    def numpy(
        self,
    ) -> GenericNDArray:
        """Return underlying numpy array.

        Returns:
            np.ndarray: Array.
        """
        return self.arr


class _MockDataWithNumpy:
    """Mock object with data.numpy() method."""

    def __init__(
        self,
        arr: GenericNDArray,
    ) -> None:
        """Initialize with nested data object.

        Args:
            arr (GenericNDArray): Underlying array.
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
        self.saved: (
            tuple[
                dict[
                    str,
                    GenericNDArray,
                ],
                str,
            ]
            | None
        ) = None

    def load_h5(
        self, path: str
    ) -> dict[
        str,
        GenericNDArray,
    ]:
        """Load h5 mock weights.

        Args:
            path (str): Filepath.

        Returns:
            dict[str, np.ndarray]: Dict of weights.
        """
        return {"backend_key": np.array([42.0])}

    def save_h5(
        self,
        weights: dict[
            str,
            GenericNDArray,
        ],
        path: str,
    ) -> None:
        """Save h5 mock weights.

        Args:
            weights (dict[str, GenericNDArray]): Weights dict.
            path (str): Filepath.
        """
        self.saved = (weights, path)


class _MockBackendNPZ:
    """Mock backend implementing custom npz save and load methods."""

    def __init__(self) -> None:
        """Initialize mock backend."""
        self.saved: (
            tuple[
                dict[
                    str,
                    GenericNDArray,
                ],
                str,
            ]
            | None
        ) = None

    def save_npz(
        self,
        weights: dict[
            str,
            GenericNDArray,
        ],
        path: str,
    ) -> None:
        """Save npz mock weights.

        Args:
            weights (dict[str, GenericNDArray]): Weights dict.
            path (str): Filepath.
        """
        self.saved = (weights, path)

    def load_npz(
        self, path: str
    ) -> dict[
        str,
        GenericNDArray,
    ]:
        """Load npz mock weights.

        Args:
            path (str): Filepath.

        Returns:
            dict[str, np.ndarray]: Dict of weights.
        """
        return {"backend_key": np.array([100.0])}


class _MockBackendSafetensors:
    """Mock backend implementing custom from_buffer method."""

    def from_buffer(
        self, buffer: bytes, dtype: str, shape: Sequence[int]
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
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


def test_safetensors_format_ext(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify SafetensorsWeightFormat save, load, and header parsing branches.

    Args:
        tmp_path (Path): Pytest temporary path.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture.

    Returns:
        None
    """
    fmt = SafetensorsWeightFormat()
    filepath = str(tmp_path / "model.safetensors")

    from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
    from ml_switcheroo_compiler.core.shape import OpValue

    # 1. Save with skipped values exercising each conditional branch of hasattr checks
    w1_arr = np.array([1.0, 2.0], dtype=np.float32)
    data: dict[str, OpValue] = {
        "no_dtype": "string_not_array",
        "no_shape": _NoShapeObject(),
        "no_tobytes": _NoTobytesObject(),
        "w1": w1_arr,
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
    np.testing.assert_allclose(ensure_ndarray(loaded_hook["w1"]), w1_arr)

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
