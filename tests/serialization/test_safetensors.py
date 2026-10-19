"""Tests for test_safetensors."""

from __future__ import annotations

import json
import os
import struct
import tempfile
from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.serialization.formats.safetensors import SafetensorsWeightFormat

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_safetensors_format() -> None:
    """Test function test_safetensors_format."""
    fmt = SafetensorsWeightFormat()
    data = {
        "a": np.array([1.0, 2.0], dtype=np.float32),
        "b": np.array([3, 4], dtype=np.int32),
        "c": "not_a_tensor",  # Should be ignored because no dtype
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "weights.safetensors")
        cast("EagerCallableProtocol", fmt.save)(data, filepath)

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


def test_safetensors_with_backend_mock(mocker: EagerTensorProtocol) -> None:
    """Test function test_safetensors_with_backend_mock.

    Args:
        mocker: Test parameter `mocker`.
    """
    fmt = SafetensorsWeightFormat()

    class MockBackend:
        """Test class MockBackend."""

        def from_buffer(self, buffer: EagerValue, dtype: EagerValue, shape: EagerValue) -> EagerValue:
            """Test function from_buffer.

            Args:
                buffer: Test parameter `buffer`.
                dtype: Test parameter `dtype`.
                shape: Test parameter `shape`.
            """
            return "mocked_tensor"

    cast("EagerCallableProtocol", mocker.patch)("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=cast("EagerCallableProtocol", MockBackend)())

    data = {
        "a": np.array([1.0, 2.0], dtype=np.float32),
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "weights_backend.safetensors")
        fmt.save(data, filepath)

        loaded = fmt.load(filepath)
        assert cast("EagerTensorProtocol", loaded["a"]) == cast("EagerTensorProtocol", "mocked_tensor")


def test_safetensors_missing_attrs() -> None:
    """Test function test_safetensors_missing_attrs."""
    fmt = SafetensorsWeightFormat()

    class MissingDtype:
        """Test class MissingDtype."""

        pass

    class MissingShape:
        """Test class MissingShape."""

        dtype = "float32"

    class MissingTobytes:
        """Test class MissingTobytes."""

        dtype = "float32"
        shape = (1,)

    data = {
        "a": MissingDtype(),
        "b": MissingShape(),
        "c": MissingTobytes(),
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "weights_missing.safetensors")
        cast("EagerCallableProtocol", fmt.save)(data, filepath)
        assert os.path.exists(filepath)
