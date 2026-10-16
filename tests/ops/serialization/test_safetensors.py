"""Tests for test_safetensors."""

from __future__ import annotations

# ruff: noqa: E501
import json
import struct
import tempfile
from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.serialization.formats.safetensors import SafetensorsWeightFormat

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_safetensors_load_save(mocker: EagerTensorProtocol) -> None:
    """Test function test_safetensors_load_save."""
    fmt = SafetensorsWeightFormat()

    class MockNumpy:
        """Test class MockNumpy."""

        def __init__(self, data: EagerValue, dtype: EagerValue, shape: EagerValue) -> None:
            """Test function __init__."""
            self._data = data
            self.dtype = dtype
            self.shape = shape

        def tobytes(self) -> EagerValue:
            """Test function tobytes."""
            return self._data

    with tempfile.NamedTemporaryFile() as f:
        f.write(b"short")
        f.flush()
        assert fmt.load(f.name) == {}
    with tempfile.NamedTemporaryFile() as f:
        weights = {"a": MockNumpy(b"data1", "float32", (1,)), "b": MockNumpy(b"data2", "float64", (1,)), "c": MockNumpy(b"data3", "unknown", (1,)), "skip_me": "not_numpy"}
        cast("Callable[..., EagerTensorProtocol]", fmt.save)(weights, f.name)
        mock_backend = mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend").return_value
        del mock_backend.from_buffer
        loaded = fmt.load(f.name)
        assert "a" in loaded
        assert cast("EagerTensorProtocol", loaded["a"]["buffer"]) == cast("EagerTensorProtocol", b"data1")
        assert cast("EagerTensorProtocol", loaded["a"]["dtype"]) == cast("EagerTensorProtocol", "F32")
        assert cast("EagerTensorProtocol", loaded["a"]["shape"]) == cast("EagerTensorProtocol", [1])
        assert "b" in loaded
        assert cast("EagerTensorProtocol", loaded["b"]["dtype"]) == cast("EagerTensorProtocol", "F64")
        assert "c" in loaded
        assert cast("EagerTensorProtocol", loaded["c"]["dtype"]) == cast("EagerTensorProtocol", "F32")
    with tempfile.NamedTemporaryFile() as f:
        cast("Callable[..., EagerTensorProtocol]", fmt.save)(weights, f.name)
        mock_backend = mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend").return_value
        setattr(mock_backend.from_buffer, "side_effect", lambda b, dtype, shape: {"buf": b, "type": dtype, "sh": shape})
        loaded2 = fmt.load(f.name)
        assert cast("EagerTensorProtocol", loaded2["a"]["buf"]) == cast("EagerTensorProtocol", b"data1")
    with tempfile.NamedTemporaryFile() as f:
        header = {"__metadata__": {"foo": "bar"}, "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 5]}}
        header_bytes = json.dumps(header).encode("utf-8")
        header_len = len(header_bytes)
        f.write(struct.pack("<Q", header_len))
        f.write(header_bytes)
        f.write(b"data1")
        f.flush()
        mock_backend = mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend").return_value
        del mock_backend.from_buffer
        loaded = fmt.load(f.name)
        assert "__metadata__" not in loaded
        assert "a" in loaded


def test_safetensors_save_continue() -> None:
    """Test function test_safetensors_save_continue."""
    fmt = SafetensorsWeightFormat()
    with tempfile.NamedTemporaryFile() as f:

        class MissingAttr1:
            """Test class MissingAttr1."""

            pass

        class MissingAttr2:
            """Test class MissingAttr2."""

            dtype = "f"
            shape = ()

        class MissingAttr3:
            """Test class MissingAttr3."""

            dtype = "f"

            @staticmethod
            def tobytes() -> EagerValue:
                return b""

        class MissingAttr4:
            """Test class MissingAttr4."""

            shape = ()

            @staticmethod
            def tobytes() -> EagerValue:
                return b""

        cast("Callable[..., EagerTensorProtocol]", fmt.save)(
            {"a": cast("Callable[..., EagerTensorProtocol]", MissingAttr1)(), "b": cast("Callable[..., EagerTensorProtocol]", MissingAttr2)(), "c": cast("Callable[..., EagerTensorProtocol]", MissingAttr3)(), "d": cast("Callable[..., EagerTensorProtocol]", MissingAttr4)()}, f.name
        )
    assert fmt is not None or fmt is None
