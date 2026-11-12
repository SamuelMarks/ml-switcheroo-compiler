"""Tests for test_safetensors."""

from __future__ import annotations

# ruff: noqa: E501
import json
import os
import struct
import tempfile
import typing

from ml_switcheroo_compiler.serialization.formats.safetensors import SafetensorsWeightFormat
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_mutable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_safetensors_load_save(mocker: EagerTensorProtocol) -> None:
    """Test function test_safetensors_load_save.

    Args:
        mocker: Test parameter `mocker`.
    """
    fmt = SafetensorsWeightFormat()

    class MockNumpy:
        """Test class MockNumpy."""

        def __init__(self, data: EagerValue, dtype: EagerValue, shape: EagerValue) -> None:
            """Test function __init__.

            Args:
                data: Test parameter `data`.
                dtype: Test parameter `dtype`.
                shape: Test parameter `shape`.
            """
            self._data = data
            self.dtype = dtype
            self.shape = shape

        def tobytes(self) -> EagerValue:
            """Test function tobytes.

            Returns:
                EagerValue: Computed output value from `tobytes`.
            """
            return self._data

    with tempfile.NamedTemporaryFile() as f:
        f.write(b"short")
        f.flush()
        assert fmt.load(f.name) == {}
    with tempfile.NamedTemporaryFile() as f:
        weights = {"a": MockNumpy(b"data1", "float32", (1,)), "b": MockNumpy(b"data2", "float64", (1,)), "c": MockNumpy(b"data3", "unknown", (1,)), "skip_me": "not_numpy"}
        ensure_callable(fmt.save)(weights, f.name)
        mock_backend = mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend").return_value
        del mock_backend.from_buffer
        loaded = fmt.load(f.name)
        assert "a" in loaded
        assert ensure_eager_tensor(loaded["a"]["buffer"]) == ensure_eager_tensor(b"data1")
        assert ensure_eager_tensor(loaded["a"]["dtype"]) == ensure_eager_tensor("F32")
        assert ensure_eager_tensor(loaded["a"]["shape"]) == ensure_eager_tensor([1])
        assert "b" in loaded
        assert ensure_eager_tensor(loaded["b"]["dtype"]) == ensure_eager_tensor("F64")
        assert "c" in loaded
        assert ensure_eager_tensor(loaded["c"]["dtype"]) == ensure_eager_tensor("F32")
    with tempfile.NamedTemporaryFile() as f:
        ensure_callable(fmt.save)(weights, f.name)
        mock_backend = mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend").return_value
        ensure_mutable(mock_backend.from_buffer).side_effect = ensure_eager_tensor(lambda b, dtype, shape: {"buf": b, "type": dtype, "sh": shape})
        loaded2 = fmt.load(f.name)
        assert ensure_eager_tensor(loaded2["a"]["buf"]) == ensure_eager_tensor(b"data1")
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
                """Return empty bytes.

                Returns:
                    EagerValue: Computed output value from `tobytes`.
                """
                return b""

        class MissingAttr4:
            """Test class MissingAttr4."""

            shape = ()

            @staticmethod
            def tobytes() -> EagerValue:
                """Return empty bytes.

                Returns:
                    EagerValue: Computed output value from `tobytes`.
                """
                return b""

        ensure_callable(fmt.save)({"a": ensure_callable(MissingAttr1)(), "b": ensure_callable(MissingAttr2)(), "c": ensure_callable(MissingAttr3)(), "d": ensure_callable(MissingAttr4)()}, f.name)
        assert os.path.exists(f.name)
