"""Tests for test_safetensors."""

from __future__ import annotations

import json
import os
import struct
import tempfile

import pytest
from pytest_mock import MockerFixture

from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.serialization.formats.safetensors import SafetensorsWeightFormat


def test_safetensors_load_save(mocker: MockerFixture) -> None:
    """Test function test_safetensors_load_save.

    Args:
        mocker (MockerFixture): Test parameter `mocker`.
    """
    fmt = SafetensorsWeightFormat()

    class MockNumpy:
        """Test class MockNumpy."""

        def __init__(self, data: bytes, dtype: str, shape: tuple[int, ...]) -> None:
            """Test function __init__.

            Args:
                data (bytes): Test parameter `data`.
                dtype (str): Test parameter `dtype`.
                shape (tuple[int, ...]): Test parameter `shape`.
            """
            self._data = data
            self.dtype = dtype
            self.shape = shape

        def tobytes(self) -> bytes:
            """Test function tobytes.

            Returns:
                bytes: Computed output value from `tobytes`.
            """
            return self._data

    with tempfile.NamedTemporaryFile() as f:
        f.write(b"short")
        f.flush()
        assert fmt.load(f.name) == {}
    with tempfile.NamedTemporaryFile() as f:
        weights: dict[str, OpValue] = {
            "a": MockNumpy(b"data1", "float32", (1,)),
            "b": MockNumpy(b"data2", "float64", (1,)),
            "c": MockNumpy(b"data3", "unknown", (1,)),
            "skip_me": "not_numpy",
        }
        fmt.save(weights, f.name)
        mock_backend = mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend").return_value
        del mock_backend.from_buffer
        loaded = fmt.load(f.name)
        assert "a" in loaded
        assert loaded["a"]["buffer"] == b"data1"
        assert loaded["a"]["dtype"] == "F32"
        assert loaded["a"]["shape"] == [1]
        assert "b" in loaded
        assert loaded["b"]["dtype"] == "F64"
        assert "c" in loaded
        assert loaded["c"]["dtype"] == "F32"
    with tempfile.NamedTemporaryFile() as f:
        fmt.save(weights, f.name)
        mock_backend = mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend").return_value
        pytest.MonkeyPatch().setattr(
            mock_backend.from_buffer,
            "side_effect",
            lambda b, dtype, shape: {"buf": b, "type": dtype, "sh": shape},
            raising=False,
        )
        loaded2 = fmt.load(f.name)
        assert loaded2["a"]["buf"] == b"data1"
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

            ndim: int = 0

        class MissingAttr2:
            """Test class MissingAttr2."""

            dtype = "f"
            shape = ()

        class MissingAttr3:
            """Test class MissingAttr3."""

            dtype = "f"

            @staticmethod
            def tobytes() -> bytes:
                """Return empty bytes.

                Returns:
                    bytes: Computed output value from `tobytes`.
                """
                return b""

        class MissingAttr4:
            """Test class MissingAttr4."""

            shape = ()

            @staticmethod
            def tobytes() -> bytes:
                """Return empty bytes.

                Returns:
                    bytes: Computed output value from `tobytes`.
                """
                return b""

        incomplete_weights: dict[str, OpValue] = {
            "a": MissingAttr1(),
            "b": MissingAttr2(),
            "c": MissingAttr3(),
            "d": MissingAttr4(),
        }
        fmt.save(incomplete_weights, f.name)
        assert os.path.exists(f.name)
