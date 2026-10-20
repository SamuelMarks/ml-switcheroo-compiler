# ruff: noqa: E501
"""Tests for H5 weight format loading and saving."""

from __future__ import annotations

import tempfile
import typing

import numpy as np

from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_h5_load_save(mocker: EagerTensorProtocol) -> None:
    """Test function test_h5_load_save.

    Args:
        mocker: Test parameter `mocker`.
    """
    fmt = H5WeightFormat()
    mock_backend = mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend").return_value
    del mock_backend.load_h5
    del mock_backend.save_h5
    with tempfile.NamedTemporaryFile(suffix=".h5") as f:

        class MockNumpy:
            """Test class MockNumpy."""

            def numpy(self) -> EagerValue:
                """Test function numpy."""
                return np.array([1, 2])

        class MockData:
            """Test class MockData."""

            def __init__(self) -> None:
                """Test function __init__."""
                self.data = MockNumpy()

        class MockList:
            """Test class MockList."""

            def tolist(self) -> EagerValue:
                """Test function tolist."""
                return [3, 4]

        weights = {"a": np.array([1, 2]), "b": MockNumpy(), "c": MockData(), "d": MockList()}
        typing.cast("EagerCallableProtocol", fmt.save)(weights, f.name)
        loaded = fmt.load(f.name)
        assert typing.cast("EagerCallableProtocol", np.array_equal)(loaded["a"], [1, 2])
        assert typing.cast("EagerCallableProtocol", np.array_equal)(loaded["b"], [1, 2])
        assert typing.cast("EagerCallableProtocol", np.array_equal)(loaded["c"], [1, 2])
        assert typing.cast("EagerCallableProtocol", np.array_equal)(loaded["d"], [3, 4])


def test_h5_load_save_backend(mocker: EagerTensorProtocol) -> None:
    """Test function test_h5_load_save_backend.

    Args:
        mocker: Test parameter `mocker`.
    """
    fmt = H5WeightFormat()
    mock_backend = mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend").return_value
    setattr(mock_backend.load_h5, "return_value", {"a": 1})
    assert typing.cast("EagerTensorProtocol", fmt.load("test.h5")) == typing.cast("EagerTensorProtocol", {"a": 1})
    typing.cast("EagerCallableProtocol", fmt.save)({"a": 1}, "test.h5")
    mock_backend.save_h5.assert_called_with({"a": 1}, "test.h5")
