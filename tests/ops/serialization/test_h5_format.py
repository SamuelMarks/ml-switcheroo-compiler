"""Tests for H5 weight format loading and saving."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest
from pytest_mock import MockerFixture

from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.serialization.formats.h5 import H5WeightFormat


def test_h5_load_save(mocker: MockerFixture) -> None:
    """Test function test_h5_load_save.

    Args:
        mocker (MockerFixture): Test parameter `mocker`.
    """
    fmt = H5WeightFormat()
    mock_backend = mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend").return_value
    del mock_backend.load_h5
    del mock_backend.save_h5
    with tempfile.NamedTemporaryFile(suffix=".h5") as f:

        class MockNumpy:
            """Test class MockNumpy."""

            ndim: int = 1

            def numpy(self) -> OpValue:
                """Test function numpy.

                Returns:
                    OpValue: Computed output value from `numpy`.
                """
                return np.array([1, 2])

        class MockData:
            """Test class MockData."""

            def __init__(self) -> None:
                """Test function __init__."""
                self.data: MockNumpy = MockNumpy()

        class MockList:
            """Test class MockList."""

            ndim: int = 1

            def tolist(self) -> OpValue:
                """Test function tolist.

                Returns:
                    OpValue: Computed output value from `tolist`.
                """
                return [3, 4]

        weights: dict[str, OpValue] = {
            "a": np.array([1, 2]),
            "b": MockNumpy(),
            "c": MockData(),
            "d": MockList(),
        }
        fmt.save(weights, f.name)
        loaded = fmt.load(f.name)
        assert np.array_equal(ensure_ndarray(loaded["a"]), [1, 2])
        assert np.array_equal(ensure_ndarray(loaded["b"]), [1, 2])
        assert np.array_equal(ensure_ndarray(loaded["c"]), [1, 2])
        assert np.array_equal(ensure_ndarray(loaded["d"]), [3, 4])


def test_h5_load_save_backend(mocker: MockerFixture) -> None:
    """Test function test_h5_load_save_backend.

    Args:
        mocker (MockerFixture): Test parameter `mocker`.
    """
    fmt = H5WeightFormat()
    mock_backend = mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend").return_value
    pytest.MonkeyPatch().setattr(mock_backend.load_h5, "return_value", {"a": 1}, raising=False)
    assert fmt.load("test.h5") == {"a": 1}
    weights: dict[str, OpValue] = {"a": 1}
    fmt.save(weights, "test.h5")
    mock_backend.save_h5.assert_called_with(weights, "test.h5")
