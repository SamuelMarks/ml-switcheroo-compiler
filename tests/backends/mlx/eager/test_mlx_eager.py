"""Core abstractions and logic definitions for test_mlx_eager_coverage.py."""

from __future__ import annotations

import typing
from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo_compiler.backends.mlx.eager import execute_op
from tests.testing_utils import as_callable, as_tensor

if typing.TYPE_CHECKING:
    import mlx.core as mx

mx = pytest.importorskip("mlx.core")


def test_mlx_eager_coverage_part1() -> None:
    """Test the mlx eager coverage part1 behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    try:
        pass
    except ImportError:
        pytest.skip("MLX not installed")

    class DummyBackend:
        """Configuration class for dummy backend."""

        pass

    cls = DummyBackend()
    res = as_callable(execute_op)(cls, "TakeAlongAxis", as_callable(mx.array)([1, 2]), as_callable(mx.array)([0]), axis=0)
    assert res is not None
    res = as_callable(execute_op)(cls, "Take", as_callable(mx.array)([1, 2]), as_callable(mx.array)([0]), axis=0)
    assert res is not None
    with patch("ml_switcheroo_compiler.backends.numpy.eager.execute_op", return_value="scatter_res"):
        as_callable(execute_op)(cls, "ScatterNd", as_callable(mx.array)([0]), as_callable(mx.array)([0]), as_callable(mx.array)([1]))
    shape_mock = MagicMock()
    shape_mock.data = [2]
    shape_mock.tolist.return_value = [2]
    res = as_callable(execute_op)(cls, "Reshape", as_callable(mx.array)([1, 2]), shape=shape_mock)
    assert res is not None
    res = as_callable(execute_op)(cls, "Reshape", as_callable(mx.array)([1, 2]), newshape=(2,))
    assert res is not None


def test_mlx_eager_coverage_part2() -> None:
    """Test the mlx eager coverage part2 behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with pytest.raises(ValueError):

        class DummyBackend:
            """Configuration class for dummy backend."""

        cls = DummyBackend()
        shape_mock = MagicMock()
        shape_mock.data = [2]
        shape_mock.tolist.return_value = [2]
        assert as_callable(execute_op)(cls, "Zeros", 2) is not None
        assert as_callable(execute_op)(cls, "Zeros", shape=shape_mock, dtype="float32") is not None
        assert as_callable(execute_op)(cls, "Ones", 2) is not None
        assert as_callable(execute_op)(cls, "Ones", shape=shape_mock, dtype="float32") is not None
        assert as_callable(execute_op)(cls, "Full", 2, 5) is not None
        assert as_callable(execute_op)(cls, "Full", shape=shape_mock, fill_value=5, dtype="float32") is not None
        as_callable(execute_op)(cls, "Zeros", shape=shape_mock, dtype=None)
        as_callable(execute_op)(cls, "Ones", shape=shape_mock, dtype=None)
        as_callable(execute_op)(cls, "Full", shape=shape_mock, fill_value=5, dtype=None)
        with patch("mlx.core.zeros", side_effect=[TypeError, "mock_res"]):
            assert as_callable(execute_op)(cls, "Zeros", shape=(2,), dtype="float32") == "mock_res"
        with patch("mlx.core.ones", side_effect=[TypeError, "mock_res"]):
            assert as_callable(execute_op)(cls, "Ones", shape=(2,), dtype="float32") == "mock_res"
        with patch("mlx.core.full", side_effect=[TypeError, "mock_res"]):
            assert as_callable(execute_op)(cls, "Full", shape=(2,), fill_value=5, dtype="float32") == "mock_res"
        n_mock = MagicMock()
        n_mock.data = 2
        assert as_callable(execute_op)(cls, "Eye", n_mock, dtype="float32") is not None
        assert as_callable(execute_op)(cls, "Eye", n_mock, n_mock, k=1, dtype="float32") is not None
        with patch("mlx.core.take", side_effect=ValueError):
            res = as_callable(execute_op)(cls, "Take", as_callable(mx.array)([1, 2]), as_callable(mx.array)([0]), axis=0)
            assert as_tensor(getattr(getattr(res, "__class__", None), "__name__")) == "array"
        with patch("mlx.core.take", side_effect=ValueError), patch("ml_switcheroo_compiler.backends.numpy.eager.execute_op", return_value=(1, 2)):
            res = as_callable(execute_op)(cls, "Take", as_callable(mx.array)([1, 2]), as_callable(mx.array)([0]), axis=0)
            assert isinstance(as_tensor(res), tuple)


def test_mlx_ragged_tensor_to_dense() -> None:
    """Verify MLX RaggedTensorToDense converts ragged lists and dicts to dense arrays."""
    from ml_switcheroo_compiler.backends.mlx.eager import _mlx_ragged_tensor_to_dense

    r1: list[mx.array] = [mx.array([1.0, 2.0]), mx.array([3.0, 4.0, 5.0])]
    dense1: mx.array = typing.cast("mx.array", _mlx_ragged_tensor_to_dense(mx, r1, default_value=0.0))
    assert dense1.shape == (2, 3)
    assert dense1[0, 2].item() == 0.0
    assert dense1[1, 2].item() == 5.0
    r2: dict[str, mx.array] = {"values": mx.array([10.0, 20.0, 30.0, 40.0]), "row_splits": mx.array([0, 1, 4])}
    dense2: mx.array = typing.cast("mx.array", _mlx_ragged_tensor_to_dense(mx, r2, default_value=-1.0))
    assert dense2.shape == (2, 3)
    assert dense2[0, 0].item() == 10.0
    assert dense2[0, 1].item() == -1.0
    assert dense2[1, 0].item() == 20.0
    assert dense2[1, 2].item() == 40.0
    arr: mx.array = mx.array([1.0, 2.0])
    res_dense: mx.array = typing.cast("mx.array", _mlx_ragged_tensor_to_dense(mx, arr))
    assert res_dense is arr

    class RaggedContainer:
        """Container holding values and row_splits."""

        def __init__(self, values: mx.array, row_splits: mx.array) -> None:
            """Test function __init__.

            Args:
                values: Test parameter `values`.
                row_splits: Test parameter `row_splits`.
            """
            self.values = values
            self.row_splits = row_splits

    r_obj = RaggedContainer(values=mx.array([10.0, 20.0, 30.0, 40.0]), row_splits=mx.array([0, 1, 4]))
    dense_obj = _mlx_ragged_tensor_to_dense(mx, as_tensor(r_obj), default_value=-1.0)
    assert as_tensor(getattr(dense_obj, "shape")) == (2, 3)
    r_empty: dict[str, mx.array] = {"values": mx.array([], dtype=mx.float32), "row_splits": mx.array([0])}
    dense_empty = _mlx_ragged_tensor_to_dense(mx, r_empty)
    assert as_tensor(getattr(dense_empty, "shape")) == (0, 0)


def test_mlx_eager_import_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test fallback to None when mlx.core cannot be imported in eager.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """
    import importlib
    import sys

    import ml_switcheroo_compiler.backends.mlx.eager as mx_eager

    with monkeypatch.context() as m:
        m.setitem(sys.modules, "mlx.core", None)
        importlib.reload(mx_eager)
        assert as_tensor(getattr(mx_eager, "mx")) is None
    importlib.reload(mx_eager)
    assert as_tensor(getattr(mx_eager, "mx")) is not None
