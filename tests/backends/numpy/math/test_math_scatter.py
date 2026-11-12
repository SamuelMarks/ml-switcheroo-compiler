"""Tests for numpy eager math scatter ops."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_scatter import (
    _band_part,
    _np_dynamic_slice,
    _np_dynamic_update_slice,
    _np_gather_nd,
    _np_scatter,
    _np_scatter_nd,
    _np_take_along_axis,
    _np_tensor_scatter_add,
    _np_tensor_scatter_max,
    _np_tensor_scatter_min,
    _np_tensor_scatter_update,
)
from tests.testing_utils import ensure_callable, ensure_eager_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_np_tensor_scatter_update() -> None:
    """Test function test_np_tensor_scatter_update."""
    tensor = np.zeros((2, 2))
    indices = np.array([[0, 0], [1, 1]])
    updates = np.array([1, 4])

    @global_eager_registry.register("TensorScatterUpdate")
    def dummy_update(bm: EagerValue, t: EagerValue, idx: EagerValue, u: EagerValue) -> EagerValue:
        """Test function dummy_update.

        Args:
            bm: Test parameter `bm`.
            t: Test parameter `t`.
            idx: Test parameter `idx`.
            u: Test parameter `u`.

        Returns:
            EagerValue: Computed output value from `dummy_update`.
        """
        out = ensure_callable(np.copy)(t)
        idx_tuple = tuple(np.moveaxis(np.asarray(idx), -1, 0))
        out[idx_tuple] = u
        return out

    res = _np_tensor_scatter_update(ensure_eager_tensor(np), tensor, indices, updates)
    ensure_callable(np.testing.assert_allclose)(res, [[1, 0], [0, 4]])


def test_np_tensor_scatter_add() -> None:
    """Test function test_np_tensor_scatter_add."""
    tensor = np.zeros((2, 2))
    indices = np.array([[0, 0], [1, 1]])
    updates = np.array([1, 4])
    res = _np_tensor_scatter_add(ensure_eager_tensor(np), tensor, indices, updates)
    ensure_callable(np.testing.assert_allclose)(res, [[1, 0], [0, 4]])


def test_np_tensor_scatter_max() -> None:
    """Test function test_np_tensor_scatter_max."""
    tensor = np.zeros((2, 2))
    indices = np.array([[0, 0], [1, 1]])
    updates = np.array([1, 4])
    res = _np_tensor_scatter_max(ensure_eager_tensor(np), tensor, indices, updates)
    ensure_callable(np.testing.assert_allclose)(res, [[1, 0], [0, 4]])


def test_np_tensor_scatter_min() -> None:
    """Test function test_np_tensor_scatter_min."""
    tensor = np.ones((2, 2)) * 5
    indices = np.array([[0, 0], [1, 1]])
    updates = np.array([1, 4])
    res = _np_tensor_scatter_min(ensure_eager_tensor(np), tensor, indices, updates)
    ensure_callable(np.testing.assert_allclose)(res, [[1, 5], [5, 4]])


def test_np_scatter_nd() -> None:
    """Test function test_np_scatter_nd."""
    indices = np.array([[0, 0], [1, 1]])
    updates = np.array([1, 4])
    res = _np_scatter_nd(ensure_eager_tensor(np), indices, updates, shape=(2, 2))
    ensure_callable(np.testing.assert_allclose)(res, [[1, 0], [0, 4]])


def test_np_scatter() -> None:
    """Test function test_np_scatter."""
    x = np.zeros((2, 2))
    index = np.array([[0, 0], [1, 1]])
    src = np.array([[1, 2], [3, 4]])
    res = _np_scatter(ensure_eager_tensor(np), x, index, src, dim=1)
    ensure_callable(np.testing.assert_allclose)(res, [[2, 0], [0, 4]])


def test_band_part() -> None:
    """Test function test_band_part."""
    res = ensure_callable(_band_part)([[1, 2], [3, 4]], 1, 1)
    ensure_callable(np.testing.assert_allclose)(res, [[1, 2], [3, 4]])


def test_np_gather_nd() -> None:
    """Test function test_np_gather_nd."""
    params = np.array([[1, 2], [3, 4]])
    indices = np.array([[0, 0], [1, 1]])
    res = _np_gather_nd(ensure_eager_tensor(np), params, indices)
    ensure_callable(np.testing.assert_allclose)(res, [1, 4])


def test_np_take_along_axis() -> None:
    """Test function test_np_take_along_axis."""
    x = np.array([[1, 2], [3, 4]])
    indices = np.array([[1, 0], [0, 1]])
    res = _np_take_along_axis(ensure_eager_tensor(np), x, indices, axis=1)
    ensure_callable(np.testing.assert_allclose)(res, ensure_callable(np.take_along_axis)(x, indices, axis=1))


def test_np_dynamic_slice() -> None:
    """Test function test_np_dynamic_slice."""
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    res = _np_dynamic_slice(ensure_eager_tensor(np), x, start_indices=[1, 1], slice_sizes=[2, 2])
    ensure_callable(np.testing.assert_allclose)(res, [[5, 6], [8, 9]])


def test_np_dynamic_update_slice() -> None:
    """Test function test_np_dynamic_update_slice."""
    x = np.zeros((3, 3))
    res = _np_dynamic_update_slice(ensure_eager_tensor(np), x, np.ones((2, 2)), [1, 1])
    assert (
        ensure_eager_tensor(res)[
            (
                1,
                1,
            )
        ]
        == 1.0
    )
    assert (
        ensure_eager_tensor(res)[
            (
                0,
                0,
            )
        ]
        == 0.0
    )
