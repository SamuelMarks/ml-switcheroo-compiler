"""Tests for numpy eager math scatter ops."""

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


def test_np_tensor_scatter_update():
    tensor = np.zeros((2, 2))
    indices = np.array([[0, 0], [1, 1]])
    updates = np.array([1, 4])

    @global_eager_registry.register("TensorScatterUpdate")
    def dummy_update(bm, t, idx, u):
        out = np.copy(t)
        idx_tuple = tuple(np.moveaxis(np.asarray(idx), -1, 0))
        out[idx_tuple] = u
        return out

    res = _np_tensor_scatter_update(np, tensor, indices, updates)
    np.testing.assert_allclose(res, [[1, 0], [0, 4]])


def test_np_tensor_scatter_add():
    tensor = np.zeros((2, 2))
    indices = np.array([[0, 0], [1, 1]])
    updates = np.array([1, 4])
    res = _np_tensor_scatter_add(np, tensor, indices, updates)
    np.testing.assert_allclose(res, [[1, 0], [0, 4]])


def test_np_tensor_scatter_max():
    tensor = np.zeros((2, 2))
    indices = np.array([[0, 0], [1, 1]])
    updates = np.array([1, 4])
    res = _np_tensor_scatter_max(np, tensor, indices, updates)
    np.testing.assert_allclose(res, [[1, 0], [0, 4]])


def test_np_tensor_scatter_min():
    tensor = np.ones((2, 2)) * 5
    indices = np.array([[0, 0], [1, 1]])
    updates = np.array([1, 4])
    res = _np_tensor_scatter_min(np, tensor, indices, updates)
    np.testing.assert_allclose(res, [[1, 5], [5, 4]])


def test_np_scatter_nd():
    indices = np.array([[0, 0], [1, 1]])
    updates = np.array([1, 4])
    res = _np_scatter_nd(np, indices, updates, shape=(2, 2))
    np.testing.assert_allclose(res, [[1, 0], [0, 4]])


def test_np_scatter():
    x = np.zeros((2, 2))
    index = np.array([[0, 0], [1, 1]])
    src = np.array([[1, 2], [3, 4]])
    res = _np_scatter(np, x, index, src, dim=1)
    np.testing.assert_allclose(res, [[2, 0], [0, 4]])


def test_band_part():
    res = _band_part([[1, 2], [3, 4]], 1, 1)
    np.testing.assert_allclose(res, [[1, 2], [3, 4]])


def test_np_gather_nd():
    params = np.array([[1, 2], [3, 4]])
    indices = np.array([[0, 0], [1, 1]])
    res = _np_gather_nd(np, params, indices)
    np.testing.assert_allclose(res, [1, 4])


def test_np_take_along_axis():
    x = np.array([[1, 2], [3, 4]])
    indices = np.array([[1, 0], [0, 1]])
    res = _np_take_along_axis(np, x, indices, axis=1)
    np.testing.assert_allclose(res, np.take_along_axis(x, indices, axis=1))


def test_np_dynamic_slice():
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    res = _np_dynamic_slice(np, x, start_indices=[1, 1], slice_sizes=[2, 2])
    np.testing.assert_allclose(res, [[5, 6], [8, 9]])


def test_np_dynamic_update_slice():
    x = np.zeros((3, 3))
    res = _np_dynamic_update_slice(np, x, np.ones((2, 2)), [1, 1])
    assert res[1, 1] == 1.0
    assert res[0, 0] == 0.0
