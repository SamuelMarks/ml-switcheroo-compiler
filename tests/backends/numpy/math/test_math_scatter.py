"""Behavioral unit tests for math scatter."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager.reference_math import math_scatter as scatter_mod
from ml_switcheroo_compiler.backends.eager.reference_math.math_scatter import _band_part
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.backends.numpy.eager.advanced_indexing import (
    _np_gather_nd,
    _np_scatter,
    _np_scatter_nd,
    _np_take_along_axis,
    _np_tensor_scatter_add,
    _np_tensor_scatter_max,
    _np_tensor_scatter_min,
    _np_tensor_scatter_update,
)
from ml_switcheroo_compiler.backends.numpy.eager.indexing import _np_dynamic_slice, _np_dynamic_update_slice
from tests.testing_utils import require_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_np_tensor_scatter_update() -> None:
    """Test function test_np_tensor_scatter_update."""
    tensor = np.zeros((2, 2))
    indices = np.array([[0, 0], [1, 1]])
    updates = np.array([1, 4])

    @global_eager_registry.register("TensorScatterUpdate")
    def dummy_update(bm: BackendArg, t: BackendArg, idx: BackendArg, u: BackendArg) -> BackendArg:
        """Test function dummy_update.

        Args:
            bm: Test parameter `bm`.
            t: Test parameter `t`.
            idx: Test parameter `idx`.
            u: Test parameter `u`.

        Returns:
            BackendArg: Computed output value from `dummy_update`.
        """
        out = np.copy(ensure_ndarray(t))
        idx_tuple = tuple(np.moveaxis(ensure_ndarray(idx), -1, 0))
        out[idx_tuple] = ensure_ndarray(u)
        return out

    res = _np_tensor_scatter_update(np, tensor, indices, updates)
    np.testing.assert_allclose(ensure_ndarray(res), [[1, 0], [0, 4]])


def test_np_tensor_scatter_add() -> None:
    """Test function test_np_tensor_scatter_add."""
    tensor = np.zeros((2, 2))
    indices = np.array([[0, 0], [1, 1]])
    updates = np.array([1, 4])
    res = _np_tensor_scatter_add(np, tensor, indices, updates)
    np.testing.assert_allclose(ensure_ndarray(res), [[1, 0], [0, 4]])


def test_np_tensor_scatter_max() -> None:
    """Test function test_np_tensor_scatter_max."""
    tensor = np.zeros((2, 2))
    indices = np.array([[0, 0], [1, 1]])
    updates = np.array([1, 4])
    res = _np_tensor_scatter_max(np, tensor, indices, updates)
    np.testing.assert_allclose(ensure_ndarray(res), [[1, 0], [0, 4]])


def test_np_tensor_scatter_min() -> None:
    """Test function test_np_tensor_scatter_min."""
    tensor = np.ones((2, 2)) * 5
    indices = np.array([[0, 0], [1, 1]])
    updates = np.array([1, 4])
    res = _np_tensor_scatter_min(np, tensor, indices, updates)
    np.testing.assert_allclose(ensure_ndarray(res), [[1, 5], [5, 4]])


def test_np_scatter_nd() -> None:
    """Test function test_np_scatter_nd."""
    indices = np.array([[0, 0], [1, 1]])
    updates = np.array([1, 4])
    res = _np_scatter_nd(np, indices, updates, shape=(2, 2))
    np.testing.assert_allclose(ensure_ndarray(res), [[1, 0], [0, 4]])


def test_np_scatter() -> None:
    """Test function test_np_scatter."""
    x = np.zeros((2, 2))
    index = np.array([[0, 0], [1, 1]])
    src = np.array([[1, 2], [3, 4]])
    res = _np_scatter(np, x, index, src, dim=1)
    np.testing.assert_allclose(ensure_ndarray(res), [[2, 0], [0, 4]])


def test_band_part() -> None:
    """Test function test_band_part."""
    res = _band_part([[1, 2], [3, 4]], 1, 1)
    np.testing.assert_allclose(res, [[1, 2], [3, 4]])


def test_np_gather_nd() -> None:
    """Test function test_np_gather_nd."""
    params = np.array([[1, 2], [3, 4]])
    indices = np.array([[0, 0], [1, 1]])
    res = _np_gather_nd(np, params, indices)
    np.testing.assert_allclose(ensure_ndarray(res), [1, 4])


def test_np_take_along_axis() -> None:
    """Test function test_np_take_along_axis."""
    x = np.array([[1, 2], [3, 4]])
    indices = np.array([[1, 0], [0, 1]])
    res = _np_take_along_axis(np, x, indices, axis=1)
    np.testing.assert_allclose(ensure_ndarray(res), np.take_along_axis(x, indices, axis=1))


def test_np_dynamic_slice() -> None:
    """Test function test_np_dynamic_slice."""
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    res = _np_dynamic_slice(np, x, start_indices=[1, 1], slice_sizes=[2, 2])
    np.testing.assert_allclose(ensure_ndarray(res), [[5, 6], [8, 9]])


def test_np_dynamic_update_slice() -> None:
    """Test function test_np_dynamic_update_slice."""
    x = np.zeros((3, 3))
    res = _np_dynamic_update_slice(np, x, np.ones((2, 2)), [1, 1])
    assert (
        ensure_ndarray(res)[
            (
                1,
                1,
            )
        ]
        == 1.0
    )
    assert (
        ensure_ndarray(res)[
            (
                0,
                0,
            )
        ]
        == 0.0
    )


def test_math_scatter_ops() -> None:
    """Test scatter and gather operations in math_scatter.

    Returns:
        None
    """
    target = np.zeros((3, 3), dtype=np.float32)
    indices = np.array([[0, 0], [1, 1]])
    updates = np.array([5.0, 10.0], dtype=np.float32)

    # TensorScatterUpdate
    sc_up = scatter_mod._np_tensor_scatter_update(np, target, indices, updates)
    assert (
        sc_up[
            (
                0,
                0,
            )
        ]
        == 5.0
        and sc_up[
            (
                1,
                1,
            )
        ]
        == 10.0
    )
    assert require_callable(numpy_eager_registry.get)("TensorScatterUpdate")(np, target, indices, updates) is not None

    # TensorScatterAdd
    sc_add = scatter_mod._np_tensor_scatter_add(np, target, indices, updates)
    assert (
        sc_add[
            (
                0,
                0,
            )
        ]
        == 5.0
        and sc_add[
            (
                1,
                1,
            )
        ]
        == 10.0
    )
    assert require_callable(numpy_eager_registry.get)("TensorScatterAdd")(np, target, indices, updates) is not None

    # TensorScatterMax
    sc_max = scatter_mod._np_tensor_scatter_max(np, target, indices, updates)
    assert (
        sc_max[
            (
                0,
                0,
            )
        ]
        == 5.0
        and sc_max[
            (
                1,
                1,
            )
        ]
        == 10.0
    )
    assert require_callable(numpy_eager_registry.get)("TensorScatterMax")(np, target, indices, updates) is not None

    # TensorScatterMin
    base = np.full((3, 3), 20.0, dtype=np.float32)
    sc_min = scatter_mod._np_tensor_scatter_min(np, base, indices, updates)
    assert (
        sc_min[
            (
                0,
                0,
            )
        ]
        == 5.0
        and sc_min[
            (
                1,
                1,
            )
        ]
        == 10.0
    )
    assert require_callable(numpy_eager_registry.get)("TensorScatterMin")(np, base, indices, updates) is not None

    # ScatterNd
    sc_nd = scatter_mod._np_scatter_nd(np, indices, updates, (3, 3))
    assert (
        sc_nd[
            (
                0,
                0,
            )
        ]
        == 5.0
        and sc_nd[
            (
                1,
                1,
            )
        ]
        == 10.0
    )
    assert require_callable(numpy_eager_registry.get)("ScatterNd")(np, indices, updates, (3, 3)) is not None

    # Scatter
    inp = np.zeros(3)
    idx = np.array([0, 1])
    src = np.array([1.0, 2.0])
    sc_res = scatter_mod._np_scatter(np, inp, idx, src, dim=0)
    assert sc_res[0] == 1.0 and sc_res[1] == 2.0
    assert require_callable(numpy_eager_registry.get)("Scatter")(np, inp, idx, src) is not None

    # _band_part
    band = scatter_mod._band_part(np.eye(3), 1, 1)
    assert band.shape == (3, 3)

    # GatherNd
    params = np.array([[10, 20], [30, 40]])
    g_idx = np.array([[0, 1], [1, 0]])
    g_res = scatter_mod._np_gather_nd(np, params, g_idx)
    assert np.array_equal(g_res, [20, 30])
    assert np.array_equal(require_callable(numpy_eager_registry.get)("GatherNd")(np, params, g_idx), [20, 30])

    # TakeAlongAxis
    take_res = scatter_mod._np_take_along_axis(np, params, np.array([[1, 0], [0, 1]]), axis=1)
    assert np.array_equal(take_res, [[20, 10], [30, 40]])
    assert require_callable(numpy_eager_registry.get)("TakeAlongAxis")(np, params, np.array([[0]]), axis=0) is not None

    # DynamicSlice
    dyn_slice = scatter_mod._np_dynamic_slice(np, params, [0, 0], [1, 2])
    assert np.array_equal(dyn_slice, [[10, 20]])
    assert require_callable(numpy_eager_registry.get)("DynamicSlice")(np, params, [0, 0], [1, 2]) is not None

    # DynamicUpdateSlice
    up_slice = scatter_mod._np_dynamic_update_slice(np, params, np.array([[99, 88]]), [0, 0])
    assert (
        up_slice[
            (
                0,
                0,
            )
        ]
        == 99
        and up_slice[
            (
                0,
                1,
            )
        ]
        == 88
    )
    assert (
        require_callable(numpy_eager_registry.get)("DynamicUpdateSlice")(np, params, np.array([[99]]), [0, 0])
        is not None
    )
