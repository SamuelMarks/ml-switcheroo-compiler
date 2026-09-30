"""Tests for test_math_scatter_coverage."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import math_scatter as scatter_mod


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
    assert np.array_equal(sc_up, target)
    assert numpy_eager_registry.get("TensorScatterUpdate")(np, target, indices, updates) is not None

    # TensorScatterAdd
    sc_add = scatter_mod._np_tensor_scatter_add(np, target, indices, updates)
    assert sc_add[0, 0] == 5.0 and sc_add[1, 1] == 10.0
    assert numpy_eager_registry.get("TensorScatterAdd")(np, target, indices, updates) is not None

    # TensorScatterMax
    sc_max = scatter_mod._np_tensor_scatter_max(np, target, indices, updates)
    assert sc_max[0, 0] == 5.0 and sc_max[1, 1] == 10.0
    assert numpy_eager_registry.get("TensorScatterMax")(np, target, indices, updates) is not None

    # TensorScatterMin
    base = np.full((3, 3), 20.0, dtype=np.float32)
    sc_min = scatter_mod._np_tensor_scatter_min(np, base, indices, updates)
    assert sc_min[0, 0] == 5.0 and sc_min[1, 1] == 10.0
    assert numpy_eager_registry.get("TensorScatterMin")(np, base, indices, updates) is not None

    # ScatterNd
    sc_nd = scatter_mod._np_scatter_nd(np, indices, updates, (3, 3))
    assert sc_nd[0, 0] == 5.0 and sc_nd[1, 1] == 10.0
    assert numpy_eager_registry.get("ScatterNd")(np, indices, updates, (3, 3)) is not None

    # Scatter
    inp = np.zeros(3)
    idx = np.array([0, 1])
    src = np.array([1.0, 2.0])
    sc_res = scatter_mod._np_scatter(np, inp, idx, src, dim=0)
    assert sc_res[0] == 1.0 and sc_res[1] == 2.0
    assert numpy_eager_registry.get("Scatter")(np, inp, idx, src) is not None

    # _band_part
    band = scatter_mod._band_part(np.eye(3), 1, 1)
    assert band.shape == (3, 3)

    # GatherNd
    params = np.array([[10, 20], [30, 40]])
    g_idx = np.array([[0, 1], [1, 0]])
    g_res = scatter_mod._np_gather_nd(np, params, g_idx)
    assert np.array_equal(g_res, [20, 30])
    assert np.array_equal(numpy_eager_registry.get("GatherNd")(np, params, g_idx), [20, 30])

    # TakeAlongAxis
    take_res = scatter_mod._np_take_along_axis(np, params, np.array([[1, 0], [0, 1]]), axis=1)
    assert np.array_equal(take_res, [[20, 10], [30, 40]])
    assert numpy_eager_registry.get("TakeAlongAxis")(np, params, np.array([[0]]), axis=0) is not None

    # DynamicSlice
    dyn_slice = scatter_mod._np_dynamic_slice(np, params, [0, 0], [1, 2])
    assert np.array_equal(dyn_slice, [[10, 20]])
    assert numpy_eager_registry.get("DynamicSlice")(np, params, [0, 0], [1, 2]) is not None

    # DynamicUpdateSlice
    up_slice = scatter_mod._np_dynamic_update_slice(np, params, np.array([[99, 88]]), [0, 0])
    assert up_slice[0, 0] == 99 and up_slice[0, 1] == 88
    assert numpy_eager_registry.get("DynamicUpdateSlice")(np, params, np.array([[99]]), [0, 0]) is not None
