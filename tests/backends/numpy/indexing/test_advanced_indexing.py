"""Tests for numpy eager advanced indexing ops."""

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.advanced_indexing import (
    _gather_nd,
    _np_gather_nd,
    _np_scatter,
    _np_scatter_add,
    _np_scatter_apply,
    _np_scatter_max,
    _np_scatter_min,
    _np_scatter_mul,
    _np_scatter_nd,
    _np_take_along_axis,
    _np_tensor_scatter_add,
    _np_tensor_scatter_max,
    _np_tensor_scatter_min,
    _np_tensor_scatter_update,
    _scatter,
    _scatter_add,
    _scatter_nd,
    _tensor_scatter_add,
    _tensor_scatter_max,
    _tensor_scatter_min,
    _tensor_scatter_update,
)


def test_gather_nd_helper() -> None:
    """Test _gather_nd."""
    x = np.array([[1, 2], [3, 4]])
    indices = np.array([[0, 0], [1, 1]])
    res = _gather_nd(x, indices)
    np.testing.assert_allclose(res, [1, 4])


def test_scatter_nd_helper() -> None:
    """Test _scatter_nd."""
    indices = np.array([[0, 0], [1, 1]])
    updates = np.array([1, 4])
    res = _scatter_nd(indices, updates, (2, 2))
    expected = np.array([[1, 0], [0, 4]])
    np.testing.assert_allclose(res, expected)


def test_scatter_helper() -> None:
    """Test _scatter."""
    x = np.zeros((2, 2))
    index = np.array([[0, 0], [1, 1]])
    src = np.array([[1, 2], [3, 4]])
    res = _scatter(x, index, src, dim=1)
    # put_along_axis replaces
    expected = np.array([[2, 0], [0, 4]])
    np.testing.assert_allclose(res, expected)


def test_scatter_add_helper() -> None:
    """Test _scatter_add."""
    x = np.zeros((2, 2))
    index = np.array([[0, 0], [1, 1]])
    src = np.array([[1, 2], [3, 4]])
    res = _scatter_add(x, index, src, dim=1)
    expected = np.array([[3, 0], [0, 7]])
    np.testing.assert_allclose(res, expected)


def test_tensor_scatter_update_helper() -> None:
    """Test _tensor_scatter_update."""
    tensor = np.zeros((2, 2))
    indices = np.array([[0, 0], [1, 1]])
    updates = np.array([1, 4])
    res1 = _tensor_scatter_update(tensor, indices, updates)
    expected = np.array([[1, 0], [0, 4]])
    np.testing.assert_allclose(res1, expected)

    # test list indices
    res2 = _tensor_scatter_update(tensor, [[0, 0], [1, 1]], updates)
    np.testing.assert_allclose(res2, expected)


def test_tensor_scatter_add_helper() -> None:
    """Test _tensor_scatter_add."""
    tensor = np.zeros((2, 2))
    indices = np.array([[0, 0], [1, 1]])
    updates = np.array([1, 4])
    res1 = _tensor_scatter_add(tensor, indices, updates)
    expected = np.array([[1, 0], [0, 4]])
    np.testing.assert_allclose(res1, expected)

    # test list indices
    res2 = _tensor_scatter_add(tensor, [[0, 0], [1, 1]], updates)
    np.testing.assert_allclose(res2, expected)


def test_tensor_scatter_max_helper() -> None:
    """Test _tensor_scatter_max."""
    tensor = np.zeros((2, 2))
    indices = np.array([[0, 0], [1, 1]])
    updates = np.array([1, 4])
    res1 = _tensor_scatter_max(tensor, indices, updates)
    expected = np.array([[1, 0], [0, 4]])
    np.testing.assert_allclose(res1, expected)

    # test list indices
    res2 = _tensor_scatter_max(tensor, [[0, 0], [1, 1]], updates)
    np.testing.assert_allclose(res2, expected)


def test_tensor_scatter_min_helper() -> None:
    """Test _tensor_scatter_min."""
    tensor = np.ones((2, 2)) * 5
    indices = np.array([[0, 0], [1, 1]])
    updates = np.array([1, 4])
    res1 = _tensor_scatter_min(tensor, indices, updates)
    expected = np.array([[1, 5], [5, 4]])
    np.testing.assert_allclose(res1, expected)

    # test list indices
    res2 = _tensor_scatter_min(tensor, [[0, 0], [1, 1]], updates)
    np.testing.assert_allclose(res2, expected)


def test_np_take_along_axis() -> None:
    """Test _np_take_along_axis."""
    x = np.array([[1, 2], [3, 4]])
    indices = np.array([[1, 0], [0, 1]])
    res = _np_take_along_axis(np, x, indices, axis=1)
    expected = np.take_along_axis(x, indices, axis=1)
    np.testing.assert_allclose(res, expected)


def test_np_tensor_scatter_update() -> None:
    """Test _np_tensor_scatter_update."""
    tensor = np.zeros((2, 2))
    indices = np.array([[0, 0], [1, 1]])
    updates = np.array([1, 4])

    # We need to register a dummy implementation in global registry for testing
    @global_eager_registry.register("TensorScatterUpdate")
    def _dummy_update(backend_module, t, idx, u):
        return _tensor_scatter_update(t, idx, u)

    res = _np_tensor_scatter_update(np, tensor, indices, updates)
    expected = np.array([[1, 0], [0, 4]])
    np.testing.assert_allclose(res, expected)


def test_np_tensor_scatter_add() -> None:
    """Test _np_tensor_scatter_add."""
    tensor = np.zeros((2, 2))
    indices = np.array([[0, 0], [1, 1]])
    updates = np.array([1, 4])
    res = _np_tensor_scatter_add(np, tensor, indices, updates)
    expected = np.array([[1, 0], [0, 4]])
    np.testing.assert_allclose(res, expected)


def test_np_tensor_scatter_max() -> None:
    """Test _np_tensor_scatter_max."""
    tensor = np.zeros((2, 2))
    indices = np.array([[0, 0], [1, 1]])
    updates = np.array([1, 4])
    res = _np_tensor_scatter_max(np, tensor, indices, updates)
    expected = np.array([[1, 0], [0, 4]])
    np.testing.assert_allclose(res, expected)


def test_np_tensor_scatter_min() -> None:
    """Test _np_tensor_scatter_min."""
    tensor = np.ones((2, 2)) * 5
    indices = np.array([[0, 0], [1, 1]])
    updates = np.array([1, 4])
    res = _np_tensor_scatter_min(np, tensor, indices, updates)
    expected = np.array([[1, 5], [5, 4]])
    np.testing.assert_allclose(res, expected)


def test_np_gather_nd() -> None:
    """Test _np_gather_nd."""
    x = np.array([[1, 2], [3, 4]])
    indices = np.array([[0, 0], [1, 1]])
    res = _np_gather_nd(np, x, indices)
    np.testing.assert_allclose(res, [1, 4])


def test_np_scatter_nd() -> None:
    """Test _np_scatter_nd."""
    indices = np.array([[0, 0], [1, 1]])
    updates = np.array([1, 4])
    res = _np_scatter_nd(np, indices, updates, (2, 2))
    expected = np.array([[1, 0], [0, 4]])
    np.testing.assert_allclose(res, expected)


def test_np_scatter() -> None:
    """Test _np_scatter."""
    x = np.zeros((2, 2))
    index = np.array([[0, 0], [1, 1]])
    src = np.array([[1, 2], [3, 4]])
    res = _np_scatter(np, x, index, src, dim=1)
    expected = np.array([[2, 0], [0, 4]])
    np.testing.assert_allclose(res, expected)


def test_np_scatter_add() -> None:
    """Test _np_scatter_add."""
    x = np.zeros((2, 2))
    index = np.array([[0, 0], [1, 1]])
    src = np.array([[1, 2], [3, 4]])
    res = _np_scatter_add(np, x, index, src, dim=1)
    expected = np.array([[2, 0], [0, 4]])
    np.testing.assert_allclose(res, expected)


def test_np_scatter_apply() -> None:
    """Test _np_scatter_apply."""
    x = np.zeros((2, 2))
    res = _np_scatter_apply(np, None, x)
    np.testing.assert_allclose(res, x)


def test_np_scatter_max() -> None:
    """Test _np_scatter_max."""
    tensor = np.zeros((2, 2))
    indices = np.array([[0, 0], [1, 1]])
    updates = np.array([1, 4])
    res = _np_scatter_max(np, tensor, indices, updates)
    expected = np.array([[1, 0], [0, 4]])
    np.testing.assert_allclose(res, expected)


def test_np_scatter_min() -> None:
    """Test _np_scatter_min."""
    tensor = np.ones((2, 2)) * 5
    indices = np.array([[0, 0], [1, 1]])
    updates = np.array([1, 4])
    res = _np_scatter_min(np, tensor, indices, updates)
    expected = np.array([[1, 5], [5, 4]])
    np.testing.assert_allclose(res, expected)


def test_np_scatter_mul() -> None:
    """Test _np_scatter_mul."""
    tensor = np.ones((2, 2)) * 2
    indices = np.array([[0, 0], [1, 1]])
    updates = np.array([2, 4])
    res = _np_scatter_mul(np, tensor, indices, updates)
    expected = np.array([[4, 2], [2, 8]])
    np.testing.assert_allclose(res, expected)
