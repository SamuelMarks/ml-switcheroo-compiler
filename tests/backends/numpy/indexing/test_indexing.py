"""Tests for numpy eager indexing ops."""

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.indexing import (
    IndexingContext,
    IndexTarget,
    _dynamic_update_slice,
    _eager_indexindim,
    _np_dynamic_index_in_dim,
    _np_dynamic_slice,
    _np_dynamic_slice_in_dim,
    _np_dynamic_update_index_in_dim,
    _np_dynamic_update_slice,
    _np_dynamic_update_slice_in_dim,
    _np_getitem,
    _np_slice,
    _np_slice_in_dim,
    _np_unstack,
    gather_eager,
    stack_eager,
)


def test_index_target_and_context() -> None:
    """Test IndexTarget and IndexingContext."""
    target = IndexTarget(operand=1, update=2, index=0)
    assert target.operand == 1
    assert target.update == 2
    assert target.index == 0

    ctx = IndexingContext(axis=1)
    assert ctx.axis == 1


def test_dynamic_update_slice() -> None:
    """Test _dynamic_update_slice."""
    x = np.zeros((3, 3))
    update = np.ones((2, 2))
    start_indices = [1, 1]
    res = _dynamic_update_slice(x, update, start_indices)
    assert res[1, 1] == 1.0
    assert res[0, 0] == 0.0

    class MockItem:
        def __init__(self, val):
            self.data = val

        def item(self):
            return self.data

    start_indices_mock = [MockItem(1), MockItem(1)]
    res_mock = _dynamic_update_slice(x, update, start_indices_mock)
    assert res_mock[1, 1] == 1.0


def test_np_dynamic_update_slice() -> None:
    """Test _np_dynamic_update_slice."""
    x = np.zeros((3, 3))
    update = np.ones((2, 2))
    start_indices = [1, 1]
    res = _np_dynamic_update_slice(np, x, update, start_indices)
    assert res[1, 1] == 1.0


def test_np_unstack() -> None:
    """Test _np_unstack."""
    x = np.array([[1, 2], [3, 4]])
    res = _np_unstack(np, x, axis=0)
    assert len(res) == 2
    np.testing.assert_allclose(res[0], [1, 2])
    np.testing.assert_allclose(res[1], [3, 4])


def test_np_dynamic_slice() -> None:
    """Test _np_dynamic_slice."""
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    start_indices = [1, 1]
    slice_sizes = [2, 2]
    res = _np_dynamic_slice(np, x, start_indices, slice_sizes)
    expected = np.array([[5, 6], [8, 9]])
    np.testing.assert_allclose(res, expected)


def test_np_dynamic_slice_in_dim() -> None:
    """Test _np_dynamic_slice_in_dim."""
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    ctx = IndexingContext(axis=1, start_index=1, slice_size=2)
    res = _np_dynamic_slice_in_dim(np, x, ctx)
    expected = np.array([[2, 3], [5, 6], [8, 9]])
    np.testing.assert_allclose(res, expected)


def test_np_dynamic_update_slice_in_dim() -> None:
    """Test _np_dynamic_update_slice_in_dim."""
    x = np.zeros((3, 3))
    update = np.ones((3, 2))
    ctx = IndexingContext(axis=1, start_index=1)
    res = _np_dynamic_update_slice_in_dim(np, x, update, ctx)
    assert res[0, 1] == 1.0
    assert res[0, 0] == 0.0


def test_np_dynamic_index_in_dim() -> None:
    """Test _np_dynamic_index_in_dim."""
    x = np.array([[1, 2, 3], [4, 5, 6]])
    # keepdims = True
    ctx_true = IndexingContext(axis=1, keepdims=True)
    res_true = _np_dynamic_index_in_dim(np, x, 1, ctx_true)
    assert res_true.shape == (2, 1)

    # keepdims = False
    ctx_false = IndexingContext(axis=1, keepdims=False)
    res_false = _np_dynamic_index_in_dim(np, x, 1, ctx_false)
    assert res_false.shape == (2,)


def test_np_dynamic_update_index_in_dim() -> None:
    """Test _np_dynamic_update_index_in_dim."""
    x = np.zeros((3, 3))
    update = np.ones((3,))
    target = IndexTarget(operand=x, update=update, index=1)
    ctx = IndexingContext(axis=1)
    res = _np_dynamic_update_index_in_dim(np, target, ctx)
    assert res[0, 1] == 1.0
    assert res[0, 0] == 0.0


def test_np_slice_in_dim() -> None:
    """Test _np_slice_in_dim."""
    x = np.array([[1, 2, 3], [4, 5, 6]])
    ctx = IndexingContext(axis=1, start_index=1, limit_index=3, stride=1)
    res = _np_slice_in_dim(np, x, ctx)
    expected = np.array([[2, 3], [5, 6]])
    np.testing.assert_allclose(res, expected)


def test_np_slice() -> None:
    """Test _np_slice."""
    x = np.array([[1, 2, 3], [4, 5, 6]])
    ctx = IndexingContext(axis=1, start_index=1, limit_index=3, stride=1)
    res = _np_slice(np, x, ctx)
    expected = np.array([[2, 3], [5, 6]])
    np.testing.assert_allclose(res, expected)


def test_np_getitem() -> None:
    """Test _np_getitem."""
    x = np.array([1, 2, 3])
    # _safe_parse_key("1") evaluates to 1
    res = _np_getitem(np, x, "1")
    assert res == 2


def test_eager_indexindim() -> None:
    """Test _eager_indexindim."""
    x = np.array([[1, 2], [3, 4]])
    res = _eager_indexindim(np, x, [0, 1], 1)
    np.testing.assert_allclose(res, [[1, 2], [3, 4]])


def test_gather_eager() -> None:
    """Test gather_eager."""
    x = np.array([[1, 2], [3, 4]])
    idx = np.array([[0, 0], [1, 1]])

    # Test args
    res = gather_eager(np, x, 1, idx)
    expected = np.take_along_axis(x, idx, axis=1)
    np.testing.assert_allclose(res, expected)

    # Test kwargs
    res_kwargs = gather_eager(np, x, dim=1, index=idx)
    np.testing.assert_allclose(res_kwargs, expected)

    # test .numpy() unwrap
    class MockTensor:
        def __init__(self, data):
            self.data = data

        def numpy(self):
            return self.data

    t_x = MockTensor(x)
    t_idx = MockTensor(idx)
    res_mock = gather_eager(np, t_x, 1, t_idx)
    np.testing.assert_allclose(res_mock, expected)


def test_stack_eager() -> None:
    """Test stack_eager."""
    a = np.array([1, 2])
    b = np.array([3, 4])

    # Test args
    res = stack_eager(np, [a, b], 0)
    expected = np.stack([a, b], axis=0)
    np.testing.assert_allclose(res, expected)

    # Test kwargs
    res_kwargs = stack_eager(np, tensors=[a, b], dim=0)
    np.testing.assert_allclose(res_kwargs, expected)

    # Test axis kwarg
    res_axis = stack_eager(np, tensors=[a, b], axis=0)
    np.testing.assert_allclose(res_axis, expected)

    # Test .numpy() unwrap
    class MockTensor:
        def __init__(self, data):
            self.data = data

        def numpy(self):
            return self.data

    t_a = MockTensor(a)
    t_b = MockTensor(b)
    res_mock = stack_eager(np, [t_a, t_b], 0)
    np.testing.assert_allclose(res_mock, expected)
