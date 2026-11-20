"""Tests for numpy eager indexing ops."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
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
from tests.testing_utils import MockTestStub, coerce_operand_list, extract_test_shape

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


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

    class MockItem(MockTestStub):
        """Test class MockItem."""

        def __init__(self, val: BackendArg) -> None:
            """Test function __init__.

            Args:
                val: Test parameter `val`.
            """
            self.data = val

        def item(self) -> BackendArg:
            """Test function item.

            Returns:
                BackendArg: Computed output value from `item`.
            """
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
    assert (
        ensure_ndarray(res)[
            (
                1,
                1,
            )
        ]
        == 1.0
    )


def test_np_unstack() -> None:
    """Test _np_unstack."""
    x = np.array([[1, 2], [3, 4]])
    res = _np_unstack(np, x, axis=0)
    assert len(coerce_operand_list(res)) == 2
    np.testing.assert_allclose(ensure_ndarray(coerce_operand_list(res)[0]), [1, 2])
    np.testing.assert_allclose(ensure_ndarray(coerce_operand_list(res)[1]), [3, 4])


def test_np_dynamic_slice() -> None:
    """Test _np_dynamic_slice."""
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    start_indices = [1, 1]
    slice_sizes = [2, 2]
    res = _np_dynamic_slice(np, x, start_indices, slice_sizes)
    expected = np.array([[5, 6], [8, 9]])
    np.testing.assert_allclose(ensure_ndarray(res), expected)


def test_np_dynamic_slice_in_dim() -> None:
    """Test _np_dynamic_slice_in_dim."""
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    ctx = IndexingContext(axis=1, start_index=1, slice_size=2)
    res = _np_dynamic_slice_in_dim(np, x, ctx)
    expected = np.array([[2, 3], [5, 6], [8, 9]])
    np.testing.assert_allclose(ensure_ndarray(res), expected)


def test_np_dynamic_update_slice_in_dim() -> None:
    """Test _np_dynamic_update_slice_in_dim."""
    x = np.zeros((3, 3))
    update = np.ones((3, 2))
    ctx = IndexingContext(axis=1, start_index=1)
    res = _np_dynamic_update_slice_in_dim(np, x, update, ctx)
    assert (
        ensure_ndarray(res)[
            (
                0,
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


def test_np_dynamic_index_in_dim() -> None:
    """Test _np_dynamic_index_in_dim."""
    x = np.array([[1, 2, 3], [4, 5, 6]])
    # keepdims = True
    ctx_true = IndexingContext(axis=1, keepdims=True)
    res_true = _np_dynamic_index_in_dim(np, x, 1, ctx_true)
    assert extract_test_shape(res_true) == (2, 1)

    # keepdims = False
    ctx_false = IndexingContext(axis=1, keepdims=False)
    res_false = _np_dynamic_index_in_dim(np, x, 1, ctx_false)
    assert extract_test_shape(res_false) == (2,)


def test_np_dynamic_update_index_in_dim() -> None:
    """Test _np_dynamic_update_index_in_dim."""
    x = np.zeros((3, 3))
    update = np.ones((3,))
    target = IndexTarget(operand=x, update=update, index=1)
    ctx = IndexingContext(axis=1)
    res = _np_dynamic_update_index_in_dim(np, target, ctx)
    assert (
        ensure_ndarray(res)[
            (
                0,
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


def test_np_slice_in_dim() -> None:
    """Test _np_slice_in_dim."""
    x = np.array([[1, 2, 3], [4, 5, 6]])
    ctx = IndexingContext(axis=1, start_index=1, limit_index=3, stride=1)
    res = _np_slice_in_dim(np, x, ctx)
    expected = np.array([[2, 3], [5, 6]])
    np.testing.assert_allclose(ensure_ndarray(res), expected)


def test_np_slice() -> None:
    """Test _np_slice."""
    x = np.array([[1, 2, 3], [4, 5, 6]])
    ctx = IndexingContext(axis=1, start_index=1, limit_index=3, stride=1)
    res = _np_slice(np, x, ctx)
    expected = np.array([[2, 3], [5, 6]])
    np.testing.assert_allclose(ensure_ndarray(res), expected)


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
    np.testing.assert_allclose(ensure_ndarray(res), [[1, 2], [3, 4]])


def test_gather_eager() -> None:
    """Test gather_eager."""
    x = np.array([[1, 2], [3, 4]])
    idx = np.array([[0, 0], [1, 1]])

    # Test args
    res = gather_eager(np, x, 1, idx)
    expected = np.take_along_axis(x, idx, axis=1)
    np.testing.assert_allclose(ensure_ndarray(res), expected)

    # Test kwargs
    res_kwargs = gather_eager(np, x, dim=1, index=idx)
    np.testing.assert_allclose(ensure_ndarray(res_kwargs), expected)

    # test .numpy() unwrap
    class MockTensor:
        """Test class MockTensor."""

        def __init__(self, data: BackendArg) -> None:
            """Test function __init__.

            Args:
                data: Test parameter `data`.
            """
            self.data = data

        def numpy(self) -> BackendArg:
            """Test function numpy.

            Returns:
                BackendArg: Computed output value from `numpy`.
            """
            return self.data

    t_x = MockTensor(x)
    t_idx = MockTensor(idx)
    res_mock = gather_eager(np, t_x, 1, t_idx)
    np.testing.assert_allclose(ensure_ndarray(res_mock), expected)


def test_stack_eager() -> None:
    """Test stack_eager."""
    a = np.array([1, 2])
    b = np.array([3, 4])

    # Test args
    res = stack_eager(np, [a, b], 0)
    expected = np.stack([a, b], axis=0)
    np.testing.assert_allclose(ensure_ndarray(res), expected)

    # Test kwargs
    res_kwargs = stack_eager(np, tensors=[a, b], dim=0)
    np.testing.assert_allclose(ensure_ndarray(res_kwargs), expected)

    # Test axis kwarg
    res_axis = stack_eager(np, tensors=[a, b], axis=0)
    np.testing.assert_allclose(ensure_ndarray(res_axis), expected)

    # Test .numpy() unwrap
    class MockTensor:
        """Test class MockTensor."""

        def __init__(self, data: BackendArg) -> None:
            """Test function __init__.

            Args:
                data: Test parameter `data`.
            """
            self.data = data

        def numpy(self) -> BackendArg:
            """Test function numpy.

            Returns:
                BackendArg: Computed output value from `numpy`.
            """
            return self.data

    t_a = MockTensor(a)
    t_b = MockTensor(b)
    res_mock = stack_eager(np, [t_a, t_b], 0)
    np.testing.assert_allclose(ensure_ndarray(res_mock), expected)


def test_dynamic_update_slice_with_item() -> None:
    """Test function test_dynamic_update_slice_with_item."""

    class MockItem(MockTestStub):
        # has item but not data
        """Test class MockItem."""

        def item(self) -> BackendArg:
            """Test function item.

            Returns:
                BackendArg: Computed output value from `item`.
            """
            return 1

    x = np.zeros((3, 3))
    update = np.ones((2, 2))
    start_indices = [MockItem(), MockItem()]
    res = _dynamic_update_slice(x, update, start_indices)
    assert res[1, 1] == 1.0


def test_backends_numpy_indexing_setitem() -> None:
    """Test _np_setitem in backends/numpy/eager/indexing.py."""
    from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry

    func = numpy_eager_registry.get("SetItem")
    assert func is not None
    x = np.array([10, 20, 30])
    res = func(np, x, 99, "1")
    assert res[1] == 99
    assert res[0] == 10
