"""Tests for test_shape_ops."""

from __future__ import annotations

import typing

# ruff: noqa: E501
import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.backends.numpy.eager.array_manipulation import (
    _np_append,
    _np_atleast_1d,
    _np_atleast_2d,
    _np_atleast_3d,
    _np_band_part,
    _np_block,
    _np_column_stack,
    _np_delete,
    _np_diag,
    _np_diag_indices,
    _np_diag_indices_from,
    _np_diagflat,
    _np_dsplit,
    _np_dstack,
    _np_expand_dims,
    _np_fill_diagonal,
    _np_hsplit,
    _np_hstack,
    _np_insert,
    _np_moveaxis,
    _np_permute,
    _np_reshape,
    _np_resize,
    _np_roll,
    _np_rot90,
    _np_squeeze,
    _np_swapaxes,
    _np_tile,
    _np_transpose,
    _np_tril,
    _np_triu,
    _np_unstack,
    _np_vsplit,
    _np_vstack,
    gather_eager,
    stack_eager,
)

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_resize() -> None:
    """Test function test_resize."""
    x = np.ones((1, 2, 3))
    res = _np_resize(np, x, (4, 5))
    assert typing.cast("EagerTensorProtocol", getattr(res, "shape")) == (4, 5, 3)
    assert np.all(res == 1)


def test_band_part() -> None:
    """Test function test_band_part."""
    x = np.array([[1, 2], [3, 4]])
    res = _np_band_part(np, x, 1, 1)
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(res, x)


def test_diag() -> None:
    """Test function test_diag."""
    x = np.array([[1, 2], [3, 4]])
    res = _np_diag(np, x, diagonal=0)
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(res, [1, 4])
    res2 = _np_diag(np, x, k=1)
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(res2, [2])


def test_unstack() -> None:
    """Test function test_unstack."""
    x = np.array([[1, 2], [3, 4]])
    res = _np_unstack(np, x, axis=0)
    assert typing.cast("EagerCallableProtocol", len)(res) == 2
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(typing.cast("EagerTensorProtocol", getattr(res, "__getitem__")(0)), [1, 2])


def test_reshape() -> None:
    """Test function test_reshape."""
    x = np.ones((2, 2))
    res = _np_reshape(np, x, (4,))
    assert typing.cast("EagerTensorProtocol", getattr(res, "shape")) == (4,)
    res_kwargs = _np_reshape(np, x, shape=(4,))
    assert typing.cast("EagerTensorProtocol", getattr(res_kwargs, "shape")) == (4,)
    res_newshape = _np_reshape(np, x, newshape=(4,))
    assert typing.cast("EagerTensorProtocol", getattr(res_newshape, "shape")) == (4,)


def test_squeeze() -> None:
    """Test function test_squeeze."""
    x = np.ones((1, 2, 1))
    res = _np_squeeze(np, x)
    assert typing.cast("EagerTensorProtocol", getattr(res, "shape")) == (2,)
    res_args = _np_squeeze(np, x, 0)
    assert typing.cast("EagerTensorProtocol", getattr(res_args, "shape")) == (2, 1)
    res_kwargs = _np_squeeze(np, x, axis=2)
    assert typing.cast("EagerTensorProtocol", getattr(res_kwargs, "shape")) == (1, 2)


def test_transpose() -> None:
    """Test function test_transpose."""
    x = np.ones((1, 2, 3))
    res = _np_transpose(np, x, (2, 1, 0))
    assert typing.cast("EagerTensorProtocol", getattr(res, "shape")) == (3, 2, 1)
    res_axes = _np_transpose(np, x, axes=(2, 1, 0))
    assert typing.cast("EagerTensorProtocol", getattr(res_axes, "shape")) == (3, 2, 1)
    res_dims = _np_transpose(np, x, dims=(2, 1, 0))
    assert typing.cast("EagerTensorProtocol", getattr(res_dims, "shape")) == (3, 2, 1)


def test_rot90() -> None:
    """Test function test_rot90."""
    x = np.array([[1, 2], [3, 4]])
    res = _np_rot90(np, x)
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(res, typing.cast("EagerCallableProtocol", np.rot90)(x))


def test_gather_eager() -> None:
    """Test function test_gather_eager."""
    x = np.array([[1, 2], [3, 4]])
    idx = np.array([[0, 0], [1, 1]])
    res = gather_eager(np, x, 1, idx)
    expected = np.take_along_axis(x, idx, axis=1)
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(res, expected)
    res_kwargs = gather_eager(np, x, axis=1, index=idx)
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(res_kwargs, expected)

    class MockTensor:
        """Test class MockTensor."""

        def __init__(self, data: EagerValue) -> None:
            """Test function __init__.

            Args:
                data: Test parameter `data`.
            """
            self.data = data

        def numpy(self) -> EagerValue:
            """Test function numpy."""
            return self.data

    t_x = MockTensor(x)
    t_idx = MockTensor(idx)
    res_mock = gather_eager(np, typing.cast("EagerValue", t_x), 1, typing.cast("EagerValue", t_idx))
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(res_mock, expected)


def test_stack_eager() -> None:
    """Test function test_stack_eager."""
    a = np.array([1, 2])
    b = np.array([3, 4])
    res = stack_eager(np, [a, b], 0)
    expected = np.stack([a, b], axis=0)
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(res, expected)
    res_kwargs = stack_eager(np, tensors=[a, b], dim=0)
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(res_kwargs, expected)
    res_axis = stack_eager(np, tensors=[a, b], axis=0)
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(res_axis, expected)

    class MockTensor:
        """Test class MockTensor."""

        def __init__(self, data: EagerValue) -> None:
            """Test function __init__.

            Args:
                data: Test parameter `data`.
            """
            self.data = data

        def numpy(self) -> EagerValue:
            """Test function numpy."""
            return self.data

    t_a = MockTensor(a)
    t_b = MockTensor(b)
    res_mock = stack_eager(np, [typing.cast("EagerValue", t_a), typing.cast("EagerValue", t_b)], 0)
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(res_mock, expected)


def test_tile() -> None:
    """Test function test_tile."""
    x = np.array([1, 2])
    res = _np_tile(np, x, 2)
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(res, [1, 2, 1, 2])


def test_permute() -> None:
    """Test function test_permute."""
    x = np.ones((1, 2, 3))
    res = _np_permute(np, x, (2, 1, 0))
    assert typing.cast("EagerTensorProtocol", getattr(res, "shape")) == (3, 2, 1)
    res_dims = _np_permute(np, x, dims=(2, 1, 0))
    assert typing.cast("EagerTensorProtocol", getattr(res_dims, "shape")) == (3, 2, 1)


def test_triu() -> None:
    """Test function test_triu."""
    x = np.ones((3, 3))
    res = _np_triu(np, x, diagonal=1)
    expected = np.triu(x, k=1)
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(res, expected)


def test_tril() -> None:
    """Test function test_tril."""
    x = np.ones((3, 3))
    res = _np_tril(np, x, diagonal=-1)
    expected = np.tril(x, k=-1)
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(res, expected)


def test_expand_dims() -> None:
    """Test function test_expand_dims."""
    x = np.array([1, 2])
    res = _np_expand_dims(np, x, 0)
    assert typing.cast("EagerTensorProtocol", getattr(res, "shape")) == (1, 2)
    res_kwargs = _np_expand_dims(np, x, axis=1)
    assert typing.cast("EagerTensorProtocol", getattr(res_kwargs, "shape")) == (2, 1)


def test_atleast_1d() -> None:
    """Test function test_atleast_1d."""
    res = _np_atleast_1d(np, 1)
    assert typing.cast("EagerTensorProtocol", getattr(res, "shape")) == (1,)


def test_atleast_2d() -> None:
    """Test function test_atleast_2d."""
    res = _np_atleast_2d(np, 1)
    assert typing.cast("EagerTensorProtocol", getattr(res, "shape")) == (1, 1)


def test_atleast_3d() -> None:
    """Test function test_atleast_3d."""
    res = _np_atleast_3d(np, 1)
    assert typing.cast("EagerTensorProtocol", getattr(res, "shape")) == (1, 1, 1)


def test_append() -> None:
    """Test function test_append."""
    res = _np_append(np, [1, 2, 3], [[4, 5, 6], [7, 8, 9]])
    assert typing.cast("EagerTensorProtocol", getattr(res, "shape")) == (9,)


def test_column_stack() -> None:
    """Test function test_column_stack."""
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    res = _np_column_stack(np, (a, b))
    assert typing.cast("EagerTensorProtocol", getattr(res, "shape")) == (3, 2)
    res_kwargs = _np_column_stack(np, tup=(a, b))
    assert typing.cast("EagerTensorProtocol", getattr(res_kwargs, "shape")) == (3, 2)


def test_dsplit() -> None:
    """Test function test_dsplit."""
    a = np.ones((2, 2, 2))
    res = _np_dsplit(np, a, 2)
    assert typing.cast("EagerCallableProtocol", len)(res) == 2


def test_dstack() -> None:
    """Test function test_dstack."""
    a = np.ones((2, 2))
    b = np.ones((2, 2))
    res = _np_dstack(np, (a, b))
    assert typing.cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 2, 2)
    res_kwargs = _np_dstack(np, tup=(a, b))
    assert typing.cast("EagerTensorProtocol", getattr(res_kwargs, "shape")) == (2, 2, 2)


def test_hsplit() -> None:
    """Test function test_hsplit."""
    a = np.ones((2, 2))
    res = _np_hsplit(np, a, 2)
    assert typing.cast("EagerCallableProtocol", len)(res) == 2


def test_hstack() -> None:
    """Test function test_hstack."""
    a = np.ones((2, 2))
    b = np.ones((2, 2))
    res = _np_hstack(np, (a, b))
    assert typing.cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 4)
    res_kwargs = _np_hstack(np, tup=(a, b))
    assert typing.cast("EagerTensorProtocol", getattr(res_kwargs, "shape")) == (2, 4)


def test_vsplit() -> None:
    """Test function test_vsplit."""
    a = np.ones((2, 2))
    res = _np_vsplit(np, a, 2)
    assert typing.cast("EagerCallableProtocol", len)(res) == 2


def test_vstack() -> None:
    """Test function test_vstack."""
    a = np.ones((2, 2))
    b = np.ones((2, 2))
    res = _np_vstack(np, (a, b))
    assert typing.cast("EagerTensorProtocol", getattr(res, "shape")) == (4, 2)
    res_kwargs = _np_vstack(np, tup=(a, b))
    assert typing.cast("EagerTensorProtocol", getattr(res_kwargs, "shape")) == (4, 2)


def test_moveaxis() -> None:
    """Test function test_moveaxis."""
    a = np.ones((1, 2, 3))
    res = _np_moveaxis(np, a, 0, -1)
    assert typing.cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 3, 1)


def test_swapaxes() -> None:
    """Test function test_swapaxes."""
    a = np.ones((1, 2, 3))
    res = _np_swapaxes(np, a, 0, -1)
    assert typing.cast("EagerTensorProtocol", getattr(res, "shape")) == (3, 2, 1)


def test_roll() -> None:
    """Test function test_roll."""
    a = np.array([1, 2, 3])
    res = _np_roll(np, a, 1)
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(res, [3, 1, 2])


def test_block() -> None:
    """Test function test_block."""
    a = np.ones((2, 2))
    res = _np_block(np, [[a, a], [a, a]])
    assert typing.cast("EagerTensorProtocol", getattr(res, "shape")) == (4, 4)


def test_delete() -> None:
    """Test function test_delete."""
    a = np.array([1, 2, 3])
    res = _np_delete(np, a, 1)
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(res, [1, 3])


def test_diag_indices() -> None:
    """Test function test_diag_indices."""
    res = _np_diag_indices(np, 2)
    assert typing.cast("EagerCallableProtocol", len)(res) == 2
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(typing.cast("EagerTensorProtocol", getattr(res, "__getitem__")(0)), [0, 1])


def test_diag_indices_from() -> None:
    """Test function test_diag_indices_from."""
    a = np.ones((2, 2))
    res = _np_diag_indices_from(np, a)
    assert typing.cast("EagerCallableProtocol", len)(res) == 2
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(typing.cast("EagerTensorProtocol", getattr(res, "__getitem__")(0)), [0, 1])


def test_diagflat() -> None:
    """Test function test_diagflat."""
    a = np.array([1, 2])
    res = _np_diagflat(np, a)
    assert typing.cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 2)


def test_fill_diagonal() -> None:
    """Test function test_fill_diagonal."""
    a = np.zeros((3, 3))
    res = _np_fill_diagonal(np, a, 1)
    assert (
        typing.cast(
            "EagerTensorProtocol",
            getattr(res, "__getitem__")(
                (
                    0,
                    0,
                )
            ),
        )
        == 1
    )
    b = np.zeros((3, 3))
    res2 = _np_fill_diagonal(np, a=b, val=2)
    assert (
        typing.cast(
            "EagerTensorProtocol",
            getattr(res2, "__getitem__")(
                (
                    0,
                    0,
                )
            ),
        )
        == 2
    )


def test_insert() -> None:
    """Test function test_insert."""
    a = np.array([1, 2])
    res = _np_insert(np, a, 1, 3)
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(res, [1, 3, 2])
