"""Tests for _tensordot_einsum_routing and its helpers."""

from unittest.mock import MagicMock, patch

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.linalg.einsum_frontend import (
    _generate_tensordot_einsum_strings,
    _tensordot_einsum_routing,
    _validate_tensordot_axes,
)


def test_validate_tensordot_axes() -> object:
    """Function docstring."""
    axes = ([1, 2], [0, 1])
    assert _validate_tensordot_axes(axes) == ([1, 2], [0, 1])


def test_generate_tensordot_einsum_strings() -> object:
    """Function docstring."""
    # shape_a: (A, B, C), shape_b: (B, C, D)
    # axes_a: [1, 2], axes_b: [0, 1]
    # 'abc', 'bcd'
    # a: 3 dims, b: 3 dims
    # alphabet = abcdef...
    # a_letters = ['a', 'b', 'c']
    # b_letters = ['d', 'e', 'f']
    # zip([1, 2], [0, 1]) -> b_letters[0] = a_letters[1] ('b')
    #                     -> b_letters[1] = a_letters[2] ('c')
    # a_str: "abc", b_str: "bcf"
    # contracted: {'b', 'c'}
    # out_str: "a" + "f" = "af"
    a_str, b_str, out_str = _generate_tensordot_einsum_strings(shape_a=(2, 3, 4), shape_b=(3, 4, 5), axes_a=[1, 2], axes_b=[0, 1])
    assert a_str == "abc"
    assert b_str == "bcf"
    assert out_str == "af"


@patch("ml_switcheroo_compiler.ops.linalg.einsum_frontend.einsum")
def test_tensordot_einsum_routing(mock_einsum: object) -> object:
    """Function docstring."""
    a = MagicMock(spec=Tensor)
    a.shape = (2, 3, 4)
    b = MagicMock(spec=Tensor)
    b.shape = (3, 4, 5)

    mock_einsum.return_value = "fake_tensor"

    res = _tensordot_einsum_routing(a, b, ([1, 2], [0, 1]))
    assert res == "fake_tensor"
    mock_einsum.assert_called_once_with("abc,bcf->af", a, b)


def test_generate_tensordot_einsum_strings_early_return() -> object:
    """Function docstring."""
    # Scalar products
    a_str, b_str, out_str = _generate_tensordot_einsum_strings((), (), [], [])
    assert a_str == ""
    assert b_str == ""
    assert out_str == ""
