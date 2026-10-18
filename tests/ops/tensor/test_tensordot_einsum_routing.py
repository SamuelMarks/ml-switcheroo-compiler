"""Tests for _tensordot_einsum_routing and its helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.linalg.einsum_frontend import _generate_tensordot_einsum_strings, _tensordot_einsum_routing, _validate_tensordot_axes

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_validate_tensordot_axes() -> None:
    """Test the validate tensordot axes behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    axes = ([1, 2], [0, 1])
    assert _validate_tensordot_axes(axes) == ([1, 2], [0, 1])


def test_generate_tensordot_einsum_strings() -> None:
    """Test the generate tensordot einsum strings behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    (a_str, b_str, out_str) = _generate_tensordot_einsum_strings(shape_a=(2, 3, 4), shape_b=(3, 4, 5), axes_a=[1, 2], axes_b=[0, 1])
    assert a_str == "abc"
    assert b_str == "bcf"
    assert out_str == "af"


@patch("ml_switcheroo_compiler.ops.linalg.einsum_frontend.einsum")
def test_tensordot_einsum_routing(mock_einsum: EagerTensorProtocol) -> None:
    """Test the tensordot einsum routing behavior.

    Args:
        mock_einsum (EagerTensorProtocol): The mock_einsum parameter.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    a = MagicMock(spec=Tensor)
    a.shape = (2, 3, 4)
    b = MagicMock(spec=Tensor)
    b.shape = (3, 4, 5)
    setattr(mock_einsum, "return_value", "fake_tensor")
    res = _tensordot_einsum_routing(a, b, ([1, 2], [0, 1]))
    assert res == "fake_tensor"
    mock_einsum.assert_called_once_with("abc,bcf->af", a, b)


def test_generate_tensordot_einsum_strings_early_return() -> None:
    """Test the generate tensordot einsum strings early return behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    (a_str, b_str, out_str) = _generate_tensordot_einsum_strings((), (), [], [])
    assert a_str == ""
    assert b_str == ""
    assert out_str == ""
