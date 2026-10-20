"""Tests for test_special."""

from __future__ import annotations

from ml_switcheroo_compiler.ops.binary.special import Atan2, Divmod, Isclose


def test_divmod_infer_shape_fallback() -> None:
    """Test the divmod infer shape fallback behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    op = Divmod()
    assert op.infer_shape(None, (1, 2)) is None


def test_isclose_infer_shape_fallback() -> None:
    """Test the isclose infer shape fallback behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    op = Isclose()
    assert op.infer_shape(None, (1, 2)) is None


def test_atan2_infer_shape_fallback() -> None:
    """Test the atan2 infer shape fallback behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    op = Atan2()
    assert op.infer_shape(None, (1, 2)) is None
