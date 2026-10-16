"""Tests for test_conv_ops."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.ops.configs import ConvConfig
from ml_switcheroo_compiler.ops.linalg.conv_ops import (
    ConvGeneralDilated,
    ConvGeneralDilatedLocal,
    ConvGeneralDilatedPatches,
    Convolve,
    ConvTranspose,
    ConvTransposeShapeTuple,
    ConvWithGeneralPadding,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class DummyTensor:
    """Test class DummyTensor."""

    def __init__(self, shape: EagerValue) -> None:
        """Test function __init__."""
        self.shape = shape
        self.dtype = "float32"
        self.config = None


def test_conv_ops_infer_shape() -> None:
    # ConvGeneralDilated
    """Test function test_conv_ops_infer_shape."""
    op = ConvGeneralDilated()
    lhs = DummyTensor((1, 1))
    rhs = DummyTensor((1, 1))
    config = ConvConfig(window_strides=[], padding=[])

    cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(lhs, rhs, config)
    cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(lhs=lhs, rhs=rhs, config=config)
    cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(lhs, rhs)
    cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(lhs, rhs=rhs, config=config)

    # Missing shapes
    class NoShape:
        """Test class NoShape."""

        pass

    cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(cast("Callable[..., EagerTensorProtocol]", NoShape)(), cast("Callable[..., EagerTensorProtocol]", NoShape)())

    # Convolve
    op2 = Convolve()
    cast("Callable[..., EagerTensorProtocol]", op2.infer_shape)(lhs, rhs)

    # ConvGeneralDilatedLocal
    op3 = ConvGeneralDilatedLocal()
    cast("Callable[..., EagerTensorProtocol]", op3.infer_shape)(lhs)
    op3.infer_shape()

    # ConvGeneralDilatedPatches
    op4 = ConvGeneralDilatedPatches()
    cast("Callable[..., EagerTensorProtocol]", op4.infer_shape)(lhs)
    op4.infer_shape()

    # ConvWithGeneralPadding
    op5 = ConvWithGeneralPadding()
    cast("Callable[..., EagerTensorProtocol]", op5.infer_shape)(lhs)
    op5.infer_shape()

    # ConvTransposeShapeTuple
    op6 = ConvTransposeShapeTuple()
    op6.infer_shape()

    # ConvTranspose
    op7 = ConvTranspose()
    lhs = DummyTensor((1, 4, 4, 3))
    rhs = DummyTensor((3, 3, 3, 2))
    assert cast("Callable[..., EagerTensorProtocol]", op7.infer_shape)(lhs, rhs, strides=2, padding="VALID") == (1, 9, 9, 2)
    assert cast("Callable[..., EagerTensorProtocol]", op7.infer_shape)(lhs, rhs, strides=2, padding="SAME") == (1, 8, 8, 2)
    assert cast("Callable[..., EagerTensorProtocol]", op7.infer_shape)(lhs, rhs, strides=(2, 2), padding="VALID") == (1, 9, 9, 2)


def test_conv_exact_shape_deduction() -> None:
    """Verify exact output shape calculation for ConvGeneralDilated and Convolve."""
    # 1. Test 2D ConvGeneralDilated (NCHW lhs, OIHW rhs)
    op = ConvGeneralDilated()
    lhs = DummyTensor((2, 3, 32, 32))
    rhs = DummyTensor((16, 3, 3, 3))
    cfg = ConvConfig(window_strides=[2, 2], padding=[(1, 1), (1, 1)])
    # Spatial formula: (32 + 2 - 3) // 2 + 1 = 31 // 2 + 1 = 16
    assert cast("Callable[..., EagerTensorProtocol]", op.infer_shape)(lhs, rhs, cfg) == (2, 16, 16, 16)

    # 2. Test 1D Convolve modes
    conv1d = Convolve()
    a = DummyTensor((10,))
    v = DummyTensor((4,))
    assert cast("Callable[..., EagerTensorProtocol]", conv1d.infer_shape)(a, v, mode="full") == (13,)
    assert cast("Callable[..., EagerTensorProtocol]", conv1d.infer_shape)(a, v, mode="same") == (10,)
    assert cast("Callable[..., EagerTensorProtocol]", conv1d.infer_shape)(a, v, mode="valid") == (7,)
    assert cast("Callable[..., EagerTensorProtocol]", conv1d.infer_shape)(a, v, mode="unknown_default") == (13,)
    assert cast("Callable[..., EagerTensorProtocol]", conv1d.infer_shape)(None, v) == ()
    assert cast("Callable[..., EagerTensorProtocol]", conv1d.infer_shape)(a, None) == ()
