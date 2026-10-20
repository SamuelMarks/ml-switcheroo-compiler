"""Tests for test_conv_ops."""

from __future__ import annotations

import typing

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

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerValue


class DummyTensor:
    """Test class DummyTensor."""

    def __init__(self, shape: EagerValue) -> None:
        """Test function __init__.

        Args:
            shape: Test parameter `shape`.
        """
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

    typing.cast("EagerCallableProtocol", op.infer_shape)(lhs, rhs, config)
    typing.cast("EagerCallableProtocol", op.infer_shape)(lhs=lhs, rhs=rhs, config=config)
    typing.cast("EagerCallableProtocol", op.infer_shape)(lhs, rhs)
    typing.cast("EagerCallableProtocol", op.infer_shape)(lhs, rhs=rhs, config=config)

    # Missing shapes
    class NoShape:
        """Test class NoShape."""

        pass

    typing.cast("EagerCallableProtocol", op.infer_shape)(typing.cast("EagerCallableProtocol", NoShape)(), typing.cast("EagerCallableProtocol", NoShape)())

    # Convolve
    op2 = Convolve()
    typing.cast("EagerCallableProtocol", op2.infer_shape)(lhs, rhs)

    # ConvGeneralDilatedLocal
    op3 = ConvGeneralDilatedLocal()
    typing.cast("EagerCallableProtocol", op3.infer_shape)(lhs)
    op3.infer_shape()

    # ConvGeneralDilatedPatches
    op4 = ConvGeneralDilatedPatches()
    typing.cast("EagerCallableProtocol", op4.infer_shape)(lhs)
    op4.infer_shape()

    # ConvWithGeneralPadding
    op5 = ConvWithGeneralPadding()
    typing.cast("EagerCallableProtocol", op5.infer_shape)(lhs)
    op5.infer_shape()

    # ConvTransposeShapeTuple
    op6 = ConvTransposeShapeTuple()
    op6.infer_shape()

    # ConvTranspose
    op7 = ConvTranspose()
    lhs = DummyTensor((1, 4, 4, 3))
    rhs = DummyTensor((3, 3, 3, 2))
    assert typing.cast("EagerCallableProtocol", op7.infer_shape)(lhs, rhs, strides=2, padding="VALID") == (1, 9, 9, 2)
    assert typing.cast("EagerCallableProtocol", op7.infer_shape)(lhs, rhs, strides=2, padding="SAME") == (1, 8, 8, 2)
    assert typing.cast("EagerCallableProtocol", op7.infer_shape)(lhs, rhs, strides=(2, 2), padding="VALID") == (1, 9, 9, 2)


def test_conv_exact_shape_deduction() -> None:
    """Verify exact output shape calculation for ConvGeneralDilated and Convolve."""
    # 1. Test 2D ConvGeneralDilated (NCHW lhs, OIHW rhs)
    op = ConvGeneralDilated()
    lhs = DummyTensor((2, 3, 32, 32))
    rhs = DummyTensor((16, 3, 3, 3))
    cfg = ConvConfig(window_strides=[2, 2], padding=[(1, 1), (1, 1)])
    # Spatial formula: (32 + 2 - 3) // 2 + 1 = 31 // 2 + 1 = 16
    assert typing.cast("EagerCallableProtocol", op.infer_shape)(lhs, rhs, cfg) == (2, 16, 16, 16)

    # 2. Test 1D Convolve modes
    conv1d = Convolve()
    a = DummyTensor((10,))
    v = DummyTensor((4,))
    assert typing.cast("EagerCallableProtocol", conv1d.infer_shape)(a, v, mode="full") == (13,)
    assert typing.cast("EagerCallableProtocol", conv1d.infer_shape)(a, v, mode="same") == (10,)
    assert typing.cast("EagerCallableProtocol", conv1d.infer_shape)(a, v, mode="valid") == (7,)
    assert typing.cast("EagerCallableProtocol", conv1d.infer_shape)(a, v, mode="unknown_default") == (13,)
    assert typing.cast("EagerCallableProtocol", conv1d.infer_shape)(None, v) == ()
    assert typing.cast("EagerCallableProtocol", conv1d.infer_shape)(a, None) == ()
