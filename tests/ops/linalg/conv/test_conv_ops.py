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
from tests.testing_utils import ensure_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


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

    ensure_callable(op.infer_shape)(lhs, rhs, config)
    ensure_callable(op.infer_shape)(lhs=lhs, rhs=rhs, config=config)
    ensure_callable(op.infer_shape)(lhs, rhs)
    ensure_callable(op.infer_shape)(lhs, rhs=rhs, config=config)

    # Missing shapes
    class NoShape:
        """Test class NoShape."""

        pass

    ensure_callable(op.infer_shape)(ensure_callable(NoShape)(), ensure_callable(NoShape)())

    # Convolve
    op2 = Convolve()
    ensure_callable(op2.infer_shape)(lhs, rhs)

    # ConvGeneralDilatedLocal
    op3 = ConvGeneralDilatedLocal()
    ensure_callable(op3.infer_shape)(lhs)
    op3.infer_shape()

    # ConvGeneralDilatedPatches
    op4 = ConvGeneralDilatedPatches()
    ensure_callable(op4.infer_shape)(lhs)
    op4.infer_shape()

    # ConvWithGeneralPadding
    op5 = ConvWithGeneralPadding()
    ensure_callable(op5.infer_shape)(lhs)
    op5.infer_shape()

    # ConvTransposeShapeTuple
    op6 = ConvTransposeShapeTuple()
    op6.infer_shape()

    # ConvTranspose
    op7 = ConvTranspose()
    lhs = DummyTensor((1, 4, 4, 3))
    rhs = DummyTensor((3, 3, 3, 2))
    assert ensure_callable(op7.infer_shape)(lhs, rhs, strides=2, padding="VALID") == (1, 9, 9, 2)
    assert ensure_callable(op7.infer_shape)(lhs, rhs, strides=2, padding="SAME") == (1, 8, 8, 2)
    assert ensure_callable(op7.infer_shape)(lhs, rhs, strides=(2, 2), padding="VALID") == (1, 9, 9, 2)


def test_conv_exact_shape_deduction() -> None:
    """Verify exact output shape calculation for ConvGeneralDilated and Convolve."""
    # 1. Test 2D ConvGeneralDilated (NCHW lhs, OIHW rhs)
    op = ConvGeneralDilated()
    lhs = DummyTensor((2, 3, 32, 32))
    rhs = DummyTensor((16, 3, 3, 3))
    cfg = ConvConfig(window_strides=[2, 2], padding=[(1, 1), (1, 1)])
    # Spatial formula: (32 + 2 - 3) // 2 + 1 = 31 // 2 + 1 = 16
    assert ensure_callable(op.infer_shape)(lhs, rhs, cfg) == (2, 16, 16, 16)

    # 2. Test 1D Convolve modes
    conv1d = Convolve()
    a = DummyTensor((10,))
    v = DummyTensor((4,))
    assert ensure_callable(conv1d.infer_shape)(a, v, mode="full") == (13,)
    assert ensure_callable(conv1d.infer_shape)(a, v, mode="same") == (10,)
    assert ensure_callable(conv1d.infer_shape)(a, v, mode="valid") == (7,)
    assert ensure_callable(conv1d.infer_shape)(a, v, mode="unknown_default") == (13,)
    assert ensure_callable(conv1d.infer_shape)(None, v) == ()
    assert ensure_callable(conv1d.infer_shape)(a, None) == ()
