"""Test raw operations shape inference."""

from __future__ import annotations

from ml_switcheroo_compiler.ir.shape_system import SymInt
from ml_switcheroo_compiler.ops.raw_ops import RawConv2D, RawMatMul, RawMerge, RawOp, RawSwitch


class MockTensor:
    """Mock tensor."""

    def __init__(self, shape):
        """Init mock tensor."""
        self.shape = shape


def test_raw_op_shape() -> None:
    """Test RawOp shape inference."""
    op = RawOp()
    t = MockTensor((2, 3))
    assert op.infer_shape(t) == (2, 3)
    assert op.infer_shape(a=t) == (2, 3)


def test_raw_switch_shape() -> None:
    """Test RawSwitch shape inference."""
    op = RawSwitch()
    t = MockTensor((2, 3))
    assert op.infer_shape(t) == ((2, 3), (2, 3))


def test_raw_merge_shape() -> None:
    """Test RawMerge shape inference."""
    op = RawMerge()
    t = MockTensor((2, 3))
    assert op.infer_shape(t) == ((2, 3), ())


def test_raw_conv2d_shape() -> None:
    """Test RawConv2D shape inference."""
    op = RawConv2D()
    inp = MockTensor((4, 32, 32, 3))
    filters = MockTensor((3, 3, 3, 16))

    # VALID padding, strides 1
    assert op.infer_shape(inp, filters, padding="VALID", strides=[1, 1, 1, 1]) == (4, 30, 30, 16)

    # SAME padding, strides 2
    assert op.infer_shape(inp, filters, padding="SAME", strides=[1, 2, 2, 1]) == (4, 16, 16, 16)

    # With symbolic dimension
    sym_n = SymInt("N")
    inp_sym = MockTensor((sym_n, 32, 32, 3))
    assert op.infer_shape(inp_sym, filters, padding="SAME")[0] is sym_n


def test_raw_matmul_shape() -> None:
    """Test RawMatMul shape inference."""
    op = RawMatMul()
    a = MockTensor((4, 5, 6))
    b = MockTensor((4, 6, 7))

    assert op.infer_shape(a, b) == (4, 5, 7)

    a_T = MockTensor((4, 6, 5))
    assert op.infer_shape(a_T, b, transpose_a=True) == (4, 5, 7)

    b_T = MockTensor((4, 7, 6))
    assert op.infer_shape(a, b_T, transpose_b=True) == (4, 5, 7)

    # Broadcasting
    a_b = MockTensor((2, 1, 5, 6))
    b_b = MockTensor((3, 6, 7))
    assert op.infer_shape(a_b, b_b) == (2, 3, 5, 7)
