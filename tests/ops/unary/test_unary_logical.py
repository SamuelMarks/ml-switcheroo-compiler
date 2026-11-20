"""Tests for ops unary logical coverage."""

from __future__ import annotations

from ml_switcheroo_ir import DType

import ml_switcheroo_compiler.ops.unary.logical as unary_logical
from ml_switcheroo_compiler.ops.unary.base import UnaryMathOp
from tests.testing_utils import MockNodeStub


def test_unary_logical_functions() -> None:
    """Test unary logical OpDef shape inference and execution."""

    class DummyNode(MockNodeStub):
        """Test class DummyNode."""

        def __init__(self) -> None:
            """Initialize dummy node with shape and dtype."""
            super().__init__()
            self.shape = (10, 20)
            self.dtype = DType.float32

    a = DummyNode()
    unary_math_ops: tuple[UnaryMathOp, ...] = (
        unary_logical.BitwiseNot(),
        unary_logical.Isfinite(),
        unary_logical.Isinf(),
        unary_logical.Isnan(),
        unary_logical.Isneginf(),
        unary_logical.Isposinf(),
        unary_logical.LogicalNot(),
        unary_logical.BitwiseCount(),
        unary_logical.IsNonDecreasing(),
        unary_logical.IsStrictlyIncreasing(),
        unary_logical.Clz(),
        unary_logical.PopulationCount(),
        unary_logical.BitcastConvertType(),
        unary_logical.ReducePrecision(),
        unary_logical.Iscomplex(),
        unary_logical.Isreal(),
    )
    evaluated = 0
    for op in unary_math_ops:
        assert op.infer_shape(a) == (10, 20)
        evaluated += 1

    assert unary_logical.Packbits().infer_shape(a) == (10, 20)
    assert unary_logical.Unpackbits().infer_shape(a) == (10, 20)
    assert unary_logical.Iscomplexobj().infer_shape(a) == ()
    assert unary_logical.Isrealobj().infer_shape(a) == ()
    assert unary_logical.Issubdtype().infer_shape(a) == ()
    assert unary_logical.Isin().infer_shape(a) == (10, 20)
    assert unary_logical.Ediff1d().infer_shape(a) == (199,)
    evaluated += 7
    assert evaluated >= 4
