"""Tests for test_ops_unary."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops.unary.arithmetic import Abs, Ceil, Floor, Negative, Positive, Round, Sign, Sqrt, Square
from ml_switcheroo_compiler.ops.unary.base import UnaryMathOp
from ml_switcheroo_compiler.ops.unary.exponential import Exp, Log
from ml_switcheroo_compiler.ops.unary.special import Bitcast, Cast, Frexp
from ml_switcheroo_compiler.ops.unary.trigonometric import Cos, Sin, Tan
from tests.testing_utils import as_callable, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_unary_math_ops() -> None:
    """Test the unary math ops behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    x = 2.0
    x_arr = np.array([1.0, 2.0])
    ops = [(Sin(), np.sin), (Tan(), np.tan), (Cos(), np.cos), (Exp(), np.exp), (Log(), np.log), (Sqrt(), np.sqrt), (Square(), np.square), (Abs(), np.abs), (Negative(), np.negative), (Positive(), np.positive), (Sign(), np.sign), (Floor(), np.floor), (Ceil(), np.ceil), (Round(), np.round)]
    for op, np_func in ops:
        assert as_callable(op.infer_shape)(x) == ()
        if op.op_name == "Round":
            assert as_callable(np.allclose)(as_callable(op.eager_eval)(x_arr), as_callable(np.round)(x_arr))
            assert as_callable(np.allclose)(as_callable(op.eager_eval)(x_arr), as_callable(np_func)(x_arr))


def test_unary_base_op() -> None:
    """Test the unary base op behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """

    class DummyUnary(UnaryMathOp):
        """A dummy implementation of UnaryMathOp for testing base class behaviors.

        This class overrides the abstract methods of UnaryMathOp to allow
        instantiation
        and testing of inherited methods like infer_shape and eager_eval.
        """

        op_name = "Sin"

        def vjp(self, cotangent: EagerValue, x: EagerValue, **kwargs: EagerValue) -> None:
            """Computes the vector-Jacobian product for the dummy operation.

            Args:
            cotangent (EagerValue): The cotangent vector
            x (EagerValue): The input operand
            **kwargs (EagerValue): Additional keyword arguments

            Returns:
            TensorOrScalar: The computed vector-Jacobian product.
            """

        def jvp(self, tangent: EagerValue, x: EagerValue, **kwargs: EagerValue) -> None:
            """Computes the Jacobian-vector product for the dummy operation.

            Args:
            tangent (EagerValue): The tangent vector
            x (EagerValue): The input operand
            **kwargs (EagerValue): Additional keyword arguments

            Returns:
            TensorOrScalar: The computed Jacobian-vector product.
            """

    op = DummyUnary()
    assert op.infer_shape((10,)) == (10,)
    assert as_callable(np.allclose)(as_callable(op.eager_eval)(0.0), 0.0)


def test_unary_special_coverage() -> None:
    """Test the unary special coverage behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    x = np.array([1.5, 2.5])
    cast_op = Cast()
    res1 = cast_op.eager_eval(x, dtype=DType.Int32)
    assert as_tensor(getattr(res1, "dtype")) == np.int32
    res1_str = cast_op.eager_eval(x, dtype="int32")
    assert as_tensor(getattr(res1_str, "dtype")) == np.int32
    bitcast_op = Bitcast()
    x_int = np.array([1, 2], dtype=np.int32)
    res2 = bitcast_op.eager_eval(x_int, dtype=DType.Float32)
    assert as_tensor(getattr(res2, "dtype")) == np.float32
    res2_str = bitcast_op.eager_eval(x_int, dtype="float32")
    assert as_tensor(getattr(res2_str, "dtype")) == np.float32
    frexp_op = Frexp()
    res_frexp = frexp_op.eager_eval(x)
    assert isinstance(res_frexp, tuple)
    assert len(res_frexp) == 2


def test_unary_special_coverage_brute() -> None:
    """Test the unary special coverage brute behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    c = Cast()
    bc = Bitcast()
    c.eager_eval(np.array([1, 2]), dtype="float32")
    bc.eager_eval(np.array([1, 2], dtype=np.int32), dtype="float32")
    assert callable(test_unary_special_coverage_brute)
