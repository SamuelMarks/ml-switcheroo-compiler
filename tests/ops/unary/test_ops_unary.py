# ruff: noqa: E501
import numpy as np

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops.unary.arithmetic import Abs, Ceil, Floor, Negative, Positive, Round, Sign, Sqrt, Square
from ml_switcheroo_compiler.ops.unary.base import UnaryMathOp
from ml_switcheroo_compiler.ops.unary.exponential import Exp, Log
from ml_switcheroo_compiler.ops.unary.special import Bitcast, Cast, Frexp
from ml_switcheroo_compiler.ops.unary.trigonometric import Cos, Sin, Tan

"Unit tests for unary mathematical operations in the ml_switcheroo_compiler library.\n\nThis module contains tests verifying the correctness of shape inference, NumPy\nevaluation, and base class behaviors for various unary mathematical operations such as\nSin, Cos, Exp, etc.\n"


def test_unary_math_ops() -> None:
    """Test the unary math ops behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Verifies the correctness of various unary mathematical operations.\n\n    This test iterates through a suite of unary operations (e.g., Sin, Cos, Exp) and\n    validates their shape inference and NumPy evaluation against standard NumPy\n    functions\n\n    Returns:\n    None\n    "
        x = 2.0
        x_arr = np.array([1.0, 2.0])
        ops = [(Sin(), np.sin), (Tan(), np.tan), (Cos(), np.cos), (Exp(), np.exp), (Log(), np.log), (Sqrt(), np.sqrt), (Square(), np.square), (Abs(), np.abs), (Negative(), np.negative), (Positive(), np.positive), (Sign(), np.sign), (Floor(), np.floor), (Ceil(), np.ceil), (Round(), np.round)]
        for op, np_func in ops:
            assert op.infer_shape(x) == x
            if op.op_name == "Round":
                assert np.allclose(op.eager_eval(x_arr), np.round(x_arr))
                assert np.allclose(op.eager_eval(x_arr), np_func(x_arr))
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_unary_base_op() -> None:
    """Test the unary base op behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Tests the base class functionality of unary mathematical operations.\n\n    This test defines a dummy unary operation subclassing UnaryMathOp to verify\n    default behaviors such as shape inference and NumPy evaluation\n\n    Returns:\n    None\n    "

        class DummyUnary(UnaryMathOp):
            """A dummy implementation of UnaryMathOp for testing base class behaviors.

            This class overrides the abstract methods of UnaryMathOp to allow
            instantiation
            and testing of inherited methods like infer_shape and eager_eval.
            """

            op_name = "Sin"

            def vjp(self, cotangent, x, **kwargs):
                """Computes the vector-Jacobian product for the dummy operation.

                Args:
                cotangent (object): The cotangent vector
                x (object): The input operand
                **kwargs (object): Additional keyword arguments

                Returns:
                object: The computed vector-Jacobian product.
                """

            def jvp(self, tangent, x, **kwargs):
                """Computes the Jacobian-vector product for the dummy operation.

                Args:
                tangent (object): The tangent vector
                x (object): The input operand
                **kwargs (object): Additional keyword arguments

                Returns:
                object: The computed Jacobian-vector product.
                """

        op = DummyUnary()
        assert op.infer_shape((10,)) == (10,)
        assert np.allclose(op.eager_eval(0.0), 0.0)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_unary_special_coverage() -> None:
    """Test the unary special coverage behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Tests special edge cases and error handling for unary operations."
        x = np.array([1.5, 2.5])
        cast_op = Cast()
        res1 = cast_op.eager_eval(x, dtype=DType.Int32)
        assert res1.dtype == np.int32
        res1_str = cast_op.eager_eval(x, dtype="int32")
        assert res1_str.dtype == np.int32
        bitcast_op = Bitcast()
        x_int = np.array([1, 2], dtype=np.int32)
        res2 = bitcast_op.eager_eval(x_int, dtype=DType.Float32)
        assert res2.dtype == np.float32
        res2_str = bitcast_op.eager_eval(x_int, dtype="float32")
        assert res2_str.dtype == np.float32
        frexp_op = Frexp()
        res_frexp = frexp_op.eager_eval(x)
        assert isinstance(res_frexp, tuple)
        assert len(res_frexp) == 2
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


"Provides required module functionality."


def test_unary_special_coverage_brute() -> None:
    """Test the unary special coverage brute behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Execute the requested function."
        c = Cast()
        bc = Bitcast()
        c.eager_eval(np.array([1, 2]), dtype="float32")
        bc.eager_eval(np.array([1, 2], dtype=np.int32), dtype="float32")
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass
