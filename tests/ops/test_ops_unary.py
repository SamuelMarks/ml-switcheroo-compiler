"""Unit tests for unary mathematical operations in the ml_switcheroo_compiler library.

This module contains tests verifying the correctness of shape inference, NumPy
evaluation, and base class behaviors for various unary mathematical operations such as
Sin, Cos, Exp, etc.
"""

import numpy as np

from ml_switcheroo_compiler.ops.unary.math import (
    Abs,
    Ceil,
    Cos,
    Exp,
    Floor,
    Log,
    Negative,
    Positive,
    Round,
    Sign,
    Sin,
    Tan,
    Sqrt,
    Square,
    UnaryMathOp,
)


def test_unary_math_ops() -> None:
    """Verifies the correctness of various unary mathematical operations.

    This test iterates through a suite of unary operations (e.g., Sin, Cos, Exp) and
    validates their shape inference and NumPy evaluation against standard NumPy
    functions

    Returns:
    None
    """
    x = 2.0
    x_arr = np.array([1.0, 2.0])

    ops = [
        (Sin(), np.sin),
        (Tan(), np.tan),
        (Cos(), np.cos),
        (Exp(), np.exp),
        (Log(), np.log),
        (Sqrt(), np.sqrt),
        (Square(), np.square),
        (Abs(), np.abs),
        (Negative(), np.negative),
        (Positive(), np.positive),
        (Sign(), np.sign),
        (Floor(), np.floor),
        (Ceil(), np.ceil),
        (Round(), np.round),
    ]

    for op, np_func in ops:
        # Check infer_shape
        assert op.infer_shape(x) == x

        # Check eager_eval
        if op.op_name == "Round":
            assert np.allclose(op.eager_eval(x_arr), np.round(x_arr))
            assert np.allclose(op.eager_eval(x_arr), np_func(x_arr))

        # Check emitters


def test_unary_base_op() -> None:
    """Tests the base class functionality of unary mathematical operations.

    This test defines a dummy unary operation subclassing UnaryMathOp to verify
    default behaviors such as shape inference and NumPy evaluation

    Returns:
    None
    """

    class DummyUnary(UnaryMathOp):
        """A dummy implementation of UnaryMathOp for testing base class behaviors.

        This class overrides the abstract methods of UnaryMathOp to allow
        instantiation
        and testing of inherited methods like infer_shape and eager_eval.
        """

        op_name = "Sin"

        def vjp(self, cotangent: object, x: object, **kwargs: object) -> object:
            """Computes the vector-Jacobian product for the dummy operation.

            Args:
            cotangent (object): The cotangent vector
            x (object): The input operand
            **kwargs (object): Additional keyword arguments

            Returns:
            object: The computed vector-Jacobian product.
            """

        def jvp(self, tangent: object, x: object, **kwargs: object) -> object:
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


def test_unary_special_coverage() -> None:
    """Tests special edge cases and error handling for unary operations."""
    import numpy as np

    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.ops.unary.special import Bitcast, Cast, Frexp

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
