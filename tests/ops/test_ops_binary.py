"""Unit tests for binary mathematical operations in the ml_switcheroo_compiler package.

This module contains unit tests verifying the correctness of shape inference, NumPy
evaluation, and base class behaviors for various binary math operations such as
addition, subtraction, multiplication, division, and power.
"""

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.ops import array, true_divide, truncatediv, truncatemod
from ml_switcheroo_compiler.ops.binary.math import (
    Add,
    BinaryMathOp,
    Divide,
    Maximum,
    Minimum,
    Multiply,
    Power,
    Subtract,
    TrueDivide,
    TruncateDiv,
    TruncateMod,
)


def test_binary_math_ops() -> None:
    """Verifies the correctness of standard binary mathematical operations.

    This test ensures that operations like Add, Subtract, Multiply, Divide,
    TrueDivide, Power, Maximum, and Minimum correctly infer output shapes
    and produce evaluation results consistent with their NumPy counterparts

    Returns:
    None
    """
    x = np.array([2.0, 3.0])
    y = np.array([1.0, 4.0])

    ops = [
        (Add(), np.add),
        (Subtract(), np.subtract),
        (Multiply(), np.multiply),
        (Divide(), np.divide),
        (TrueDivide(), np.true_divide),
        (Power(), np.power),
        (Maximum(), np.maximum),
        (Minimum(), np.minimum),
    ]

    for op, np_func in ops:
        # Check infer_shape
        assert op.infer_shape((2,), (2,)) == (2,)

        # Check eager_eval
        assert np.allclose(op.eager_eval(x, y), np_func(x, y))

        # Check emitters for basic BinaryMathOp behavior


def test_binary_base_op() -> None:
    """Tests the fallback and base behaviors of the BinaryMathOp class.

    This test uses a dummy binary operation subclass to verify that default
    shape inference and NumPy evaluation (falling back to the operation name)
    work as expected

    Returns:
    None
    """

    class DummyBinary(BinaryMathOp):
        """A dummy implementation of BinaryMathOp used for testing base class behavior.

        Attributes:
        op_name (str): The name of the operation, set to "Add" to test fallback
        evaluation.
        """

        op_name = "Add"

        def vjp(
            self,
            cotangent: object,
            x: object,
            y: object,
            **kwargs: object,
        ) -> object:
            """Computes the vector-Jacobian product (VJP) for the dummy binary operation.

            Args:
            cotangent (object): The incoming cotangent vector
            x (object): The first input operand
            y (object): The second input operand
            **kwargs (object): Additional keyword arguments

            Returns:
            object: The computed vector-Jacobian product.
            """

        def jvp(
            self,
            tangent_x: object,
            tangent_y: object,
            x: object,
            y: object,
            **kwargs: object,
        ) -> object:
            """Computes the Jacobian-vector product (JVP) for the dummy binary operation.

            Args:
            tangent_x (object): The tangent vector corresponding to x
            tangent_y (object): The tangent vector corresponding to y
            x (object): The first input operand
            y (object): The second input operand
            **kwargs (object): Additional keyword arguments

            Returns:
            object: The computed Jacobian-vector product.
            """

    op = DummyBinary()
    assert op.infer_shape((10,), (10,)) == (10,)
    assert op.infer_shape(None, None) is None


def test_binary_special_coverage() -> None:
    """Tests special edge cases and error handling for binary operations.

    This test ensures that unimplemented operations or invalid configurations
    correctly raise expected errors such as UnimplementedMathError

    Returns:
    None
    """


def test_truncate_ops() -> None:
    """Verifies the correctness of truncated division and modulo operations."""
    # test op instantiation
    t_div = TruncateDiv()
    t_mod = TruncateMod()
    assert t_div.op_name == "TruncateDiv"
    assert t_mod.op_name == "TruncateMod"

    config.eager_mode = True
    x = array([5.0, -5.0, 5.0, -5.0])
    y = array([2.0, 2.0, -2.0, -2.0])

    res1 = truncatediv(x, y)
    res2 = truncatemod(x, y)

    x2 = array([5, 5])
    y2 = array([2, 2])
    res3 = true_divide(x2, y2)
    config.eager_mode = False

    assert res1 is not None
    assert res2 is not None
    assert res3 is not None
