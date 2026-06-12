"""Tests for binary operations."""

import numpy as np
from ml_switcheroo.ops.binary.math import (
    BinaryMathOp,
    Add,
    Subtract,
    Multiply,
    Divide,
    TrueDivide,
    Power,
    Maximum,
    Minimum,
)


def test_binary_math_ops() -> None:
    """Docstring."""
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

        # Check numpy_eval
        assert np.allclose(op.numpy_eval(x, y), np_func(x, y))

        # Check emitters for basic BinaryMathOp behavior
        if op.op_name != "True_Divide":
            assert op.emit_jax("x", "y") == f"jnp.{op.op_name.lower()}(x, y)"
        else:
            assert op.emit_jax("x", "y") == "jnp.true_divide(x, y)"
            assert op.emit_tensorflow("x", "y") == "tf.math.truediv(x, y)"
            assert op.emit_mlx("x", "y") == "mx.divide(x, y)"

        # Check VJP/JVP types
        vjp_out = op.vjp("dz", "x", "y")
        jvp_out = op.jvp("dx", "dy", "x", "y")

        assert isinstance(vjp_out, tuple)
        assert len(vjp_out) == 2
        assert isinstance(jvp_out, str)


def test_binary_base_op() -> None:
    """Docstring."""

    class DummyBinary(BinaryMathOp):
        """Docstring."""

        op_name = "Add"

        def vjp(
            self, cotangent: object, x: object, y: object, **kwargs: object
        ) -> object:
            """Docstring."""
            pass

        def jvp(
            self,
            tangent_x: object,
            tangent_y: object,
            x: object,
            y: object,
            **kwargs: object,
        ) -> object:
            """Docstring."""
            pass

    op = DummyBinary()
    assert op.infer_shape((10,), (10,)) == (10,)
    assert op.infer_shape(None, None) is None
    assert np.allclose(op.numpy_eval(1.0, 2.0), 3.0)
    assert op.emit_jax("x", "y") == "jnp.add(x, y)"
