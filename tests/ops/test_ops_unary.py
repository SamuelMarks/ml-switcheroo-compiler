"""Tests for unary operations."""

import numpy as np
from ml_switcheroo.ops.unary.math import (
    UnaryMathOp,
    Sin,
    Cos,
    Exp,
    Log,
    Sqrt,
    Square,
    Abs,
    Negative,
    Positive,
    Sign,
    Floor,
    Ceil,
    Round,
)


def test_unary_math_ops() -> None:
    """Docstring."""
    x = 2.0
    x_arr = np.array([1.0, 2.0])

    ops = [
        (Sin(), np.sin),
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

        # Check numpy_eval
        if op.op_name == "Round":
            assert np.allclose(op.numpy_eval(x_arr), np.round(x_arr))
        else:
            assert np.allclose(op.numpy_eval(x_arr), np_func(x_arr))

        # Check emitters
        assert op.emit_jax("x") == f"jnp.{op.op_name.lower()}(x)"
        assert op.emit_pytorch("x") == f"torch.{op.op_name.lower()}(x)"
        assert op.emit_mlx("x") == f"mx.{op.op_name.lower()}(x)"
        assert op.emit_keras("x") == f"keras.ops.{op.op_name.lower()}(x)"
        assert op.emit_tensorflow("x") == f"tf.math.{op.op_name.lower()}(x)"

        # Check VJP/JVP (just string formatting for now based on the implementation)
        vjp_out = op.vjp("dy", "x")
        jvp_out = op.jvp("dx", "x")

        assert isinstance(vjp_out, tuple)
        assert isinstance(jvp_out, str)


def test_unary_base_op() -> None:
    """Docstring."""

    class DummyUnary(UnaryMathOp):
        """Docstring."""

        op_name = "Sin"

        def vjp(self, cotangent: object, x: object, **kwargs: object) -> object:
            """Docstring."""
            pass

        def jvp(self, tangent: object, x: object, **kwargs: object) -> object:
            """Docstring."""
            pass

    op = DummyUnary()
    assert op.infer_shape((10,)) == (10,)
    assert np.allclose(op.numpy_eval(0.0), 0.0)
    assert op.emit_jax("x") == "jnp.sin(x)"
