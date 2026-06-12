"""Tests for creation operations."""

import numpy as np
from ml_switcheroo.ops.creation.basic import (
    Zeros,
    Ones,
    Full,
    Arange,
)


def test_creation_ops() -> None:
    """Docstring."""
    shape = (2, 3)

    ops = [
        (Zeros(), np.zeros),
        (Ones(), np.ones),
    ]

    for op, np_func in ops:
        assert op.infer_shape(shape) == shape
        assert np.array_equal(op.numpy_eval(shape), np_func(shape))
        assert op.emit_jax("shape") == f"jnp.{op.op_name.lower()}(shape)"
        assert op.emit_pytorch("shape") == f"torch.{op.op_name.lower()}(shape)"
        assert op.emit_mlx("shape") == f"mx.{op.op_name.lower()}(shape)"
        assert op.emit_keras("shape") == f"keras.ops.{op.op_name.lower()}(shape)"
        assert op.emit_tensorflow("shape") == f"tf.{op.op_name.lower()}(shape)"
        assert op.vjp("dz", "shape") == ()
        assert op.jvp("dx", "shape") == "0"


def test_full_op() -> None:
    """Docstring."""
    op = Full()
    shape = (2, 2)
    val = 5.0
    assert op.infer_shape(shape, val) == shape
    assert np.array_equal(op.numpy_eval(shape, val), np.full(shape, val))
    assert op.emit_jax("shape", "val") == "jnp.full(shape, val)"
    assert op.emit_tensorflow("shape", "val") == "tf.fill(shape, val)"


def test_arange_op() -> None:
    """Docstring."""
    op = Arange()
    assert op.infer_shape(10) is None
    assert np.array_equal(op.numpy_eval(5), np.arange(5))
    assert op.emit_jax("start", "stop") == "jnp.arange(start, stop)"
    assert op.emit_tensorflow("5") == "tf.range(5)"
