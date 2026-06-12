"""Tests for shape operations."""

import numpy as np
from ml_switcheroo.ops.shape.basic import (
    Reshape,
    Transpose,
    BroadcastTo,
)


def test_reshape_op() -> None:
    """Docstring."""
    op = Reshape()
    x = np.array([1, 2, 3, 4])
    newshape = (2, 2)

    assert op.infer_shape(x.shape, newshape) == newshape
    assert np.array_equal(op.numpy_eval(x, newshape), np.reshape(x, newshape))
    assert op.emit_jax("x", "shape") == "jnp.reshape(x, shape)"
    assert op.emit_pytorch("x", "shape") == "torch.reshape(x, shape)"
    assert op.emit_mlx("x", "shape") == "mx.reshape(x, shape)"
    assert op.emit_keras("x", "shape") == "keras.ops.reshape(x, shape)"
    assert op.emit_tensorflow("x", "shape") == "tf.reshape(x, shape)"

    assert len(op.vjp("dz", "x", "shape")) == 1
    assert isinstance(op.jvp("dx", "x", "shape"), str)


def test_transpose_op() -> None:
    """Docstring."""
    op = Transpose()
    x = np.random.randn(2, 3)

    assert op.infer_shape(x.shape) is None
    assert op.infer_shape(x.shape, (1, 0)) == (3, 2)
    assert np.array_equal(op.numpy_eval(x), np.transpose(x))
    assert np.array_equal(op.numpy_eval(x, axes=(1, 0)), np.transpose(x, axes=(1, 0)))

    assert op.emit_jax("x") == "jnp.transpose(x)"
    assert op.emit_jax("x", "(1, 0)") == "jnp.transpose(x, (1, 0))"
    assert op.emit_pytorch("x") == "x.t()"
    assert op.emit_pytorch("x", "(1, 0)") == "torch.permute(x, (1, 0))"
    assert op.emit_tensorflow("x") == "tf.transpose(x)"
    assert op.emit_tensorflow("x", "(1, 0)") == "tf.transpose(x, perm=(1, 0))"
    assert op.emit_mlx("x") == "mx.transpose(x)"
    assert op.emit_keras("x") == "keras.ops.transpose(x)"

    assert len(op.vjp("dz", "x")) == 1
    assert len(op.vjp("dz", "x", "(1, 0)")) == 1
    assert isinstance(op.jvp("dx", "x"), str)
    assert isinstance(op.jvp("dx", "x", "(1, 0)"), str)


def test_broadcast_to_op() -> None:
    """Docstring."""
    op = BroadcastTo()
    x = np.array([1, 2])
    shape = (2, 2)

    assert op.infer_shape(x.shape, shape) == shape
    assert np.array_equal(op.numpy_eval(x, shape), np.broadcast_to(x, shape))

    assert op.emit_jax("x", "shape") == "jnp.broadcast_to(x, shape)"
    assert op.emit_pytorch("x", "shape") == "x.expand(shape)"
    assert op.emit_mlx("x", "shape") == "mx.broadcast_to(x, shape)"
    assert op.emit_keras("x", "shape") == "keras.ops.broadcast_to(x, shape)"
    assert op.emit_tensorflow("x", "shape") == "tf.broadcast_to(x, shape)"

    assert len(op.vjp("dz", "x", "shape")) == 1
    assert isinstance(op.jvp("dx", "x", "shape"), str)
