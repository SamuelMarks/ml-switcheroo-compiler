"""Tests for linalg operations."""

import numpy as np
from ml_switcheroo.ops.linalg.basic import (
    Matmul,
    Dot,
    Einsum,
)


def test_matmul_op() -> None:
    """Docstring."""
    op = Matmul()
    a = np.random.randn(2, 3)
    b = np.random.randn(3, 4)

    assert op.infer_shape(a.shape, b.shape) == (2, 4)
    assert op.infer_shape(None, None) is None

    res = op.numpy_eval(a, b)
    assert np.allclose(res, np.matmul(a, b))

    assert op.emit_jax("a", "b") == "jnp.matmul(a, b)"
    assert op.emit_pytorch("a", "b") == "torch.matmul(a, b)"
    assert op.emit_mlx("a", "b") == "mx.matmul(a, b)"
    assert op.emit_keras("a", "b") == "keras.ops.matmul(a, b)"
    assert op.emit_tensorflow("a", "b") == "tf.linalg.matmul(a, b)"

    vjp_out = op.vjp("dz", "a", "b")
    assert len(vjp_out) == 2


def test_dot_op() -> None:
    """Docstring."""
    op = Dot()
    a = np.random.randn(3)
    b = np.random.randn(3)

    assert op.infer_shape(a.shape, b.shape) is None

    res = op.numpy_eval(a, b)
    assert np.allclose(res, np.dot(a, b))

    assert op.emit_jax("a", "b") == "jnp.dot(a, b)"
    assert op.emit_tensorflow("a", "b") == "tf.tensordot(a, b, axes=1)"

    vjp_out = op.vjp("dz", "a", "b")
    assert len(vjp_out) == 2


def test_einsum_op() -> None:
    """Docstring."""
    op = Einsum()
    a = np.random.randn(2, 3)
    b = np.random.randn(3, 4)
    subscripts = "ij,jk->ik"

    assert op.infer_shape(subscripts, a.shape, b.shape) is None

    res = op.numpy_eval(subscripts, a, b)
    assert np.allclose(res, np.einsum(subscripts, a, b))

    assert op.emit_jax("subs", "a", "b") == "jnp.einsum(subs, a, b)"

    vjp_out = op.vjp("dz", subscripts, "a", "b")
    assert len(vjp_out) == 2
