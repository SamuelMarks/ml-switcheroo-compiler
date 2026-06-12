"""Tests for reduction operations."""

import numpy as np
from ml_switcheroo.ops.reductions.basic import (
    Sum,
    Mean,
    Max,
    Min,
)


def test_reduction_ops() -> None:
    """Docstring."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])

    ops = [
        (Sum(), np.sum),
        (Mean(), np.mean),
        (Max(), np.max),
        (Min(), np.min),
    ]

    for op, np_func in ops:
        assert op.infer_shape(x.shape) is None
        assert np.allclose(op.numpy_eval(x), np_func(x))
        assert np.allclose(op.numpy_eval(x, axis=0), np_func(x, axis=0))
        assert np.allclose(
            op.numpy_eval(x, axis=1, keepdims=True), np_func(x, axis=1, keepdims=True)
        )

        assert op.emit_jax("x") == f"jnp.{op.op_name.lower()}(x)"
        assert op.emit_jax("x", axis=0) == f"jnp.{op.op_name.lower()}(x, axis=0)"
        assert (
            op.emit_pytorch("x", axis=0, keepdims=True)
            == f"torch.{op.op_name.lower()}(x, dim=0, keepdim=True)"
        )
        assert (
            op.emit_tensorflow("x", axis=0)
            == f"tf.reduce_{op.op_name.lower()}(x, axis=0)"
        )
        assert op.emit_keras("x") == f"keras.ops.{op.op_name.lower()}(x)"
        assert op.emit_mlx("x", axis=0) == f"mx.{op.op_name.lower()}(x, axis=0)"

        vjp_out = op.vjp("dz", "x")
        assert len(vjp_out) == 1
        assert isinstance(op.jvp("dx", "x"), str)
