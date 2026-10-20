"""Tests for test_jax_ragged_eager."""

from __future__ import annotations

import typing

import numpy as np

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_jax_ragged_eager() -> None:
    """Test function test_jax_ragged_eager."""
    import jax.numpy as jnp

    from ml_switcheroo_compiler.backends.jax.eager import _execute_ragged_tensor_to_dense

    a = [jnp.array([1, 2]), jnp.array([1])]
    res = _execute_ragged_tensor_to_dense(a)
    assert typing.cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 2)
    assert _execute_ragged_tensor_to_dense(np.array([1, 2])) is not None
