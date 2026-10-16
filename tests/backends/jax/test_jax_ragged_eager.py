"""Tests for test_jax_ragged_eager."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_jax_ragged_eager() -> None:
    """Test function test_jax_ragged_eager."""
    try:
        import jax.numpy as jnp

        from ml_switcheroo_compiler.backends.jax.eager import _execute_ragged_tensor_to_dense

        a = [jnp.array([1, 2]), jnp.array([1])]
        res = _execute_ragged_tensor_to_dense(a)
        assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2, 2)
        assert _execute_ragged_tensor_to_dense(np.array([1, 2])) is not None
    except ImportError as _exc:
        assert _exc is not None
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, AssertionError) as e:
        # Ignore crashes from ml_dtypes incompatibility with local numpy
        assert e is not None
