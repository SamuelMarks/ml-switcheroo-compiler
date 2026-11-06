"""Tests for test_jax_types."""

from __future__ import annotations

import typing

import pytest

from ml_switcheroo_compiler.backends.jax.types import array, asarray, item, zeros
from tests.testing_utils import ensure_callable, ensure_eager_tensor

if typing.TYPE_CHECKING:
    pass


def test_jax_types() -> None:
    """Test function test_jax_types."""
    try:
        import jax.numpy as jnp
    except ImportError:
        pytest.skip("JAX not available")

    res_zeros = ensure_callable(zeros)(None, (2, 2))
    assert ensure_eager_tensor(res_zeros).shape == (2, 2)

    res_array = ensure_callable(array)(None, [1, 2], "float32")
    assert ensure_eager_tensor(res_array).shape == (2,)

    res_asarray = ensure_callable(asarray)(None, [3, 4])
    assert ensure_eager_tensor(res_asarray).shape == (2,)

    res_item = ensure_callable(item)(None, ensure_callable(jnp.array)(5))
    assert res_item == 5
