"""Tests for test_jax_types."""

from __future__ import annotations

import typing

import pytest

from ml_switcheroo_compiler.backends.jax.types import array, asarray, item, zeros

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol


def test_jax_types() -> None:
    """Test function test_jax_types."""
    try:
        import jax.numpy as jnp
    except ImportError:
        pytest.skip("JAX not available")

    res_zeros = typing.cast("EagerCallableProtocol", zeros)(None, (2, 2))
    assert typing.cast("EagerTensorProtocol", getattr(res_zeros, "shape")) == (2, 2)

    res_array = typing.cast("EagerCallableProtocol", array)(None, [1, 2], "float32")
    assert typing.cast("EagerTensorProtocol", getattr(res_array, "shape")) == (2,)

    res_asarray = typing.cast("EagerCallableProtocol", asarray)(None, [3, 4])
    assert typing.cast("EagerTensorProtocol", getattr(res_asarray, "shape")) == (2,)

    res_item = typing.cast("EagerCallableProtocol", item)(None, typing.cast("EagerCallableProtocol", jnp.array)(5))
    assert res_item == 5
