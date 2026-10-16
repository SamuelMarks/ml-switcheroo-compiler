"""Tests for test_jax_types."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from ml_switcheroo_compiler.backends.jax.types import array, asarray, item, zeros

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_jax_types() -> None:
    """Test function test_jax_types."""
    try:
        import jax.numpy as jnp
    except ImportError:
        pytest.skip("JAX not available")

    res_zeros = cast("Callable[..., EagerTensorProtocol]", zeros)(None, (2, 2))
    assert cast("EagerTensorProtocol", getattr(res_zeros, "shape")) == (2, 2)

    res_array = cast("Callable[..., EagerTensorProtocol]", array)(None, [1, 2], "float32")
    assert cast("EagerTensorProtocol", getattr(res_array, "shape")) == (2,)

    res_asarray = cast("Callable[..., EagerTensorProtocol]", asarray)(None, [3, 4])
    assert cast("EagerTensorProtocol", getattr(res_asarray, "shape")) == (2,)

    res_item = cast("Callable[..., EagerTensorProtocol]", item)(None, cast("Callable[..., EagerTensorProtocol]", jnp.array)(5))
    assert res_item == 5
