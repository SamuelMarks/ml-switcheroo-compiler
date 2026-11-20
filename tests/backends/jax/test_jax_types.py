"""Tests for test_jax_types."""

from __future__ import annotations

import typing

import pytest

from ml_switcheroo_compiler.backends.jax.types import array, asarray, item, zeros
from tests.testing_utils import extract_test_shape

if typing.TYPE_CHECKING:
    pass


def test_jax_types() -> None:
    """Test function test_jax_types."""
    try:
        import jax.numpy as jnp
    except ImportError:
        pytest.skip("JAX not available")

    res_zeros = zeros(None, (2, 2))
    assert extract_test_shape(res_zeros) == (2, 2)

    res_array = array(None, [1, 2], "float32")
    assert extract_test_shape(res_array) == (2,)

    res_asarray = asarray(None, [3, 4])
    assert extract_test_shape(res_asarray) == (2,)

    res_item = item(None, jnp.array(5))
    assert res_item == 5
