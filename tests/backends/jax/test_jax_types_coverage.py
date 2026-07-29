import pytest

from ml_switcheroo_compiler.backends.jax.types import array, asarray, item, zeros


def test_jax_types():
    try:
        import jax.numpy as jnp
    except ImportError:
        pytest.skip("JAX not available")

    res_zeros = zeros(None, (2, 2))
    assert res_zeros.shape == (2, 2)

    res_array = array(None, [1, 2], "float32")
    assert res_array.shape == (2,)

    res_asarray = asarray(None, [3, 4])
    assert res_asarray.shape == (2,)

    res_item = item(None, jnp.array(5))
    assert res_item == 5
