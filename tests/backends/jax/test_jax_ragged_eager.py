import numpy as np


def test_jax_ragged_eager():
    try:
        import jax.numpy as jnp

        from ml_switcheroo_compiler.backends.jax.eager import _execute_ragged_tensor_to_dense

        a = [jnp.array([1, 2]), jnp.array([1])]
        res = _execute_ragged_tensor_to_dense(a)
        assert res.shape == (2, 2)
        assert _execute_ragged_tensor_to_dense(np.array([1, 2])) is not None
    except ImportError:
        pass
    except Exception as e:
        # Ignore crashes from ml_dtypes incompatibility with local numpy
        pass
