"""Module docstring."""

import numpy as np

import ml_switcheroo_compiler.backends.eager_registry as reg
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.random.state import rng_bit_generator, rng_uniform


def test_random_state_eager() -> object:
    """Function docstring."""
    # Register stubs
    reg.numpy_eager_registry.register("RngBitGenerator")(lambda m, key, **kw: np.zeros(kw.get("shape")))
    reg.numpy_eager_registry.register("RngUniform")(lambda m, a, b, **kw: np.zeros(kw.get("shape")))

    with ConfigContext(eager_mode=True, backend="numpy"):
        r1 = rng_bit_generator(None, (2, 2), "uint32")
        r2 = rng_uniform(0, 1, (2, 2), "float32")

        assert r1.shape == (2, 2)
        assert r2.shape == (2, 2)
