"""Module docstring."""

import math

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


def _gelu(x: object, *args: object, **kwargs: object) -> object:
    r"""Execute _gelu.\n\n    Args:\n        cls (Any): The class.\n        x (Any): Argument x.\n        *args (Any): Argument *args.\n        **kwargs (Any): Argument **kwargs.\n\n    Returns:\n    Any: The result.\n."""
    erf_vec = np.vectorize(math.erf)
    return (0.5 * x) * (1 + erf_vec(x / np.sqrt(2.0)))


@numpy_eager_registry.register("Relu")
def _np_relu(backend_module: object, x: object, *args: object, **kwargs: object) -> object:
    return backend_module.maximum(x, 0.0)
