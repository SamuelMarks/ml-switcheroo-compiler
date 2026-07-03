"""Distributed ops eager."""

# Dummy implementation for tests
import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("AxisIndex")
def _np_axis_index(backend_module: object, **kwargs: object) -> object:
    """Function docstring."""
    return np.array(0)
