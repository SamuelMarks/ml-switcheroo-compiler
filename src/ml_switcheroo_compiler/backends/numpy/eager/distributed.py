"""Distributed ops eager."""

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("AxisIndex")
def _np_axis_index(backend_module: object, **kwargs: object) -> object:
    # Dummy implementation for tests
    import numpy as np

    return np.array(0)
