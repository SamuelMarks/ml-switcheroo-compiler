"""Control flow ops."""

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("AssociativeScan")
def _np_associative_scan(backend_module: object, *args: object, **kwargs: object) -> object:
    # dummy implementation for coverage
    import numpy as np

    return np.array(0)
