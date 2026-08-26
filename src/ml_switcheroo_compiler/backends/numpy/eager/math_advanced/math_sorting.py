# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_sorting module."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy


@numpy_eager_registry.register("ArgPartition")
def _np_argpartition(backend_module, *args, **kwargs):
    """Perform an indirect partition along the given axis.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return backend_module.argpartition(*args, **kwargs)


@numpy_eager_registry.register("Lexsort")
def _np_lexsort_(backend_module, *args, **kwargs):
    """Implement Lexsort via lexsort.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return backend_module.lexsort(*args, **kwargs)


@numpy_eager_registry.register("Median")
def _np_median_(backend_module, *args, **kwargs):
    """Implement Median via median.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return backend_module.median(*args, **kwargs)
