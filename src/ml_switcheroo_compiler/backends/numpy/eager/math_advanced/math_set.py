# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_set module."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy


@numpy_eager_registry.register("Union1d")
def _np_union1d(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Find the union of two one-dimensional arrays.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return np.union1d(np.asarray(args[0]), np.asarray(args[1]), **kwargs)


@numpy_eager_registry.register("Intersect1d")
def _np_intersect1d_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Intersect1d via intersect1d.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.intersect1d(*args, **kwargs)


@numpy_eager_registry.register("Isin")
def _np_isin_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Isin via isin.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.isin(*args, **kwargs)
