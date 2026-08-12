# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Math Ops."""

from collections.abc import Sequence
from typing import Any, Optional

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy

from .math_misc_ext import _get_np_arg, _get_sc


@numpy_eager_registry.register("Xlogy")
def _np_xlogy(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_xlogy operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return _xlogy(*args, **kwargs)


@numpy_eager_registry.register("Logsumexp")
def _np_logsumexp(backend_module: Any, a: Any, axis: Any = None, keepdims: bool = False, **kwargs: Any) -> Any:  # noqa: D417
    """Evaluate _np_logsumexp logic eagerly backed by NumPy.

    Args:
        backend_module (object): The backend_module parameter.
        a (object): The a parameter.
        axis (object): The axis parameter.
        keepdims (bool): The keepdims parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import numpy as np

    a = np.array(a)
    a_max = np.amax(a, axis=axis, keepdims=True)
    if not keepdims:
        a_max_s = np.squeeze(a_max, axis=axis)
    else:
        a_max_s = a_max
    out = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=keepdims))
    out += a_max_s
    return out


@numpy_eager_registry.register("Log1P")
def _np_log1p2(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_log1p2 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.log1p(*args, **kwargs)


@numpy_eager_registry.register("Unsqueeze")
def _np_expand_dims_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Unsqueeze via expand_dims.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.expand_dims(*args, **kwargs)


@numpy_eager_registry.register("LogSoftmax")
def _np_log_softmax(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_log_softmax operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    a = _get_np_arg(args, 0)
    if a is None:
        return None
    axis = kwargs.get("axis", -1)
    c = np.max(a, axis=axis, keepdims=True)
    return a - c - np.log(np.sum(np.exp(a - c), axis=axis, keepdims=True))
