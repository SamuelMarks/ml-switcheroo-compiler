"""Math Ops."""

from collections.abc import Sequence  # noqa: F401
from types import ModuleType
from typing import (
    Optional,  # noqa: F401
    cast,
)

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy

from .math_general import _get_np_arg, _get_sc  # noqa: F401


@numpy_eager_registry.register("Xlogy")
def _np_xlogy(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_xlogy operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _xlogy(*args, **kwargs)


@numpy_eager_registry.register("Logsumexp")
def _np_logsumexp(backend_module: ModuleType, a: EagerValue, axis: EagerValue = None, keepdims: bool = False, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_logsumexp logic eagerly backed by NumPy.

    Args:
        backend_module (object): The backend_module parameter.
        a (object): The a parameter.
        axis (object): The axis parameter.
        keepdims (bool): The keepdims parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy as np

    a = np.array(a)
    a_max = getattr(np, "amax")(a, axis=axis, keepdims=True)
    if not keepdims:
        a_max_s = getattr(np, "squeeze")(a_max, axis=axis)
    else:
        a_max_s = a_max
    out = np.log(getattr(np, "sum")(np.exp(a - a_max), axis=axis, keepdims=keepdims))
    out += a_max_s
    return cast(EagerValue, out)


@numpy_eager_registry.register("Log1P")
def _np_log1p2(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_log1p2 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.log1p(*args, **kwargs))


@numpy_eager_registry.register("Unsqueeze")
def _np_expand_dims_(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Unsqueeze via expand_dims.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, backend_module.expand_dims(*args, **kwargs))


@numpy_eager_registry.register("LogSoftmax")
def _np_log_softmax(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_log_softmax operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    a = _get_np_arg(args, 0)
    if a is None:
        return None
    axis = kwargs.get("axis", -1)
    c = getattr(np, "max")(a, axis=axis, keepdims=True)
    return cast(EagerValue, a - c - np.log(getattr(np, "sum")(np.exp(a - c), axis=axis, keepdims=True)))
