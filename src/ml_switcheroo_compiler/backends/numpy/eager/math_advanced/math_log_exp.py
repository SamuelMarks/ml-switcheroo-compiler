"""Math Ops."""

from __future__ import annotations

import typing
from collections.abc import Sequence  # noqa: F401

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol

from .math_general import _get_np_arg, _get_sc  # noqa: F401


@numpy_eager_registry.register("Xlogy")
def _np_xlogy(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np xlogy operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return _xlogy(*args, **kwargs)


@numpy_eager_registry.register("Logsumexp")
def _np_logsumexp(backend_module: BackendModuleProtocol, a: EagerValue, axis: EagerValue = None, keepdims: bool = False, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_logsumexp logic eagerly backed by NumPy.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        a (EagerValue): The a parameter.
        axis (EagerValue): The axis parameter.
        keepdims (bool): The keepdims parameter.
        **kwargs (EagerValue): Keyword args.

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
    return typing.cast(EagerValue, out)


@numpy_eager_registry.register("Log1P")
def _np_log1p2(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np log1p2 operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return typing.cast(EagerValue, backend_module.log1p(*args, **kwargs))


@numpy_eager_registry.register("Unsqueeze")
def _np_expand_dims_(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Unsqueeze via expand_dims.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Variable positional arguments.
        **kwargs (EagerValue): Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return typing.cast(EagerValue, backend_module.expand_dims(*args, **kwargs))


@numpy_eager_registry.register("LogSoftmax")
def _np_log_softmax(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np log softmax operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    a = _get_np_arg(args, 0)
    if a is None:
        return None
    axis = kwargs.get("axis", -1)
    c = getattr(np, "max")(a, axis=axis, keepdims=True)
    return typing.cast(EagerValue, a - c - np.log(getattr(np, "sum")(np.exp(a - c), axis=axis, keepdims=True)))
