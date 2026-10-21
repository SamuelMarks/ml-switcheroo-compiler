# ruff: noqa: E402
"""Implementation of math logical operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.eager.core_math_ops.math_logical`)."""

from __future__ import annotations

import builtins  # noqa: F401
import typing  # noqa: F401

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    EagerValue,
    as_tensor,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, as_mod


@global_eager_registry.register("NanToNum")
def _nan_to_num(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager nan to num operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    x = args[0]
    kwargs.pop("copy", None)
    nan = kwargs.get("nan", 0.0)
    posinf = kwargs.get("posinf", None)
    neginf = kwargs.get("neginf", None)
    if hasattr(backend_module, "nan_to_num"):
        return backend_module.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
    return _resolve_eager_mod(backend_module).nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


_ACTIVE_AXIS_INDICES: dict[EagerValue, int] = {}
_INFEED_QUEUES: dict[str, EagerValue] = {}


def set_axis_index(axis: EagerValue, index: int) -> None:
    """Set active SPMD axis index for execution.

    Args:
        axis: Axis identifier (string or integer).
        index (int): Active rank or thread index.
    """
    _ACTIVE_AXIS_INDICES[axis] = index


def clear_axis_indices() -> None:
    """Clear active SPMD axis indices."""
    _ACTIVE_AXIS_INDICES.clear()


def register_infeed_queue(name: str, queue_or_iterator: EagerValue) -> None:
    """Register infeed data generator or queue.

    Args:
        name (str): Queue identifier.
        queue_or_iterator: Host iterator or list of batches.
    """
    _INFEED_QUEUES[name] = queue_or_iterator


def clear_infeed_queues() -> None:
    """Clear registered infeed queues."""
    _INFEED_QUEUES.clear()


@global_eager_registry.register("AxisIndex")
def _axis_index(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager axis index operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args (axis identifier).
        **kwargs: Keyword args (axis_name).

    Returns:
        int: Active axis index.
    """
    axis_name = args[0] if args else kwargs.get("axis_name", 0)
    return _ACTIVE_AXIS_INDICES.get(axis_name, 0)


@global_eager_registry.register("DivideNoNan")
def _divide_no_nan(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager divide no nan operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = backend_module
    func = getattr(backend_module, "math", backend_module)
    func2 = getattr(func, "divide_no_nan", None)
    if func2:
        return as_tensor(func2(*args, **kwargs))
    (x, y) = (as_tensor(args[0]), as_tensor(args[1]))
    safe_y = mod.where(y.__eq__(0), 1.0, y)
    return mod.where(y.__eq__(0), 0, x / safe_y)


@global_eager_registry.register("Finfo")
def _finfo(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager finfo operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return backend_module.finfo(*args, **kwargs)


@global_eager_registry.register("Hardswish")
def _hardswish(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager hardswish operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = backend_module
    x = as_tensor(args[0])
    return x * mod.clip(x + 3, 0, 6) / 6


@global_eager_registry.register("Heaviside")
def _heaviside(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager heaviside operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = backend_module
    func = getattr(backend_module, "heaviside", None)
    if func:
        return as_tensor(func(*args, **kwargs))
    x, y = as_tensor(args[0]), args[1]
    return mod.where(x < 0, 0.0, mod.where(x > 0, 1.0, y))


@global_eager_registry.register("Histogram")
def _histogram(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager histogram operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return backend_module.histogram(*args, **kwargs)


@global_eager_registry.register("Histogram2D")
def _histogram2d(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager histogram2d operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return backend_module.histogram2d(*args, **kwargs)


@global_eager_registry.register("HistogramBinedges")
def _histogrambinedges(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager histogrambinedges operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return backend_module.histogram_bin_edges(*args, **kwargs)


@global_eager_registry.register("HistogramDd")
def _histogramdd(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager histogramdd operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return backend_module.histogramdd(*args, **kwargs)


@global_eager_registry.register("Iinfo")
def _iinfo(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager iinfo operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return backend_module.iinfo(*args, **kwargs)


@global_eager_registry.register("Infeed")
def _infeed(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager infeed operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args (queue_name).
        **kwargs: Keyword args (queue_name).

    Returns:
        EagerValue: Ingested tensor from host queue.
    """
    queue_name = str(args[0]) if args and isinstance(args[0], (str, int)) else str(kwargs.get("queue_name", "default"))
    if queue_name in _INFEED_QUEUES:
        it = _INFEED_QUEUES[queue_name]
        try:
            val = getattr(it, "__next__")() if hasattr(it, "__next__") else as_mod(it).pop(0)
            return val
        except (StopIteration, IndexError):
            return 0
    return 0


@global_eager_registry.register("Isclose")
def _isclose(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager isclose operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return backend_module.isclose(*args, **kwargs)


@global_eager_registry.register("Iscomplex")
def _iscomplex(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager iscomplex operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "iscomplex", None)
    if func:
        return as_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).iscomplex(*args, **kwargs)


@global_eager_registry.register("Iscomplexobj")
def _iscomplexobj(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager iscomplexobj operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "iscomplexobj", None)
    if func:
        return as_tensor(func(*args, **kwargs))
    x = args[0]
    return _resolve_eager_mod(backend_module).iscomplexobj(x)


@global_eager_registry.register("Isnan")
def _isnan(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager isnan operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return backend_module.isnan(*args, **kwargs)


@global_eager_registry.register("Isneginf")
def _isneginf(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager isneginf operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = backend_module
    func = getattr(backend_module, "isneginf", None)
    if func:
        return as_tensor(func(*args, **kwargs))
    x = as_tensor(args[0])
    return mod.logical_and(mod.isinf(x), x < 0)


@global_eager_registry.register("Isreal")
def _isreal(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager isreal operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "isreal", None)
    if func:
        return as_tensor(func(*args, **kwargs))
    return _resolve_eager_mod(backend_module).isreal(*args, **kwargs)


@global_eager_registry.register("Isrealobj")
def _isrealobj(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager isrealobj operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "isrealobj", None)
    if func:
        return as_tensor(func(*args, **kwargs))
    x = args[0]
    return _resolve_eager_mod(backend_module).isrealobj(x)


@global_eager_registry.register("Isscalar")
def _isscalar(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager isscalar operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    func = getattr(backend_module, "isscalar", None)
    if func:
        return as_tensor(func(*args, **kwargs))
    x = args[0]
    return _resolve_eager_mod(backend_module).isscalar(x)


@global_eager_registry.register("Issubdtype")
def _issubdtype(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager issubdtype operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return _resolve_eager_mod(backend_module).issubdtype(*args, **kwargs)


@global_eager_registry.register("Kaiser")
def _kaiser(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager kaiser operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return backend_module.array(_resolve_eager_mod(backend_module).kaiser(*args, **kwargs))


@global_eager_registry.register("Mish")
def _mish(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager mish operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = backend_module
    x = as_tensor(args[0])
    return x * mod.tanh(mod.log1p(mod.exp(x)))


@global_eager_registry.register("MultiplyNoNan")
def _multiply_no_nan(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager multiply no nan operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = backend_module
    func = getattr(backend_module, "math", backend_module)
    func2 = getattr(func, "multiply_no_nan", None)
    if func2:
        return as_tensor(func2(*args, **kwargs))
    (x, y) = (as_tensor(args[0]), as_tensor(args[1]))
    return mod.where(y.__eq__(0), 0, x * y)


@global_eager_registry.register("NpHeaviside")
def _np_heaviside(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np heaviside operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    mod = backend_module
    func = getattr(backend_module, "heaviside", None)
    if func:
        return as_tensor(func(*args, **kwargs))
    x, y = as_tensor(args[0]), args[1]
    return mod.where(x < 0, 0.0, mod.where(x > 0, 1.0, y))


@global_eager_registry.register("NpTakealongaxis")
def _np_takealongaxis(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np takealongaxis operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return backend_module.take_along_axis(*args, **kwargs)


@global_eager_registry.register("Piecewise")
def _piecewise(backend_module: BackendModuleProtocol, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager piecewise operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return backend_module.piecewise(*args, **kwargs)
