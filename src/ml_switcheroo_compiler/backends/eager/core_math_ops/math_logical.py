# ruff: noqa: E402
"""Core utilities."""

from __future__ import annotations

import builtins  # noqa: F401
import typing  # noqa: F401
from types import ModuleType
from typing import cast

from ml_switcheroo_compiler.backends.eager_registry import (
    EagerTensorProtocol,
    EagerValue,
    global_eager_registry,
)
from ml_switcheroo_compiler.distributed.host_collectives import np


@global_eager_registry.register("NanToNum")
def _nan_to_num(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _nan_to_num operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    x = args[0]
    kwargs.pop("copy", None)
    nan = kwargs.get("nan", 0.0)
    posinf = kwargs.get("posinf", None)
    neginf = kwargs.get("neginf", None)
    if hasattr(backend_module, "nan_to_num"):
        return cast(EagerValue, cast(ModuleType, backend_module).nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf))
    return cast(EagerValue, np.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf))


_ACTIVE_AXIS_INDICES: dict[EagerValue, int] = {}
_INFEED_QUEUES: dict[str, AttributeValue] = {}


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
def _axis_index(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _axis_index operation.

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
def _divide_no_nan(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _divide_no_nan operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    mod = cast(ModuleType, backend_module)
    func = getattr(backend_module, "math", backend_module)
    func2 = getattr(func, "divide_no_nan", None)
    if func2:
        return cast(EagerValue, func2(*args, **kwargs))
    (x, y) = (cast(EagerTensorProtocol, args[0]), cast(EagerTensorProtocol, args[1]))
    safe_y = mod.where(y.__eq__(0), 1.0, y)
    return cast(EagerValue, mod.where(y.__eq__(0), 0, x / safe_y))


@global_eager_registry.register("Finfo")
def _finfo(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _finfo operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, cast(ModuleType, backend_module).finfo(*args, **kwargs))


@global_eager_registry.register("Hardswish")
def _hardswish(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _hardswish operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    mod = cast(ModuleType, backend_module)
    x = cast(EagerTensorProtocol, args[0])
    return cast(EagerValue, x * mod.clip(x + 3, 0, 6) / 6)


@global_eager_registry.register("Heaviside")
def _heaviside(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _heaviside operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    mod = cast(ModuleType, backend_module)
    func = getattr(backend_module, "heaviside", None)
    if func:
        return cast(EagerValue, func(*args, **kwargs))
    x, y = cast(EagerTensorProtocol, args[0]), args[1]
    return cast(EagerValue, mod.where(x < 0, 0.0, mod.where(x > 0, 1.0, y)))


@global_eager_registry.register("Histogram")
def _histogram(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _histogram operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, cast(ModuleType, backend_module).histogram(*args, **kwargs))


@global_eager_registry.register("Histogram2D")
def _histogram2d(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _histogram2d operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, cast(ModuleType, backend_module).histogram2d(*args, **kwargs))


@global_eager_registry.register("HistogramBinedges")
def _histogrambinedges(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _histogrambinedges operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, cast(ModuleType, backend_module).histogram_bin_edges(*args, **kwargs))


@global_eager_registry.register("HistogramDd")
def _histogramdd(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _histogramdd operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, cast(ModuleType, backend_module).histogramdd(*args, **kwargs))


@global_eager_registry.register("Iinfo")
def _iinfo(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _iinfo operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, cast(ModuleType, backend_module).iinfo(*args, **kwargs))


@global_eager_registry.register("Infeed")
def _infeed(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _infeed operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args (queue_name).
        **kwargs: Keyword args (queue_name).

    Returns:
        Any: Ingested tensor from host queue.
    """
    queue_name = str(args[0]) if args and isinstance(args[0], (str, int)) else str(kwargs.get("queue_name", "default"))
    if queue_name in _INFEED_QUEUES:
        it = _INFEED_QUEUES[queue_name]
        try:
            val = next(it) if hasattr(it, "__next__") else cast(ModuleType, it).pop(0)
            return cast(EagerValue, val)
        except (StopIteration, IndexError):
            return 0
    return 0


@global_eager_registry.register("Isclose")
def _isclose(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _isclose operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, cast(ModuleType, backend_module).isclose(*args, **kwargs))


@global_eager_registry.register("Iscomplex")
def _iscomplex(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _iscomplex operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "iscomplex", None)
    if func:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, np.iscomplex(*args, **kwargs))


@global_eager_registry.register("Iscomplexobj")
def _iscomplexobj(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _iscomplexobj operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "iscomplexobj", None)
    if func:
        return cast(EagerValue, func(*args, **kwargs))
    x = args[0]
    return cast(EagerValue, np.iscomplexobj(x))


@global_eager_registry.register("Isnan")
def _isnan(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _isnan operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, cast(ModuleType, backend_module).isnan(*args, **kwargs))


@global_eager_registry.register("Isneginf")
def _isneginf(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _isneginf operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    mod = cast(ModuleType, backend_module)
    func = getattr(backend_module, "isneginf", None)
    if func:
        return cast(EagerValue, func(*args, **kwargs))
    x = cast(EagerTensorProtocol, args[0])
    return cast(EagerValue, mod.logical_and(mod.isinf(x), x < 0))


@global_eager_registry.register("Isreal")
def _isreal(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _isreal operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "isreal", None)
    if func:
        return cast(EagerValue, func(*args, **kwargs))
    return cast(EagerValue, np.isreal(*args, **kwargs))


@global_eager_registry.register("Isrealobj")
def _isrealobj(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _isrealobj operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "isrealobj", None)
    if func:
        return cast(EagerValue, func(*args, **kwargs))
    x = args[0]
    return cast(EagerValue, np.isrealobj(x))


@global_eager_registry.register("Isscalar")
def _isscalar(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _isscalar operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    func = getattr(backend_module, "isscalar", None)
    if func:
        return cast(EagerValue, func(*args, **kwargs))
    x = args[0]
    return cast(EagerValue, np.isscalar(x))


@global_eager_registry.register("Issubdtype")
def _issubdtype(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _issubdtype operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, np.issubdtype(*args, **kwargs))


@global_eager_registry.register("Kaiser")
def _kaiser(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _kaiser operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, cast(ModuleType, backend_module).array(np.kaiser(*args, **kwargs)))


@global_eager_registry.register("Mish")
def _mish(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _mish operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    mod = cast(ModuleType, backend_module)
    x = cast(EagerTensorProtocol, args[0])
    return cast(EagerValue, x * mod.tanh(mod.log1p(mod.exp(x))))


@global_eager_registry.register("MultiplyNoNan")
def _multiply_no_nan(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _multiply_no_nan operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    mod = cast(ModuleType, backend_module)
    func = getattr(backend_module, "math", backend_module)
    func2 = getattr(func, "multiply_no_nan", None)
    if func2:
        return cast(EagerValue, func2(*args, **kwargs))
    (x, y) = (cast(EagerTensorProtocol, args[0]), cast(EagerTensorProtocol, args[1]))
    return cast(EagerValue, mod.where(y.__eq__(0), 0, x * y))


@global_eager_registry.register("NpHeaviside")
def _np_heaviside(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_heaviside operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    mod = cast(ModuleType, backend_module)
    func = getattr(backend_module, "heaviside", None)
    if func:
        return cast(EagerValue, func(*args, **kwargs))
    x, y = cast(EagerTensorProtocol, args[0]), args[1]
    return cast(EagerValue, mod.where(x < 0, 0.0, mod.where(x > 0, 1.0, y)))


@global_eager_registry.register("NpTakealongaxis")
def _np_takealongaxis(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_takealongaxis operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, cast(ModuleType, backend_module).take_along_axis(*args, **kwargs))


@global_eager_registry.register("Piecewise")
def _piecewise(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _piecewise operation.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            object: Result.
    """
    return cast(EagerValue, cast(ModuleType, backend_module).piecewise(*args, **kwargs))


from ml_switcheroo_ir import AttributeValue
