"""Operations and utilities for `ml_switcheroo_compiler.backends.eager.core_math_ops.math_logical`."""

from __future__ import annotations

import builtins  # noqa: F401
import types
import typing  # noqa: F401
from collections.abc import Iterator

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


@global_eager_registry.register("NanToNum")
def _nan_to_num(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager nan to num operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    x = args[0]
    kwargs.pop("copy", None)
    nan = kwargs.get("nan", 0.0)
    posinf = kwargs.get("posinf", None)
    neginf = kwargs.get("neginf", None)
    if adapt_backend_module(backend_module).has_symbol("nan_to_num"):
        return adapt_backend_module(backend_module).require_symbol("nan_to_num")(
            x, nan=nan, posinf=posinf, neginf=neginf
        )
    return _resolve_eager_mod(backend_module).nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


_ACTIVE_AXIS_INDICES: dict[BackendArg, int] = {}

_INFEED_QUEUES: dict[str, BackendArg] = {}


def set_axis_index(axis: BackendArg, index: int) -> None:
    """Set active SPMD axis index for execution.

    Args:
        axis: Axis identifier (string or integer).
        index (int): Active rank or thread index.
    """
    _ACTIVE_AXIS_INDICES[axis] = index


def clear_axis_indices() -> None:
    """Clear active SPMD axis indices."""
    _ACTIVE_AXIS_INDICES.clear()


def register_infeed_queue(name: str, queue_or_iterator: BackendArg) -> None:
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
def _axis_index(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager axis index operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args (axis identifier).
        **kwargs: Keyword args (axis_name).

    Returns:
        int: Active axis index.
    """
    axis_name = args[0] if args else kwargs.get("axis_name", 0)
    return _ACTIVE_AXIS_INDICES.get(axis_name, 0)


@global_eager_registry.register("DivideNoNan")
def _divide_no_nan(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager divide no nan operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    func = adapt_backend_module(backend_module).get_symbol("math", backend_module)
    func2 = adapt_backend_module(func).get_symbol("divide_no_nan", None)
    if func2:
        return func2(*args, **kwargs)
    (x, y) = (args[0], args[1])
    safe_y = adapt_backend_module(mod).require_symbol("where")(y == 0, 1.0, y)
    return adapt_backend_module(mod).require_symbol("where")(y == 0, 0, x / safe_y)


@global_eager_registry.register("Finfo")
def _finfo(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager finfo operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("finfo")(*args, **kwargs)


@global_eager_registry.register("Hardswish")
def _hardswish(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager hardswish operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    from ml_switcheroo_compiler.backends.eager_registry import coerce_numeric

    x = coerce_numeric(args[0])
    return x * adapt_backend_module(mod).require_symbol("clip")(x + 3, 0, 6) / 6


@global_eager_registry.register("Heaviside")
def _heaviside(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager heaviside operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    func = adapt_backend_module(backend_module).get_symbol("heaviside", None)
    if func:
        return func(*args, **kwargs)
    from ml_switcheroo_compiler.backends.eager_registry import coerce_numeric

    x, y = coerce_numeric(args[0]), args[1]
    return adapt_backend_module(mod).require_symbol("where")(
        x < 0, 0.0, adapt_backend_module(mod).require_symbol("where")(x > 0, 1.0, y)
    )


@global_eager_registry.register("Histogram")
def _histogram(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager histogram operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("histogram")(*args, **kwargs)


@global_eager_registry.register("Histogram2D")
def _histogram2d(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager histogram2d operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("histogram2d")(*args, **kwargs)


@global_eager_registry.register("HistogramBinedges")
def _histogrambinedges(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager histogrambinedges operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("histogram_bin_edges")(*args, **kwargs)


@global_eager_registry.register("HistogramDd")
def _histogramdd(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager histogramdd operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("histogramdd")(*args, **kwargs)


@global_eager_registry.register("Iinfo")
def _iinfo(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager iinfo operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("iinfo")(*args, **kwargs)


@global_eager_registry.register("Infeed")
def _infeed(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager infeed operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args (queue_name).
        **kwargs: Keyword args (queue_name).

    Returns:
        BackendArg: Ingested tensor from host queue.
    """

    @typing.runtime_checkable
    class _PopQueueProto(typing.Protocol):
        """Protocol for custom host queues supporting `.pop(0)`."""

        def pop(self, index: int) -> BackendArg:
            """Pop and return element at `index`.

            Args:
                index (int): Queue index.

            Returns:
                BackendArg: Popped batch value.
            """
            ...

    queue_name = str(args[0]) if args and isinstance(args[0], (str, int)) else str(kwargs.get("queue_name", "default"))
    if queue_name in _INFEED_QUEUES:
        it: BackendArg | Iterator[BackendArg] | _PopQueueProto = _INFEED_QUEUES[queue_name]
        try:
            if isinstance(it, Iterator):
                return next(it)
            if isinstance(it, list):
                popped: BackendArg = it.pop(0)
                return popped
            if isinstance(it, _PopQueueProto):
                return it.pop(0)
        except (StopIteration, IndexError):
            return 0
    return 0


@global_eager_registry.register("Isclose")
def _isclose(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager isclose operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("isclose")(*args, **kwargs)


@global_eager_registry.register("Iscomplex")
def _iscomplex(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager iscomplex operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("iscomplex", None)
    if func:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).iscomplex(*args, **kwargs)


@global_eager_registry.register("Iscomplexobj")
def _iscomplexobj(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager iscomplexobj operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("iscomplexobj", None)
    if func:
        return func(*args, **kwargs)
    x = args[0]
    return _resolve_eager_mod(backend_module).iscomplexobj(x)


@global_eager_registry.register("Isnan")
def _isnan(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager isnan operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("isnan")(*args, **kwargs)


@global_eager_registry.register("Isneginf")
def _isneginf(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager isneginf operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    func = adapt_backend_module(backend_module).get_symbol("isneginf", None)
    if func:
        return func(*args, **kwargs)
    from ml_switcheroo_compiler.backends.eager_registry import coerce_numeric

    x = coerce_numeric(args[0])
    return adapt_backend_module(mod).require_symbol("logical_and")(
        adapt_backend_module(mod).require_symbol("isinf")(x), x < 0
    )


@global_eager_registry.register("Isreal")
def _isreal(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager isreal operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("isreal", None)
    if func:
        return func(*args, **kwargs)
    return _resolve_eager_mod(backend_module).isreal(*args, **kwargs)


@global_eager_registry.register("Isrealobj")
def _isrealobj(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager isrealobj operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("isrealobj", None)
    if func:
        return func(*args, **kwargs)
    x = args[0]
    return _resolve_eager_mod(backend_module).isrealobj(x)


@global_eager_registry.register("Isscalar")
def _isscalar(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager isscalar operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    func = adapt_backend_module(backend_module).get_symbol("isscalar", None)
    if func:
        return func(*args, **kwargs)
    x = args[0]
    return _resolve_eager_mod(backend_module).isscalar(x)


@global_eager_registry.register("Issubdtype")
def _issubdtype(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager issubdtype operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return _resolve_eager_mod(backend_module).issubdtype(*args, **kwargs)


@global_eager_registry.register("Kaiser")
def _kaiser(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager kaiser operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("array")(
        _resolve_eager_mod(backend_module).kaiser(*args, **kwargs)
    )


@global_eager_registry.register("Mish")
def _mish(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager mish operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    x = args[0]
    return x * adapt_backend_module(mod).require_symbol("tanh")(
        adapt_backend_module(mod).require_symbol("log1p")(adapt_backend_module(mod).require_symbol("exp")(x))
    )


@global_eager_registry.register("MultiplyNoNan")
def _multiply_no_nan(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager multiply no nan operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    func = adapt_backend_module(backend_module).get_symbol("math", backend_module)
    func2 = adapt_backend_module(func).get_symbol("multiply_no_nan", None)
    if func2:
        return func2(*args, **kwargs)
    from ml_switcheroo_compiler.backends.eager_registry import coerce_numeric

    (x, y) = (coerce_numeric(args[0]), coerce_numeric(args[1]))
    return adapt_backend_module(mod).require_symbol("where")(y == 0, 0, x * y)


@global_eager_registry.register("NpHeaviside")
def _np_heaviside(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np heaviside operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    mod = backend_module
    func = adapt_backend_module(backend_module).get_symbol("heaviside", None)
    if func:
        return func(*args, **kwargs)
    from ml_switcheroo_compiler.backends.eager_registry import coerce_numeric

    x, y = coerce_numeric(args[0]), args[1]
    return adapt_backend_module(mod).require_symbol("where")(
        x < 0, 0.0, adapt_backend_module(mod).require_symbol("where")(x > 0, 1.0, y)
    )


@global_eager_registry.register("NpTakealongaxis")
def _np_takealongaxis(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np takealongaxis operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("take_along_axis")(*args, **kwargs)


@global_eager_registry.register("Piecewise")
def _piecewise(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager piecewise operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("piecewise")(*args, **kwargs)
