# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core utilities."""

from __future__ import annotations

from typing import Any

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("NanToNum")
def _nan_to_num(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _nan_to_num operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    x = args[0]
    kwargs.pop("copy", None)
    nan = kwargs.get("nan", 0.0)
    posinf = kwargs.get("posinf", None)
    neginf = kwargs.get("neginf", None)
    if hasattr(backend_module, "nan_to_num"):
        return backend_module.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
    return None


@global_eager_registry.register("Isclose")
def _isclose(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _isclose operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "isclose", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("IsComplex")
def _iscomplex(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _iscomplex operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "iscomplex", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("IsReal")
def _isreal(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _isreal operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "isreal", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Kaiser")
def _kaiser(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _kaiser operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "kaiser", None)
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Heaviside")
def _heaviside(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _heaviside operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "heaviside", getattr(backend_module, "step", None))
    if func:
        return func(*args, **kwargs)
    x, h0 = args[0], args[1]
    return backend_module.where(x < 0, 0.0, backend_module.where(x > 0, 1.0, h0))


@global_eager_registry.register("AxisIndex")
def _axis_index(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _axis_index operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.array(0) if hasattr(backend_module, "array") else 0


@global_eager_registry.register("DivideNoNan")
def _divide_no_nan(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _divide_no_nan operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "divide_no_nan", None)
    if func:
        return func(*args, **kwargs)
    x, y = args[0], args[1]
    res = backend_module.divide(x, y)
    return backend_module.where(y == 0, 0.0, res)


@global_eager_registry.register("MultiplyNoNan")
def _multiply_no_nan(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _multiply_no_nan operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "multiply_no_nan", None)
    if func:
        return func(*args, **kwargs)
    x, y = args[0], args[1]
    res = backend_module.multiply(x, y)
    return backend_module.where(backend_module.isnan(res), 0.0, res)


@global_eager_registry.register("Iscomplexobj")
def _iscomplexobj(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _iscomplexobj operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.iscomplexobj(*args, **kwargs)


@global_eager_registry.register("Isrealobj")
def _isrealobj(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _isrealobj operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.isrealobj(*args, **kwargs)


@global_eager_registry.register("Issubdtype")
def _issubdtype(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _issubdtype operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.issubdtype(*args, **kwargs)


@global_eager_registry.register("Finfo")
def _finfo(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _finfo operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.finfo(*args, **kwargs)


@global_eager_registry.register("Iinfo")
def _iinfo(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _iinfo operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.iinfo(*args, **kwargs)


@global_eager_registry.register("HardSwish")
def _hardswish(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _hardswish operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    x = args[0]
    return x * backend_module.clip(x + 3, 0, 6) / 6


@global_eager_registry.register("Histogram")
def _histogram(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _histogram operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.histogram(*args, **kwargs)


@global_eager_registry.register("Histogram2d")
def _histogram2d(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _histogram2d operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.histogram2d(*args, **kwargs)


@global_eager_registry.register("HistogramBinEdges")
def _histogrambinedges(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _histogrambinedges operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.histogram_bin_edges(*args, **kwargs)


@global_eager_registry.register("Histogramdd")
def _histogramdd(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _histogramdd operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.histogramdd(*args, **kwargs)


@global_eager_registry.register("Infeed")
def _infeed(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _infeed operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    if hasattr(backend_module, "lax") and hasattr(backend_module.lax, "infeed"):
        return backend_module.lax.infeed(*args, **kwargs)
    return args[0] if args else None


@global_eager_registry.register("Isscalar")
def _isscalar(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _isscalar operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.isscalar(*args, **kwargs)


@global_eager_registry.register("Mish")
def _mish(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _mish operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    x = args[0]
    return x * backend_module.tanh(backend_module.log1p(backend_module.exp(x)))


@global_eager_registry.register("Piecewise")
def _piecewise(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _piecewise operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.piecewise(*args, **kwargs)


@global_eager_registry.register("Isnan")
def _isnan(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _isnan operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "isnan", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.isnan(backend_module.asarray(args[0]))


@global_eager_registry.register("Isneginf")
def _isneginf(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _isneginf operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "isneginf", None)
    if func:
        return func(*args, **kwargs)

    return backend_module.isneginf(backend_module.asarray(args[0]))


@global_eager_registry.register("Heaviside")
def _np_heaviside(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_heaviside operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "heaviside", getattr(backend_module, "heaviside", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return np.heaviside(args[0], args[1])


@global_eager_registry.register("TakeAlongAxis")
def _np_takealongaxis(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_takealongaxis operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "takealongaxis", getattr(backend_module, "takealongaxis", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return np.take_along_axis(*args, **kwargs)
