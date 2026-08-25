# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core utilities."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("NanToNum")
def _nan_to_num(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _nan_to_num operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    x: typing.Any = args[0]
    kwargs.pop("copy", None)
    nan: typing.Any = kwargs.get("nan", 0.0)
    posinf: typing.Any = kwargs.get("posinf", None)
    neginf: typing.Any = kwargs.get("neginf", None)
    if hasattr(backend_module, "nan_to_num"):
        return backend_module.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
    import numpy as np

    return np.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


@global_eager_registry.register("AxisIndex")
def _axis_index(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _axis_index operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    return 0


@global_eager_registry.register("DivideNoNan")
def _divide_no_nan(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _divide_no_nan operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    func: typing.Any = getattr(backend_module, "math", backend_module)
    func2: typing.Any = getattr(func, "divide_no_nan", None)
    if func2:
        return func2(*args, **kwargs)
    (x, y) = (args[0], args[1])
    return backend_module.where(y == 0, 0, x / y)


@global_eager_registry.register("Finfo")
def _finfo(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _finfo operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    return backend_module.finfo(*args, **kwargs)


@global_eager_registry.register("Hardswish")
def _hardswish(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _hardswish operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    x: typing.Any = args[0]
    return x * backend_module.clip(x + 3, 0, 6) / 6


@global_eager_registry.register("Heaviside")
def _heaviside(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _heaviside operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    return backend_module.heaviside(*args, **kwargs)


@global_eager_registry.register("Histogram")
def _histogram(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _histogram operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    return backend_module.histogram(*args, **kwargs)


@global_eager_registry.register("Histogram2D")
def _histogram2d(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _histogram2d operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    return backend_module.histogram2d(*args, **kwargs)


@global_eager_registry.register("HistogramBinedges")
def _histogrambinedges(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _histogrambinedges operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    return backend_module.histogram_bin_edges(*args, **kwargs)


@global_eager_registry.register("HistogramDd")
def _histogramdd(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _histogramdd operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    return backend_module.histogramdd(*args, **kwargs)


@global_eager_registry.register("Iinfo")
def _iinfo(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _iinfo operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    return backend_module.iinfo(*args, **kwargs)


@global_eager_registry.register("Infeed")
def _infeed(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _infeed operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    return 0


@global_eager_registry.register("Isclose")
def _isclose(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _isclose operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    return backend_module.isclose(*args, **kwargs)


@global_eager_registry.register("Iscomplex")
def _iscomplex(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _iscomplex operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    func: typing.Any = getattr(backend_module, "iscomplex", None)
    if func:
        return func(*args, **kwargs)
    x: typing.Any = args[0]
    return backend_module.imag(x) != 0


@global_eager_registry.register("Iscomplexobj")
def _iscomplexobj(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _iscomplexobj operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    func: typing.Any = getattr(backend_module, "iscomplexobj", None)
    if func:
        return func(*args, **kwargs)
    x: typing.Any = args[0]
    import numpy as np

    return np.iscomplexobj(x)


@global_eager_registry.register("Isnan")
def _isnan(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _isnan operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    return backend_module.isnan(*args, **kwargs)


@global_eager_registry.register("Isneginf")
def _isneginf(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _isneginf operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    func: typing.Any = getattr(backend_module, "isneginf", None)
    if func:
        return func(*args, **kwargs)
    x: typing.Any = args[0]
    return backend_module.logical_and(backend_module.isinf(x), x < 0)


@global_eager_registry.register("Isreal")
def _isreal(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _isreal operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    func: typing.Any = getattr(backend_module, "isreal", None)
    if func:
        return func(*args, **kwargs)
    x: typing.Any = args[0]
    return backend_module.imag(x) == 0


@global_eager_registry.register("Isrealobj")
def _isrealobj(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _isrealobj operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    func: typing.Any = getattr(backend_module, "isrealobj", None)
    if func:
        return func(*args, **kwargs)
    x: typing.Any = args[0]
    import numpy as np

    return np.isrealobj(x)


@global_eager_registry.register("Isscalar")
def _isscalar(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _isscalar operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    func: typing.Any = getattr(backend_module, "isscalar", None)
    if func:
        return func(*args, **kwargs)
    x: typing.Any = args[0]
    import numpy as np

    return np.isscalar(x)


@global_eager_registry.register("Issubdtype")
def _issubdtype(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _issubdtype operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    import numpy as np

    return np.issubdtype(*args, **kwargs)


@global_eager_registry.register("Kaiser")
def _kaiser(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _kaiser operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    import numpy as np

    return backend_module.array(np.kaiser(*args, **kwargs))


@global_eager_registry.register("Mish")
def _mish(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _mish operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    x: typing.Any = args[0]
    return x * backend_module.tanh(backend_module.log1p(backend_module.exp(x)))


@global_eager_registry.register("MultiplyNoNan")
def _multiply_no_nan(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _multiply_no_nan operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    func: typing.Any = getattr(backend_module, "math", backend_module)
    func2: typing.Any = getattr(func, "multiply_no_nan", None)
    if func2:
        return func2(*args, **kwargs)
    (x, y) = (args[0], args[1])
    return backend_module.where(y == 0, 0, x * y)


@global_eager_registry.register("NpHeaviside")
def _np_heaviside(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _np_heaviside operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    return backend_module.heaviside(*args, **kwargs)


@global_eager_registry.register("NpTakealongaxis")
def _np_takealongaxis(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _np_takealongaxis operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    return backend_module.take_along_axis(*args, **kwargs)


@global_eager_registry.register("Piecewise")
def _piecewise(backend_module: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Evaluate _piecewise operation.

    Args:
        backend_module (typing.Any): The backend_module parameter.
        *args (typing.Any): Positional args.
        **kwargs (typing.Any): Keyword args.

    Returns:
            typing.Any: Result.
    """
    return backend_module.piecewise(*args, **kwargs)
