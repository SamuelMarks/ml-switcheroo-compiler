# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_creation module."""

from __future__ import annotations

from typing import Any

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("Fromfunction")
def _fromfunction(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _fromfunction operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.fromfunction(*args, **kwargs)


@global_eager_registry.register("FromDlpack")
def _from_dlpack(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _from_dlpack operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.from_dlpack(*args, **kwargs)


@global_eager_registry.register("Frompyfunc")
def _frompyfunc(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _frompyfunc operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.frompyfunc(*args, **kwargs)


@global_eager_registry.register("Geomspace")
def _geomspace(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _geomspace operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.geomspace(*args, **kwargs)


@global_eager_registry.register("Mgrid")
def _mgrid(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _mgrid operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.mgrid(*args, **kwargs)


@global_eager_registry.register("Ogrid")
def _ogrid(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _ogrid operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.ogrid(*args, **kwargs)


@global_eager_registry.register("Fromiter")
def _fromiter(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _fromiter operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "fromiter", None)
    if "dtype" not in kwargs and len(args) < 2:
        kwargs["dtype"] = float
    if func:
        return func(*args, **kwargs)
    x = args[0]
    dtype = kwargs.pop("dtype", float)
    return backend_module.fromiter(x, dtype=dtype)


@global_eager_registry.register("Fromfunction")
def _np_fromfunction(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_fromfunction operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "fromfunction", getattr(backend_module, "fromfunction", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.fromfunction(*args, **kwargs)


@global_eager_registry.register("Fromiter")
def _np_fromiter(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_fromiter operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "fromiter", getattr(backend_module, "fromiter", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.fromiter(*args, **kwargs)


@global_eager_registry.register("Frompyfunc")
def _np_frompyfunc(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_frompyfunc operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "frompyfunc", getattr(backend_module, "frompyfunc", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.frompyfunc(*args, **kwargs)
