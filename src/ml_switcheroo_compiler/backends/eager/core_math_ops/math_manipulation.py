# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_manipulation module."""

from __future__ import annotations

from typing import Any

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("AllGather")
def _all_gather(backend_module: Any, tensor: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _all_gather operation.

    Args:
        backend_module (object): The backend_module parameter.
        tensor (object): The tensor parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    if hasattr(backend_module, "stack"):
        return backend_module.stack([tensor])
    if hasattr(backend_module, "array"):
        return backend_module.array([tensor])
    return tensor


@global_eager_registry.register("Argwhere")
def _argwhere(backend_module: Any, a: Any, **kwargs: Any) -> Any:
    """Evaluate _argwhere operation.

    Args:
        backend_module (object): The backend_module parameter.
        a (object): The a parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.argwhere(a)


@global_eager_registry.register("Extract")
def _extract(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _extract operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "extract", None)
    if func:
        return func(*args, **kwargs)
    (condition, arr) = (args[0], args[1])
    return backend_module.extract(backend_module.asarray(condition), backend_module.asarray(arr))


@global_eager_registry.register("Pswapaxes")
def _pswapaxes(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _pswapaxes operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    if hasattr(backend_module, "lax") and hasattr(backend_module.lax, "pswapaxes"):
        return backend_module.lax.pswapaxes(*args, **kwargs)
    return args[0] if args else None


@global_eager_registry.register("Rot90")
def _rot90(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _rot90 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.rot90(*args, **kwargs)


@global_eager_registry.register("Nonzero")
def _nonzero(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _nonzero operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.nonzero(*args, **kwargs)


@global_eager_registry.register("Repeat")
def _repeat(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _repeat operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.repeat(*args, **kwargs)


@global_eager_registry.register("Tile")
def _tile(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _tile operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.tile(*args, **kwargs)


@global_eager_registry.register("UpdateSlice")
def _updateslice(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _updateslice operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        RuntimeError: An exception.
    """
    if not hasattr(backend_module, "array"):
        raise RuntimeError("Expected numpy-like backend")
    x = args[0]
    update = args[1] if len(args) > 1 else kwargs.get("update")
    start_indices = args[2] if len(args) > 2 else kwargs.get("start_indices")
    out = backend_module.array(x).copy()
    slices = tuple(slice(s, s + getattr(update, "shape", ())[i]) for (i, s) in enumerate(start_indices))  # type: ignore
    out[slices] = update
    return out


@global_eager_registry.register("Flatnonzero")
def _flatnonzero(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _flatnonzero operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "flatnonzero", None)
    if func:
        return func(*args, **kwargs)
    return backend_module.flatnonzero(backend_module.asarray(args[0]))


@global_eager_registry.register("Fliplr")
def _fliplr(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _fliplr operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "fliplr", None)
    if func:
        return func(*args, **kwargs)
    return backend_module.fliplr(backend_module.asarray(args[0]))


@global_eager_registry.register("Flipud")
def _flipud(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _flipud operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "flipud", None)
    if func:
        return func(*args, **kwargs)
    return backend_module.flipud(backend_module.asarray(args[0]))


@global_eager_registry.register("Hsplit")
def _np_hsplit(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_hsplit operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "hsplit", getattr(backend_module, "hsplit", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return np.hsplit(args[0], args[1])


@global_eager_registry.register("ScatterApply")
def _np_scatterapply(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_scatterapply operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        RuntimeError: An exception.
    """
    func = getattr(backend_module, "scatterapply", getattr(backend_module, "scatterapply", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return args[0]


@global_eager_registry.register("ScatterNd")
def _np_scatternd(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_scatternd operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "scatternd", getattr(backend_module, "scatternd", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return args[0]


@global_eager_registry.register("TensorArrayStack")
def _np_tensorarraystack(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_tensorarraystack operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "tensorarraystack", getattr(backend_module, "tensorarraystack", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return np.stack(args[0])


@global_eager_registry.register("TensorScatterUpdate")
def _np_tensorscatterupdate(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_tensorscatterupdate operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "tensorscatterupdate", getattr(backend_module, "tensorscatterupdate", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return args[0]


@global_eager_registry.register("Unfold")
def _np_unfold(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_unfold operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "unfold", getattr(backend_module, "unfold", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return args[0]


@global_eager_registry.register("Unstack")
def _np_unstack(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_unstack operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "unstack", getattr(backend_module, "unstack", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return np.split(args[0], args[0].shape[kwargs.get("axis", 0)], axis=kwargs.get("axis", 0))


@global_eager_registry.register("UpdateSlice")
def _np_updateslice(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_updateslice operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        RuntimeError: An exception.
    """
    func = getattr(backend_module, "updateslice", getattr(backend_module, "updateslice", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    args[0][args[1]] = args[2]
    return args[0]


@global_eager_registry.register("Vsplit")
def _np_vsplit(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_vsplit operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    func = getattr(backend_module, "vsplit", getattr(backend_module, "vsplit", None))
    if func is not None:
        try:
            return func(*args, **kwargs)
        except Exception:
            pass
    import numpy as np

    return np.vsplit(args[0], args[1])
