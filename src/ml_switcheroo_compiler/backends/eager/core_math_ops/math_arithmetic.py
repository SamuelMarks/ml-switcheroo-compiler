# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core utilities."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry


@global_eager_registry.register("TrueDivide")
def _true_divide(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _true_divide operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func: object = getattr(backend_module, "divide", getattr(backend_module, "true_divide", None))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("Fmod")
def _fmod(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _fmod operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func: object = getattr(backend_module, "fmod", getattr(backend_module, "remainder", getattr(backend_module, "mod", None)))
    return func(*args, **kwargs) if func else None


@global_eager_registry.register("AccumulateN")
def _accumulate_n(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _accumulate_n operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    inputs: object = args[0] if len(args) > 0 else kwargs.get("inputs", [])
    if not inputs:
        import numpy as np

        return np.zeros(())
    res: object = inputs[0]
    for i in range(1, len(inputs)):
        res: object = res + inputs[i]
    return res


@global_eager_registry.register("AssignAdd")
def _assign_add(backend_module: object, ref: object, value: object, **kwargs: object) -> object:
    """Evaluate _assign_add operation.

    Args:
        backend_module (object): The backend_module parameter.
        ref (object): The ref parameter.
        value (object): The value parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return ref + value


@global_eager_registry.register("AssignSub")
def _assign_sub(backend_module: object, ref: object, value: object, **kwargs: object) -> object:
    """Evaluate _assign_sub operation.

    Args:
        backend_module (object): The backend_module parameter.
        ref (object): The ref parameter.
        value (object): The value parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return ref - value


@global_eager_registry.register("AddN")
def _add_n(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _add_n operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    inputs: object = args[0] if len(args) > 0 else kwargs.get("inputs", [])
    if not inputs:
        import numpy as np

        return np.zeros(())
    res: object = inputs[0]
    for i in range(1, len(inputs)):
        res: object = res + inputs[i]
    return res


@global_eager_registry.register("Modf")
def _modf(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _modf operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.modf(*args, **kwargs)


@global_eager_registry.register("RavelMultiIndex")
def _ravelmultiindex(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _ravelmultiindex operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.ravel_multi_index(*args, **kwargs)


@global_eager_registry.register("ScatterMul")
def _np_scattermul(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_scattermul operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func: object = getattr(backend_module, "scattermul", getattr(backend_module, "scattermul", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return args[0]


@global_eager_registry.register("StringSubstr")
def _np_stringsubstr(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_stringsubstr operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func: object = getattr(backend_module, "stringsubstr", getattr(backend_module, "stringsubstr", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.asarray(args[0], dtype=str)


@global_eager_registry.register("TensorScatterSub")
def _np_tensorscattersub(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_tensorscattersub operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func: object = getattr(backend_module, "tensorscattersub", getattr(backend_module, "tensorscattersub", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return args[0]


@global_eager_registry.register("TruncateDiv")
def _np_truncatediv(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_truncatediv operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func: object = getattr(backend_module, "truncatediv", getattr(backend_module, "truncatediv", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.trunc(np.divide(args[0], args[1]))


@global_eager_registry.register("TruncateMod")
def _np_truncatemod(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_truncatemod operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func: object = getattr(backend_module, "truncatemod", getattr(backend_module, "truncatemod", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.mod(args[0], args[1])


@global_eager_registry.register("Xdivy")
def _np_xdivy(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_xdivy operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    func: object = getattr(backend_module, "xdivy", getattr(backend_module, "xdivy", None))
    if func is not None:
        return func(*args, **kwargs)
    import numpy as np

    return np.where(args[0] == 0, 0, np.divide(args[0], args[1]))
