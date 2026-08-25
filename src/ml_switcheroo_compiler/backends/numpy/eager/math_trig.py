# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Extracted math functions for numpy eager."""

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("Sin")
def _np_sin(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_sin operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.sin(*args, **kwargs)


@numpy_eager_registry.register("Cos")
def _np_cos(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_cos operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.cos(*args, **kwargs)


@numpy_eager_registry.register("Acos")
def _np_acos(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_acos operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.arccos(*args, **kwargs)


@numpy_eager_registry.register("Acosh")
def _np_acosh(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_acosh operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.arccosh(*args, **kwargs)


@numpy_eager_registry.register("Asin")
def _np_asin(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_asin operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.arcsin(*args, **kwargs)


@numpy_eager_registry.register("Asinh")
def _np_asinh(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_asinh operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.arcsinh(*args, **kwargs)


@numpy_eager_registry.register("Atan")
def _np_atan(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_atan operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.arctan(*args, **kwargs)


@numpy_eager_registry.register("Atanh")
def _np_atanh(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_atanh operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.arctanh(*args, **kwargs)


@numpy_eager_registry.register("Atan2")
def _np_atan2(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_atan2 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.arctan2(*args, **kwargs)


@numpy_eager_registry.register("Deg2Rad")
def _np_deg2rad(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_deg2rad operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.deg2rad(*args, **kwargs)


@numpy_eager_registry.register("Rad2Deg")
def _np_rad2deg(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate _np_rad2deg operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.rad2deg(*args, **kwargs)
