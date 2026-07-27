# ruff: noqa: E501
"""Extracted math functions for numpy eager."""

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("Sin")
def _np_sin(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the sin logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.sin(*args, **kwargs)


@numpy_eager_registry.register("Cos")
def _np_cos(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the cos logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.cos(*args, **kwargs)


@numpy_eager_registry.register("Acos")
def _np_acos(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the acos logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.arccos(*args, **kwargs)


@numpy_eager_registry.register("Acosh")
def _np_acosh(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the acosh logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.arccosh(*args, **kwargs)


@numpy_eager_registry.register("Asin")
def _np_asin(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the asin logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.arcsin(*args, **kwargs)


@numpy_eager_registry.register("Asinh")
def _np_asinh(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the asinh logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.arcsinh(*args, **kwargs)


@numpy_eager_registry.register("Atan")
def _np_atan(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the atan logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.arctan(*args, **kwargs)


@numpy_eager_registry.register("Atanh")
def _np_atanh(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the atanh logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.arctanh(*args, **kwargs)


@numpy_eager_registry.register("Atan2")
def _np_atan2(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the atan2 logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.arctan2(*args, **kwargs)


@numpy_eager_registry.register("Deg2Rad")
def _np_deg2rad(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the deg2rad logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.deg2rad(*args, **kwargs)


@numpy_eager_registry.register("Rad2Deg")
def _np_rad2deg(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the rad2deg logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.rad2deg(*args, **kwargs)
