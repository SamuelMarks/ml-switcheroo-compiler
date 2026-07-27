# ruff: noqa: E501
"""Numpy eager unary math ops."""

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("Exp")
def _np_exp(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the exp logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.exp(*args, **kwargs)


@numpy_eager_registry.register("Log")
def _np_log(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the log logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.log(*args, **kwargs)


@numpy_eager_registry.register("Log1p")
def _np_log1p(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the log1p logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.log1p(*args, **kwargs)


@numpy_eager_registry.register("Round")
def _np_round(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the round logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.round(*args, **kwargs)


@numpy_eager_registry.register("Erf")
def _np_erf(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the erf logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    import math

    x = args[0]
    if hasattr(backend_module, "asarray"):
        x = backend_module.asarray(x)
    if getattr(x, "ndim", 0) == 0:
        return math.erf(float(x))
    return backend_module.vectorize(math.erf)(x)


@numpy_eager_registry.register("Erfc")
def _np_erfc(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the erfc logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    import math

    x = args[0]
    if hasattr(backend_module, "asarray"):
        x = backend_module.asarray(x)
    if getattr(x, "ndim", 0) == 0:
        return math.erfc(float(x))
    return backend_module.vectorize(math.erfc)(x)


@numpy_eager_registry.register("Erfinv")
def _np_erfinv(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the erfinv logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    import scipy.special as sc

    return backend_module.array(sc.erfinv(args[0]))


@numpy_eager_registry.register("Igamma")
def _np_igamma(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the igamma logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    import scipy.special as sc

    a = args[0]
    x = args[1] if len(args) > 1 else kwargs.get("x")
    return backend_module.array(sc.gammainc(a, x))


@numpy_eager_registry.register("Igammac")
def _np_igammac(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the igammac logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    import scipy.special as sc

    a = args[0]
    x = args[1] if len(args) > 1 else kwargs.get("x")
    return backend_module.array(sc.gammaincc(a, x))


@numpy_eager_registry.register("BitwiseNot")
def _np_bitwise_not(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the bitwise not logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.bitwise_not(*args, **kwargs)


@numpy_eager_registry.register("Angle")
def _np_angle(backend_module: object, x: object, **kwargs: object) -> object:
    """Evaluate the angle logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        x (object): Required parameter for x.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return np.angle(x)


@numpy_eager_registry.register("Expm1")
def _np_expm1(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the expm1 logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.expm1(*args, **kwargs)


@numpy_eager_registry.register("Log10")
def _np_log10(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the log10 logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.log10(*args, **kwargs)


@numpy_eager_registry.register("Log2")
def _np_log2(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the log2 logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.log2(*args, **kwargs)


@numpy_eager_registry.register("Exp2")
def _np_exp2(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the exp2 logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.exp2(*args, **kwargs)


@numpy_eager_registry.register("Signbit")
def _np_signbit(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the signbit logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.signbit(*args, **kwargs)


@numpy_eager_registry.register("Isnan")
def _np_isnan(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the isnan logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.isnan(*args, **kwargs)


@numpy_eager_registry.register("Isinf")
def _np_isinf(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the isinf logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.isinf(*args, **kwargs)


@numpy_eager_registry.register("Isfinite")
def _np_isfinite(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the isfinite logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.isfinite(*args, **kwargs)
