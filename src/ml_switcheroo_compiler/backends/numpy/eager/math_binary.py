# ruff: noqa: E501
"""Numpy eager binary math ops."""

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("Add")
def _np_add(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the add logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.add(*args, **kwargs)


@numpy_eager_registry.register("Subtract")
def _np_subtract(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the subtract logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.subtract(*args, **kwargs)


@numpy_eager_registry.register("Multiply")
def _np_multiply(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the multiply logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.multiply(*args, **kwargs)


@numpy_eager_registry.register("TrueDivide")
def _np_true_divide(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the true divide logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.divide(*args, **kwargs)


@numpy_eager_registry.register("Maximum")
def _np_maximum(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the maximum logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.maximum(*args, **kwargs)


@numpy_eager_registry.register("Minimum")
def _np_minimum(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the minimum logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.minimum(*args, **kwargs)


@numpy_eager_registry.register("BitwiseAnd")
def _np_bitwise_and(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the bitwise and logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.bitwise_and(*args, **kwargs)


@numpy_eager_registry.register("BitwiseOr")
def _np_bitwise_or(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the bitwise or logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.bitwise_or(*args, **kwargs)


@numpy_eager_registry.register("BitwiseXor")
def _np_bitwise_xor(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the bitwise xor logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.bitwise_xor(*args, **kwargs)


@numpy_eager_registry.register("LeftShift")
def _np_left_shift(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the left shift logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.left_shift(*args, **kwargs)


@numpy_eager_registry.register("RightShift")
def _np_right_shift(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the right shift logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.right_shift(*args, **kwargs)


@numpy_eager_registry.register("Logaddexp")
def _np_logaddexp(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the logaddexp logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.logaddexp(*args, **kwargs)


@numpy_eager_registry.register("Logaddexp2")
def _np_logaddexp2(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the logaddexp2 logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.logaddexp2(*args, **kwargs)


@numpy_eager_registry.register("NanToNum")
def _np_nan_to_num(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the nan to num logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.nan_to_num(*args, **kwargs)


@numpy_eager_registry.register("Frexp")
def _np_frexp(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the frexp logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.frexp(*args, **kwargs)


@numpy_eager_registry.register("Clip")
def _np_clip(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the clip logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.clip(*args, **kwargs)


@numpy_eager_registry.register("Amax")
def _np_amax(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the amax logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.amax(*args, **kwargs)


@numpy_eager_registry.register("Amin")
def _np_amin(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the amin logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.amin(*args, **kwargs)


@numpy_eager_registry.register("Logit")
def _np_logit(backend_module: object, x: object, eps: object = None, *args: object, **kwargs: object) -> object:
    """Evaluate the logit logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        x (object): Required parameter for x.
        eps (object): Required parameter for eps.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return backend_module.log(x / (1.0 - x))


@numpy_eager_registry.register("Polygamma")
def _np_polygamma(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the polygamma logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    import scipy.special as sc

    n = args[0]
    if len(args) > 1:
        x = args[1]
    elif "x" in kwargs:
        x = kwargs["x"]
    else:
        return backend_module.zeros_like(n)
    return backend_module.array(sc.polygamma(n, x))


@numpy_eager_registry.register("Zeta")
def _np_zeta(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the zeta logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    import scipy.special as sc

    x = args[0]
    if len(args) > 1:
        q = args[1]
    elif "q" in kwargs:
        q = kwargs["q"]
    else:
        return backend_module.zeros_like(x)
    return backend_module.array(sc.zeta(x, q))


@numpy_eager_registry.register("Remainder")
def _eager_remainder(backend_module: object, *args: object, **kwargs: object) -> object:
    return backend_module.remainder(*args, **kwargs)
