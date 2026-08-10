# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Numpy eager binary math ops."""

from typing import Any

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("Add")
def _np_add(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_add operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.add(*args, **kwargs)


@numpy_eager_registry.register("Subtract")
def _np_subtract(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_subtract operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.subtract(*args, **kwargs)


@numpy_eager_registry.register("Multiply")
def _np_multiply(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_multiply operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.multiply(*args, **kwargs)


@numpy_eager_registry.register("TrueDivide")
def _np_true_divide(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_true_divide operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.divide(*args, **kwargs)


@numpy_eager_registry.register("Maximum")
def _np_maximum(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_maximum operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.maximum(*args, **kwargs)


@numpy_eager_registry.register("Minimum")
def _np_minimum(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_minimum operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.minimum(*args, **kwargs)


@numpy_eager_registry.register("BitwiseAnd")
def _np_bitwise_and(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_bitwise_and operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.bitwise_and(*args, **kwargs)


@numpy_eager_registry.register("BitwiseOr")
def _np_bitwise_or(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_bitwise_or operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.bitwise_or(*args, **kwargs)


@numpy_eager_registry.register("BitwiseXor")
def _np_bitwise_xor(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_bitwise_xor operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.bitwise_xor(*args, **kwargs)


@numpy_eager_registry.register("LeftShift")
def _np_left_shift(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_left_shift operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.left_shift(*args, **kwargs)


@numpy_eager_registry.register("RightShift")
def _np_right_shift(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_right_shift operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.right_shift(*args, **kwargs)


@numpy_eager_registry.register("Logaddexp")
def _np_logaddexp(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_logaddexp operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.logaddexp(*args, **kwargs)


@numpy_eager_registry.register("Logaddexp2")
def _np_logaddexp2(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_logaddexp2 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.logaddexp2(*args, **kwargs)


@numpy_eager_registry.register("NanToNum")
def _np_nan_to_num(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_nan_to_num operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.nan_to_num(*args, **kwargs)


@numpy_eager_registry.register("Frexp")
def _np_frexp(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_frexp operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.frexp(*args, **kwargs)


@numpy_eager_registry.register("Clip")
def _np_clip(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_clip operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.clip(*args, **kwargs)


@numpy_eager_registry.register("Amax")
def _np_amax(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_amax operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.amax(*args, **kwargs)


@numpy_eager_registry.register("Amin")
def _np_amin(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_amin operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.amin(*args, **kwargs)


@numpy_eager_registry.register("Logit")
def _np_logit(backend_module: Any, x: Any, eps: Any = None, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_logit operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        eps (object): The eps parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.log(x / (1.0 - x))


@numpy_eager_registry.register("Polygamma")
def _np_polygamma(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_polygamma operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
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
def _np_zeta(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_zeta operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
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
def _eager_remainder(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _eager_remainder operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.remainder(*args, **kwargs)
