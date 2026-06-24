"""Math Ops."""

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@numpy_eager_registry.register("Add")
def _np_add(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.add(*args, **kwargs)


@numpy_eager_registry.register("Subtract")
def _np_subtract(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.subtract(*args, **kwargs)


@numpy_eager_registry.register("Multiply")
def _np_multiply(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.multiply(*args, **kwargs)


@numpy_eager_registry.register("TrueDivide")
def _np_true_divide(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.divide(*args, **kwargs)


@numpy_eager_registry.register("Exp")
def _np_exp(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.exp(*args, **kwargs)


@numpy_eager_registry.register("Log")
def _np_log(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.log(*args, **kwargs)


@numpy_eager_registry.register("NanToNum")
def _np_nan_to_num(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.nan_to_num(*args, **kwargs)


@numpy_eager_registry.register("Signbit")
def _np_signbit(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.signbit(*args, **kwargs)


@numpy_eager_registry.register("Frexp")
def _np_frexp(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.frexp(*args, **kwargs)


@numpy_eager_registry.register("Maximum")
def _np_maximum(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.maximum(*args, **kwargs)


@numpy_eager_registry.register("Minimum")
def _np_minimum(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.minimum(*args, **kwargs)


@numpy_eager_registry.register("BitwiseAnd")
def _np_bitwise_and(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.bitwise_and(*args, **kwargs)


@numpy_eager_registry.register("BitwiseOr")
def _np_bitwise_or(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.bitwise_or(*args, **kwargs)


@numpy_eager_registry.register("BitwiseXor")
def _np_bitwise_xor(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.bitwise_xor(*args, **kwargs)


@numpy_eager_registry.register("BitwiseNot")
def _np_bitwise_not(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.bitwise_not(*args, **kwargs)


@numpy_eager_registry.register("LeftShift")
def _np_left_shift(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.left_shift(*args, **kwargs)


@numpy_eager_registry.register("RightShift")
def _np_right_shift(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.right_shift(*args, **kwargs)


@numpy_eager_registry.register("Erf")
def _np_erf(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    import scipy.special

    return scipy.special.erf(*args, **kwargs)


@numpy_eager_registry.register("Erfc")
def _np_erfc(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    import scipy.special

    return scipy.special.erfc(*args, **kwargs)


@numpy_eager_registry.register("Erfinv")
def _np_erfinv(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    import scipy.special

    return scipy.special.erfinv(*args, **kwargs)


@numpy_eager_registry.register("Exp2")
def _np_exp2(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.exp2(*args, **kwargs)


@numpy_eager_registry.register("Expm1")
def _np_expm1(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.expm1(*args, **kwargs)


@numpy_eager_registry.register("Log1p")
def _np_log1p(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.log1p(*args, **kwargs)


@numpy_eager_registry.register("Log2")
def _np_log2(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.log2(*args, **kwargs)


@numpy_eager_registry.register("Log10")
def _np_log10(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.log10(*args, **kwargs)


@numpy_eager_registry.register("Logaddexp")
def _np_logaddexp(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.logaddexp(*args, **kwargs)


@numpy_eager_registry.register("Logaddexp2")
def _np_logaddexp2(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.logaddexp2(*args, **kwargs)


@numpy_eager_registry.register("Round")
def _np_round(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.round(*args, **kwargs)


@numpy_eager_registry.register("Isnan")
def _np_isnan(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.isnan(*args, **kwargs)


@numpy_eager_registry.register("Isinf")
def _np_isinf(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.isinf(*args, **kwargs)


@numpy_eager_registry.register("Isfinite")
def _np_isfinite(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.isfinite(*args, **kwargs)


@numpy_eager_registry.register("Clip")
def _np_clip(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.clip(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("Amax")
def _np_amax(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.amax(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("Amin")
def _np_amin(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.amin(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("Logit")
def _np_logit(
    backend_module: object, x: object, eps: object = None, *args: object, **kwargs: object
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
        eps: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.log(x / (1.0 - x))


@numpy_eager_registry.register("Xlogy")
def _np_xlogy(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    from ml_switcheroo_compiler.backends.numpy.eager.math_extras import _xlogy

    return _xlogy(*args, **kwargs)


@numpy_eager_registry.register("Mvlgamma")
def _np_mvlgamma(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    from ml_switcheroo_compiler.backends.numpy.eager.shape import _mvlgamma

    return _mvlgamma(*args, **kwargs)


@numpy_eager_registry.register("Pmean")
def _np_pmean(
    backend_module: object, x: object, axis_name: object, *args: object, **kwargs: object
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
        axis_name: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return x


@numpy_eager_registry.register("Logsumexp")
def _np_logsumexp(
    backend_module: object,
    a: object,
    axis: object = None,
    keepdims: bool = False,
    **kwargs: object,
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        a: Arg.
        axis: Arg.
        keepdims: Arg.
        kwargs: Arg.
    """
    import scipy.special

    return scipy.special.logsumexp(a, axis=axis, keepdims=keepdims)


@numpy_eager_registry.register("SegmentSum")
def _np_segment_sum(
    backend_module: object,
    data: object,
    segment_ids: object,
    num_segments: object = None,
    **kwargs: object,
) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        data: Arg.
        segment_ids: Arg.
        num_segments: Arg.
        kwargs: Arg.
    """
    import numpy as np

    num_segments = num_segments if num_segments is not None else np.max(segment_ids) + 1
    out = np.zeros((num_segments,) + data.shape[1:], dtype=data.dtype)
    np.add.at(out, segment_ids, data)
    return out


@numpy_eager_registry.register("Psum")
def _np_psum(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return args[0]


@numpy_eager_registry.register("Log1P")
def _np_log1p2(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.log1p(*args, **kwargs)


@numpy_eager_registry.register("Rsqrt")
def _np_rsqrt(backend_module: object, x: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
        args: Arg.
        kwargs: Arg.
    """
    import numpy as np

    # Robust rsqrt to avoid division by zero warnings/NaNs in edge cases if required
    # But usually just `np.reciprocal(np.sqrt(x))` or `1.0 / np.sqrt(x)` works.
    # To be extremely robust to 0 or negatives producing uncatchable warnings:
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / np.sqrt(x)
