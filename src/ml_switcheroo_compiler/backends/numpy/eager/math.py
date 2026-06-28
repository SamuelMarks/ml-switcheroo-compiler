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


@numpy_eager_registry.register("TruncateDiv")
def _np_truncate_div(backend_module: object, *args: object, **kwargs: object) -> object:
    import numpy as np

    x, y = args
    return np.trunc(np.divide(x, y))


@numpy_eager_registry.register("TruncateMod")
def _np_truncate_mod(backend_module: object, *args: object, **kwargs: object) -> object:
    import numpy as np

    x, y = args
    return np.fmod(x, y)


@numpy_eager_registry.register("Igamma")
def _np_igamma(backend_module: object, *args: object, **kwargs: object) -> object:
    import scipy.special

    return scipy.special.gammainc(*args, **kwargs)


@numpy_eager_registry.register("Igammac")
def _np_igammac(backend_module: object, *args: object, **kwargs: object) -> object:
    import scipy.special

    return scipy.special.gammaincc(*args, **kwargs)


@numpy_eager_registry.register("Betainc")
def _np_betainc(backend_module: object, *args: object, **kwargs: object) -> object:
    import scipy.special

    return scipy.special.betainc(*args, **kwargs)


@numpy_eager_registry.register("Polygamma")
def _np_polygamma(backend_module: object, *args: object, **kwargs: object) -> object:
    import scipy.special

    return scipy.special.polygamma(*args, **kwargs)


@numpy_eager_registry.register("Zeta")
def _np_zeta(backend_module: object, *args: object, **kwargs: object) -> object:
    import scipy.special

    return scipy.special.zeta(*args, **kwargs)


@numpy_eager_registry.register("BesselI0e")
def _np_bessel_i0e(backend_module: object, *args: object, **kwargs: object) -> object:
    import scipy.special

    return scipy.special.i0e(*args, **kwargs)


@numpy_eager_registry.register("BesselI1e")
def _np_bessel_i1e(backend_module: object, *args: object, **kwargs: object) -> object:
    import scipy.special

    return scipy.special.i1e(*args, **kwargs)


@numpy_eager_registry.register("Clz")
def _np_clz(backend_module: object, x: object, *args: object, **kwargs: object) -> object:
    import numpy as np

    # A bit hacky but works for integer types up to 64 bit
    x_arr = np.asarray(x)
    if x_arr.dtype == np.uint32 or x_arr.dtype == np.int32:
        return 32 - np.ceil(np.log2(np.maximum(x_arr, 1) + 0.5)).astype(int)
    elif x_arr.dtype == np.uint64 or x_arr.dtype == np.int64:
        return 64 - np.ceil(np.log2(np.maximum(x_arr, 1) + 0.5)).astype(int)
    elif x_arr.dtype == np.uint8 or x_arr.dtype == np.int8:
        return 8 - np.ceil(np.log2(np.maximum(x_arr, 1) + 0.5)).astype(int)
    else:
        return 32 - np.ceil(np.log2(np.maximum(x_arr, 1) + 0.5)).astype(int)


@numpy_eager_registry.register("PopulationCount")
def _np_population_count(
    backend_module: object, x: object, *args: object, **kwargs: object
) -> object:
    import numpy as np

    x_arr = np.asarray(x)
    return np.array([bin(n).count("1") for n in x_arr.flat]).reshape(x_arr.shape)


@numpy_eager_registry.register("BitcastConvertType")
def _np_bitcast_convert_type(
    backend_module: object, x: object, new_dtype: object, *args: object, **kwargs: object
) -> object:
    import numpy as np

    dt = getattr(np, str(new_dtype).split(".")[-1], np.float32)
    return np.asarray(x).view(dt)


@numpy_eager_registry.register("ReducePrecision")
def _np_reduce_precision(
    backend_module: object,
    x: object,
    exponent_bits: int,
    mantissa_bits: int,
    *args: object,
    **kwargs: object,
) -> object:
    import numpy as np

    # Very crude approximation: just convert to float16 and back
    return np.asarray(x).astype(np.float16).astype(np.asarray(x).dtype)


@numpy_eager_registry.register("SortKeyVal")
def _np_sort_key_val(
    backend_module: object,
    keys: object,
    values: object,
    axis: int = -1,
    *args: object,
    **kwargs: object,
) -> object:
    import numpy as np

    keys_arr = np.asarray(keys)
    values_arr = np.asarray(values)
    idx = np.argsort(keys_arr, axis=axis)
    sorted_keys = np.take_along_axis(keys_arr, idx, axis=axis)
    sorted_values = np.take_along_axis(values_arr, idx, axis=axis)
    return sorted_keys, sorted_values


@numpy_eager_registry.register("Angle")
def _np_angle(backend_module: object, x: object, **kwargs: object) -> object:
    import numpy as np

    return np.angle(x)
