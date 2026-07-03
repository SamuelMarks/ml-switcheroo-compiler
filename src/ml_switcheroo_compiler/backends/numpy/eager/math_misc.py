"""Math Ops."""

import numpy as np
import scipy.special

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy
from ml_switcheroo_compiler.backends.numpy.eager.shape import _mvlgamma


@numpy_eager_registry.register("Xlogy")
def _np_xlogy(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return _xlogy(*args, **kwargs)


@numpy_eager_registry.register("Mvlgamma")
def _np_mvlgamma(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return _mvlgamma(*args, **kwargs)


@numpy_eager_registry.register("Pmean")
def _np_pmean(backend_module: object, x: object, axis_name: object, *args: object, **kwargs: object) -> object:
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
    # Robust rsqrt to avoid division by zero warnings/NaNs in edge cases if required
    # But usually just `np.reciprocal(np.sqrt(x))` or `1.0 / np.sqrt(x)` works.
    # To be extremely robust to 0 or negatives producing uncatchable warnings:
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / np.sqrt(x)


@numpy_eager_registry.register("TruncateDiv")
def _np_truncate_div(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    x, y = args
    return np.trunc(np.divide(x, y))


@numpy_eager_registry.register("TruncateMod")
def _np_truncate_mod(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    x, y = args
    return np.fmod(x, y)


@numpy_eager_registry.register("Betainc")
def _np_betainc(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    return scipy.special.betainc(*args, **kwargs)


@numpy_eager_registry.register("BesselI0e")
def _np_bessel_i0e(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    return scipy.special.i0e(*args, **kwargs)


@numpy_eager_registry.register("BesselI1e")
def _np_bessel_i1e(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    return scipy.special.i1e(*args, **kwargs)


@numpy_eager_registry.register("Clz")
def _np_clz(backend_module: object, x: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    x_arr = np.asarray(x)
    if not np.issubdtype(x_arr.dtype, np.integer):
        raise TypeError("Clz requires integer inputs.")

    bit_width = x_arr.itemsize * 8

    @np.vectorize
    def _clz_scalar(val: object) -> object:
        """Function docstring."""
        val = int(val)
        if val < 0:
            val = val & ((1 << bit_width) - 1)
        return bit_width - val.bit_length()

    res = _clz_scalar(x_arr)
    return res.astype(x_arr.dtype)


@numpy_eager_registry.register("PopulationCount")
def _np_population_count(backend_module: object, x: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    x_arr = np.asarray(x)
    return np.array([bin(n).count("1") for n in x_arr.flat]).reshape(x_arr.shape)


@numpy_eager_registry.register("BitcastConvertType")
def _np_bitcast_convert_type(backend_module: object, x: object, new_dtype: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
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
    """Function docstring."""
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
    """Function docstring."""
    keys_arr = np.asarray(keys)
    values_arr = np.asarray(values)
    idx = np.argsort(keys_arr, axis=axis)
    sorted_keys = np.take_along_axis(keys_arr, idx, axis=axis)
    sorted_values = np.take_along_axis(values_arr, idx, axis=axis)
    return sorted_keys, sorted_values
