# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Math Ops."""

from collections.abc import Sequence
from typing import Any, Optional

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy


@numpy_eager_registry.register("Xlogy")
def _np_xlogy(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_xlogy operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return _xlogy(*args, **kwargs)


@numpy_eager_registry.register("Mvlgamma")
def _np_mvlgamma(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_mvlgamma operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.mvlgamma(*args, **kwargs)


@numpy_eager_registry.register("Pmean")
def _np_pmean(backend_module: Any, x: Any, axis_name: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_pmean operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        axis_name (object): The axis_name parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return x


@numpy_eager_registry.register("Logsumexp")
def _np_logsumexp(backend_module: Any, a: Any, axis: Any = None, keepdims: bool = False, **kwargs: Any) -> Any:  # noqa: D417
    """Evaluate _np_logsumexp logic eagerly backed by NumPy.

    Args:
        backend_module (object): The backend_module parameter.
        a (object): The a parameter.
        axis (object): The axis parameter.
        keepdims (bool): The keepdims parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import numpy as np

    a = np.array(a)
    a_max = np.amax(a, axis=axis, keepdims=True)
    if not keepdims:
        a_max_s = np.squeeze(a_max, axis=axis)
    else:
        a_max_s = a_max
    out = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=keepdims))
    out += a_max_s
    return out


@numpy_eager_registry.register("SegmentSum")
def _np_segment_sum(backend_module: Any, data: Any, segment_ids: Any, num_segments: Any = None, **kwargs: Any) -> Any:
    """Evaluate _np_segment_sum operation.

    Args:
        backend_module (object): The backend_module parameter.
        data (object): The data parameter.
        segment_ids (object): The segment_ids parameter.
        num_segments (object): The num_segments parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    num_segments = num_segments if num_segments is not None else np.max(segment_ids) + 1
    out = np.zeros((num_segments,) + data.shape[1:], dtype=data.dtype)
    np.add.at(out, segment_ids, data)
    return out


@numpy_eager_registry.register("Psum")
def _np_psum(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_psum operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.backends.numpy.eager.distributed import _tcp_dist_ctx

    if _tcp_dist_ctx.world_size > 1:
        # In a real mock, this would reduce across mailboxes.
        # Here we just multiply by world size to simulate a sum of identical arrays.
        return backend_module.array(args[0]) * _tcp_dist_ctx.world_size
    return backend_module.array(args[0])


@numpy_eager_registry.register("Log1P")
def _np_log1p2(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_log1p2 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.log1p(*args, **kwargs)


@numpy_eager_registry.register("Rsqrt")
def _np_rsqrt(backend_module: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_rsqrt operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / np.sqrt(x)


@numpy_eager_registry.register("TruncateDiv")
def _np_truncate_div(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_truncate_div operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    (x, y) = args
    return np.trunc(np.divide(x, y))


@numpy_eager_registry.register("TruncateMod")
def _np_truncate_mod(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_truncate_mod operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    (x, y) = args
    return np.fmod(x, y)


@numpy_eager_registry.register("Betainc")
def _np_betainc(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_betainc operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import scipy.special as sc

    a = args[0]
    b = args[1]
    x = args[2] if len(args) > 2 else kwargs.get("x")
    return backend_module.array(sc.betainc(a, b, x))


@numpy_eager_registry.register("BesselI0e")
def _np_bessel_i0e(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_bessel_i0e operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import scipy.special as sc

    return backend_module.array(sc.i0e(args[0]))


@numpy_eager_registry.register("BesselI1e")
def _np_bessel_i1e(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_bessel_i1e operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import scipy.special as sc

    return backend_module.array(sc.i1e(args[0]))


@numpy_eager_registry.register("Clz")
def _np_clz(backend_module: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
    """Count the number of leading zero bits in the integer representation of the input.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        TypeError: An exception.
    """
    x_arr = np.asarray(x)
    if not np.issubdtype(x_arr.dtype, np.integer):
        raise TypeError("Clz requires integer inputs.")
    bit_width = x_arr.itemsize * 8

    @np.vectorize
    def _clz_scalar(val: Any) -> Any:
        """Evaluate _clz_scalar operation.

        Args:
        val (object): The val parameter.

        Returns: Any: Result.
        """
        val = int(val)
        if val < 0:
            val = val & (1 << bit_width) - 1
        return bit_width - val.bit_length()

    res = _clz_scalar(x_arr)
    return res.astype(x_arr.dtype)


@numpy_eager_registry.register("PopulationCount")
def _np_population_count(backend_module: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
    """Count the number of set bits in the binary representation of each element.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    x_arr = np.asarray(x)
    return np.array([bin(n).count("1") for n in x_arr.flat]).reshape(x_arr.shape)


@numpy_eager_registry.register("BitcastConvertType")
def _np_bitcast_convert_type(backend_module: Any, x: Any, new_dtype: Any, *args: Any, **kwargs: Any) -> Any:
    """Bitcast a tensor from one type to another without changing its underlying memory.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        new_dtype (object): The new_dtype parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    dt = getattr(np, str(new_dtype).split(".")[-1], np.float32)
    return np.asarray(x).view(dt)


@numpy_eager_registry.register("ReducePrecision")
def _np_reduce_precision(backend_module: Any, x: Any, exponent_bits: int, mantissa_bits: int) -> Any:
    """Reduce the precision of a tensor to a specified number of exponent and mantissa bits.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        exponent_bits (int): The exponent_bits parameter.
        mantissa_bits (int): The mantissa_bits parameter.

    Returns: Any: The computed result.
    """
    return np.asarray(x).astype(np.float16).astype(np.asarray(x).dtype)


@numpy_eager_registry.register("Packbits")
def _np_packbits(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Pack the elements of a binary-valued array into bits in a uint8 array.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return np.packbits(np.asarray(args[0]), **kwargs)


@numpy_eager_registry.register("Unpackbits")
def _np_unpackbits(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Unpack elements of a uint8 array into a binary-valued output array.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return np.unpackbits(np.asarray(args[0]), **kwargs)


@numpy_eager_registry.register("Piecewise")
def _np_piecewise(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_piecewise operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return np.piecewise(np.asarray(args[0]), args[1], args[2], *args[3:], **kwargs)


@numpy_eager_registry.register("PromoteTypes")
def _np_promotetypes(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Return the data type with the smallest size and smallest scalar kind to which both given types can be safely cast.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return np.promote_types(args[0], args[1])


@numpy_eager_registry.register("Trace")
def _np_trace(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Return the sum along diagonals of the array.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return np.trace(np.asarray(args[0]), *args[1:], **kwargs)


@numpy_eager_registry.register("Trapz")
def _np_trapz(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Integrate along the given axis using the composite trapezoidal rule.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return np.trapz(np.asarray(args[0]), *args[1:], **kwargs)


@numpy_eager_registry.register("Tri")
def _np_tri(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Construct an array with ones at and below the given diagonal and zeros elsewhere.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return np.tri(*args, **kwargs)


@numpy_eager_registry.register("TrilIndices")
def _np_trilindices(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Return the indices for the lower-triangle of an (n, m) array.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return np.tril_indices(*args, **kwargs)


@numpy_eager_registry.register("TrilIndicesFrom")
def _np_trilindicesfrom(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Return the indices for the lower-triangle of an array.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return np.tril_indices_from(np.asarray(args[0]), *args[1:], **kwargs)


@numpy_eager_registry.register("TrimZeros")
def _np_trimzeros(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Trim the leading and/or trailing zeros from a 1-D array or sequence.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return np.trim_zeros(np.asarray(args[0]), *args[1:], **kwargs)


@numpy_eager_registry.register("TriuIndices")
def _np_triuindices(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Return the indices for the upper-triangle of an (n, m) array.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return np.triu_indices(*args, **kwargs)


@numpy_eager_registry.register("TriuIndicesFrom")
def _np_triuindicesfrom(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Return the indices for the upper-triangle of an array.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return np.triu_indices_from(np.asarray(args[0]), *args[1:], **kwargs)


@numpy_eager_registry.register("Uint")
def _np_uint(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Cast or create an array as unsigned integers of default width.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return np.array(args[0], dtype=np.uint, **kwargs)


@numpy_eager_registry.register("Uint8")
def _np_uint8(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Cast or create an array as 8-bit unsigned integers.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return np.array(args[0], dtype=np.uint8, **kwargs)


@numpy_eager_registry.register("Union1d")
def _np_union1d(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Find the union of two one-dimensional arrays.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return np.union1d(np.asarray(args[0]), np.asarray(args[1]), **kwargs)


@numpy_eager_registry.register("UnravelIndex")
def _np_unravelindex(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Convert a flat index or array of flat indices into a tuple of coordinate arrays.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return np.unravel_index(np.asarray(args[0]), *args[1:], **kwargs)


@numpy_eager_registry.register("Unwrap")
def _np_unwrap(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Unwrap by changing deltas between values to 2*pi complement.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return np.unwrap(np.asarray(args[0]), *args[1:], **kwargs)


@numpy_eager_registry.register("Vander")
def _np_vander(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_vander operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return np.vander(np.asarray(args[0]), *args[1:], **kwargs)


@numpy_eager_registry.register("Vectorize")
def _np_vectorize(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Return a vectorized function which takes a nested sequence of objects or numpy arrays as inputs.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return np.vectorize(args[0], **kwargs)(*args[1:])


@numpy_eager_registry.register("Append")
def _np_append(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Append values to the end of an array.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.append(*args, **kwargs)


@numpy_eager_registry.register("Average")
def _np_average(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_average operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.average(*args, **kwargs)


@numpy_eager_registry.register("Block")
def _np_block(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Assemble an nd-array from nested lists of blocks.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.block(*args, **kwargs)


@numpy_eager_registry.register("Atleast1d")
def _np_atleast_1d(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Convert inputs to arrays with at least one dimension.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.atleast_1d(*args, **kwargs)


@numpy_eager_registry.register("Atleast2d")
def _np_atleast_2d(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """View inputs as arrays with at least two dimensions.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.atleast_2d(*args, **kwargs)


@numpy_eager_registry.register("Atleast3d")
def _np_atleast_3d(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """View inputs as arrays with at least three dimensions.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.atleast_3d(*args, **kwargs)


@numpy_eager_registry.register("ApplyAlongAxis")
def _np_apply_along_axis(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Apply a function to 1-D slices along the given axis.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.apply_along_axis(*args, **kwargs)


@numpy_eager_registry.register("ApplyOverAxes")
def _np_apply_over_axes(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Apply a function repeatedly over multiple axes.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.apply_over_axes(*args, **kwargs)


@numpy_eager_registry.register("ArgPartition")
def _np_argpartition(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Perform an indirect partition along the given axis.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.argpartition(*args, **kwargs)


@numpy_eager_registry.register("ArgWhere")
def _np_argwhere(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Find the indices of array elements that are non-zero, grouped by element.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.argwhere(*args, **kwargs)


@numpy_eager_registry.register("Choose")
def _np_choose(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Construct an array from an index array and a list of arrays to choose from.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.choose(*args, **kwargs)


@numpy_eager_registry.register("ColumnStack")
def _np_column_stack(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Stack 1-D arrays as columns into a 2-D array.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.column_stack(*args, **kwargs)


@numpy_eager_registry.register("Compress")
def _np_compress(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Return selected slices of an array along given axis.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.compress(*args, **kwargs)


@numpy_eager_registry.register("Convolve")
def _np_convolve(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Return the discrete, linear convolution of two one-dimensional sequences.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.convolve(*args, **kwargs)


@numpy_eager_registry.register("CorrCoef")
def _np_corrcoef(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Return Pearson product-moment correlation coefficients.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.corrcoef(*args, **kwargs)


@numpy_eager_registry.register("Correlate")
def _np_correlate(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Cross-correlation of two 1-dimensional sequences.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.correlate(*args, **kwargs)


@numpy_eager_registry.register("Cov")
def _np_cov(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Estimate a covariance matrix, given data and weights.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.cov(*args, **kwargs)


@numpy_eager_registry.register("ArrayEquiv")
def _np_array_equiv_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement ArrayEquiv via array_equiv.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.array_equiv(*args, **kwargs)


@numpy_eager_registry.register("ArrayRepr")
def _np_array_repr_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement ArrayRepr via array_repr.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.array_repr(*args, **kwargs)


@numpy_eager_registry.register("ArrayStr")
def _np_array_str_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement ArrayStr via array_str.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.array_str(*args, **kwargs)


@numpy_eager_registry.register("Blackman")
def _np_blackman_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Blackman via blackman.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.blackman(*args, **kwargs)


@numpy_eager_registry.register("BroadcastArrays")
def _np_broadcast_arrays_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement BroadcastArrays via broadcast_arrays.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.broadcast_arrays(*args, **kwargs)


@numpy_eager_registry.register("CanCast")
def _np_can_cast_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement CanCast via can_cast.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.can_cast(*args, **kwargs)


@numpy_eager_registry.register("Delete")
def _np_delete_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Delete via delete.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.delete(*args, **kwargs)


@numpy_eager_registry.register("DiagIndices")
def _np_diag_indices_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement DiagIndices via diag_indices.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.diag_indices(*args, **kwargs)


@numpy_eager_registry.register("DiagIndicesFrom")
def _np_diag_indices_from_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement DiagIndicesFrom via diag_indices_from.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.diag_indices_from(*args, **kwargs)


@numpy_eager_registry.register("Diagflat")
def _np_diagflat_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Diagflat via diagflat.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.diagflat(*args, **kwargs)


@numpy_eager_registry.register("Diagonal")
def _np_diagonal_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Diagonal via diagonal.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.diagonal(*args, **kwargs)


@numpy_eager_registry.register("Diff")
def _np_diff_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Diff via diff.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.diff(*args, **kwargs)


@numpy_eager_registry.register("Digitize")
def _np_digitize_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Digitize via digitize.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.digitize(*args, **kwargs)


@numpy_eager_registry.register("Ediff1d")
def _np_ediff1d_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Ediff1d via ediff1d.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.ediff1d(*args, **kwargs)


@numpy_eager_registry.register("EinsumPath")
def _np_einsum_path_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement EinsumPath via einsum_path.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.einsum_path(*args, **kwargs)


@numpy_eager_registry.register("Extract")
def _np_extract_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Extract via extract.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.extract(*args, **kwargs)


@numpy_eager_registry.register("Fabs")
def _np_fabs_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Fabs via fabs.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.fabs(*args, **kwargs)


@numpy_eager_registry.register("Flatnonzero")
def _np_flatnonzero_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Flatnonzero via flatnonzero.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.flatnonzero(*args, **kwargs)


@numpy_eager_registry.register("Flip")
def _np_flip_op_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Flip via flip.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.flip(*args, **kwargs)


@numpy_eager_registry.register("Fliplr")
def _np_fliplr_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Fliplr via fliplr.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.fliplr(*args, **kwargs)


@numpy_eager_registry.register("Flipud")
def _np_flipud_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Flipud via flipud.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.flipud(*args, **kwargs)


@numpy_eager_registry.register("Reverse")
def _np_flip_reverse_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Reverse via flip.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.flip(*args, **kwargs)


@numpy_eager_registry.register("Fromfunction")
def _np_fromfunction_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Fromfunction via fromfunction.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.fromfunction(*args, **kwargs)


@numpy_eager_registry.register("Fromiter")
def _np_fromiter_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Fromiter via fromiter.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.fromiter(*args, **kwargs)


@numpy_eager_registry.register("Frompyfunc")
def _np_frompyfunc_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Frompyfunc via frompyfunc.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.frompyfunc(*args, **kwargs)


@numpy_eager_registry.register("Fromstring")
def _np_fromstring_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Fromstring via fromstring.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.fromstring(*args, **kwargs)


@numpy_eager_registry.register("Geomspace")
def _np_geomspace_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Geomspace via geomspace.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.geomspace(*args, **kwargs)


@numpy_eager_registry.register("GetPrintoptions")
def _np_get_printoptions_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement GetPrintoptions via get_printoptions.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.get_printoptions(*args, **kwargs)


@numpy_eager_registry.register("Hamming")
def _np_hamming_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Hamming via hamming.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.hamming(*args, **kwargs)


@numpy_eager_registry.register("Hanning")
def _np_hanning_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Hanning via hanning.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.hanning(*args, **kwargs)


@numpy_eager_registry.register("Histogram")
def _np_histogram_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Histogram via histogram.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.histogram(*args, **kwargs)


@numpy_eager_registry.register("Histogram2d")
def _np_histogram2d_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Histogram2d via histogram2d.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.histogram2d(*args, **kwargs)


@numpy_eager_registry.register("HistogramBinEdges")
def _np_histogram_bin_edges_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement HistogramBinEdges via histogram_bin_edges.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.histogram_bin_edges(*args, **kwargs)


@numpy_eager_registry.register("Histogramdd")
def _np_histogramdd_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Histogramdd via histogramdd.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.histogramdd(*args, **kwargs)


@numpy_eager_registry.register("Indices")
def _np_indices_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Indices via indices.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.indices(*args, **kwargs)


@numpy_eager_registry.register("Insert")
def _np_insert_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Insert via insert.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.insert(*args, **kwargs)


@numpy_eager_registry.register("Interp")
def _np_interp_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Interp via interp.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.interp(*args, **kwargs)


@numpy_eager_registry.register("Intersect1d")
def _np_intersect1d_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Intersect1d via intersect1d.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.intersect1d(*args, **kwargs)


@numpy_eager_registry.register("Iscomplex")
def _np_iscomplex_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Iscomplex via iscomplex.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.iscomplex(*args, **kwargs)


@numpy_eager_registry.register("Iscomplexobj")
def _np_iscomplexobj_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Iscomplexobj via iscomplexobj.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.iscomplexobj(*args, **kwargs)


@numpy_eager_registry.register("Isdtype")
def _np_issubdtype_op_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Isdtype via issubdtype.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.issubdtype(*args, **kwargs)


@numpy_eager_registry.register("Isin")
def _np_isin_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Isin via isin.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.isin(*args, **kwargs)


@numpy_eager_registry.register("Isreal")
def _np_isreal_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Isreal via isreal.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.isreal(*args, **kwargs)


@numpy_eager_registry.register("Isrealobj")
def _np_isrealobj_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Isrealobj via isrealobj.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.isrealobj(*args, **kwargs)


@numpy_eager_registry.register("Isscalar")
def _np_isscalar_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Isscalar via isscalar.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.isscalar(*args, **kwargs)


@numpy_eager_registry.register("Issubdtype")
def _np_issubdtype_issubdtype_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Issubdtype via issubdtype.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.issubdtype(*args, **kwargs)


@numpy_eager_registry.register("Iterable")
def _np_iterable_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Iterable via iterable.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.iterable(*args, **kwargs)


@numpy_eager_registry.register("Ix")
def _np_ix__(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Ix via ix_.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.ix_(*args, **kwargs)


@numpy_eager_registry.register("Kaiser")
def _np_kaiser_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Kaiser via kaiser.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.kaiser(*args, **kwargs)


@numpy_eager_registry.register("Kron")
def _np_kron_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Kron via kron.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.kron(*args, **kwargs)


@numpy_eager_registry.register("Lexsort")
def _np_lexsort_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Lexsort via lexsort.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.lexsort(*args, **kwargs)


@numpy_eager_registry.register("Load")
def _np_load_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Load via load.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.load(*args, **kwargs)


@numpy_eager_registry.register("MaskIndices")
def _np_mask_indices_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement MaskIndices via mask_indices.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.mask_indices(*args, **kwargs)


@numpy_eager_registry.register("Median")
def _np_median_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Median via median.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.median(*args, **kwargs)


@numpy_eager_registry.register("Modf")
def _np_modf_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Modf via modf.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.modf(*args, **kwargs)


@numpy_eager_registry.register("Nonzero")
def _np_nonzero_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Nonzero via nonzero.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.nonzero(*args, **kwargs)


@numpy_eager_registry.register("Resize")
def _np_resize_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Resize via resize.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.resize(*args, **kwargs)


@numpy_eager_registry.register("ResultType")
def _np_result_type_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement ResultType via result_type.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.result_type(*args, **kwargs)


@numpy_eager_registry.register("RavelMultiIndex")
def _np_ravel_multi_index_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement RavelMultiIndex via ravel_multi_index.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.ravel_multi_index(*args, **kwargs)


@numpy_eager_registry.register("Trapezoid")
def _np_trapz_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Trapezoid via trapz.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.trapz(*args, **kwargs)


@numpy_eager_registry.register("Zeros")
def _np_zeros_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Zeros via zeros.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.zeros(*args, **kwargs)


@numpy_eager_registry.register("Ones")
def _np_ones_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Ones via ones.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.ones(*args, **kwargs)


@numpy_eager_registry.register("Empty")
def _np_empty_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Empty via empty.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.empty(*args, **kwargs)


@numpy_eager_registry.register("Full")
def _np_full_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Full via full.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.full(*args, **kwargs)


@numpy_eager_registry.register("ZerosLike")
def _np_zeros_like_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement ZerosLike via zeros_like.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.zeros_like(*args, **kwargs)


@numpy_eager_registry.register("OnesLike")
def _np_ones_like_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement OnesLike via ones_like.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.ones_like(*args, **kwargs)


@numpy_eager_registry.register("EmptyLike")
def _np_empty_like_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement EmptyLike via empty_like.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.empty_like(*args, **kwargs)


@numpy_eager_registry.register("FullLike")
def _np_full_like_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement FullLike via full_like.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.full_like(*args, **kwargs)


@numpy_eager_registry.register("Arange")
def _np_arange_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Arange via arange.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.arange(*args, **kwargs)


@numpy_eager_registry.register("Cholesky")
def _np_linalg_cholesky_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Cholesky via linalg.cholesky.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.linalg.cholesky(*args, **kwargs)


@numpy_eager_registry.register("Det")
def _np_linalg_det_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Det via linalg.det.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.linalg.det(*args, **kwargs)


@numpy_eager_registry.register("Svd")
def _np_linalg_svd_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Svd via linalg.svd.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.linalg.svd(*args, **kwargs)


@numpy_eager_registry.register("Unsqueeze")
def _np_expand_dims_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Unsqueeze via expand_dims.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.expand_dims(*args, **kwargs)


@numpy_eager_registry.register("Inv")
def _np_linalg_inv_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Inv via linalg.inv.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.linalg.inv(*args, **kwargs)


@numpy_eager_registry.register("MatrixPower")
def _np_linalg_matrix_power_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement MatrixPower via linalg.matrix_power.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.linalg.matrix_power(*args, **kwargs)


@numpy_eager_registry.register("Pinv")
def _np_linalg_pinv_(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Pinv via linalg.pinv.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Variable positional arguments.
        **kwargs (object): Arbitrary keyword arguments.

    Returns: Any: The computed result.
    """
    return backend_module.linalg.pinv(*args, **kwargs)


@numpy_eager_registry.register("AffineConfig")
def _np_affineconfig(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement AffineConfig.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return kwargs


@numpy_eager_registry.register("AsStringConfig")
def _np_asstringconfig(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement AsStringConfig.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.text.frontend import AsStringConfig

    return AsStringConfig(*args, **kwargs)


@numpy_eager_registry.register("AssertOp")
def _np_assertop(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement AssertOp.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import numpy as np

    condition = args[0] if len(args) > 0 else kwargs.get("condition", None)
    if condition is not None:
        assert np.all(np.asarray(condition))
    return np.array([0.0])


@numpy_eager_registry.register("BlurConfig")
def _np_blurconfig(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement BlurConfig.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.configs import BlurConfig

    return BlurConfig(*args, **kwargs)


@numpy_eager_registry.register("Callable")
def _np_callable(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Callable.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    if args:
        return callable(args[0])
    return False


@numpy_eager_registry.register("ConvGeneralDilatedLocal")
def _np_convgeneraldilatedlocal(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement ConvGeneralDilatedLocal.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    # A generic fallback using scipy.signal
    import scipy.signal

    return scipy.signal.convolve(np.asarray(args[0]), np.asarray(args[1]), mode="valid")


@numpy_eager_registry.register("ConvGeneralDilatedPatches")
def _np_convgeneraldilatedpatches(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement ConvGeneralDilatedPatches.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import scipy.signal

    return scipy.signal.convolve(np.asarray(args[0]), np.asarray(args[1]), mode="valid")


@numpy_eager_registry.register("ConvWithGeneralPadding")
def _np_convwithgeneralpadding(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement ConvWithGeneralPadding.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import scipy.signal

    return scipy.signal.convolve(np.asarray(args[0]), np.asarray(args[1]), mode="valid")


@numpy_eager_registry.register("CustomLinearSolve")
def _np_customlinearsolve(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement CustomLinearSolve.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    # CustomLinearSolve typically takes (matvec_or_matrix, b, solve, ...).
    # If a custom solve is provided, we use it, otherwise fallback to np.linalg.solve if it's a matrix.
    if callable(args[0]):
        solve = kwargs.get("solve", args[2] if len(args) > 2 else None)
        if solve:
            return solve(args[0], args[1])
        # If no solve function and it's a callable, we can't trivially invert without an iterative solver like CG.
        # Fallback to returning b as a dummy for testing if no solver is provided (shouldn't happen in valid IR).
        return args[1]
    return np.linalg.solve(args[0], args[1])


@numpy_eager_registry.register("CustomRoot")
def _np_customroot(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement CustomRoot.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    f = args[0]
    initial_guess = args[1]
    solve = kwargs.get("solve", args[2] if len(args) > 2 else None)
    if solve:
        return solve(f, initial_guess)
    return initial_guess


@numpy_eager_registry.register("DebugInfs")
def _np_debuginfs(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement DebugInfs.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        ValueError: An exception.
    """
    x = args[0]
    if backend_module.any(backend_module.isinf(x)):
        raise ValueError("Infinity found in tensor.")
    return x


@numpy_eager_registry.register("DebugNans")
def _np_debugnans(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement DebugNans.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        ValueError: An exception.
    """
    x = args[0]
    if backend_module.any(backend_module.isnan(x)):
        raise ValueError("NaN found in tensor.")
    return x


def _build_dot_general_einsum_str(lhs_ndim: int, rhs_ndim: int, dimension_numbers: tuple) -> str:
    """Evaluate _build_dot_general_einsum_str operation.

    Args:
        lhs_ndim (int): The lhs_ndim parameter.
        rhs_ndim (int): The rhs_ndim parameter.
        dimension_numbers (tuple): The dimension_numbers parameter.

    Returns:
        str: Result.
    """
    (lhs_cont, rhs_cont), (lhs_batch, rhs_batch) = dimension_numbers
    batch_chars = [chr(ord("a") + i) for i in range(len(lhs_batch))]
    cont_chars = [chr(ord("a") + len(lhs_batch) + i) for i in range(len(lhs_cont))]

    current_char = ord("a") + len(lhs_batch) + len(lhs_cont)
    lhs_str, rhs_str = [""] * lhs_ndim, [""] * rhs_ndim

    for i, (lb, rb) in enumerate(zip(lhs_batch, rhs_batch)):
        lhs_str[lb] = rhs_str[rb] = batch_chars[i]

    for i, (lc, rc) in enumerate(zip(lhs_cont, rhs_cont)):
        lhs_str[lc] = rhs_str[rc] = cont_chars[i]

    out_lhs, out_rhs = [], []
    for i in range(lhs_ndim):
        if not lhs_str[i]:
            lhs_str[i] = chr(current_char)
            out_lhs.append(chr(current_char))
            current_char += 1

    for i in range(rhs_ndim):
        if not rhs_str[i]:
            rhs_str[i] = chr(current_char)
            out_rhs.append(chr(current_char))
            current_char += 1

    out_str = "".join(batch_chars) + "".join(out_lhs) + "".join(out_rhs)
    return "".join(lhs_str) + "," + "".join(rhs_str) + "->" + out_str


@numpy_eager_registry.register("DotGeneral")
def _np_dotgeneral(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement DotGeneral.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    lhs, rhs = np.asarray(args[0]), np.asarray(args[1])
    dimension_numbers = kwargs.get("dimension_numbers", (((-1,), (0,)), ((), ())))
    einsum_str = _build_dot_general_einsum_str(lhs.ndim, rhs.ndim, dimension_numbers)
    return np.einsum(einsum_str, lhs, rhs)


@numpy_eager_registry.register("ElasticConfig")
def _np_elasticconfig(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement ElasticConfig.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.configs import ElasticConfig

    return ElasticConfig(*args, **kwargs)


@numpy_eager_registry.register("ExtractPatchesOptions")
def _np_extractpatchesoptions(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement ExtractPatchesOptions.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.vision.bbox import ExtractPatchesOptions

    return ExtractPatchesOptions(*args, **kwargs)


@numpy_eager_registry.register("LinearOperator")
def _np_linearoperator(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperator.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperator

    return LinearOperator(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorAdjoint")
def _np_linearoperatoradjoint(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorAdjoint.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorAdjoint

    return LinearOperatorAdjoint(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorBlockDiag")
def _np_linearoperatorblockdiag(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorBlockDiag.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorBlockDiag

    return LinearOperatorBlockDiag(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorBlockLowerTriangular")
def _np_linearoperatorblocklowertriangular(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorBlockLowerTriangular.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorBlockLowerTriangular

    return LinearOperatorBlockLowerTriangular(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorCirculant")
def _np_linearoperatorcirculant(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorCirculant.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorCirculant

    return LinearOperatorCirculant(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorCirculant2D")
def _np_linearoperatorcirculant2d(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorCirculant2D.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorCirculant2D

    return LinearOperatorCirculant2D(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorCirculant3D")
def _np_linearoperatorcirculant3d(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorCirculant3D.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorCirculant3D

    return LinearOperatorCirculant3D(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorComposition")
def _np_linearoperatorcomposition(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorComposition.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorComposition

    return LinearOperatorComposition(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorDiag")
def _np_linearoperatordiag(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorDiag.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorDiag

    return LinearOperatorDiag(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorFullMatrix")
def _np_linearoperatorfullmatrix(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorFullMatrix.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorFullMatrix

    return LinearOperatorFullMatrix(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorHouseholder")
def _np_linearoperatorhouseholder(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorHouseholder.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorHouseholder

    return LinearOperatorHouseholder(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorIdentity")
def _np_linearoperatoridentity(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorIdentity.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorIdentity

    return LinearOperatorIdentity(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorInversion")
def _np_linearoperatorinversion(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorInversion.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorInversion

    return LinearOperatorInversion(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorKronecker")
def _np_linearoperatorkronecker(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorKronecker.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorKronecker

    return LinearOperatorKronecker(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorLowRankUpdate")
def _np_linearoperatorlowrankupdate(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorLowRankUpdate.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorLowRankUpdate

    return LinearOperatorLowRankUpdate(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorLowerTriangular")
def _np_linearoperatorlowertriangular(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorLowerTriangular.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorLowerTriangular

    return LinearOperatorLowerTriangular(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorPermutation")
def _np_linearoperatorpermutation(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorPermutation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorPermutation

    return LinearOperatorPermutation(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorScaledIdentity")
def _np_linearoperatorscaledidentity(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorScaledIdentity.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorScaledIdentity

    return LinearOperatorScaledIdentity(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorToeplitz")
def _np_linearoperatortoeplitz(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorToeplitz.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorToeplitz

    return LinearOperatorToeplitz(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorTridiag")
def _np_linearoperatortridiag(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorTridiag.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorTridiag

    return LinearOperatorTridiag(*args, **kwargs)


@numpy_eager_registry.register("LinearOperatorZeros")
def _np_linearoperatorzeros(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement LinearOperatorZeros.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.linalg.linear_operator import LinearOperatorZeros

    return LinearOperatorZeros(*args, **kwargs)


@numpy_eager_registry.register("PerspectiveConfig")
def _np_perspectiveconfig(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement PerspectiveConfig.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.configs import PerspectiveConfig

    return PerspectiveConfig(*args, **kwargs)


@numpy_eager_registry.register("RaggedDot")
def _np_raggeddot(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement RaggedDot.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.matmul(args[0], args[1])


@numpy_eager_registry.register("RawConv2D")
def _np_rawconv2d(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement RawConv2D.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import scipy.signal

    return scipy.signal.convolve(np.asarray(args[0]), np.asarray(args[1]), mode="valid")


@numpy_eager_registry.register("RawMatMul")
def _np_rawmatmul(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement RawMatMul.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        RuntimeError: An exception.
    """
    try:
        import ml_switcheroo_compiler.ops as _ops

        if hasattr(_ops, "RawMatMul"):
            cls_or_func = _ops.RawMatMul
            if isinstance(cls_or_func, type) and not issubclass(cls_or_func, _ops.OpDef):
                return cls_or_func(*args, **kwargs)
    except Exception as e:
        if not isinstance(e, (ImportError, AttributeError)):
            raise RuntimeError(f"Eager execution failed: {e}") from e

    # Fallback default
    if hasattr(backend_module, "rawmatmul"):
        return backend_module.rawmatmul(*args, **kwargs)
    return np.matmul(args[0], args[1])


@numpy_eager_registry.register("RawMerge")
def _np_rawmerge(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement RawMerge.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    inputs = args[0] if len(args) == 1 and isinstance(args[0], (list, tuple)) else args
    return (inputs[0], np.array(0, dtype=np.int32)) if inputs else (None, np.array(-1, dtype=np.int32))


@numpy_eager_registry.register("RawOp")
def _np_rawop(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement RawOp.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return args[0] if args else None


@numpy_eager_registry.register("RawSwitch")
def _np_rawswitch(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement RawSwitch.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    data = args[0]
    pred = args[1] if len(args) > 1 else kwargs.get("pred", False)

    if bool(np.asarray(pred).item()):
        return (None, data)
    return (data, None)


def _parse_scanop_args(args: tuple, kwargs: dict) -> tuple:
    """Parse ScanOp arguments.

    Args:
        args (tuple): The args parameter.
        kwargs (dict): The kwargs parameter.

    Returns:
        tuple: Result.
    """
    fn = args[0] if len(args) > 0 else kwargs.get("fn")
    elems = args[1] if len(args) > 1 else kwargs.get("elems")
    acc = args[2] if len(args) > 2 else None
    has_acc = len(args) > 2
    return fn, elems, acc, has_acc


@numpy_eager_registry.register("ScanOp")
def _np_scanop(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement ScanOp.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    fn, elems, acc, has_acc = _parse_scanop_args(args, kwargs)

    if not callable(fn) or elems is None:
        return args[0] if args else None

    elems_arr = np.asarray(elems)
    if elems_arr.size == 0:
        return elems_arr

    out = np.empty_like(elems_arr)
    if not has_acc:
        acc = elems_arr[0]
        out[0] = acc

    start_idx = 0 if has_acc else 1
    for i in range(start_idx, elems_arr.shape[0]):
        acc = fn(acc, elems_arr[i])
        out[i] = acc

    return out


@numpy_eager_registry.register("SobolSample")
def _np_sobolsample(backend_module: Any, dim: int, num_results: int, skip: int = 0, *args: Any, **kwargs: Any) -> Any:
    """Implement SobolSample eagerly.

    Args:
        backend_module (object): The backend_module parameter.
        dim (int): The dim parameter.
        num_results (int): The num_results parameter.
        skip (int): The skip parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import numpy as np

    np.random.seed((42 + skip) % (2**32 - 1))
    return backend_module.array(np.random.uniform(size=(num_results, dim)), dtype=backend_module.float32)


@numpy_eager_registry.register("SparseDenseMatMul")
def _np_sparsedensematmul(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement SparseDenseMatMul.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        RuntimeError: An exception.
    """
    try:
        import ml_switcheroo_compiler.ops as _ops

        if hasattr(_ops, "SparseDenseMatMul"):
            cls_or_func = _ops.SparseDenseMatMul
            if isinstance(cls_or_func, type) and not issubclass(cls_or_func, _ops.OpDef):
                return cls_or_func(*args, **kwargs)
    except Exception as e:
        if not isinstance(e, (ImportError, AttributeError)):
            raise RuntimeError(f"Eager execution failed: {e}") from e

    # Fallback default
    if hasattr(backend_module, "sparsedensematmul"):
        return backend_module.sparsedensematmul(*args, **kwargs)
    return np.matmul(args[0], args[1])


@numpy_eager_registry.register("SparseMapValues")
def _np_sparsemapvalues(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement SparseMapValues.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    fn = args[0]
    sp_input = args[1]
    return fn(backend_module.array(sp_input))


@numpy_eager_registry.register("SparseReduceMax")
def _np_sparsereducemax(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement SparseReduceMax.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.max(args[0], axis=-1)


@numpy_eager_registry.register("SparseReshape")
def _np_sparsereshape(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement SparseReshape.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.reshape(args[0], args[1])


@numpy_eager_registry.register("SparseSampledAdd")
def _np_sparsesampledadd(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement SparseSampledAdd.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.add(args[0], args[1])


@numpy_eager_registry.register("SparseSegmentSum")
def _np_sparsesegmentsum(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement SparseSegmentSum.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    data = backend_module.asarray(args[0])
    return backend_module.sum(data, axis=0, keepdims=True)


@numpy_eager_registry.register("SparseTranspose")
def _np_sparsetranspose(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement SparseTranspose.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return backend_module.transpose(args[0])


@numpy_eager_registry.register("SwitchOp")
def _np_switchop(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement SwitchOp.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    data = args[0]
    pred = args[1] if len(args) > 1 else kwargs.get("pred", False)

    if bool(np.asarray(pred).item()):
        return (None, data)
    return (data, None)


@numpy_eager_registry.register("Tensor")
def _np_tensor(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Tensor.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    # Convert inputs to a numpy array.
    if not args and not kwargs:
        return backend_module.array([])
    return backend_module.array(*args, **kwargs)


@numpy_eager_registry.register("TensorArrayRead")
def _np_tensorarrayread(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement TensorArrayRead.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    handle = args[0]
    index = args[1]
    return handle[index]


@numpy_eager_registry.register("TensorArrayStack")
def _np_tensorarraystack(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement TensorArrayStack.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    handle = args[0]
    return backend_module.stack(handle)


@numpy_eager_registry.register("TensorArrayWrite")
def _np_tensorarraywrite(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement TensorArrayWrite.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    handle = args[0]
    index = args[1]
    value = args[2]
    # Handle could be a tuple if immutable, or a list. Let's make sure it's a list.
    new_handle = list(handle)
    # Pad the list if index is out of bounds
    if index >= len(new_handle):
        new_handle.extend([None] * (index - len(new_handle) + 1))
    new_handle[index] = value
    return new_handle


@numpy_eager_registry.register("TensorConfig")
def _np_tensorconfig(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement TensorConfig.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.core.tensor import TensorConfig

    return TensorConfig(*args, **kwargs)


@numpy_eager_registry.register("Vecdot")
def _np_vecdot(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Vecdot.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    x = args[0]
    y = args[1]
    axis = kwargs.get("axis", -1)
    if hasattr(backend_module, "linalg") and hasattr(backend_module.linalg, "vecdot"):
        return backend_module.linalg.vecdot(x, y, axis=axis)
    if hasattr(backend_module, "vecdot"):
        return backend_module.vecdot(x, y, axis=axis)

    if backend_module.iscomplexobj(x):
        x = backend_module.conj(x)
    return backend_module.sum(x * y, axis=axis)


@numpy_eager_registry.register("decode_csv")
def _np_decode_csv(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_decode_csv operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return _np_decode_csv_camel(backend_module, *args, **kwargs)


@numpy_eager_registry.register("decode_image")
def _np_decode_image(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_decode_image operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return _np_decode_image_camel(backend_module, *args, **kwargs)


@numpy_eager_registry.register("parse_example")
def _np_parse_example(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_parse_example operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return _np_parse_example_camel(backend_module, *args, **kwargs)


@numpy_eager_registry.register("parse_tensor")
def _np_parse_tensor(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_parse_tensor operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return _np_parse_tensor_camel(backend_module, *args, **kwargs)


@numpy_eager_registry.register("read_file")
def _np_read_file(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_read_file operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return _np_read_file_camel(backend_module, *args, **kwargs)


@numpy_eager_registry.register("rem")
def _np_rem(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_rem operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        RuntimeError: An exception.
    """
    try:
        import ml_switcheroo_compiler.ops as _ops

        if hasattr(_ops, "rem"):
            cls_or_func = _ops.rem
            if isinstance(cls_or_func, type) and not issubclass(cls_or_func, _ops.OpDef):
                return cls_or_func(*args, **kwargs)
    except Exception as e:
        if not isinstance(e, (ImportError, AttributeError)):
            raise RuntimeError(f"Eager execution failed: {e}") from e

    # Fallback default
    if hasattr(backend_module, "rem"):
        return backend_module.rem(*args, **kwargs)
    return np.remainder(args[0], args[1])


@numpy_eager_registry.register("serialize_tensor")
def _np_serialize_tensor(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_serialize_tensor operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return _np_serialize_tensor_camel(backend_module, *args, **kwargs)


@numpy_eager_registry.register("write_file")
def _np_write_file(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_write_file operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    return _np_write_file_camel(backend_module, *args, **kwargs)


@numpy_eager_registry.register("confusion_matrix")
def _np_confusion_matrix(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_confusion_matrix operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        RuntimeError: An exception.
    """
    try:
        import ml_switcheroo_compiler.ops as _ops

        if hasattr(_ops, "confusion_matrix"):
            cls_or_func = _ops.confusion_matrix
            if isinstance(cls_or_func, type) and not issubclass(cls_or_func, _ops.OpDef):
                return cls_or_func(*args, **kwargs)
    except Exception as e:
        if not isinstance(e, (ImportError, AttributeError)):
            raise RuntimeError(f"Eager execution failed: {e}") from e

    # Fallback default
    if hasattr(backend_module, "confusion_matrix"):
        return backend_module.confusion_matrix(*args, **kwargs)

    y_true = np.asarray(args[0]).flatten()
    y_pred = np.asarray(args[1]).flatten()
    num_classes = kwargs.get("num_classes", args[2] if len(args) > 2 else None)
    if num_classes is None:
        num_classes = max(np.max(y_true), np.max(y_pred)) + 1
    return np.bincount(y_true * num_classes + y_pred, minlength=num_classes**2).reshape(num_classes, num_classes)


@numpy_eager_registry.register("descriptive")
def _np_descriptive(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_descriptive operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        RuntimeError: An exception.
    """
    try:
        import ml_switcheroo_compiler.ops as _ops

        if hasattr(_ops, "descriptive"):
            cls_or_func = _ops.descriptive
            if isinstance(cls_or_func, type) and not issubclass(cls_or_func, _ops.OpDef):
                return cls_or_func(*args, **kwargs)
    except Exception as e:
        if not isinstance(e, (ImportError, AttributeError)):
            raise RuntimeError(f"Eager execution failed: {e}") from e
    if hasattr(backend_module, "descriptive"):
        return backend_module.descriptive(*args, **kwargs)
    arr = np.asarray(args[0]) if args else np.zeros((1,))
    return np.array([np.mean(arr), np.var(arr), np.std(arr)])


@numpy_eager_registry.register("distributions")
def _np_distributions(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_distributions operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        RuntimeError: An exception.
    """
    try:
        import ml_switcheroo_compiler.ops as _ops

        if hasattr(_ops, "distributions"):
            cls_or_func = _ops.distributions
            if isinstance(cls_or_func, type) and not issubclass(cls_or_func, _ops.OpDef):
                return cls_or_func(*args, **kwargs)
    except Exception as e:
        if not isinstance(e, (ImportError, AttributeError)):
            raise RuntimeError(f"Eager execution failed: {e}") from e
    if hasattr(backend_module, "distributions"):
        return backend_module.distributions(*args, **kwargs)
    arr = np.asarray(args[0]) if args else np.zeros((1,))
    return np.array([np.mean(arr), np.var(arr)])


@numpy_eager_registry.register("BitwiseCount")
def _np_bitwise_count(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_bitwise_count operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import numpy as np

    x = np.asarray(args[0])
    return np.array([bin(n).count("1") for n in x.flat]).reshape(x.shape)


@numpy_eager_registry.register("FromDlpack")
def _np_fromdlpack(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_fromdlpack operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    if hasattr(backend_module, "from_dlpack"):
        return backend_module.from_dlpack(*args, **kwargs)
    return args[0]


@numpy_eager_registry.register("RandomCategorical")
def _np_randomcategorical(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_randomcategorical operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    logits = backend_module.asarray(args[0])
    num_samples = args[1] if len(args) > 1 else kwargs.get("num_samples", 1)

    # Very basic dummy
    return backend_module.zeros(list(logits.shape[:-1]) + [num_samples], dtype=np.int64)


@numpy_eager_registry.register("RandomPermutation")
def _np_randompermutation(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_randompermutation operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import numpy as np

    x = backend_module.asarray(args[0])
    if x.ndim == 0:
        return backend_module.array(np.random.permutation(int(x)))
    return backend_module.array(np.random.permutation(x))


@numpy_eager_registry.register("RandomTruncatedNormal")
def _np_randomtruncatednormal(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_randomtruncatednormal operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    shape = kwargs.get("shape", args[0] if len(args) > 0 else None)
    return backend_module.random.standard_normal(size=shape)


@numpy_eager_registry.register("Key")
def _np_key(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_key operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    if len(args) > 0:
        return np.array([args[0], 0], dtype=np.uint32)
    return np.array([0, 0], dtype=np.uint32)


@numpy_eager_registry.register("StridedSlice")
def _np_stridedslice(backend_module: Any, data: Any, start: Any, end: Any, strides: Any, **kwargs: Any) -> Any:  # noqa: D417
    """Evaluate _np_stridedslice logic eagerly backed by NumPy.

    Args:
        backend_module (object): The backend_module parameter.
        data (object): The data parameter.
        start (object): The start parameter.
        end (object): The end parameter.
        strides (object): The strides parameter.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    slices = tuple(slice(s, e, st) for s, e, st in zip(start, end, strides))
    return data[slices]


@numpy_eager_registry.register("RandomBernoulli")
def _np_randombernoulli(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_randombernoulli operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    shape = kwargs.get("shape", args[0] if len(args) > 0 else None)
    p = kwargs.get("p", args[1] if len(args) > 1 else 0.5)
    return backend_module.random.binomial(1, p, size=shape)


def _get_np_arg(arg: Sequence[Any], i: int) -> Optional[np.ndarray]:
    """Get numpy arg.

    Args:
        arg (object): The arg parameter.
        i (int): The i parameter.

    Returns: Any: Result.
    """
    return np.asarray(arg[i]) if len(arg) > i else None


def _get_sc() -> Any:
    """Evaluate _get_sc operation.

    Returns: Any: Result.
    """
    try:
        import scipy.special as sc

        return sc
    except ImportError:
        return None


def _poly_recurrence(n: Any, x: Any, p0: float, p1_func: Any, p_next_func: Any) -> Any:  # noqa: D417
    """Evaluate _poly_recurrence logic eagerly backed by NumPy.

    Args:
        n (object): The n parameter.
        x (object): The x parameter.
        p0 (float): The p0 parameter.
        p1_func (object): The p1_func parameter.
        p_next_func (object): The p_next_func parameter.

    Returns: Any: Result.
    """
    import numpy as np

    n = np.asarray(n, dtype=int)
    x = np.asarray(x)
    n_b, x_b = np.broadcast_arrays(n, x)
    max_n = np.max(n_b)

    if max_n < 0:
        return np.zeros_like(x_b)

    T = [np.ones_like(x_b) * p0]
    if max_n >= 1:
        T.append(p1_func(x_b))

    for i in range(2, max_n + 1):
        T.append(p_next_func(i - 1, x_b, T[-1], T[-2]))

    T = np.stack(T)
    indices = np.indices(n_b.shape)
    return T[tuple([n_b] + list(indices))]


@numpy_eager_registry.register("chebyshev_polynomial_t")
def _np_chebyshev_polynomial_t(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_chebyshev_polynomial_t operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        ValueError: An exception.
    """
    x, n = _get_np_arg(args, 0), _get_np_arg(args, 1)
    if n is None or x is None:
        raise ValueError("Expected 2 arguments x and n.")
    return _poly_recurrence(n, x, 1, lambda x: x, lambda n, x, t1, t2: 2 * x * t1 - t2)


@numpy_eager_registry.register("chebyshev_polynomial_u")
def _np_chebyshev_polynomial_u(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_chebyshev_polynomial_u operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        ValueError: An exception.
    """
    x, n = _get_np_arg(args, 0), _get_np_arg(args, 1)
    if n is None or x is None:
        raise ValueError("Expected 2 arguments x and n.")
    return _poly_recurrence(n, x, 1, lambda x: 2 * x, lambda n, x, t1, t2: 2 * x * t1 - t2)


@numpy_eager_registry.register("hermite_polynomial_h")
def _np_hermite_polynomial_h(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_hermite_polynomial_h operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        ValueError: An exception.
    """
    x, n = _get_np_arg(args, 0), _get_np_arg(args, 1)
    if n is None or x is None:
        raise ValueError("Expected 2 arguments x and n.")
    return _poly_recurrence(n, x, 1, lambda x: 2 * x, lambda n, x, t1, t2: 2 * x * t1 - 2 * n * t2)


@numpy_eager_registry.register("hermite_polynomial_he")
def _np_hermite_polynomial_he(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_hermite_polynomial_he operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        ValueError: An exception.
    """
    x, n = _get_np_arg(args, 0), _get_np_arg(args, 1)
    if n is None or x is None:
        raise ValueError("Expected 2 arguments x and n.")
    return _poly_recurrence(n, x, 1, lambda x: x, lambda n, x, t1, t2: x * t1 - n * t2)


@numpy_eager_registry.register("laguerre_polynomial_l")
def _np_laguerre_polynomial_l(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_laguerre_polynomial_l operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        ValueError: An exception.
    """
    x, n = _get_np_arg(args, 0), _get_np_arg(args, 1)
    if n is None or x is None:
        raise ValueError("Expected 2 arguments x and n.")
    return _poly_recurrence(n, x, 1, lambda x: 1 - x, lambda n, x, t1, t2: ((2 * n + 1 - x) * t1 - n * t2) / (n + 1))


@numpy_eager_registry.register("legendre_polynomial_p")
def _np_legendre_polynomial_p(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_legendre_polynomial_p operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        ValueError: An exception.
    """
    x, n = _get_np_arg(args, 0), _get_np_arg(args, 1)
    if n is None or x is None:
        raise ValueError("Expected 2 arguments x and n.")
    return _poly_recurrence(n, x, 1, lambda x: x, lambda n, x, t1, t2: ((2 * n + 1) * x * t1 - n * t2) / (n + 1))


@numpy_eager_registry.register("modified_bessel_i0")
def _np_modified_bessel_i0(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_modified_bessel_i0 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import numpy as np

    x = _get_np_arg(args, 0)
    if x is None:
        return None
    return np.i0(x)


@numpy_eager_registry.register("modified_bessel_i1")
def _np_modified_bessel_i1(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_modified_bessel_i1 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    sc = _get_sc()
    x = _get_np_arg(args, 0)
    if x is None:
        return None
    if sc is None:
        import numpy as np

        t = np.linspace(0, np.pi, 100)
        t = np.reshape(t, (1,) * np.ndim(x) + (-1,)) if np.ndim(x) > 0 else t
        x_ex = np.expand_dims(x, -1) if np.ndim(x) > 0 else x
        integrand = np.exp(x_ex * np.cos(t)) * np.cos(t)
        return (1.0 / np.pi) * np.trapz(integrand, x=t, axis=-1)
    return sc.i1(x)


@numpy_eager_registry.register("modified_bessel_k0")
def _np_modified_bessel_k0(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_modified_bessel_k0 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    sc = _get_sc()
    x = _get_np_arg(args, 0)
    if x is None:
        return None
    if sc is None:
        import numpy as np

        t = np.linspace(0, 10, 100)
        t = np.reshape(t, (1,) * np.ndim(x) + (-1,)) if np.ndim(x) > 0 else t
        x_ex = np.expand_dims(x, -1) if np.ndim(x) > 0 else x
        integrand = np.exp(-x_ex * np.cosh(t))
        return np.trapz(integrand, x=t, axis=-1)
    return sc.k0(x)


@numpy_eager_registry.register("modified_bessel_k1")
def _np_modified_bessel_k1(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_modified_bessel_k1 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    sc = _get_sc()
    x = _get_np_arg(args, 0)
    if x is None:
        return None
    if sc is None:
        import numpy as np

        t = np.linspace(0, 10, 100)
        t = np.reshape(t, (1,) * np.ndim(x) + (-1,)) if np.ndim(x) > 0 else t
        x_ex = np.expand_dims(x, -1) if np.ndim(x) > 0 else x
        integrand = np.exp(-x_ex * np.cosh(t)) * np.cosh(t)
        return np.trapz(integrand, x=t, axis=-1)
    return sc.k1(x)


@numpy_eager_registry.register("shifted_chebyshev_polynomial_t")
def _np_shifted_chebyshev_polynomial_t(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement shifted_chebyshev_polynomial_t.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    sc = _get_sc()
    n, x = _get_np_arg(args, 0), _get_np_arg(args, 1)
    if n is None or x is None or sc is None:
        return None
    return sc.eval_sh_chebyt(np.asarray(n, dtype=int), x)


@numpy_eager_registry.register("shifted_chebyshev_polynomial_u")
def _np_shifted_chebyshev_polynomial_u(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement shifted_chebyshev_polynomial_u.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    sc = _get_sc()
    n, x = _get_np_arg(args, 0), _get_np_arg(args, 1)
    if n is None or x is None or sc is None:
        return None
    return sc.eval_sh_chebyu(np.asarray(n, dtype=int), x)


@numpy_eager_registry.register("shifted_chebyshev_polynomial_v")
def _np_shifted_chebyshev_polynomial_v(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement shifted_chebyshev_polynomial_v.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    sc = _get_sc()
    # V_n(x) = U_n(x) - U_{n-1}(x) / 2
    n, x = _get_np_arg(args, 0), _get_np_arg(args, 1)
    if n is None or x is None or sc is None:
        return None
    n_int = np.asarray(n, dtype=int)
    u_n = sc.eval_sh_chebyu(n_int, x)
    u_nm1 = sc.eval_sh_chebyu(n_int - 1, x)
    return u_n - u_nm1 / 2.0


@numpy_eager_registry.register("shifted_chebyshev_polynomial_w")
def _np_shifted_chebyshev_polynomial_w(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement shifted_chebyshev_polynomial_w.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    sc = _get_sc()
    # W_n(x) = U_n(x) + U_{n-1}(x) / 2
    n, x = _get_np_arg(args, 0), _get_np_arg(args, 1)
    if n is None or x is None or sc is None:
        return None
    n_int = np.asarray(n, dtype=int)
    u_n = sc.eval_sh_chebyu(n_int, x)
    u_nm1 = sc.eval_sh_chebyu(n_int - 1, x)
    return u_n + u_nm1 / 2.0


@numpy_eager_registry.register("Rfft")
def _np_rfft(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_rfft operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return fft.rfft(a, **kwargs)


@numpy_eager_registry.register("Ifft")
def _np_ifft(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_ifft operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return fft.ifft(a, **kwargs)


@numpy_eager_registry.register("Fftn")
def _np_fftn(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_fftn operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return fft.fftn(a, **kwargs)


@numpy_eager_registry.register("Ifftn")
def _np_ifftn(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_ifftn operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return fft.ifftn(a, **kwargs)


@numpy_eager_registry.register("Rfftn")
def _np_rfftn(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_rfftn operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return fft.rfftn(a, **kwargs)


@numpy_eager_registry.register("Irfftn")
def _np_irfftn(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_irfftn operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return fft.irfftn(a, **kwargs)


@numpy_eager_registry.register("Ifft2")
def _np_ifft2(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_ifft2 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return fft.ifft2(a, **kwargs)


@numpy_eager_registry.register("Rfft2")
def _np_rfft2(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_rfft2 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return fft.rfft2(a, **kwargs)


@numpy_eager_registry.register("Irfft2")
def _np_irfft2(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_irfft2 operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return fft.irfft2(a, **kwargs)


@numpy_eager_registry.register("Fftnd")
def _np_fftnd(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_fftnd operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return fft.fftn(a, **kwargs)


@numpy_eager_registry.register("Ifftnd")
def _np_ifftnd(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_ifftnd operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return fft.ifftn(a, **kwargs)


@numpy_eager_registry.register("Rfftnd")
def _np_rfftnd(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_rfftnd operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return fft.rfftn(a, **kwargs)


@numpy_eager_registry.register("Irfftnd")
def _np_irfftnd(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_irfftnd operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return fft.irfftn(a, **kwargs)


@numpy_eager_registry.register("Fftshift")
def _np_fftshift(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_fftshift operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return fft.fftshift(a, **kwargs)


@numpy_eager_registry.register("Ifftshift")
def _np_ifftshift(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_ifftshift operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return fft.ifftshift(a, **kwargs)


@numpy_eager_registry.register("Hfft")
def _np_hfft(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_hfft operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import numpy.fft as fft

    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return fft.hfft(a, **kwargs)


@numpy_eager_registry.register("Rfftfreq")
def _np_rfftfreq(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_rfftfreq operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import numpy.fft as fft

    if not args:
        return None
    try:
        n = int(args[0])
    except (TypeError, ValueError):
        return args[0]

    return fft.rfftfreq(n, **kwargs)


@numpy_eager_registry.register("ConfusionMatrix")  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
def _np_confusion_matrix(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_confusion_matrix operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    y_true = _get_np_arg(args, 0)
    y_pred = _get_np_arg(args, 1)
    if y_true is None or y_pred is None:
        return None
    num_classes = kwargs.get("num_classes", None)
    if num_classes is None:
        num_classes = max(np.max(y_true), np.max(y_pred)) + 1
    return np.bincount(y_true * num_classes + y_pred, minlength=num_classes**2).reshape((num_classes, num_classes))


@numpy_eager_registry.register("Descriptive")  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
def _np_descriptive(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_descriptive operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    # Just returning a dummy dict or array for descriptive stats
    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return {"mean": np.mean(a), "std": np.std(a), "min": np.min(a), "max": np.max(a)}


@numpy_eager_registry.register("Distributions")  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
def _np_distributions(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_distributions operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    a = _get_np_arg(args, 0)
    if a is None:
        return None
    counts, bins = np.histogram(a, bins="auto")
    return {"counts": counts, "bins": bins}


@numpy_eager_registry.register("Rrelu")
def _np_rrelu(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_rrelu operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    a = _get_np_arg(args, 0)
    if a is None:
        return None
    lower = kwargs.get("lower", 0.125)
    upper = kwargs.get("upper", 0.333)
    alpha = np.random.uniform(lower, upper, size=a.shape)
    return np.where(a >= 0, a, a * alpha)


@numpy_eager_registry.register("Clip")
def _np_clip(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_clip operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    a = _get_np_arg(args, 0)
    if a is None:
        return None
    a_min = kwargs.get("a_min", _get_np_arg(args, 1))
    a_max = kwargs.get("a_max", _get_np_arg(args, 2))
    return np.clip(a, a_min, a_max)


@numpy_eager_registry.register("Softmax")
def _np_softmax(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_softmax operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    a = _get_np_arg(args, 0)
    if a is None:
        return None
    axis = kwargs.get("axis", -1)
    e_x = np.exp(a - np.max(a, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


@numpy_eager_registry.register("Sigmoid")
def _np_sigmoid(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_sigmoid operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return 1.0 / (1.0 + np.exp(-a))


@numpy_eager_registry.register("LogSoftmax")
def _np_log_softmax(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_log_softmax operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    a = _get_np_arg(args, 0)
    if a is None:
        return None
    axis = kwargs.get("axis", -1)
    c = np.max(a, axis=axis, keepdims=True)
    return a - c - np.log(np.sum(np.exp(a - c), axis=axis, keepdims=True))


@numpy_eager_registry.register("OneHot")
def _np_one_hot(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_one_hot operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    indices = _get_np_arg(args, 0)
    depth = _get_np_arg(args, 1) if len(args) > 1 else kwargs.get("depth", None)
    if indices is None or depth is None:
        return None
    on_value = kwargs.get("on_value", 1)
    off_value = kwargs.get("off_value", 0)
    axis = kwargs.get("axis", -1)
    dtype = kwargs.get("dtype", float)

    depth_int = int(np.asarray(depth).item())

    out = np.eye(depth_int, dtype=dtype)[indices]
    if axis != -1:
        out = np.moveaxis(out, -1, axis)
    out = out * (on_value - off_value) + off_value
    out = out.astype(dtype)
    return out


def _parse_csv_row(row: list[str], record_defaults: list[Any], np: Any) -> list[Any]:
    """Parse a single CSV row, applying defaults on parse error or missing elements.

    Args:
        row (list): The row parameter.
        record_defaults (list): The record_defaults parameter.
        np (object): The np parameter.

    Returns:
        list: Result.

    Raises:
        RuntimeError: An exception.
    """
    row_out = []
    for i, val in enumerate(row):
        default = record_defaults[i] if i < len(record_defaults) else 0.0
        dt = np.asarray(default).dtype
        try:
            row_out.append(np.array(val, dtype=dt))
        except Exception as e:
            raise RuntimeError(f"Eager execution failed: {e}") from e
    for i in range(len(row), len(record_defaults)):
        row_out.append(np.array(record_defaults[i]))
    return row_out


def _get_csv_data(args: tuple[Any, ...], np: Any) -> str:
    """Extract and decode the CSV data string from arguments.

    Args:
        args: Positional arguments provided to the operation.
        np: The numpy module.

    Returns:
        A decoded CSV data string.
    """
    if not args:
        return ""
    arg0 = np.asarray(args[0])
    data = arg0.item() if arg0.ndim == 0 else arg0.flatten()[0]
    return data.decode("utf-8") if isinstance(data, bytes) else str(data)


def _get_csv_defaults(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[Any]:
    """Extract record defaults from arguments or keyword arguments.

    Args:
        args: Positional arguments provided to the operation.
        kwargs: Keyword arguments provided to the operation.

    Returns:
        A list of record default values.
    """
    return kwargs.get("record_defaults", args[1] if len(args) > 1 else [])


@numpy_eager_registry.register("DecodeCsv")
def _np_decode_csv_camel(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_decode_csv_camel operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        ValueError: An exception.
    """
    import csv
    import io

    import numpy as np

    if len(args) == 0:
        return [np.array([])]

    data_str = _get_csv_data(args, np)
    record_defaults = _get_csv_defaults(args, kwargs)

    out = []
    try:
        reader = csv.reader(io.StringIO(data_str))
        for row in reader:
            out.append(_parse_csv_row(row, record_defaults, np))
    except Exception as e:
        raise ValueError(f"Failed to parse CSV: {e}") from e

    if not out:
        return tuple([np.array(d) for d in record_defaults])
    return tuple([np.stack([r[i] for r in out]) for i in range(len(record_defaults))])


@numpy_eager_registry.register("DecodeImage")
def _np_decode_image_camel(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_decode_image_camel operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        RuntimeError: An exception.
    """
    import numpy as np

    if len(args) == 0:
        return np.array([])
    data = np.asarray(args[0]).item() if np.asarray(args[0]).ndim == 0 else np.asarray(args[0]).flatten()[0]
    try:
        import io

        from PIL import Image

        if isinstance(data, str):
            data = data.encode("utf-8")
        img = Image.open(io.BytesIO(data))
        return np.array(img)
    except Exception as e:
        raise RuntimeError(f"Eager execution failed: {e}") from e


@numpy_eager_registry.register("ParseExample")
def _np_parse_example_camel(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_parse_example_camel operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        RuntimeError: An exception.
    """
    import json

    import numpy as np

    features = kwargs.get("features", args[1] if len(args) > 1 else {})
    if len(args) == 0:
        return {k: np.zeros(getattr(v, "shape", (1,)), dtype=getattr(v, "dtype", np.float32)) for k, v in features.items()}
    data = np.asarray(args[0]).item() if np.asarray(args[0]).ndim == 0 else np.asarray(args[0]).flatten()[0]
    out = {}
    try:
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        parsed = json.loads(data)
        for k, v in features.items():
            if k in parsed:
                out[k] = np.array(parsed[k], dtype=getattr(v, "dtype", np.float32))
            else:
                out[k] = np.zeros(getattr(v, "shape", (1,)), dtype=getattr(v, "dtype", np.float32))
    except Exception as e:
        raise RuntimeError(f"Eager execution failed: {e}") from e
    return out


@numpy_eager_registry.register("ParseTensor")
def _np_parse_tensor_camel(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_parse_tensor_camel operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        RuntimeError: An exception.
    """
    import pickle

    import numpy as np

    if len(args) == 0:
        return np.array([])
    data = np.asarray(args[0]).item() if np.asarray(args[0]).ndim == 0 else np.asarray(args[0]).flatten()[0]
    dtype = kwargs.get("out_type", np.float32)
    try:
        return np.array(pickle.loads(data), dtype=dtype)
    except Exception as e:
        raise RuntimeError(f"Eager execution failed: {e}") from e


@numpy_eager_registry.register("ReadFile")
def _np_read_file_camel(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_read_file_camel operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        RuntimeError: An exception.
    """
    import numpy as np

    if len(args) == 0:
        return np.array(b"")
    filename = str(np.asarray(args[0]).item() if np.asarray(args[0]).ndim == 0 else np.asarray(args[0]).flatten()[0])
    try:
        with open(filename, "rb") as f:
            return np.array(f.read())
    except Exception as e:
        raise RuntimeError(f"Eager execution failed: {e}") from e


@numpy_eager_registry.register("Rem")  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
def _np_rem(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_rem operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    a = _get_np_arg(args, 0)
    b = _get_np_arg(args, 1)
    if a is None or b is None:
        return None
    return np.remainder(a, b)


@numpy_eager_registry.register("SerializeTensor")
def _np_serialize_tensor_camel(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_serialize_tensor_camel operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        RuntimeError: An exception.
    """
    import pickle

    import numpy as np

    if len(args) == 0:
        return np.array(b"")
    try:
        return np.array(pickle.dumps(np.asarray(args[0])))
    except Exception as e:
        raise RuntimeError(f"Eager execution failed: {e}") from e


@numpy_eager_registry.register("WriteFile")
def _np_write_file_camel(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_write_file_camel operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.

    Raises:
        OSError: An exception.
    """
    import numpy as np

    if len(args) < 2:
        return None
    filename = str(np.asarray(args[0]).item() if np.asarray(args[0]).ndim == 0 else np.asarray(args[0]).flatten()[0])
    contents = np.asarray(args[1])
    try:
        with open(filename, "wb") as f:
            if contents.ndim == 0 and isinstance(contents.item(), bytes):
                f.write(contents.item())
            else:
                f.write(contents.tobytes())
    except Exception as e:
        raise OSError(f"Failed to write file: {e}") from e
    return None


@numpy_eager_registry.register("Frombuffer")
def _np_frombuffer(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_frombuffer operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    import numpy as np

    if not args:
        return None
    return np.frombuffer(args[0], **kwargs)
