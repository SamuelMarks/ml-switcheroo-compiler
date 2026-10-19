"""Implementation of math general operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_general`)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence  # noqa: F401
from typing import cast

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy  # noqa: F401


@numpy_eager_registry.register("TruncateDiv")
def _np_truncate_div(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np truncate div operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    (x, y) = args[0], args[1]
    return cast(EagerValue, getattr(np, "trunc")(getattr(np, "divide")(x, y)))


@numpy_eager_registry.register("TruncateMod")
def _np_truncate_mod(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np truncate mod operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    (x, y) = args[0], args[1]
    return cast(EagerValue, getattr(np, "fmod")(x, y))


@numpy_eager_registry.register("Betainc")
def _np_betainc(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np betainc operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    try:
        import scipy.special

        return cast(EagerValue, getattr(backend_module, "array")(scipy.special.betainc(*args, **kwargs)))
    except ImportError as e:
        raise ImportError("scipy is required for Betainc in numpy eager backend") from e


@numpy_eager_registry.register("Trapz")
def _np_trapz(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Integrate along the given axis using the composite trapezoidal rule.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    trapz_fn = cast(Callable[..., EagerValue], getattr(np, "trapezoid", getattr(np, "trapz", None)))
    return trapz_fn(np.asarray(args[0]), *args[1:], **kwargs)


@numpy_eager_registry.register("Uint")
def _np_uint(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Cast or create an array as unsigned integers of default width.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(np, "array")(args[0], dtype=np.uint, **kwargs))


@numpy_eager_registry.register("Uint8")
def _np_uint8(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Cast or create an array as 8-bit unsigned integers.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(np, "array")(args[0], dtype=np.uint8, **kwargs))


@numpy_eager_registry.register("UnravelIndex")
def _np_unravelindex(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Convert a flat index or array of flat indices into a tuple of coordinate arrays.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(np, "unravel_index")(np.asarray(args[0]), *args[1:], **kwargs))


@numpy_eager_registry.register("Unwrap")
def _np_unwrap(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Unwrap by changing deltas between values to 2*pi complement.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(np, "unwrap")(np.asarray(args[0]), *args[1:], **kwargs))


@numpy_eager_registry.register("Vectorize")
def _np_vectorize(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Return a vectorized function which takes a nested sequence of Anys or numpy arrays as inputs.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(np, "vectorize")(args[0], **kwargs)(*args[1:]))


@numpy_eager_registry.register("Average")
def _np_average(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np average operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(EagerValue, getattr(backend_module, "average")(*args, **kwargs))


@numpy_eager_registry.register("Block")
def _np_block(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Assemble an nd-array from nested lists of blocks.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(backend_module, "block")(*args, **kwargs))


@numpy_eager_registry.register("Atleast1d")
def _np_atleast_1d(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Convert inputs to arrays with at least one dimension.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(backend_module, "atleast_1d")(*args, **kwargs))


@numpy_eager_registry.register("Atleast2d")
def _np_atleast_2d(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """View inputs as arrays with at least two dimensions.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(backend_module, "atleast_2d")(*args, **kwargs))


@numpy_eager_registry.register("Atleast3d")
def _np_atleast_3d(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """View inputs as arrays with at least three dimensions.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(backend_module, "atleast_3d")(*args, **kwargs))


@numpy_eager_registry.register("ApplyOverAxes")
def _np_apply_over_axes(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Apply a function repeatedly over multiple axes.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(backend_module, "apply_over_axes")(*args, **kwargs))


@numpy_eager_registry.register("CorrCoef")
def _np_corrcoef(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Return Pearson product-moment correlation coefficients.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(backend_module, "corrcoef")(*args, **kwargs))


@numpy_eager_registry.register("Cov")
def _np_cov(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Estimate a covariance matrix, given data and weights.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(backend_module, "cov")(*args, **kwargs))


@numpy_eager_registry.register("ArrayRepr")
def _np_array_repr_(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement ArrayRepr via array_repr.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(backend_module, "array_repr")(*args, **kwargs))


@numpy_eager_registry.register("ArrayStr")
def _np_array_str_(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement ArrayStr via array_str.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(backend_module, "array_str")(*args, **kwargs))


@numpy_eager_registry.register("Diff")
def _np_diff_(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Diff via diff.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(backend_module, "diff")(*args, **kwargs))


@numpy_eager_registry.register("Digitize")
def _np_digitize_(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Digitize via digitize.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(backend_module, "digitize")(*args, **kwargs))


@numpy_eager_registry.register("Ediff1d")
def _np_ediff1d_(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Ediff1d via ediff1d.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(backend_module, "ediff1d")(*args, **kwargs))


@numpy_eager_registry.register("EinsumPath")
def _np_einsum_path_(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement EinsumPath via einsum_path.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(backend_module, "einsum_path")(*args, **kwargs))


@numpy_eager_registry.register("Fabs")
def _np_fabs_(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Fabs via fabs.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(backend_module, "fabs")(*args, **kwargs))


@numpy_eager_registry.register("Interp")
def _np_interp_(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Interp via interp.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(backend_module, "interp")(*args, **kwargs))


@numpy_eager_registry.register("Iterable")
def _np_iterable_(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Iterable via iterable.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(backend_module, "iterable")(*args, **kwargs))


@numpy_eager_registry.register("Ix")
def _np_ix__(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Ix via ix_.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(backend_module, "ix_")(*args, **kwargs))


@numpy_eager_registry.register("Kron")
def _np_kron_(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Kron via kron.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(backend_module, "kron")(*args, **kwargs))


@numpy_eager_registry.register("Load")
def _np_load_(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Load via load.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(backend_module, "load")(*args, **kwargs))


@numpy_eager_registry.register("Modf")
def _np_modf_(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Modf via modf.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(backend_module, "modf")(*args, **kwargs))


@numpy_eager_registry.register("RavelMultiIndex")
def _np_ravel_multi_index_(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement RavelMultiIndex via ravel_multi_index.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    return cast(EagerValue, getattr(backend_module, "ravel_multi_index")(*args, **kwargs))


@numpy_eager_registry.register("Trapezoid")
def _np_trapz_(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Trapezoid via trapz.

    Args:
        backend_module: The backend_module parameter.
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: np.ndarray: The computed result.
    """
    trapz_fn = getattr(backend_module, "trapezoid", getattr(backend_module, "trapz", None))
    if trapz_fn is not None:
        return cast(EagerValue, trapz_fn(*args, **kwargs))
    trapz_fn = cast(Callable[..., EagerValue], getattr(np, "trapezoid", getattr(np, "trapz", None)))
    return trapz_fn(*args, **kwargs)


@numpy_eager_registry.register("Callable")
def _np_callable(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Callable.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    if args:
        return callable(args[0])
    return False


@numpy_eager_registry.register("DebugInfs")
def _np_debuginfs(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement DebugInfs.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.

    Raises:
        ValueError: An exception.
    """
    x = args[0]
    if getattr(backend_module, "any")(getattr(backend_module, "isinf")(x)):
        raise ValueError("Infinity found in tensor.")
    return x


@numpy_eager_registry.register("DebugNans")
def _np_debugnans(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement DebugNans.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.

    Raises:
        ValueError: An exception.
    """
    x = args[0]
    if getattr(backend_module, "any")(getattr(backend_module, "isnan")(x)):
        raise ValueError("NaN found in tensor.")
    return x


def _build_dot_general_einsum_str(lhs_ndim: int, rhs_ndim: int, dimension_numbers: Sequence[Sequence[Sequence[int]]]) -> str:
    """Execute the eager build dot general einsum str operation on the active backend array.

    Args:
        lhs_ndim (int): The lhs_ndim parameter.
        rhs_ndim (int): The rhs_ndim parameter.
        dimension_numbers: The dimension_numbers parameter.

    Returns:
        str: Result.
    """
    lhs_cont, rhs_cont = dimension_numbers[0][0], dimension_numbers[0][1]
    lhs_batch, rhs_batch = dimension_numbers[1][0], dimension_numbers[1][1]
    batch_chars = [chr(ord("a") + i) for i in range(len(lhs_batch))]
    cont_chars = [chr(ord("a") + len(lhs_batch) + i) for i in range(len(lhs_cont))]
    current_char = ord("a") + len(lhs_batch) + len(lhs_cont)
    (lhs_str, rhs_str) = ([""] * lhs_ndim, [""] * rhs_ndim)
    for i, (lb, rb) in enumerate(zip(lhs_batch, rhs_batch)):
        lhs_str[lb] = rhs_str[rb] = batch_chars[i]
    for i, (lc, rc) in enumerate(zip(lhs_cont, rhs_cont)):
        lhs_str[lc] = rhs_str[rc] = cont_chars[i]
    (out_lhs, out_rhs) = ([], [])
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
def _np_dotgeneral(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement DotGeneral.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    (lhs, rhs) = (np.asarray(args[0]), np.asarray(args[1]))
    dimension_numbers = kwargs.get("dimension_numbers", (((-1,), (0,)), ((), ()))) if hasattr(kwargs, "get") else (((-1,), (0,)), ((), ()))
    einsum_str = cast("Callable[..., str]", _build_dot_general_einsum_str)(lhs.ndim, rhs.ndim, dimension_numbers)
    return cast(EagerValue, np.einsum(einsum_str, lhs, rhs))


@numpy_eager_registry.register("RaggedDot")
def _np_raggeddot(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement RaggedDot.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return cast(EagerValue, getattr(backend_module, "matmul")(args[0], args[1]))


@numpy_eager_registry.register("Tensor")
def _np_tensor(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Implement Tensor.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    if not args and (not kwargs):
        return cast(EagerValue, getattr(backend_module, "array")([]))
    return cast(EagerValue, getattr(backend_module, "array")(*args, **kwargs))


@numpy_eager_registry.register("serialize_tensor")
def _np_serialize_tensor(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np serialize tensor operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    return _np_serialize_tensor_camel(backend_module, *args, **kwargs)


@numpy_eager_registry.register("Key")
def _np_key(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np key operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    if len(args) > 0:
        return np.array([args[0], 0], dtype=np.uint32)
    return np.array([0, 0], dtype=np.uint32)


def _get_np_arg(arg: Sequence[EagerValue], i: int) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]] | None:
    """Get numpy arg.

    Args:
        arg: The arg parameter.
        i (int): The i parameter.

    Returns:
            np.ndarray[tuple[int, ...], np.dtype[np.generic]] | None: Result.
    """
    return np.asarray(arg[i]) if len(arg) > i else None


def _get_sc() -> EagerValue:
    """Execute the eager get sc operation on the active backend array.

    Returns:
        EagerValue: Result.
    """
    try:
        import scipy.special as sc

        return cast(EagerValue, sc)
    except ImportError:
        return None


@numpy_eager_registry.register("Clip")
def _np_clip(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np clip operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    a = _get_np_arg(args, 0)
    if a is None:
        return None
    a_min = kwargs.get("a_min", _get_np_arg(args, 1)) if hasattr(kwargs, "get") else _get_np_arg(args, 1)
    a_max = kwargs.get("a_max", _get_np_arg(args, 2)) if hasattr(kwargs, "get") else _get_np_arg(args, 2)
    return cast(EagerValue, getattr(np, "clip")(a, a_min, a_max))


@numpy_eager_registry.register("OneHot")
def _np_one_hot(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np one hot operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.
    """
    indices = _get_np_arg(args, 0)
    depth = _get_np_arg(args, 1) if len(args) > 1 else kwargs.get("depth", None) if hasattr(kwargs, "get") else None
    if indices is None or depth is None:
        return None
    on_value = float(cast(float, kwargs.get("on_value", 1))) if hasattr(kwargs, "get") else 1.0
    off_value = float(cast(float, kwargs.get("off_value", 0))) if hasattr(kwargs, "get") else 0.0
    axis = kwargs.get("axis", -1) if hasattr(kwargs, "get") else -1
    dtype = kwargs.get("dtype", float) if hasattr(kwargs, "get") else float
    depth_int = int(np.asarray(depth).item())
    out = getattr(np, "eye")(depth_int, dtype=dtype)[indices]
    if axis != -1:
        out = getattr(np, "moveaxis")(out, -1, axis)
    out = out * (on_value - off_value) + off_value
    out = out.astype(dtype)
    return cast(EagerValue, out)


@numpy_eager_registry.register("SerializeTensor")
def _np_serialize_tensor_camel(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np serialize tensor camel operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.

    Raises:
        RuntimeError: An exception.
    """
    import pickle

    import numpy as np

    if len(args) == 0:
        return np.array(b"")
    try:
        return np.array(pickle.dumps(np.asarray(args[0])))
    except Exception:
        # Ignore errors and fallback
        logging.getLogger(__name__).debug("Failed to import ops", exc_info=True)
    return np.array(b"")


@numpy_eager_registry.register("rem")
def _np_rem_2(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np rem 2 operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.

    Raises:
        RuntimeError: An exception.
    """
    try:
        import ml_switcheroo_compiler.ops as _ops

        if hasattr(_ops, "rem"):
            cls_or_func = _ops.rem
            if isinstance(cls_or_func, type) and (not issubclass(cls_or_func, getattr(_ops, "OpDef", type))):
                return cls_or_func(*args, **kwargs)
    except Exception:
        # Ignore errors and fallback
        logging.getLogger(__name__).debug("Failed to import ops", exc_info=True)
    if hasattr(backend_module, "rem"):
        return cast(EagerValue, getattr(backend_module, "rem")(*args, **kwargs))
    return cast(EagerValue, getattr(np, "remainder")(args[0], args[1])) if len(args) > 1 else None


@numpy_eager_registry.register("descriptive")
def _np_descriptive_2(backend_module: EagerValue, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np descriptive 2 operation on the active backend array.

    Args:
        backend_module: The backend_module parameter.
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            EagerValue: Result.

    Raises:
        RuntimeError: An exception.
    """
    try:
        import ml_switcheroo_compiler.ops as _ops

        if hasattr(_ops, "descriptive"):
            cls_or_func = _ops.descriptive
            if isinstance(cls_or_func, type) and (not issubclass(cls_or_func, getattr(_ops, "OpDef", type))):
                return cast(EagerValue, cls_or_func(*args, **kwargs))
    except Exception:
        # Ignore errors and fallback
        logging.getLogger(__name__).debug("Failed to import ops", exc_info=True)
    if hasattr(backend_module, "descriptive"):
        return cast(EagerValue, getattr(backend_module, "descriptive")(*args, **kwargs))
    arr = np.asarray(args[0]) if args else np.zeros((1,))
    return np.array([np.mean(arr), np.var(arr), np.std(arr)])


numpy_eager_registry.register("rem")(_np_rem_2)
numpy_eager_registry.register("descriptive")(_np_descriptive_2)
