# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""math_misc_ext module."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_nan import _xlogy


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
    ((lhs_cont, rhs_cont), (lhs_batch, rhs_batch)) = dimension_numbers
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
def _np_dotgeneral(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement DotGeneral.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    (lhs, rhs) = (np.asarray(args[0]), np.asarray(args[1]))
    dimension_numbers = kwargs.get("dimension_numbers", (((-1,), (0,)), ((), ())))
    einsum_str = _build_dot_general_einsum_str(lhs.ndim, rhs.ndim, dimension_numbers)
    return np.einsum(einsum_str, lhs, rhs)


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


@numpy_eager_registry.register("Tensor")
def _np_tensor(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Implement Tensor.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    if not args and (not kwargs):
        return backend_module.array([])
    return backend_module.array(*args, **kwargs)


@numpy_eager_registry.register("Rem")
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


@numpy_eager_registry.register("Descriptive")
def _np_descriptive(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    """Evaluate _np_descriptive operation.

    Args:
        backend_module (object): The backend_module parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    a = _get_np_arg(args, 0)
    if a is None:
        return None
    return {"mean": np.mean(a), "std": np.std(a), "min": np.min(a), "max": np.max(a)}


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


def _get_np_arg(arg: Sequence[Any], i: int) -> np.ndarray | None:
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
            if isinstance(cls_or_func, type) and (not issubclass(cls_or_func, _ops.OpDef)):
                return cls_or_func(*args, **kwargs)
    except Exception as e:
        if not isinstance(e, (ImportError, AttributeError)):
            raise RuntimeError(f"Eager execution failed: {e}") from e
    if hasattr(backend_module, "rem"):
        return backend_module.rem(*args, **kwargs)
    return np.remainder(args[0], args[1])


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
            if isinstance(cls_or_func, type) and (not issubclass(cls_or_func, _ops.OpDef)):
                return cls_or_func(*args, **kwargs)
    except Exception as e:
        if not isinstance(e, (ImportError, AttributeError)):
            raise RuntimeError(f"Eager execution failed: {e}") from e
    if hasattr(backend_module, "descriptive"):
        return backend_module.descriptive(*args, **kwargs)
    arr = np.asarray(args[0]) if args else np.zeros((1,))
    return np.array([np.mean(arr), np.var(arr), np.std(arr)])


@numpy_eager_registry.register("Rem")
def _np_rem(backend_module: Any, *args: Any, **kwargs: Any) -> Any:
    a = _get_np_arg(args, 0)
    b = _get_np_arg(args, 1)
    if a is None or b is None:
        return None
    return np.remainder(a, b)
