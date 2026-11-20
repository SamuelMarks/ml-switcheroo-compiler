"""Generic backend-parametric eager reference operations for `reference_math.math_general`."""

from __future__ import annotations

import logging
import types
from collections.abc import Callable, Sequence  # noqa: F401

from ml_switcheroo_compiler.backends.eager.reference_math.math_nan import _xlogy  # noqa: F401
from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    coerce_float,
    coerce_int,
    coerce_int_list,
    coerce_str,
    global_eager_registry,
    require_callable,
    unpack_pair,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


@global_eager_registry.register("TruncateDiv")
def _np_truncate_div(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np truncate div operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    (x, y) = args[0], args[1]
    return adapt_backend_module(arr_mod).require_symbol("trunc")(
        adapt_backend_module(arr_mod).require_symbol("divide")(x, y)
    )


@global_eager_registry.register("TruncateMod")
def _np_truncate_mod(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np truncate mod operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    (x, y) = args[0], args[1]
    return adapt_backend_module(arr_mod).require_symbol("fmod")(x, y)


@global_eager_registry.register("Trapz")
def _np_trapz(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Integrate along the given axis using the composite trapezoidal rule.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    trapz_fn = adapt_backend_module(arr_mod).get_symbol(
        "trapezoid", adapt_backend_module(arr_mod).get_symbol("trapz", None)
    )
    return require_callable(trapz_fn)(require_callable(arr_mod.asarray)(args[0]), *args[1:], **kwargs)


@global_eager_registry.register("Uint")
def _np_uint(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Cast or create an array as unsigned integers of default width.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return adapt_backend_module(arr_mod).require_symbol("array")(
        args[0], dtype=adapt_backend_module(arr_mod).require_symbol("uint"), **kwargs
    )


@global_eager_registry.register("Uint8")
def _np_uint8(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Cast or create an array as 8-bit unsigned integers.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return adapt_backend_module(arr_mod).require_symbol("array")(args[0], dtype=arr_mod.uint8, **kwargs)


@global_eager_registry.register("UnravelIndex")
def _np_unravelindex(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Convert a flat index or array of flat indices into a tuple of coordinate arrays.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return adapt_backend_module(arr_mod).require_symbol("unravel_index")(arr_mod.asarray(args[0]), *args[1:], **kwargs)


@global_eager_registry.register("Unwrap")
def _np_unwrap(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Unwrap by changing deltas between values to 2*pi complement.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return adapt_backend_module(arr_mod).require_symbol("unwrap")(arr_mod.asarray(args[0]), *args[1:], **kwargs)


@global_eager_registry.register("Vectorize")
def _np_vectorize(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Return a vectorized function which takes a nested sequence of Anys or numpy arrays as inputs.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return adapt_backend_module(arr_mod).require_symbol("vectorize")(args[0], **kwargs)(*args[1:])


@global_eager_registry.register("Average")
def _np_average(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Execute the eager np average operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("average")(*args, **kwargs)


@global_eager_registry.register("Block")
def _np_block(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Assemble an nd-array from nested lists of blocks.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("block")(*args, **kwargs)


@global_eager_registry.register("Atleast1d")
def _np_atleast_1d(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Convert inputs to arrays with at least one dimension.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("atleast_1d")(*args, **kwargs)


@global_eager_registry.register("Atleast2d")
def _np_atleast_2d(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """View inputs as arrays with at least two dimensions.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("atleast_2d")(*args, **kwargs)


@global_eager_registry.register("Atleast3d")
def _np_atleast_3d(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """View inputs as arrays with at least three dimensions.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("atleast_3d")(*args, **kwargs)


@global_eager_registry.register("ApplyOverAxes")
def _np_apply_over_axes(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Apply a function repeatedly over multiple axes.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("apply_over_axes")(*args, **kwargs)


@global_eager_registry.register("CorrCoef")
def _np_corrcoef(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Return Pearson product-moment correlation coefficients.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("corrcoef")(*args, **kwargs)


@global_eager_registry.register("Cov")
def _np_cov(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Estimate a covariance matrix, given data and weights.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("cov")(*args, **kwargs)


@global_eager_registry.register("ArrayRepr")
def _np_array_repr_(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Implement ArrayRepr via array_repr.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("array_repr")(*args, **kwargs)


@global_eager_registry.register("ArrayStr")
def _np_array_str_(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Implement ArrayStr via array_str.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("array_str")(*args, **kwargs)


@global_eager_registry.register("Diff")
def _np_diff_(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Implement Diff via diff.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("diff")(*args, **kwargs)


@global_eager_registry.register("Digitize")
def _np_digitize_(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Implement Digitize via digitize.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("digitize")(*args, **kwargs)


@global_eager_registry.register("Ediff1d")
def _np_ediff1d_(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Implement Ediff1d via ediff1d.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("ediff1d")(*args, **kwargs)


@global_eager_registry.register("Fabs")
def _np_fabs_(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Implement Fabs via fabs.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("fabs")(*args, **kwargs)


@global_eager_registry.register("Interp")
def _np_interp_(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Implement Interp via interp.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("interp")(*args, **kwargs)


@global_eager_registry.register("Iterable")
def _np_iterable_(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Implement Iterable via iterable.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("iterable")(*args, **kwargs)


@global_eager_registry.register("Ix")
def _np_ix__(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Implement Ix via ix_.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("ix_")(*args, **kwargs)


@global_eager_registry.register("Kron")
def _np_kron_(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Implement Kron via kron.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("kron")(*args, **kwargs)


@global_eager_registry.register("Load")
def _np_load_(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Implement Load via load.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("load")(*args, **kwargs)


@global_eager_registry.register("Modf")
def _np_modf_(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Implement Modf via modf.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("modf")(*args, **kwargs)


@global_eager_registry.register("RavelMultiIndex")
def _np_ravel_multi_index_(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Implement RavelMultiIndex via ravel_multi_index.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    return adapt_backend_module(backend_module).require_symbol("ravel_multi_index")(*args, **kwargs)


@global_eager_registry.register("Trapezoid")
def _np_trapz_(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement Trapezoid via trapz.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Variable positional arguments.
        **kwargs: Arbitrary keyword arguments.

    Returns: EagerTensorProtocol: The computed result.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    trapz_fn = adapt_backend_module(backend_module).get_symbol(
        "trapezoid", adapt_backend_module(backend_module).get_symbol("trapz", None)
    )
    if trapz_fn is not None and callable(trapz_fn):
        return trapz_fn(*args, **kwargs)
    trapz_fn_np = adapt_backend_module(arr_mod).get_symbol(
        "trapezoid", adapt_backend_module(arr_mod).get_symbol("trapz", None)
    )
    return require_callable(trapz_fn_np)(*args, **kwargs)


@global_eager_registry.register("Callable")
def _np_callable(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Implement Callable.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    if args:
        return callable(args[0])
    return False


@global_eager_registry.register("DebugInfs")
def _np_debuginfs(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Implement DebugInfs.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.

    Raises:
        ValueError: If invalid inputs or unsupported runtime conditions trigger `ValueError`.
    """
    x = args[0]
    if adapt_backend_module(backend_module).require_symbol("any")(
        adapt_backend_module(backend_module).require_symbol("isinf")(x)
    ):
        raise ValueError("Infinity found in tensor.")
    return x


@global_eager_registry.register("DebugNans")
def _np_debugnans(backend_module: BackendArg, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
    """Implement DebugNans.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.

    Raises:
        ValueError: If invalid inputs or unsupported runtime conditions trigger `ValueError`.
    """
    x = args[0]
    if adapt_backend_module(backend_module).require_symbol("any")(
        adapt_backend_module(backend_module).require_symbol("isnan")(x)
    ):
        raise ValueError("NaN found in tensor.")
    return x


def _build_dot_general_einsum_str(
    lhs_ndim: int, rhs_ndim: int, dimension_numbers: Sequence[Sequence[Sequence[int]]]
) -> str:
    """Execute the eager build dot general einsum str operation on the active backend array.

    Args:
        lhs_ndim (int): Input `lhs_ndim` argument (lhs ndim).
        rhs_ndim (int): Input `rhs_ndim` argument (rhs ndim).
        dimension_numbers: Input `dimension_numbers` argument (dimension numbers).

    Returns:
        str: Computed `str` output.
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


@global_eager_registry.register("DotGeneral")
def _np_dotgeneral(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement DotGeneral.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    (lhs, rhs) = (arr_mod.asarray(args[0]), arr_mod.asarray(args[1]))
    raw_dn = (
        kwargs.get("dimension_numbers", (((-1,), (0,)), ((), ())))
        if adapt_backend_module(kwargs).has_symbol("get")
        else (((-1,), (0,)), ((), ()))
    )
    c_pair, b_pair = unpack_pair(raw_dn)
    lc, rc = unpack_pair(c_pair)
    lb, rb = unpack_pair(b_pair)
    dn_typed = ((coerce_int_list(lc), coerce_int_list(rc)), (coerce_int_list(lb), coerce_int_list(rb)))
    einsum_str = coerce_str(_build_dot_general_einsum_str(lhs.ndim, rhs.ndim, dn_typed))
    return arr_mod.einsum(einsum_str, lhs, rhs)


@global_eager_registry.register("RaggedDot")
def _np_raggeddot(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Implement RaggedDot.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return adapt_backend_module(backend_module).require_symbol("matmul")(args[0], args[1])


@global_eager_registry.register("Tensor")
def _np_tensor(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Implement Tensor.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    if not args and (not kwargs):
        return adapt_backend_module(backend_module).require_symbol("array")([])
    return adapt_backend_module(backend_module).require_symbol("array")(*args, **kwargs)


@global_eager_registry.register("serialize_tensor")
def _np_serialize_tensor(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np serialize tensor operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    return _np_serialize_tensor_camel(backend_module, *args, **kwargs)


@global_eager_registry.register("Key")
def _np_key(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np key operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    if len(args) > 0:
        return arr_mod.array([args[0], 0], dtype=arr_mod.uint32)
    return arr_mod.array([0, 0], dtype=arr_mod.uint32)


def _get_np_arg(arg: Sequence[BackendArg], i: int) -> EagerTensorProtocol | None:
    """Get numpy arg.

    Args:
        arg: Input `arg` argument (arg).
        i (int): Input `i` argument (i).

    Returns:
            EagerTensorProtocol | None: Computed `EagerTensorProtocol | None` output.
    """
    arr_mod = _resolve_eager_mod(None)
    return arr_mod.asarray(arg[i]) if len(arg) > i else None


@global_eager_registry.register("Clip")
def _np_clip(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np clip operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    a = _get_np_arg(args, 0)
    if a is None:
        return None
    a_min = (
        kwargs.get("a_min", _get_np_arg(args, 1))
        if adapt_backend_module(kwargs).has_symbol("get")
        else _get_np_arg(args, 1)
    )
    a_max = (
        kwargs.get("a_max", _get_np_arg(args, 2))
        if adapt_backend_module(kwargs).has_symbol("get")
        else _get_np_arg(args, 2)
    )
    return adapt_backend_module(arr_mod).require_symbol("clip")(a, a_min, a_max)


@global_eager_registry.register("OneHot")
def _np_one_hot(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np one hot operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    indices = _get_np_arg(args, 0)
    depth = (
        _get_np_arg(args, 1)
        if len(args) > 1
        else kwargs.get("depth", None)
        if adapt_backend_module(kwargs).has_symbol("get")
        else None
    )
    if indices is None or depth is None:
        return None
    on_value = float(coerce_float(kwargs.get("on_value", 1))) if adapt_backend_module(kwargs).has_symbol("get") else 1.0
    off_value = (
        float(coerce_float(kwargs.get("off_value", 0))) if adapt_backend_module(kwargs).has_symbol("get") else 0.0
    )
    axis = kwargs.get("axis", -1) if adapt_backend_module(kwargs).has_symbol("get") else -1
    dtype = kwargs.get("dtype", float) if adapt_backend_module(kwargs).has_symbol("get") else float
    depth_int = int(coerce_int(arr_mod.asarray(depth).item()))
    out = adapt_backend_module(arr_mod).require_symbol("eye")(depth_int, dtype=dtype)[indices]
    if axis != -1:
        out = adapt_backend_module(arr_mod).require_symbol("moveaxis")(out, -1, axis)
    out = out * (on_value - off_value) + off_value
    out = out.astype(dtype)
    return out


def _np_serialize_tensor_camel(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np serialize tensor camel operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.

    Raises:
        RuntimeError: If invalid inputs or unsupported runtime conditions trigger `RuntimeError`.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    import pickle

    if len(args) == 0:
        return arr_mod.array(b"")
    try:
        return arr_mod.array(pickle.dumps(arr_mod.asarray(args[0])))
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, pickle.PickleError):
        # Ignore errors and fallback
        logging.getLogger(__name__).debug("Failed to import ops", exc_info=True)
    return arr_mod.array(b"")


@global_eager_registry.register("rem")
def _np_rem_2(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np rem 2 operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.

    Raises:
        RuntimeError: If invalid inputs or unsupported runtime conditions trigger `RuntimeError`.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    try:
        import ml_switcheroo_compiler.ops as _ops

        if adapt_backend_module(_ops).has_symbol("rem"):
            cls_or_func: BackendArg = adapt_backend_module(_ops).require_symbol("rem")
            op_def_cls: BackendArg = adapt_backend_module(_ops).get_symbol("OpDef", type)
            if (
                isinstance(cls_or_func, type)
                and isinstance(op_def_cls, type)
                and not issubclass(cls_or_func, op_def_cls)
            ):
                return require_callable(cls_or_func)(*args, **kwargs)
    except (
        AttributeError,
        ImportError,
        LookupError,
        OSError,
        ArithmeticError,
        RuntimeError,
        SyntaxError,
        TypeError,
        ValueError,
    ):
        # Ignore errors and fallback
        logging.getLogger(__name__).debug("Failed to import ops", exc_info=True)
    if adapt_backend_module(backend_module).has_symbol("rem"):
        return adapt_backend_module(backend_module).require_symbol("rem")(*args, **kwargs)
    return adapt_backend_module(arr_mod).require_symbol("remainder")(args[0], args[1]) if len(args) > 1 else None


@global_eager_registry.register("descriptive")
def _np_descriptive_2(
    backend_module: BackendModuleProtocol | types.ModuleType | None,
    *args: BackendArg,
    **kwargs: BackendArg,
) -> BackendArg:
    """Execute the eager np descriptive 2 operation on the active backend array.

    Args:
        backend_module: Input `backend_module` argument (backend module).
        *args: Positional args.
        **kwargs: Keyword args.

    Returns:
            BackendArg: Computed `BackendArg` output.

    Raises:
        RuntimeError: If invalid inputs or unsupported runtime conditions trigger `RuntimeError`.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    try:
        import ml_switcheroo_compiler.ops as _ops

        if adapt_backend_module(_ops).has_symbol("descriptive"):
            cls_or_func: BackendArg = adapt_backend_module(_ops).require_symbol("descriptive")
            op_def_cls: BackendArg = adapt_backend_module(_ops).get_symbol("OpDef", type)
            if (
                isinstance(cls_or_func, type)
                and isinstance(op_def_cls, type)
                and not issubclass(cls_or_func, op_def_cls)
            ):
                return require_callable(cls_or_func)(*args, **kwargs)
    except (
        AttributeError,
        ImportError,
        LookupError,
        OSError,
        ArithmeticError,
        RuntimeError,
        SyntaxError,
        TypeError,
        ValueError,
    ):
        # Ignore errors and fallback
        logging.getLogger(__name__).debug("Failed to import ops", exc_info=True)
    if adapt_backend_module(backend_module).has_symbol("descriptive"):
        return adapt_backend_module(backend_module).require_symbol("descriptive")(*args, **kwargs)
    arr = arr_mod.asarray(args[0]) if args else arr_mod.zeros((1,))
    return arr_mod.array([arr_mod.mean(arr), arr_mod.var(arr), arr_mod.std(arr)])


global_eager_registry.register("rem")(_np_rem_2)

global_eager_registry.register("descriptive")(_np_descriptive_2)
