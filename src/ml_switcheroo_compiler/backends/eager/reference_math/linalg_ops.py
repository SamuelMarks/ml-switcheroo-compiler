"""Generic backend-parametric eager reference operations for `reference_math.linalg_ops`."""

from __future__ import annotations

import types
from collections.abc import Sequence

from ml_switcheroo_compiler.backends.eager.utils import _resolve_eager_mod
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    coerce_int,
    coerce_int_list,
    coerce_shape,
    global_eager_registry,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module


def _ensure_arr(val: BackendArg) -> EagerTensorProtocol:
    """Coerce `val` to an eager array via the active backend module.

    Args:
        val (BackendArg): Input array or scalar value.

    Returns:
        EagerTensorProtocol: Coerced backend array.
    """
    return _resolve_eager_mod(None).asarray(val)


for op_name in [
    "Qr",
    "Solve",
    "Tensorinv",
    "Tensorsolve",
    "Eig",
    "Eigh",
    "Eigvals",
    "Eigvalsh",
    "Norm",
    "Cond",
    "MultiDot",
]:

    def make_linalg_wrapper(name: str) -> BackendArg:
        """Create a wrapper for linalg operations.

        Args:
        name (str): Input `name` argument (name).

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """

        def _wrapper(
            backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
        ) -> BackendArg:
            """Execute the eager wrapper operation on the active backend array.

            Args:
            backend_module (BackendModuleProtocol | types.ModuleType | None):
                Input `backend_module` argument (backend module).
            *args (BackendArg): Positional args.
            **kwargs (BackendArg): Keyword args.

            Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
            """
            arr_mod = _resolve_eager_mod(backend_module)
            func = adapt_backend_module(arr_mod.linalg).require_symbol(
                name.lower() if name.lower() != "multidot" else "multi_dot"
            )
            return func(*args, **kwargs)

        return _wrapper

    adapt_backend_module(global_eager_registry).require_symbol("register")(op_name)(make_linalg_wrapper(op_name))


def _get_uncontracted_dims(dims: list[int], batch: Sequence[int], contracting: Sequence[int]) -> list[int]:
    """Execute the eager get uncontracted dims operation on the active backend array.

    Args:
        dims (list[int]): Input `dims` argument (dims).
        batch (Sequence[int]): Input `batch` argument (batch).
        contracting (Sequence[int]): Input `contracting` argument (contracting).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    skip_set: set[int] = set(batch) | set(contracting)
    return [dims[i] for i in range(len(dims)) if i not in skip_set]


def _parse_dot_dimension_numbers(
    dimension_numbers: BackendArg,
) -> tuple[Sequence[int], Sequence[int], Sequence[int], Sequence[int]]:
    """Execute the eager parse dot dimension numbers operation on the active backend array.

    Args:
        dimension_numbers (BackendArg): Input `dimension_numbers` argument (dimension numbers).

    Returns:
        tuple: Computed `tuple` output.
    """
    from ml_switcheroo_compiler.backends.eager_registry import unpack_pair

    contracting, batch = unpack_pair(dimension_numbers)
    a_contract, b_contract = unpack_pair(contracting)
    a_b, b_b = unpack_pair(batch)
    a_contracting: Sequence[int] = (
        coerce_int_list(a_contract) if isinstance(a_contract, list) else tuple(coerce_int_list(a_contract))
    )
    b_contracting: Sequence[int] = (
        coerce_int_list(b_contract) if isinstance(b_contract, list) else tuple(coerce_int_list(b_contract))
    )
    a_batch: Sequence[int] = coerce_int_list(a_b) if isinstance(a_b, list) else tuple(coerce_int_list(a_b))
    b_batch: Sequence[int] = coerce_int_list(b_b) if isinstance(b_b, list) else tuple(coerce_int_list(b_b))
    return (a_contracting, b_contracting, a_batch, b_batch)


def _dot_general(a: BackendArg, b: BackendArg, dimension_numbers: BackendArg) -> BackendArg:
    """Execute the eager dot general operation on the active backend array.

    Args:
        a (BackendArg): Input `a` argument (a).
        b (BackendArg): Input `b` argument (b).
        dimension_numbers (BackendArg): Input `dimension_numbers` argument (dimension numbers).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(None)
    (a_dims, b_dims, out_dims) = _build_einsum_equation(
        coerce_int(adapt_backend_module(a).require_symbol("ndim")),
        coerce_int(adapt_backend_module(b).require_symbol("ndim")),
        dimension_numbers,
    )
    return adapt_backend_module(arr_mod).require_symbol("einsum")(a, a_dims, b, b_dims, out_dims)


def _build_einsum_equation(
    a_ndim: int, b_ndim: int, dimension_numbers: BackendArg
) -> tuple[list[int], list[int], list[int]]:
    """Execute the eager build einsum equation operation on the active backend array.

    Args:
        a_ndim (int): Input `a_ndim` argument (a ndim).
        b_ndim (int): Input `b_ndim` argument (b ndim).
        dimension_numbers (BackendArg): Input `dimension_numbers` argument (dimension numbers).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    (a_contracting, b_contracting, a_batch, b_batch) = _parse_dot_dimension_numbers(dimension_numbers)
    a_dims: list[int] = list(range(a_ndim))
    b_dims: list[int] = list(range(a_ndim, a_ndim + b_ndim))
    for i, a_b in enumerate(a_batch):
        b_dims[b_batch[i]] = a_dims[a_b]
    for i, a_c in enumerate(a_contracting):
        b_dims[b_contracting[i]] = a_dims[a_c]
    out_dims: list[int] = [a_dims[i] for i in a_batch]
    out_dims.extend(_get_uncontracted_dims(a_dims, a_batch, a_contracting))
    out_dims.extend(_get_uncontracted_dims(b_dims, b_batch, b_contracting))
    return (a_dims, b_dims, out_dims)


@global_eager_registry.register("Trace")
def _np_trace(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np trace operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return adapt_backend_module(arr_mod).require_symbol("trace")(args[0], **kwargs)


@global_eager_registry.register("MatrixRank")
def _np_matrix_rank(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np matrix rank operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return adapt_backend_module(arr_mod.linalg).require_symbol("matrix_rank")(args[0], **kwargs)


@global_eager_registry.register("MatrixTranspose")
def _np_matrix_transpose(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np matrix transpose operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return adapt_backend_module(arr_mod).require_symbol("swapaxes")(args[0], -1, -2)


@global_eager_registry.register("Sqrtm")
def _np_sqrtm(a: BackendArg) -> BackendArg:
    """Sqrtm.

    Args:
        a (BackendArg): Input `a` argument (a).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return a


@global_eager_registry.register("Adjoint")
def _np_adjoint(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np adjoint operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return arr_mod.conj(adapt_backend_module(arr_mod).require_symbol("swapaxes")(args[0], -1, -2))


@global_eager_registry.register("Qr")
def _np_qr(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np qr operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    return adapt_backend_module(backend_module).require_symbol("linalg").qr(args[0], mode=kwargs.get("mode", "reduced"))


@global_eager_registry.register("Cross")
def _np_cross(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np cross operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.backends.eager_registry import coerce_opt_int

    axes = kwargs.pop("axes", None)
    if axes:
        kwargs.update(
            {
                str(k): v
                for k, v in adapt_backend_module(axes).require_symbol("items")()
                if coerce_opt_int(v) is not None
            }
        )
    if "axis" in kwargs and kwargs["axis"] is None:
        kwargs.pop("axis")
    return adapt_backend_module(backend_module).require_symbol("cross")(*args, **kwargs)


@global_eager_registry.register("Slogdet")
def _np_slogdet(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np slogdet operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return adapt_backend_module(arr_mod.linalg).require_symbol("slogdet")(*args, **kwargs)


@global_eager_registry.register("LuPivotsToPermutation")
def _np_lu_pivots(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np lu pivots operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    pivots: list[int] = coerce_int_list(args[0])
    n: int = int(coerce_int(args[1]))
    perm: EagerTensorProtocol = _ensure_arr(arr_mod.arange(n))
    for i, p in enumerate(pivots):
        perm[i], perm[p] = perm[p], perm[i]
    return perm


@global_eager_registry.register("Tridiagonal")
def _np_tridiagonal(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np tridiagonal operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    a: EagerTensorProtocol = _ensure_arr(args[0])
    diag: EagerTensorProtocol = adapt_backend_module(arr_mod).require_symbol("diagonal")(a)
    off_diag: EagerTensorProtocol = adapt_backend_module(arr_mod).require_symbol("diagonal")(a, offset=1)
    q: EagerTensorProtocol = arr_mod.eye(a.shape[0])
    return diag, off_diag, q


@global_eager_registry.register("CholeskyEx")
def _np_cholesky_ex(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np cholesky ex operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    batch_sh = coerce_shape(adapt_backend_module(args[0]).require_symbol("shape"))[:-2]
    try:
        return adapt_backend_module(arr_mod.linalg).require_symbol("cholesky")(args[0]), arr_mod.zeros(
            batch_sh, dtype=arr_mod.int32
        )
    except (ArithmeticError, RuntimeError, ValueError, Exception):
        return arr_mod.zeros_like(args[0]), arr_mod.ones(batch_sh, dtype=arr_mod.int32)


@global_eager_registry.register("InvEx")
def _np_inv_ex(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np inv ex operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    batch_sh = coerce_shape(adapt_backend_module(args[0]).require_symbol("shape"))[:-2]
    try:
        return adapt_backend_module(arr_mod.linalg).require_symbol("inv")(args[0]), arr_mod.zeros(
            batch_sh, dtype=arr_mod.int32
        )
    except (ArithmeticError, RuntimeError, ValueError, Exception):
        return arr_mod.zeros_like(args[0]), arr_mod.ones(batch_sh, dtype=arr_mod.int32)


@global_eager_registry.register("Pinv")
def _np_pinv(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np pinv operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    return adapt_backend_module(arr_mod.linalg).require_symbol("pinv")(args[0], **kwargs)


@global_eager_registry.register("SolveEx")
def _np_solve_ex(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np solve ex operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    batch_sh = coerce_shape(adapt_backend_module(args[0]).require_symbol("shape"))[:-2]
    try:
        return adapt_backend_module(arr_mod.linalg).require_symbol("solve")(args[0], args[1]), arr_mod.zeros(
            batch_sh, dtype=arr_mod.int32
        )
    except (ArithmeticError, RuntimeError, ValueError, Exception):
        return arr_mod.zeros_like(args[1]), arr_mod.ones(batch_sh, dtype=arr_mod.int32)


@global_eager_registry.register("Svd")
def _np_svd(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np svd operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    arr_mod = _resolve_eager_mod(backend_module)
    full_matrices: bool = bool(kwargs.get("full_matrices", True))
    compute_uv: bool = bool(kwargs.get("compute_uv", True))
    return adapt_backend_module(arr_mod.linalg).require_symbol("svd")(
        args[0], full_matrices=full_matrices, compute_uv=compute_uv
    )
