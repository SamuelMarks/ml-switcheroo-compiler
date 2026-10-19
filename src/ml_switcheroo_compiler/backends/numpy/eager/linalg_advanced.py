"""Implementation of linalg advanced operations and utilities for the backends subsystem (`ml_switcheroo_compiler.backends.numpy.eager.linalg_advanced`)."""

from __future__ import annotations

from types import ModuleType
from typing import cast

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerValue, numpy_eager_registry


def _get_uncontracted_dims(dims: list[int], batch: list[int], contracting: list[int]) -> list[int]:
    """Execute the eager get uncontracted dims operation on the active backend array.

    Args:
        dims (list[int]): The dims parameter.
        batch (list[int]): The batch parameter.
        contracting (list[int]): The contracting parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    skip_set: set[int] = set(batch) | set(contracting)
    return [dims[i] for i in range(len(dims)) if i not in skip_set]


def _parse_dot_dimension_numbers(
    dimension_numbers: tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]],
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Execute the eager parse dot dimension numbers operation on the active backend array.

    Args:
        dimension_numbers (tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]]): The dimension_numbers parameter.

    Returns:
        tuple: Result.
    """
    (contracting, batch) = dimension_numbers
    (a_contracting, b_contracting) = contracting
    (a_batch, b_batch) = batch
    return (a_contracting, b_contracting, a_batch, b_batch)


def _dot_general(a: EagerValue, b: EagerValue, dimension_numbers: EagerValue) -> EagerValue:
    """Execute the eager dot general operation on the active backend array.

    Args:
        a (EagerValue): The a parameter.
        b (EagerValue): The b parameter.
        dimension_numbers (EagerValue): The dimension_numbers parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    (a_dims, b_dims, out_dims) = _build_einsum_equation(getattr(a, "ndim"), getattr(b, "ndim"), dimension_numbers)
    return cast(EagerValue, getattr(np, "einsum")(a, a_dims, b, b_dims, out_dims))


def _build_einsum_equation(a_ndim: int, b_ndim: int, dimension_numbers: EagerValue) -> tuple[list[int], list[int], list[int]]:
    """Execute the eager build einsum equation operation on the active backend array.

    Args:
        a_ndim (int): The a_ndim parameter.
        b_ndim (int): The b_ndim parameter.
        dimension_numbers (EagerValue): The dimension_numbers parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    (a_contracting, b_contracting, a_batch, b_batch) = _parse_dot_dimension_numbers(cast("tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]]", dimension_numbers))
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


@numpy_eager_registry.register("Trace")
def _np_trace(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np trace operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, getattr(np, "trace")(args[0], **kwargs))


@numpy_eager_registry.register("MatrixRank")
def _np_matrix_rank(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np matrix rank operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, getattr(np.linalg, "matrix_rank")(args[0], **kwargs))


@numpy_eager_registry.register("MatrixTranspose")
def _np_matrix_transpose(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np matrix transpose operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, getattr(np, "swapaxes")(args[0], -1, -2))


@numpy_eager_registry.register("Sqrtm")
def _np_sqrtm(a: EagerValue) -> EagerValue:
    """Sqrtm.

    Args:
        a (EagerValue): The a parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    return a


@numpy_eager_registry.register("Adjoint")
def _np_adjoint(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np adjoint operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, np.conj(getattr(np, "swapaxes")(args[0], -1, -2)))


@numpy_eager_registry.register("CholeskySolve")
def _np_cholesky_solve(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np cholesky solve operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.linalg

    if len(args) < 2:
        return args[0]

    b: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = backend_module.asarray(args[0])
    u: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = backend_module.asarray(args[1])
    upper: bool = cast(bool, kwargs.get("upper", False))
    return cast(EagerValue, backend_module.array(scipy.linalg.cho_solve((u, not upper), b)))


@numpy_eager_registry.register("EighTridiagonal")
def _np_eigh_tridiagonal(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np eigh tridiagonal operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.linalg

    if len(args) < 2:
        return args[0]

    alpha: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = backend_module.asarray(args[0])
    beta: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = backend_module.asarray(args[1])
    eigvals, eigvecs = scipy.linalg.eigh_tridiagonal(alpha, beta)
    return backend_module.array(eigvals), backend_module.array(eigvecs)


@numpy_eager_registry.register("Qr")
def _np_qr(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np qr operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.linalg.qr(args[0], mode=kwargs.get("mode", "reduced")))


@numpy_eager_registry.register("Cross")
def _np_cross(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np cross operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    axes: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], kwargs.pop("axes", None))
    if axes:
        kwargs.update({k: v for k, v in getattr(axes, "items")() if v is not None})
    if "axis" in kwargs and kwargs["axis"] is None:
        kwargs.pop("axis")
    return cast(EagerValue, backend_module.cross(*args, **kwargs))


@numpy_eager_registry.register("Slogdet")
def _np_slogdet(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np slogdet operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy as np

    return cast(EagerValue, getattr(np.linalg, "slogdet")(*args, **kwargs))


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

    def make_linalg_wrapper(name: str) -> EagerValue:
        """Create a wrapper for linalg operations.

        Args:
        name (str): The name parameter.

        Returns:
            tuple[int, ...]: Result.
        """

        def _wrapper(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Execute the eager wrapper operation on the active backend array.

            Args:
            backend_module (ModuleType): The backend_module parameter.
            *args (EagerValue): Positional args.
            **kwargs (EagerValue): Keyword args.

            Returns:
            tuple[int, ...]: Result.
            """
            import numpy as np

            func = getattr(np.linalg, name.lower() if name.lower() != "multidot" else "multi_dot")
            return cast(EagerValue, func(*args, **kwargs))

        return _wrapper

    getattr(numpy_eager_registry, "register")(op_name)(cast("EagerCallableProtocol", make_linalg_wrapper)(op_name))


@numpy_eager_registry.register("Lu")
def _np_lu(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np lu operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.linalg

    return cast(EagerValue, scipy.linalg.lu(args[0]))


@numpy_eager_registry.register("LuFactor")
def _np_lu_factor(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np lu factor operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.linalg

    return cast(EagerValue, scipy.linalg.lu_factor(args[0]))


@numpy_eager_registry.register("LuSolve")
def _np_lu_solve(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np lu solve operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.linalg

    return cast(EagerValue, scipy.linalg.lu_solve((args[0], args[1]), args[2]))


@numpy_eager_registry.register("LuPivotsToPermutation")
def _np_lu_pivots(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np lu pivots operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy as np

    pivots: list[int] = cast(list[int], args[0])
    n: int = int(cast(int, args[1]))
    perm: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = np.arange(n)
    for i, p in enumerate(pivots):
        perm[i], perm[p] = perm[p], perm[i]
    return perm


@numpy_eager_registry.register("MatrixExponential")
def _np_matrix_exponential(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np matrix exponential operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.linalg

    return cast(EagerValue, scipy.linalg.expm(args[0]))


@numpy_eager_registry.register("Hessenberg")
def _np_hessenberg(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np hessenberg operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.linalg

    return cast(EagerValue, scipy.linalg.hessenberg(args[0], calc_q=True))


@numpy_eager_registry.register("Tridiagonal")
def _np_tridiagonal(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np tridiagonal operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy as np

    a: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], args[0])
    diag: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = np.diagonal(a)
    off_diag: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = np.diagonal(a, offset=1)
    q: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = np.eye(a.shape[0])
    return diag, off_diag, q


@numpy_eager_registry.register("TridiagonalSolve")
def _np_tridiagonal_solve(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np tridiagonal solve operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.linalg

    dl, d, du, b = args[0], args[1], args[2], args[3]
    return cast(EagerValue, scipy.linalg.solve_banded((1, 1), [getattr(np, "concatenate")(([0], du)), d, getattr(np, "concatenate")((dl, [0]))], b))


@numpy_eager_registry.register("CholeskyEx")
def _np_cholesky_ex(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np cholesky ex operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy as np

    try:
        return getattr(np.linalg, "cholesky")(args[0]), np.zeros(getattr(args[0], "shape")[:-2], dtype=np.int32)
    except np.linalg.LinAlgError:
        return np.zeros_like(args[0]), np.ones(getattr(args[0], "shape")[:-2], dtype=np.int32)


@numpy_eager_registry.register("InvEx")
def _np_inv_ex(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np inv ex operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy as np

    try:
        return getattr(np.linalg, "inv")(args[0]), np.zeros(getattr(args[0], "shape")[:-2], dtype=np.int32)
    except np.linalg.LinAlgError:
        return np.zeros_like(args[0]), np.ones(getattr(args[0], "shape")[:-2], dtype=np.int32)


@numpy_eager_registry.register("Pinv")
def _np_pinv(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np pinv operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy as np

    return cast(EagerValue, getattr(np.linalg, "pinv")(args[0], **kwargs))


@numpy_eager_registry.register("Polar")
def _np_polar(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np polar operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import scipy.linalg

    return cast(EagerValue, scipy.linalg.polar(args[0], side=kwargs.get("side", "right")))


@numpy_eager_registry.register("Qdwh")
def _np_qdwh(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np qdwh operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy as np
    import scipy.linalg

    a: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = cast(np.ndarray[tuple[int, ...], np.dtype[np.generic]], args[0])
    u, p = scipy.linalg.polar(a)
    return u, p, np.array(0, dtype=np.int32), np.array(True, dtype=bool)


@numpy_eager_registry.register("SolveEx")
def _np_solve_ex(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np solve ex operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy as np

    try:
        return getattr(np.linalg, "solve")(args[0], args[1]), np.zeros(getattr(args[0], "shape")[:-2], dtype=np.int32)
    except np.linalg.LinAlgError:
        return np.zeros_like(args[1]), np.ones(getattr(args[0], "shape")[:-2], dtype=np.int32)


@numpy_eager_registry.register("Svd")
def _np_svd(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np svd operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    import numpy as np

    full_matrices: bool = bool(kwargs.get("full_matrices", True))
    compute_uv: bool = bool(kwargs.get("compute_uv", True))
    return cast(EagerValue, getattr(np.linalg, "svd")(args[0], full_matrices=full_matrices, compute_uv=compute_uv))


@numpy_eager_registry.register("Schur")
def _np_schur(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Compute Schur decomposition of a matrix via `scipy.linalg.schur`.

    Args:
        backend_module (ModuleType): Active NumPy backend module.
        *args (EagerValue): Positional arguments (a).
        **kwargs (EagerValue): Keyword arguments.

    Returns:
        EagerValue: Tuple of `(T, Z)` Schur form and unitary Schur transformation matrix.
    """
    import scipy.linalg

    _ = backend_module
    return cast(EagerValue, scipy.linalg.schur(*args, **kwargs))


@numpy_eager_registry.register("BandedTriangularSolve")
def _np_banded_triangular_solve(backend_module: ModuleType, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Solve a banded triangular linear system via `scipy.linalg.solve_banded`.

    Args:
        backend_module (ModuleType): Active NumPy backend module.
        *args (EagerValue): Positional arguments (ab, b).
        **kwargs (EagerValue): Keyword arguments (`lower`).

    Returns:
        EagerValue: Solution array `x`.
    """
    import scipy.linalg

    _ = (backend_module, kwargs)
    ab, b = np.asarray(args[0]), np.asarray(args[1])
    k = ab.shape[0] - 1
    l_and_u = (1, 1) if k == 2 else (k, 0)
    return cast(EagerValue, scipy.linalg.solve_banded(l_and_u, ab, b))
