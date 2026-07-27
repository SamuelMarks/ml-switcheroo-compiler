# ruff: noqa: E501
"""Linalg extras module."""

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


def _get_uncontracted_dims(dims: list[int], batch: list[int], contracting: list[int]) -> list[int]:
    """Retrieve the uncontracted dims property or mapping.

    Args:
        dims (list): Required parameter for dims.
        batch (list): Required parameter for batch.
        contracting (list): Required parameter for contracting.

    Returns:
        list: The evaluated or processed output.
    """
    skip_set = set(batch) | set(contracting)
    return [dims[i] for i in range(len(dims)) if i not in skip_set]


def _parse_dot_dimension_numbers(dimension_numbers: object) -> tuple:
    """Parse the dot dimension numbers abstract syntax tree node into its semantic representation.

    Args:
        dimension_numbers (object): Required parameter for dimension_numbers.

    Returns:
        tuple: The evaluated or processed output.
    """
    (contracting, batch) = dimension_numbers
    (a_contracting, b_contracting) = contracting
    (a_batch, b_batch) = batch
    return (a_contracting, b_contracting, a_batch, b_batch)


def _dot_general(a: object, b: object, dimension_numbers: object) -> object:
    """Evaluate and process the dot general operation.

    Args:
        a (object): Required parameter for a.
        b (object): Required parameter for b.
        dimension_numbers (object): Required parameter for dimension_numbers.

    Returns:
        object: The evaluated or processed output.
    """
    (a_dims, b_dims, out_dims) = _build_einsum_equation(a.ndim, b.ndim, dimension_numbers)
    return np.einsum(a, a_dims, b, b_dims, out_dims)


def _build_einsum_equation(a_ndim: int, b_ndim: int, dimension_numbers: object) -> tuple[list[int], list[int], list[int]]:
    """Evaluate and process the build einsum equation operation.

    Args:
        a_ndim (int): Required parameter for a_ndim.
        b_ndim (int): Required parameter for b_ndim.
        dimension_numbers (object): Required parameter for dimension_numbers.

    Returns:
        tuple: The evaluated or processed output.
    """
    (a_contracting, b_contracting, a_batch, b_batch) = _parse_dot_dimension_numbers(dimension_numbers)
    a_dims = list(range(a_ndim))
    b_dims = list(range(a_ndim, a_ndim + b_ndim))
    for i, a_b in enumerate(a_batch):
        b_dims[b_batch[i]] = a_dims[a_b]
    for i, a_c in enumerate(a_contracting):
        b_dims[b_contracting[i]] = a_dims[a_c]
    out_dims = [a_dims[i] for i in a_batch]
    out_dims.extend(_get_uncontracted_dims(a_dims, a_batch, a_contracting))
    out_dims.extend(_get_uncontracted_dims(b_dims, b_batch, b_contracting))
    return (a_dims, b_dims, out_dims)


@numpy_eager_registry.register("Trace")
def _np_trace(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the trace logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return np.trace(args[0], **kwargs)


@numpy_eager_registry.register("MatrixRank")
def _np_matrix_rank(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the matrix rank logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return np.linalg.matrix_rank(args[0], **kwargs)


@numpy_eager_registry.register("MatrixTranspose")
def _np_matrix_transpose(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the matrix transpose logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return np.swapaxes(args[0], -1, -2)


@numpy_eager_registry.register("Sqrtm")
def _np_sqrtm(a: object) -> object:
    """Sqrtm."""
    return a


@numpy_eager_registry.register("Adjoint")
def _np_adjoint(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the adjoint logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    return np.conj(np.swapaxes(args[0], -1, -2))


@numpy_eager_registry.register("CholeskySolve")
def _np_cholesky_solve(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the cholesky solve logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    import scipy.linalg

    if len(args) < 2:
        return args[0]

    b = backend_module.asarray(args[0])
    u = backend_module.asarray(args[1])
    upper = kwargs.get("upper", False)
    return backend_module.array(scipy.linalg.cho_solve((u, not upper), b))


@numpy_eager_registry.register("EighTridiagonal")
def _np_eigh_tridiagonal(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the eigh tridiagonal logic eagerly backed by NumPy.

    Args:
        backend_module (object): Required parameter for backend_module.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    import scipy.linalg

    if len(args) < 2:
        return args[0]

    alpha = backend_module.asarray(args[0])
    beta = backend_module.asarray(args[1])
    eigvals, eigvecs = scipy.linalg.eigh_tridiagonal(alpha, beta)
    return backend_module.array(eigvals), backend_module.array(eigvecs)


@numpy_eager_registry.register("Qr")
def _np_qr(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the qr logic eagerly backed by NumPy."""
    return backend_module.linalg.qr(args[0], mode=kwargs.get("mode", "reduced"))


@numpy_eager_registry.register("Cross")
def _np_cross(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the cross operation using NumPy.

    Args:
        backend_module (object): The backend module.

        *args (object): Positional arguments.

        **kwargs (object): Keyword arguments.


    Returns:
        object: The result.

    """
    axes = kwargs.pop("axes", None)
    if axes:
        kwargs.update({k: v for k, v in axes.items() if v is not None})
    if "axis" in kwargs and kwargs["axis"] is None:
        kwargs.pop("axis")
    return backend_module.cross(*args, **kwargs)


@numpy_eager_registry.register("Slogdet")
def _np_slogdet(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the slogdet operation using NumPy.

    Args:
        backend_module (object): The backend module.

        *args (object): Positional arguments.

        **kwargs (object): Keyword arguments.


    Returns:
        object: The result.

    """
    import numpy as np

    return np.linalg.slogdet(*args, **kwargs)


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

    def make_linalg_wrapper(name: str) -> object:
        """Create a wrapper for linalg operations."""

        def _wrapper(backend_module: object, *args: object, **kwargs: object) -> object:
            """Evaluate linalg_wrapper."""
            import numpy as np

            func = getattr(np.linalg, name.lower() if name.lower() != "multidot" else "multi_dot")
            return func(*args, **kwargs)

        return _wrapper

    numpy_eager_registry.register(op_name)(make_linalg_wrapper(op_name))


@numpy_eager_registry.register("Lu")
def _np_lu(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the lu operation using NumPy.

    Args:
        backend_module (object): The backend module.

        *args (object): Positional arguments.

        **kwargs (object): Keyword arguments.


    Returns:
        object: The result.

    """
    import scipy.linalg

    return scipy.linalg.lu(args[0])


@numpy_eager_registry.register("LuFactor")
def _np_lu_factor(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the lu_factor operation using NumPy.

    Args:
        backend_module (object): The backend module.

        *args (object): Positional arguments.

        **kwargs (object): Keyword arguments.


    Returns:
        object: The result.

    """
    import scipy.linalg

    return scipy.linalg.lu_factor(args[0])


@numpy_eager_registry.register("LuSolve")
def _np_lu_solve(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the lu_solve operation using NumPy.

    Args:
        backend_module (object): The backend module.

        *args (object): Positional arguments.

        **kwargs (object): Keyword arguments.


    Returns:
        object: The result.

    """
    import scipy.linalg

    return scipy.linalg.lu_solve((args[0], args[1]), args[2])


@numpy_eager_registry.register("LuPivotsToPermutation")
def _np_lu_pivots(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the lu_pivots operation using NumPy.

    Args:
        backend_module (object): The backend module.

        *args (object): Positional arguments.

        **kwargs (object): Keyword arguments.


    Returns:
        object: The result.

    """
    import numpy as np

    pivots = args[0]
    n = args[1]
    perm = np.arange(n)
    for i, p in enumerate(pivots):
        perm[i], perm[p] = perm[p], perm[i]
    return perm


@numpy_eager_registry.register("MatrixExponential")
def _np_matrix_exponential(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the matrix_exponential operation using NumPy.

    Args:
        backend_module (object): The backend module.

        *args (object): Positional arguments.

        **kwargs (object): Keyword arguments.


    Returns:
        object: The result.

    """
    import scipy.linalg

    return scipy.linalg.expm(args[0])


@numpy_eager_registry.register("Hessenberg")
def _np_hessenberg(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the hessenberg operation using NumPy.

    Args:
        backend_module (object): The backend module.

        *args (object): Positional arguments.

        **kwargs (object): Keyword arguments.


    Returns:
        object: The result.

    """
    import scipy.linalg

    return scipy.linalg.hessenberg(args[0], calc_q=True)


@numpy_eager_registry.register("Tridiagonal")
def _np_tridiagonal(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the tridiagonal operation using NumPy.

    Args:
        backend_module (object): The backend module.

        *args (object): Positional arguments.

        **kwargs (object): Keyword arguments.


    Returns:
        object: The result.

    """
    import numpy as np

    a = args[0]
    diag = np.diagonal(a)
    off_diag = np.diagonal(a, offset=1)
    q = np.eye(a.shape[0])
    return diag, off_diag, q


@numpy_eager_registry.register("TridiagonalSolve")
def _np_tridiagonal_solve(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the tridiagonal_solve operation using NumPy.

    Args:
        backend_module (object): The backend module.

        *args (object): Positional arguments.

        **kwargs (object): Keyword arguments.


    Returns:
        object: The result.

    """
    import scipy.linalg

    dl, d, du, b = args[0], args[1], args[2], args[3]
    return scipy.linalg.solve_banded((1, 1), [np.concatenate(([0], du)), d, np.concatenate((dl, [0]))], b)


@numpy_eager_registry.register("CholeskyEx")
def _np_cholesky_ex(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the cholesky_ex operation using NumPy.

    Args:
        backend_module (object): The backend module.

        *args (object): Positional arguments.

        **kwargs (object): Keyword arguments.


    Returns:
        object: The result.

    """
    import numpy as np

    try:
        return np.linalg.cholesky(args[0]), np.zeros(args[0].shape[:-2], dtype=np.int32)
    except np.linalg.LinAlgError:
        return np.zeros_like(args[0]), np.ones(args[0].shape[:-2], dtype=np.int32)


@numpy_eager_registry.register("InvEx")
def _np_inv_ex(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the inv_ex operation using NumPy.

    Args:
        backend_module (object): The backend module.

        *args (object): Positional arguments.

        **kwargs (object): Keyword arguments.


    Returns:
        object: The result.

    """
    import numpy as np

    try:
        return np.linalg.inv(args[0]), np.zeros(args[0].shape[:-2], dtype=np.int32)
    except np.linalg.LinAlgError:
        return np.zeros_like(args[0]), np.ones(args[0].shape[:-2], dtype=np.int32)


@numpy_eager_registry.register("Pinv")
def _np_pinv(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the pinv operation using NumPy.

    Args:
        backend_module (object): The backend module.

        *args (object): Positional arguments.

        **kwargs (object): Keyword arguments.


    Returns:
        object: The result.

    """
    import numpy as np

    return np.linalg.pinv(args[0], **kwargs)


@numpy_eager_registry.register("Polar")
def _np_polar(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the polar operation using NumPy.

    Args:
        backend_module (object): The backend module.

        *args (object): Positional arguments.

        **kwargs (object): Keyword arguments.


    Returns:
        object: The result.

    """
    import scipy.linalg

    return scipy.linalg.polar(args[0], side=kwargs.get("side", "right"))


@numpy_eager_registry.register("Qdwh")
def _np_qdwh(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the qdwh operation using NumPy.

    Args:
        backend_module (object): The backend module.

        *args (object): Positional arguments.

        **kwargs (object): Keyword arguments.


    Returns:
        object: The result.

    """
    import numpy as np
    import scipy.linalg

    a = args[0]
    u, p = scipy.linalg.polar(a)
    return u, p, np.array(0, dtype=np.int32), np.array(True, dtype=bool)


@numpy_eager_registry.register("SolveEx")
def _np_solve_ex(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the solve_ex operation using NumPy.

    Args:
        backend_module (object): The backend module.

        *args (object): Positional arguments.

        **kwargs (object): Keyword arguments.


    Returns:
        object: The result.

    """
    import numpy as np

    try:
        return np.linalg.solve(args[0], args[1]), np.zeros(args[0].shape[:-2], dtype=np.int32)
    except np.linalg.LinAlgError:
        return np.zeros_like(args[1]), np.ones(args[0].shape[:-2], dtype=np.int32)


@numpy_eager_registry.register("Svd")
def _np_svd(backend_module: object, *args: object, **kwargs: object) -> object:
    """Evaluate the svd operation using NumPy.

    Args:
        backend_module (object): The backend module.

        *args (object): Positional arguments.

        **kwargs (object): Keyword arguments.


    Returns:
        object: The result.

    """
    import numpy as np

    full_matrices = kwargs.get("full_matrices", True)
    compute_uv = kwargs.get("compute_uv", True)
    return np.linalg.svd(args[0], full_matrices=full_matrices, compute_uv=compute_uv)
