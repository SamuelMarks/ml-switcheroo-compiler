"""Thin NumPy/SciPy linear algebra decomposition and solver kernel bindings."""

from __future__ import annotations

import types

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    coerce_bool,
    numpy_eager_registry,
    require_callable,
)
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module

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
            import numpy as np

            func = adapt_backend_module(np.linalg).require_symbol(
                name.lower() if name.lower() != "multidot" else "multi_dot"
            )
            return func(*args, **kwargs)

        return _wrapper

    adapt_backend_module(numpy_eager_registry).require_symbol("register")(op_name)(make_linalg_wrapper(op_name))


@numpy_eager_registry.register("CholeskySolve")
def _np_cholesky_solve(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np cholesky solve operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    if len(args) < 2:
        return args[0]

    import scipy.linalg

    b: EagerTensorProtocol = adapt_backend_module(backend_module).require_symbol("asarray")(args[0])
    u: EagerTensorProtocol = adapt_backend_module(backend_module).require_symbol("asarray")(args[1])
    upper: bool = coerce_bool(kwargs.get("upper", False))
    return adapt_backend_module(backend_module).require_symbol("array")(scipy.linalg.cho_solve((u, not upper), b))


@numpy_eager_registry.register("EighTridiagonal")
def _np_eigh_tridiagonal(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np eigh tridiagonal operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    if len(args) < 2:
        return args[0]

    import scipy.linalg

    alpha: EagerTensorProtocol = adapt_backend_module(backend_module).require_symbol("asarray")(args[0])
    beta: EagerTensorProtocol = adapt_backend_module(backend_module).require_symbol("asarray")(args[1])
    eigvals, eigvecs = scipy.linalg.eigh_tridiagonal(alpha, beta)
    return adapt_backend_module(backend_module).require_symbol("array")(eigvals), adapt_backend_module(
        backend_module
    ).require_symbol("array")(eigvecs)


@numpy_eager_registry.register("Lu")
def _np_lu(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np lu operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import scipy.linalg

    return require_callable(scipy.linalg.lu)(args[0])


@numpy_eager_registry.register("LuFactor")
def _np_lu_factor(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np lu factor operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import scipy.linalg

    return require_callable(scipy.linalg.lu_factor)(args[0])


@numpy_eager_registry.register("LuSolve")
def _np_lu_solve(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np lu solve operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import scipy.linalg

    return require_callable(scipy.linalg.lu_solve)((args[0], args[1]), args[2])


@numpy_eager_registry.register("MatrixExponential")
def _np_matrix_exponential(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np matrix exponential operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import scipy.linalg

    return require_callable(scipy.linalg.expm)(args[0])


@numpy_eager_registry.register("Hessenberg")
def _np_hessenberg(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np hessenberg operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import scipy.linalg

    return require_callable(scipy.linalg.hessenberg)(args[0], calc_q=True)


@numpy_eager_registry.register("TridiagonalSolve")
def _np_tridiagonal_solve(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np tridiagonal solve operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import scipy.linalg

    dl, d, du, b = args[0], args[1], args[2], args[3]
    return require_callable(scipy.linalg.solve_banded)(
        (1, 1),
        [
            adapt_backend_module(np).require_op("concatenate")(([0], du)),
            d,
            adapt_backend_module(np).require_op("concatenate")((dl, [0])),
        ],
        b,
    )


@numpy_eager_registry.register("Polar")
def _np_polar(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np polar operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import scipy.linalg

    return require_callable(scipy.linalg.polar)(args[0], side=kwargs.get("side", "right"))


@numpy_eager_registry.register("Qdwh")
def _np_qdwh(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Execute the eager np qdwh operation on the active backend array.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None):
            Input `backend_module` argument (backend module).
        *args (BackendArg): Positional args.
        **kwargs (BackendArg): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    import numpy as np
    import scipy.linalg

    a: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = ensure_ndarray(args[0])
    u, p = scipy.linalg.polar(a)
    return ensure_ndarray(u), ensure_ndarray(p), np.array(0, dtype=np.int32), np.array(True, dtype=bool)


@numpy_eager_registry.register("Schur")
def _np_schur(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Compute Schur decomposition of a matrix via `scipy.linalg.schur`.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active NumPy backend module.
        *args (BackendArg): Positional arguments (a).
        **kwargs (BackendArg): Keyword arguments.

    Returns:
        BackendArg: Tuple of `(T, Z)` Schur form and unitary Schur transformation matrix.
    """
    import scipy.linalg

    _ = backend_module
    return require_callable(scipy.linalg.schur)(*args, **kwargs)


@numpy_eager_registry.register("BandedTriangularSolve")
def _np_banded_triangular_solve(
    backend_module: BackendModuleProtocol | types.ModuleType | None, *args: BackendArg, **kwargs: BackendArg
) -> BackendArg:
    """Solve a banded triangular linear system via `scipy.linalg.solve_banded`.

    Args:
        backend_module (BackendModuleProtocol | types.ModuleType | None): Active NumPy backend module.
        *args (BackendArg): Positional arguments (ab, b).
        **kwargs (BackendArg): Keyword arguments (`lower`).

    Returns:
        BackendArg: Solution array `x`.
    """
    import scipy.linalg

    _ = (backend_module, kwargs)
    ab, b = np.asarray(args[0]), np.asarray(args[1])
    k = ab.shape[0] - 1
    if k == 0:
        return ensure_ndarray(b / ab[0])
    l_and_u = (1, 1) if k == 2 else (k, 0)
    return require_callable(scipy.linalg.solve_banded)(l_and_u, ab, b)
