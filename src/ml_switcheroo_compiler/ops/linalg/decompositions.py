"""Linear algebra operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.linalg.frontend import _emit_linalg_node

if TYPE_CHECKING:
    pass


def cholesky(input: Tensor) -> Tensor:
    """Computes the Cholesky decomposition of a symmetric/Hermitian positive-definite.

    matrix

    Args:
        input (Tensor): The input symmetric/Hermitian positive-definite matrix

    Returns:
    Tensor: The lower-triangular or upper-triangular Cholesky factor
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Cholesky", input.data)
        return Tensor(backend.array(data), backend.array(data).shape, input.dtype, input.device)
    return _emit_linalg_node("Cholesky", [input], {}, [()], [input.dtype])


def svd(
    input: Tensor,
    full_matrices: bool = True,
    compute_uv: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Computes the Singular Value Decomposition (SVD) of a matrix.

    Args:
        input (Tensor): The input matrix of shape (..., M, N)
        full_matrices (bool): If True, matrices U and Vh have shapes (..., M, M)
        and (..., N, N). Otherwise, shapes are (..., M, K) and (..., K, N)
        where K = min(M, N). Defaults to True
        compute_uv (bool): Whether to compute U and Vh in addition to S. Defaults to
        True

    Returns:
    tuple[Tensor, Tensor, Tensor]: A tuple containing:
        - U (Tensor): Left singular vectors
        - S (Tensor): Singular values
        - Vh (Tensor): Right singular vectors (conjugate transposed)
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        u, s, vh = backend.execute_op(
            "Svd",
            input.data,
            full_matrices=full_matrices,
            compute_uv=compute_uv,
        )
        return (
            Tensor(u, u.shape, input.dtype, input.device),
            Tensor(s, s.shape, input.dtype, input.device),
            Tensor(vh, vh.shape, input.dtype, input.device),
        )
    return _emit_linalg_node(
        "Svd",
        [input],
        {"full_matrices": full_matrices, "compute_uv": compute_uv},
        [(), (), ()],
        [input.dtype] * 3,
    )


def qr(input: Tensor, mode: str = "reduced") -> tuple[Tensor, Tensor]:
    """Computes the QR decomposition of a matrix.

    Args:
        input (Tensor): The input matrix
        mode (str): Specifies the mode of decomposition ('reduced', 'complete',
        'r', or 'raw'). Defaults to 'reduced'

    Returns:
    tuple[Tensor, Tensor]: A tuple containing:
        - Q (Tensor): The orthonormal matrix
        - R (Tensor): The upper-triangular matrix
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        q, r = backend.execute_op("Qr", input.data, mode=mode)
        return (
            Tensor(q, q.shape, input.dtype, input.device),
            Tensor(r, r.shape, input.dtype, input.device),
        )
    return _emit_linalg_node(
        "Qr",
        [input],
        {"mode": mode},
        [(), ()],
        [input.dtype] * 2,
    )


def inv(input: Tensor) -> Tensor:
    """Computes the multiplicative inverse of a square matrix.

    Args:
        input (Tensor): The square matrix to invert

    Returns:
    Tensor: The multiplicative inverse of the input matrix
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Inv", input.data)
        return Tensor(backend.array(data), backend.array(data).shape, input.dtype, input.device)
    return _emit_linalg_node("Inv", [input], {}, [()], [input.dtype])


def pinv(input: Tensor, rcond: float = 1e-15) -> Tensor:
    """Computes the Moore-Penrose pseudo-inverse of a matrix.

    Args:
        input (Tensor): The matrix to invert
        rcond (float): Cutoff for small singular values. Defaults to 1e-15

    Returns:
    Tensor: The pseudo-inverse of the input matrix
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Pinv", input.data, rcond=rcond)
        return Tensor(data, data.shape, input.dtype, input.device)
    return _emit_linalg_node(
        "Pinv",
        [input],
        {"rcond": rcond},
        [()],
        [input.dtype],
    )


def det(input: Tensor) -> Tensor:
    """Computes the determinant of a square matrix.

    Args:
        input (Tensor): The square matrix

    Returns:
    Tensor: The determinant of the input matrix
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Det", input.data)
        return Tensor(backend.array(data), backend.array(data).shape, input.dtype, input.device)
    return _emit_linalg_node("Det", [input], {}, [()], [input.dtype])


def slogdet(input: Tensor) -> tuple[Tensor, Tensor]:
    """Computes the sign and natural logarithm of the determinant of a square matrix.

    Args:
        input (Tensor): The square matrix

    Returns:
    tuple[Tensor, Tensor]: A tuple containing:
        - sign (Tensor): A number representing the sign of the determinant
    - logdet (Tensor): The natural logarithm of the absolute value of the
    determinant
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        sign, logdet = backend.execute_op("Slogdet", input.data)
        return (
            Tensor(backend.array(sign), backend.array(sign).shape, input.dtype, input.device),
            Tensor(backend.array(logdet), backend.array(logdet).shape, input.dtype, input.device),
        )
    return _emit_linalg_node("Slogdet", [input], {}, [(), ()], [input.dtype] * 2)


def eigh(input: Tensor, UPLO: str = "L") -> tuple[Tensor, Tensor]:
    """Computes the eigenvalues and eigenvectors of a complex Hermitian or real symmetric.

    matrix

    Args:
        input (Tensor): The symmetric or Hermitian matrix
        UPLO (str): Specifies whether the calculation is done with the lower ('L')
        or upper ('U') triangular part of the matrix. Defaults to 'L'

    Returns:
    tuple[Tensor, Tensor]: A tuple containing:
        - eigenvalues (Tensor): The eigenvalues in ascending order
        - eigenvectors (Tensor): The column eigenvectors
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        w, v = backend.execute_op("Eigh", input.data, UPLO=UPLO)
        return (
            Tensor(w, w.shape, input.dtype, input.device),
            Tensor(v, v.shape, input.dtype, input.device),
        )
    return _emit_linalg_node(
        "Eigh",
        [input],
        {"UPLO": UPLO},
        [(), ()],
        [input.dtype] * 2,
    )


def eigvalsh(input: Tensor, UPLO: str = "L") -> Tensor:
    """Computes the eigenvalues of a complex Hermitian or real symmetric matrix.

    Args:
        input (Tensor): The symmetric or Hermitian matrix
        UPLO (str): Specifies whether the calculation is done with the lower ('L')
        or upper ('U') triangular part of the matrix. Defaults to 'L'

    Returns:
    Tensor: The eigenvalues in ascending order
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Eigvalsh", input.data, UPLO=UPLO)
        return Tensor(backend.array(data), backend.array(data).shape, input.dtype, input.device)
    return _emit_linalg_node(
        "Eigvalsh",
        [input],
        {"UPLO": UPLO},
        [()],
        [input.dtype],
    )


def matrix_power(input: Tensor, n: int) -> Tensor:
    """Raises a square matrix to the integer power `n`.

    Args:
        input (Tensor): The square matrix
        n (int): The exponent

    Returns:
    Tensor: The matrix raised to the power `n`
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("MatrixPower", input.data, n)
        return Tensor(data, data.shape, input.dtype, input.device)
    return _emit_linalg_node("MatrixPower", [input], {"n": n}, [()], [input.dtype])


def solve(a: object, b: object) -> object:
    """Solves a linear matrix equation, or system of linear scalar equations.

    Args:
        a (object): Coefficient matrix
        b (object): Ordinate or 'dependent variable' values

    Returns:
    object: Solution to the system of linear equations
    """
    from ml_switcheroo_compiler.backends.registry import get_active_backend

    backend = get_active_backend()
    return backend.execute_op(
        "Solve",
        (a.data if hasattr(a, "device") else a),
        (b.data if hasattr(b, "device") else b),
    )


def solve_triangular(
    a: object,
    b: object,
    trans: int = 0,
    lower: bool = False,
    unit_diagonal: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> object:
    """Solves the equation `a x = b` for `x`, assuming `a` is a triangular matrix.

    Args:
        a (object): Triangular coefficient matrix
        b (object): Right-hand side matrix or vector
        trans (int): Type of system to solve. 0 for `a x = b`, 1 for `a^T x = b`,
        2 for `a^H x = b`. Defaults to 0
        lower (bool): Use only the lower triangular part of `a`. If False, use the
        upper triangular part. Defaults to False
        unit_diagonal (bool): If True, diagonal elements of `a` are assumed to be 1
        Defaults to False
        overwrite_b (bool): Allow overwriting data in `b` for speed. Defaults to False
        check_finite (bool): Whether to check that the input matrices contain only
        finite numbers. Defaults to True

    Returns:
    object: The solution matrix `x`
    """
    import scipy.linalg as spla

    return spla.solve_triangular(
        a,
        b,
        trans=trans,
        lower=lower,
        unit_diagonal=unit_diagonal,
        overwrite_b=overwrite_b,
        check_finite=check_finite,
    )


def lu(
    a: object,
    permute_l: bool = False,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> object:
    """Computes the LU decomposition of a matrix.

    Args:
        a (object): The input matrix to decompose
        permute_l (bool): If True, perform the multiplication P * L and return
        only PL and U. Defaults to False
        overwrite_a (bool): Allow overwriting data in `a` for speed. Defaults to False
        check_finite (bool): Whether to check that the input matrix contains only
        finite numbers. Defaults to True

    Returns:
    object: The LU decomposition components (P, L, U) or (PL, U) depending on
    `permute_l`
    """
    import scipy.linalg as spla

    return spla.lu(
        a,
        permute_l=permute_l,
        overwrite_a=overwrite_a,
        check_finite=check_finite,
    )


def lu_factor(
    a: object,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> object:
    """Computes pivoted LU decomposition of a matrix for use in `lu_solve`.

    Args:
        a (object): The input matrix to decompose
        overwrite_a (bool): Allow overwriting data in `a` for speed. Defaults to False
        check_finite (bool): Whether to check that the input matrix contains only
        finite numbers. Defaults to True

    Returns:
    object: A tuple (lu, piv) containing the LU factorization and pivot indices
    """
    import scipy.linalg as spla

    return spla.lu_factor(a, overwrite_a=overwrite_a, check_finite=check_finite)
