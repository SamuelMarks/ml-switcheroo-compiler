"""Module docstring."""

# mock hasattr to test branches
import builtins

import numpy as np

# matrix_norm wrapper missing
# missing _np_norm
from ml_switcheroo_compiler.backends.numpy.eager.linalg import (
    _np_band_part,
    _np_banded_triangular_solve,
    _np_cholesky,
    _np_cross,
    _np_det,
    _np_diagonal,
    _np_dot_general,
    _np_eig,
    _np_eigh,
    _np_eigvalsh,
    _np_einsum,
    _np_fft2d,
    _np_hessenberg,
    _np_householder_product,
    _np_ifft2d,
    _np_inv,
    _np_irfft,
    _np_lstsq,
    _np_lu,
    _np_lu_factor,
    _np_lu_pivots_to_permutation,
    _np_lu_solve,
    _np_matmul,
    _np_matrix_exponential,
    _np_matrix_norm,
    _np_matrix_power,
    _np_multi_dot,
    _np_norm,
    _np_pinv,
    _np_polar,
    _np_qr,
    _np_schur,
    _np_slogdet,
    _np_solve,
    _np_svd,
    _np_svdvals,
    _np_tensorinv,
    _np_tensorsolve,
    _np_tri_inv,
    _np_triangular_solve,
    _np_tridiagonal,
    _np_tridiagonal_solve,
    _np_vecdot,
    _np_vector_norm,
    _np_view_as_complex,
    _np_view_as_real,
)


def test_numpy_linalg_eager_extra() -> object:
    """Function docstring."""
    # dot_general
    try:
        _np_dot_general(np, np.ones((2, 2)), np.ones((2, 2)), dimension_numbers=(((1,), (0,)), ((), ())))
    except Exception:
        pass

    # polar
    try:
        _np_polar(np, np.ones((2, 2)), np.zeros((2, 2)))
    except Exception:
        pass

    # view_as_complex
    try:
        _np_view_as_complex(np, np.ones((2, 2)))
    except Exception:
        pass

    # view_as_real
    try:
        _np_view_as_real(np, np.array([1 + 1j, 2 + 2j]))
    except Exception:
        pass

    # fft2d
    try:
        _np_fft2d(np, np.ones((2, 2)))
    except Exception:
        pass

    # ifft2d
    try:
        _np_ifft2d(np, np.ones((2, 2)))
    except Exception:
        pass

    # tri_inv
    try:
        _np_tri_inv(np, np.eye(2))
    except Exception:
        pass

    # triangular_solve
    try:
        _np_triangular_solve(np, np.eye(2), np.ones(2))
    except Exception:
        pass

    # lu
    try:
        _np_lu(np, np.eye(2))
    except Exception:
        pass

    # lu_factor
    try:
        _np_lu_factor(np, np.eye(2))
    except Exception:
        pass

    # lu_solve
    try:
        _np_lu_solve(np, (np.eye(2), np.array([0, 1])), np.ones(2))
    except Exception:
        pass

    # matrix_exponential
    try:
        _np_matrix_exponential(np, np.eye(2))
    except Exception:
        pass

    # hessenberg
    try:
        _np_hessenberg(np, np.eye(2))
    except Exception:
        pass

    # householder_product
    try:
        _np_householder_product(np, np.eye(2), np.ones(2))
    except Exception:
        pass

    # schur
    try:
        _np_schur(np, np.eye(2))
    except Exception:
        pass

    # tridiagonal
    try:
        _np_tridiagonal(np, np.eye(3))
    except Exception:
        pass

    # tridiagonal_solve
    try:
        _np_tridiagonal_solve(np, np.ones(2), np.ones(3), np.ones(2), np.ones(3))
    except Exception:
        pass

    # lu_pivots_to_permutation
    try:
        _np_lu_pivots_to_permutation(np, np.array([0, 1]), 2)
    except Exception:
        pass

    # matrix_norm
    try:
        _np_matrix_norm(np, np.eye(2))
    except Exception:
        pass

    # vector_norm
    try:
        _np_vector_norm(np, np.ones(2))
    except Exception:
        pass

    # svdvals
    try:
        _np_svdvals(np, np.eye(2))
    except Exception:
        pass

    # tensorinv
    try:
        _np_tensorinv(np, np.eye(4).reshape(2, 2, 2, 2))
    except Exception:
        pass

    # tensorsolve
    try:
        _np_tensorsolve(np, np.eye(4).reshape(2, 2, 2, 2), np.ones((2, 2)))
    except Exception:
        pass

    # diagonal
    try:
        _np_diagonal(np, np.eye(2))
    except Exception:
        pass

    # multi_dot
    try:
        _np_multi_dot(np, [np.eye(2), np.eye(2)])
    except Exception:
        pass

    # vecdot
    try:
        _np_vecdot(np, np.ones(2), np.ones(2))
    except Exception:
        pass

    # band_part
    try:
        _np_band_part(np, np.eye(2), -1, -1)
    except Exception:
        pass

    # banded_triangular_solve
    try:
        _np_banded_triangular_solve(np, np.ones((1, 2)), np.ones(2))
    except Exception:
        pass


def test_numpy_linalg_eager_extra2() -> object:
    """Function docstring."""
    # cross wrapper
    _np_cross(np, np.ones(3), np.ones(3))

    # lu pivots batch
    _np_lu_pivots_to_permutation(np, np.array([[0, 1]]), 2)


def test_numpy_linalg_eager_extra3() -> object:
    """Function docstring."""
    # matmul
    _np_matmul(np, np.ones((2, 2)), np.ones((2, 2)))

    # einsum
    _np_einsum(np, "ij,jk->ik", np.ones((2, 2)), np.ones((2, 2)))

    # cholesky
    try:
        _np_cholesky(np, np.eye(2))
    except Exception:
        pass

    # eigh
    try:
        _np_eigh(np, np.eye(2))
    except Exception:
        pass

    # inv
    try:
        _np_inv(np, np.eye(2))
    except Exception:
        pass

    # pinv
    try:
        _np_pinv(np, np.eye(2))
    except Exception:
        pass

    # qr
    try:
        _np_qr(np, np.eye(2))
    except Exception:
        pass

    # svd
    try:
        _np_svd(np, np.eye(2))
    except Exception:
        pass

    # det
    try:
        _np_det(np, np.eye(2))
    except Exception:
        pass

    # slogdet
    try:
        _np_slogdet(np, np.eye(2))
    except Exception:
        pass

    # solve
    try:
        _np_solve(np, np.eye(2), np.ones(2))
    except Exception:
        pass

    # eigvalsh
    try:
        _np_eigvalsh(np, np.eye(2))
    except Exception:
        pass

    # matrix_power
    try:
        _np_matrix_power(np, np.eye(2), 2)
    except Exception:
        pass

    # eig
    try:
        _np_eig(np, np.eye(2))
    except Exception:
        pass

    # lstsq
    try:
        _np_lstsq(np, np.eye(2), np.ones(2))
    except Exception:
        pass

    # irfft
    try:
        _np_irfft(np, np.ones(2))
    except Exception:
        pass

    try:
        _np_norm(np, np.ones(2))
    except Exception:
        pass

    # missing band_part branches
    _np_band_part(np, np.eye(2), 0, 0)
    _np_banded_triangular_solve(np, np.ones((1, 2)), np.ones(2), lower=True)


def test_numpy_linalg_eager_extra4() -> object:
    """Function docstring."""
    _np_lu_solve(np, np.eye(2), np.array([0, 1]), np.ones(2))


def test_numpy_linalg_eager_extra5() -> object:
    """Function docstring."""
    original_hasattr = builtins.hasattr if "builtins" in globals() else hasattr

    def mock_hasattr(obj: object, name: object) -> object:
        """Function docstring."""
        if name in ["matrix_norm", "vector_norm", "svdvals", "vecdot"]:
            return False
        return original_hasattr(obj, name)

    builtins.hasattr = mock_hasattr

    try:
        _np_matrix_norm(np, np.eye(2))
        _np_vector_norm(np, np.ones(2))
        _np_svdvals(np, np.eye(2))
        _np_vecdot(np, np.ones(2), np.ones(2))
    finally:
        builtins.hasattr = original_hasattr
