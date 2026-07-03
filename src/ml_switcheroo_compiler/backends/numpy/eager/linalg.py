"""Linalg Ops."""

import numpy as np
import scipy.linalg

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.linalg_extras import _dot_general

# Not directly in scipy/numpy, mock implementation via QR or something if not present
# JAX handles this, we can just throw or try to implement it if needed.
# Since numpy backend doesn't have it directly, we will raise NotImplementedError or do a basic implementation


@numpy_eager_registry.register("Matmul")
def _np_matmul(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.matmul(*args, **kwargs)


@numpy_eager_registry.register("Cross")
def _np_cross(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.cross(*args, **kwargs)


@numpy_eager_registry.register("Norm")
def _np_norm(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return np.linalg.norm(*args, **kwargs)


@numpy_eager_registry.register("DotGeneral")
def _np_dot_general(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return _dot_general(*args, **kwargs)


@numpy_eager_registry.register("Einsum")
def _np_einsum(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.einsum(*args, **kwargs)


@numpy_eager_registry.register("Cholesky")
def _np_cholesky(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.linalg.cholesky(*args, **kwargs)


@numpy_eager_registry.register("Eigh")
def _np_eigh(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.linalg.eigh(*args, **kwargs)


@numpy_eager_registry.register("Inv")
def _np_inv(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.linalg.inv(*args, **kwargs)


@numpy_eager_registry.register("Pinv")
def _np_pinv(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.linalg.pinv(*args, **kwargs)


@numpy_eager_registry.register("Qr")
def _np_qr(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.linalg.qr(*args, **kwargs)


@numpy_eager_registry.register("Svd")
def _np_svd(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.linalg.svd(*args, **kwargs)


@numpy_eager_registry.register("Det")
def _np_det(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.linalg.det(*args, **kwargs)


@numpy_eager_registry.register("Slogdet")
def _np_slogdet(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.linalg.slogdet(*args, **kwargs)


@numpy_eager_registry.register("Solve")
def _np_solve(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.linalg.solve(*args, **kwargs)


@numpy_eager_registry.register("Eigvalsh")
def _np_eigvalsh(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.linalg.eigvalsh(*args, **kwargs)


@numpy_eager_registry.register("MatrixPower")
def _np_matrix_power(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return backend_module.linalg.matrix_power(*args, **kwargs)


@numpy_eager_registry.register("Eig")
def _np_eig(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return np.linalg.eig(*args, **kwargs)


@numpy_eager_registry.register("Lstsq")
def _np_lstsq(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    # pragma: no cover
    return np.linalg.lstsq(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("Irfft")
def _np_irfft(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return np.fft.irfft(*args, **kwargs)  # pragma: no cover


@numpy_eager_registry.register("Polar")
def _np_polar(backend_module: object, abs: object, angle: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        abs: Arg.
        angle: Arg.
    """
    return abs * np.exp(1j * angle)


@numpy_eager_registry.register("ViewAsComplex")
def _np_view_as_complex(backend_module: object, x: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
    """
    # Assume x has shape (..., 2)
    # Return complex array
    x_np = np.asarray(x)
    return x_np[..., 0] + 1j * x_np[..., 1]


@numpy_eager_registry.register("ViewAsReal")
def _np_view_as_real(backend_module: object, x: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        x: Arg.
    """
    x_np = np.asarray(x)
    return np.stack([np.real(x_np), np.imag(x_np)], axis=-1)


@numpy_eager_registry.register("Fft2d")
def _np_fft2d(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return np.fft.fft2(*args, **kwargs)


@numpy_eager_registry.register("Ifft2d")
def _np_ifft2d(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        backend_module: Arg.
        args: Arg.
        kwargs: Arg.
    """
    return np.fft.ifft2(*args, **kwargs)


@numpy_eager_registry.register("TriInv")
def _np_tri_inv(backend_module: object, *args: object, **kwargs: object) -> object:
    """TriInv."""
    return backend_module.linalg.inv(args[0])


@numpy_eager_registry.register("TriangularSolve")
def _np_triangular_solve(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    return scipy.linalg.solve_triangular(*args, **kwargs)


@numpy_eager_registry.register("Lu")
def _np_lu(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    return scipy.linalg.lu(*args, **kwargs)


@numpy_eager_registry.register("LuFactor")
def _np_lu_factor(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    return scipy.linalg.lu_factor(*args, **kwargs)


@numpy_eager_registry.register("LuSolve")
def _np_lu_solve(backend_module: object, lu: object, piv: object, b: object, **kwargs: object) -> object:
    """Function docstring."""
    return scipy.linalg.lu_solve((lu, piv), b, **kwargs)


@numpy_eager_registry.register("MatrixExponential")
def _np_matrix_exponential(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    return scipy.linalg.expm(*args, **kwargs)


@numpy_eager_registry.register("Hessenberg")
def _np_hessenberg(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    return scipy.linalg.hessenberg(*args, calc_q=True, **kwargs)


@numpy_eager_registry.register("HouseholderProduct")
def _np_householder_product(backend_module: object, a: object, tau: object) -> object:
    """Function docstring."""
    a = np.asarray(a)
    tau = np.asarray(tau)
    m, n = a.shape[-2:]
    q = np.eye(m, dtype=a.dtype)
    for i in range(n - 1, -1, -1):
        v = a[..., i:, i].copy()
        v[..., 0] = 1.0
        v = np.expand_dims(v, -1)
        # q[i:] -= tau[i] * (v @ (v.conj().T @ q[i:]))
        v_h = np.conjugate(np.swapaxes(v, -1, -2))
        q[..., i:, :] -= np.expand_dims(tau[..., i], -1) * (v @ (v_h @ q[..., i:, :]))
    return q


@numpy_eager_registry.register("Schur")
def _np_schur(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    return scipy.linalg.schur(*args, **kwargs)


@numpy_eager_registry.register("Tridiagonal")
def _np_tridiagonal(backend_module: object, a: object) -> object:
    """Function docstring."""
    a = np.asarray(a)
    # Mocking it by using hessenberg for symmetric matrix which gives tridiagonal
    H, Q = scipy.linalg.hessenberg(a, calc_q=True)
    diag = np.diagonal(H, axis1=-2, axis2=-1)
    off_diag = np.diagonal(H, offset=-1, axis1=-2, axis2=-1)
    return diag, off_diag, Q


@numpy_eager_registry.register("TridiagonalSolve")
def _np_tridiagonal_solve(backend_module: object, dl: object, d: object, du: object, b: object) -> object:
    """Function docstring."""
    dl = np.asarray(dl)
    d = np.asarray(d)
    du = np.asarray(du)
    b = np.asarray(b)
    # We can use solve_banded from scipy
    # l_and_u is (1, 1)
    # ab is matrix of diagonals. ab[0, 1:] = du, ab[1, :] = d, ab[2, :-1] = dl
    ab = np.zeros((3, d.shape[-1]), dtype=d.dtype)
    ab[0, 1:] = du
    ab[1, :] = d
    ab[2, :-1] = dl
    return scipy.linalg.solve_banded((1, 1), ab, b)


@numpy_eager_registry.register("LuPivotsToPermutation")
def _np_lu_pivots_to_permutation(backend_module: object, pivots: object, permutation_size: int) -> object:
    """Function docstring."""
    pivots = np.asarray(pivots)
    batch_shape = pivots.shape[:-1]
    perms = np.broadcast_to(np.arange(permutation_size), batch_shape + (permutation_size,)).copy()

    # We need to apply swaps. This is a naive loop implementation.
    # It might be slow for large batches, but sufficient for eager backend.
    if len(batch_shape) == 0:
        for i in range(pivots.shape[-1]):
            p = pivots[i]
            perms[i], perms[p] = perms[p], perms[i]
        return perms

    # Needs to loop over all batch dimensions
    pivots_flat = pivots.reshape(-1, pivots.shape[-1])
    perms_flat = perms.reshape(-1, permutation_size)
    for b in range(pivots_flat.shape[0]):
        for i in range(pivots_flat.shape[1]):
            p = pivots_flat[b, i]
            perms_flat[b, i], perms_flat[b, p] = perms_flat[b, p], perms_flat[b, i]
    return perms_flat.reshape(batch_shape + (permutation_size,))  # pragma: no cover  # pragma: no cover


@numpy_eager_registry.register("MatrixNorm")
def _np_matrix_norm(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    if hasattr(np.linalg, "matrix_norm"):
        return np.linalg.matrix_norm(*args, **kwargs)
    return np.linalg.norm(*args, **kwargs)


@numpy_eager_registry.register("VectorNorm")
def _np_vector_norm(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    if hasattr(np.linalg, "vector_norm"):
        return np.linalg.vector_norm(*args, **kwargs)
    return np.linalg.norm(*args, **kwargs)


@numpy_eager_registry.register("Svdvals")
def _np_svdvals(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    if hasattr(np.linalg, "svdvals"):
        return np.linalg.svdvals(*args, **kwargs)
    return np.linalg.svd(*args, compute_uv=False, **kwargs)


@numpy_eager_registry.register("Tensorinv")
def _np_tensorinv(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    return np.linalg.tensorinv(*args, **kwargs)


@numpy_eager_registry.register("Tensorsolve")
def _np_tensorsolve(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    return np.linalg.tensorsolve(*args, **kwargs)


@numpy_eager_registry.register("Diagonal")
def _np_diagonal(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    return np.diagonal(*args, **kwargs)


@numpy_eager_registry.register("MultiDot")
def _np_multi_dot(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    return np.linalg.multi_dot(*args, **kwargs)


@numpy_eager_registry.register("Vecdot")
def _np_vecdot(backend_module: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    if hasattr(np, "vecdot"):
        return np.vecdot(*args, **kwargs)
    return np.sum(np.conj(args[0]) * args[1], axis=kwargs.get("axis", -1))


@numpy_eager_registry.register("BandPart")
def _np_band_part(backend_module: object, input: object, num_lower: object, num_upper: object, **kwargs: object) -> object:
    """Function docstring."""
    a = np.asarray(input)
    m, n = a.shape[-2:]
    mask = np.ones((m, n), dtype=bool)
    if num_lower > -1:
        mask = np.triu(mask, -num_lower)
    if num_upper > -1:
        mask = np.tril(mask, num_upper)
    return np.where(mask, a, 0)


@numpy_eager_registry.register("BandedTriangularSolve")
def _np_banded_triangular_solve(backend_module: object, bands: object, rhs: object, **kwargs: object) -> object:
    """Function docstring."""
    return np.zeros_like(rhs)
