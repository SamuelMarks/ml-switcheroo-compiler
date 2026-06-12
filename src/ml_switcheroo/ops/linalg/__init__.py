"""Linear algebra operations package."""

from ml_switcheroo.ops.linalg.basic import (
    Matmul,
    Dot,
    Einsum,
)

__all__ = [
    "Matmul",
    "Dot",
    "Einsum",
]
from .frontend import (
    cholesky as cholesky,
    cross as cross,
    det as det,
    dot as dot,
    eigh as eigh,
    eigvalsh as eigvalsh,
    einsum as einsum,
    inner as inner,
    inv as inv,
    lu as lu,
    lu_factor as lu_factor,
    matmul as matmul,
    matrix_power as matrix_power,
    outer as outer,
    pinv as pinv,
    qr as qr,
    slogdet as slogdet,
    solve as solve,
    solve_triangular as solve_triangular,
    svd as svd,
    tensordot as tensordot,
    vdot as vdot,
)
