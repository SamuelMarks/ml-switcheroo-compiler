"""Linear algebra operations package."""

from ml_switcheroo_compiler.ops.linalg.basic import (
    ConvGeneralDilated,
    Dot,
    DotGeneral,
    Einsum,
    Fft,
    Matmul,
    Rfft,
)

from .frontend import (
    cross as cross,
)
from .frontend import (
    dot as dot,
)
from .frontend import (
    dot_general as dot_general,
)
from .frontend import (
    einsum as einsum,
)
from .frontend import (
    inner as inner,
)
from .frontend import (
    matmul as matmul,
)
from .frontend import (
    outer as outer,
)
from .frontend import (
    tensordot as tensordot,
)
from .frontend import (
    vdot as vdot,
)

from .decompositions import (
    cholesky as cholesky,
    det as det,
    eigh as eigh,
    eigvalsh as eigvalsh,
    inv as inv,
    lu as lu,
    lu_factor as lu_factor,
    matrix_power as matrix_power,
    pinv as pinv,
    qr as qr,
    slogdet as slogdet,
    solve as solve,
    solve_triangular as solve_triangular,
    svd as svd,
)

from .conv import (
    conv_general_dilated as conv_general_dilated,
)

from .fft import (
    fft as fft,
    rfft as rfft,
)

__all__ = [
    "ConvGeneralDilated",
    "Dot",
    "DotGeneral",
    "Einsum",
    "Fft",
    "Matmul",
    "Rfft",
    "cholesky",
    "conv_general_dilated",
    "cross",
    "det",
    "dot",
    "dot_general",
    "eigh",
    "eigvalsh",
    "einsum",
    "fft",
    "inner",
    "inv",
    "lu",
    "lu_factor",
    "matmul",
    "matrix_power",
    "outer",
    "pinv",
    "qr",
    "rfft",
    "slogdet",
    "solve",
    "solve_triangular",
    "svd",
    "tensordot",
    "vdot",
]
