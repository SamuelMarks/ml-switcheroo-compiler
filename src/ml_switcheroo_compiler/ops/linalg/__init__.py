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
    cholesky as cholesky,
)
from .frontend import (
    conv_general_dilated as conv_general_dilated,
)
from .frontend import (
    cross as cross,
)
from .frontend import (
    det as det,
)
from .frontend import (
    dot as dot,
)
from .frontend import (
    dot_general as dot_general,
)
from .frontend import (
    eigh as eigh,
)
from .frontend import (
    eigvalsh as eigvalsh,
)
from .frontend import (
    einsum as einsum,
)
from .frontend import (
    fft as fft,
)
from .frontend import (
    inner as inner,
)
from .frontend import (
    inv as inv,
)
from .frontend import (
    lu as lu,
)
from .frontend import (
    lu_factor as lu_factor,
)
from .frontend import (
    matmul as matmul,
)
from .frontend import (
    matrix_power as matrix_power,
)
from .frontend import (
    outer as outer,
)
from .frontend import (
    pinv as pinv,
)
from .frontend import (
    qr as qr,
)
from .frontend import (
    rfft as rfft,
)
from .frontend import (
    slogdet as slogdet,
)
from .frontend import (
    solve as solve,
)
from .frontend import (
    solve_triangular as solve_triangular,
)
from .frontend import (
    svd as svd,
)
from .frontend import (
    tensordot as tensordot,
)
from .frontend import (
    vdot as vdot,
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
