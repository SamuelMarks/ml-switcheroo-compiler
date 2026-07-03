"""Decompositions."""

from .cholesky import cholesky as cholesky
from .det import det as det
from .det import slogdet as slogdet
from .eig import eigh as eigh
from .eig import eigvalsh as eigvalsh
from .inv import inv as inv
from .inv import pinv as pinv
from .inv import tri_inv as tri_inv
from .lu import lu_factor as lu_factor
from .lu import lu_pivots_to_permutation as lu_pivots_to_permutation
from .misc import (
    lu as lu,
)
from .misc import (
    lu_solve as lu_solve,
)
from .misc import (
    polar as polar,
)
from .misc import (
    tridiagonal_solve as tridiagonal_solve,
)
from .norms import (
    matrix_exponential as matrix_exponential,
)
from .norms import (
    matrix_power as matrix_power,
)
from .norms import (
    norm as norm,
)
from .norms import (
    power_iteration as power_iteration,
)
from .qr import (
    hessenberg as hessenberg,
)
from .qr import (
    householder_product as householder_product,
)
from .qr import (
    qr as qr,
)
from .qr import (
    schur as schur,
)
from .qr import (
    tridiagonal as tridiagonal,
)
from .solve import solve as solve
from .solve import solve_triangular as solve_triangular
from .svd import svd as svd
