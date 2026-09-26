"""Unit tests for linear algebra decomposition and solver shape inference."""

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg.decompositions.cholesky import Cholesky, CholeskyEx
from ml_switcheroo_compiler.ops.linalg.decompositions.det import Det, Slogdet
from ml_switcheroo_compiler.ops.linalg.decompositions.eig import Eig, Eigh, Eigvals, Eigvalsh
from ml_switcheroo_compiler.ops.linalg.decompositions.inv import Inv, InvEx, TriInv
from ml_switcheroo_compiler.ops.linalg.decompositions.lu import LuFactor
from ml_switcheroo_compiler.ops.linalg.decompositions.qr import HouseholderProduct, Qr
from ml_switcheroo_compiler.ops.linalg.decompositions.solve import Solve, SolveEx
from ml_switcheroo_compiler.ops.linalg.decompositions.solvers import Lu, LuSolve, TriangularSolve
from ml_switcheroo_compiler.ops.linalg.solvers import BandedTriangularSolve, CholeskySolve, Pinv


def _make_tensor(shape: tuple[int, ...]) -> Tensor:
    """Helper to create dummy tensor with specified shape.

    Args:
        shape (tuple[int, ...]): Shape of tensor.

    Returns:
        Tensor: Dummy tensor instance.
    """
    return Tensor(None, TensorConfig(shape, "float32", "cpu"))


def test_cholesky_shape_inference() -> None:
    """Verify Cholesky and CholeskyEx shape inference across 2D, 3D, and 4D tensors."""
    op = Cholesky()
    op_ex = CholeskyEx()

    # 2D square matrix
    t2 = _make_tensor((4, 4))
    assert op.infer_shape(t2) == (4, 4)
    res_ex2 = op_ex.infer_shape(t2)
    assert res_ex2 == ((4, 4), ())

    # 3D batched
    t3 = _make_tensor((2, 5, 5))
    assert op.infer_shape(t3) == (2, 5, 5)
    res_ex3 = op_ex.infer_shape(t3)
    assert res_ex3 == ((2, 5, 5), (2,))

    # 4D multi-batch
    t4 = _make_tensor((3, 2, 6, 6))
    assert op.infer_shape(t4) == (3, 2, 6, 6)
    res_ex4 = op_ex.infer_shape(t4)
    assert res_ex4 == ((3, 2, 6, 6), (3, 2))

    # Empty / None fallback
    assert op.infer_shape() == ()
    assert op_ex.infer_shape() == ()


def test_det_and_slogdet_shape_inference() -> None:
    """Verify determinant scalar and batch shape reduction."""
    op_det = Det()
    op_slog = Slogdet()

    # 2D scalar determinant
    t2 = _make_tensor((4, 4))
    assert op_det.infer_shape(t2) == ()
    assert op_slog.infer_shape(t2) == ((), ())

    # 3D batched determinant
    t3 = _make_tensor((5, 4, 4))
    assert op_det.infer_shape(t3) == (5,)
    assert op_slog.infer_shape(t3) == ((5,), (5,))

    # 4D multi-batched determinant
    t4 = _make_tensor((3, 2, 6, 6))
    assert op_det.infer_shape(t4) == (3, 2)
    assert op_slog.infer_shape(t4) == ((3, 2), (3, 2))

    # Fallback
    assert op_det.infer_shape() == ()
    assert op_slog.infer_shape() == ()


def test_eig_decompositions_shape_inference() -> None:
    """Verify eigenvalues and eigenvectors output shape calculations."""
    op_eig = Eig()
    op_eigh = Eigh()
    op_vals = Eigvals()
    op_valsh = Eigvalsh()

    # 2D matrix
    t2 = _make_tensor((4, 4))
    assert op_eig.infer_shape(t2) == ((4,), (4, 4))
    assert op_eigh.infer_shape(t2) == ((4,), (4, 4))
    assert op_vals.infer_shape(t2) == (4,)
    assert op_valsh.infer_shape(t2) == (4,)

    # 3D batched matrix
    t3 = _make_tensor((2, 6, 6))
    assert op_eig.infer_shape(t3) == ((2, 6), (2, 6, 6))
    assert op_eigh.infer_shape(t3) == ((2, 6), (2, 6, 6))
    assert op_vals.infer_shape(t3) == (2, 6)
    assert op_valsh.infer_shape(t3) == (2, 6)

    # Fallbacks
    assert op_eig.infer_shape() == ()
    assert op_eigh.infer_shape() == ()
    assert op_vals.infer_shape() == ()
    assert op_valsh.infer_shape() == ()


def test_inv_decompositions_shape_inference() -> None:
    """Verify matrix inverse and pseudo-inverse shape inference."""
    op_inv = Inv()
    op_invex = InvEx()
    op_tri = TriInv()
    op_pinv = Pinv()

    # 2D matrix
    t2 = _make_tensor((4, 4))
    assert op_inv.infer_shape(t2) == (4, 4)
    assert op_invex.infer_shape(t2) == ((4, 4), ())
    assert op_tri.infer_shape(t2) == (4, 4)

    # 3D batched
    t3 = _make_tensor((2, 3, 3))
    assert op_inv.infer_shape(t3) == (2, 3, 3)
    assert op_invex.infer_shape(t3) == ((2, 3, 3), (2,))
    assert op_tri.infer_shape(t3) == (2, 3, 3)

    # Rectangular pseudo-inverse (M, N) -> (N, M)
    rect = _make_tensor((2, 3, 5))
    assert op_pinv.infer_shape(rect) == (2, 5, 3)

    # Fallbacks
    assert op_inv.infer_shape() == ()
    assert op_invex.infer_shape() == ()
    assert op_tri.infer_shape() == ()
    assert op_pinv.infer_shape(None) == ()


def test_lu_decompositions_shape_inference() -> None:
    """Verify LU factorization, factors P, L, U and solve shape calculations."""
    op_factor = LuFactor()
    op_lu = Lu()
    op_solve = LuSolve()

    # Rectangular matrix (3, 4, 6)
    t = _make_tensor((3, 4, 6))
    assert op_factor.infer_shape(t) == ((3, 4, 6), (3, 4))

    # P, L, U factor shapes for (2, 5, 7)
    t_lu = _make_tensor((2, 5, 7))
    assert op_lu.infer_shape(t_lu) == ((2, 5, 5), (2, 5, 5), (2, 5, 7))

    # LU Solve broadcasting
    lu_mat = _make_tensor((2, 1, 4, 4))
    rhs_mat = _make_tensor((1, 3, 4, 5))
    assert op_solve.infer_shape(lu_mat, None, rhs_mat) == (2, 3, 4, 5)

    # LU Solve 1D RHS
    rhs_1d = _make_tensor((1, 3, 4))
    assert op_solve.infer_shape(lu_mat, None, rhs_1d) == (2, 3, 4)
    # len(args) == 2 branch (line 99)
    assert op_solve.infer_shape(lu_mat, rhs_1d) == (2, 3, 4)

    # 1D / 0D rank fallbacks for LU and LuSolve
    t_1d = _make_tensor((4,))
    t_0d = _make_tensor(())
    assert op_lu.infer_shape(t_1d) == ((4,), (4,), (4,))
    assert op_solve.infer_shape(t_1d, None, rhs_1d) == (1, 3, 4)
    assert op_solve.infer_shape(lu_mat, None, t_0d) == ()

    # Fallbacks
    assert op_factor.infer_shape() == ()
    assert op_lu.infer_shape() == ()
    assert op_solve.infer_shape() == ()


def test_qr_and_householder_shape_inference() -> None:
    """Verify QR decomposition modes and Householder product shapes."""
    op_qr = Qr()
    op_hh = HouseholderProduct()

    t = _make_tensor((4, 6, 3))
    # Reduced mode: Q (4, 6, 3), R (4, 3, 3)
    assert op_qr.infer_shape(t, mode="reduced") == ((4, 6, 3), (4, 3, 3))

    # Complete mode: Q (4, 6, 6), R (4, 6, 3)
    assert op_qr.infer_shape(t, mode="complete") == ((4, 6, 6), (4, 6, 3))

    # R mode: R (4, 3, 3)
    assert op_qr.infer_shape(t, mode="r") == ((4, 3, 3),)

    # Householder product
    assert op_hh.infer_shape(t) == (4, 6, 3)
    assert op_hh.infer_shape() == ()


def test_linear_solvers_broadcasting_shape_inference() -> None:
    """Verify batch broadcasting across general and specialized linear solvers."""
    op_solve = Solve()
    op_solve_ex = SolveEx()
    op_tri_solve = TriangularSolve()
    op_chol_solve = CholeskySolve()
    op_band_solve = BandedTriangularSolve()

    a_mat = _make_tensor((2, 1, 4, 4))
    b_2d = _make_tensor((1, 3, 4, 5))
    b_1d = _make_tensor((1, 3, 4))

    # General Solve
    assert op_solve.infer_shape(a_mat, b_2d) == (2, 3, 4, 5)
    assert op_solve.infer_shape(a_mat, b_1d) == (2, 3, 4)
    assert op_solve_ex.infer_shape(a_mat, b_2d) == ((2, 3, 4, 5), (2, 3))
    assert op_solve_ex.infer_shape(a_mat, b_1d) == ((2, 3, 4), (2, 3))

    # Triangular Solve
    assert op_tri_solve.infer_shape(a_mat, b_2d) == (2, 3, 4, 5)
    assert op_tri_solve.infer_shape(a_mat, b_1d) == (2, 3, 4)

    # Cholesky Solve
    assert op_chol_solve.infer_shape(a_mat, b_2d) == (2, 3, 4, 5)
    assert op_chol_solve.infer_shape(a_mat, b_1d) == (2, 3, 4)

    # Banded Triangular Solve
    bands = _make_tensor((2, 1, 3, 4))
    assert op_band_solve.infer_shape(bands, b_2d) == (2, 3, 4, 5)
    assert op_band_solve.infer_shape(bands, b_1d) == (2, 3, 4)

    # 1D / 0D rank fallbacks
    t_1d = _make_tensor((4,))
    t_0d = _make_tensor(())
    assert op_solve.infer_shape(t_1d, b_2d) == (1, 3, 4, 5)
    assert op_solve.infer_shape(a_mat, t_0d) == ()
    assert op_solve_ex.infer_shape(t_1d, b_2d) == ((1, 3, 4, 5), ())
    assert op_solve_ex.infer_shape(a_mat, t_0d) == ((), ())
    assert op_tri_solve.infer_shape(t_1d, b_2d) == (1, 3, 4, 5)
    assert op_tri_solve.infer_shape(a_mat, t_0d) == ()

    # Fallbacks
    assert op_solve.infer_shape() == ()
    assert op_solve_ex.infer_shape() == ()
    assert op_tri_solve.infer_shape() == ()
    assert op_chol_solve.infer_shape(None, None) == ()
    assert op_band_solve.infer_shape(None, None) == ()
