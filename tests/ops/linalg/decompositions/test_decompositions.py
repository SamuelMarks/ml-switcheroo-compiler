"""Tests for test_decompositions."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg import cross
from ml_switcheroo_compiler.ops.linalg import decompositions as decomp
from ml_switcheroo_compiler.ops.linalg.products import MatrixPower
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from tests.testing_utils import ensure_callable, ensure_eager_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def _test_op(func: EagerTensorProtocol, *args: EagerTensorProtocol, **kwargs: EagerTensorProtocol) -> EagerValue:
    """Test the op behavior.

    Args:
        func (EagerTensorProtocol): The func parameter.
        *args (EagerTensorProtocol): Variable length argument list.
        **kwargs (EagerTensorProtocol): Arbitrary keyword arguments.

    Returns:
        EagerValue: The inferred shape or computed result.
    """
    with ConfigContext(eager_mode=True):
        out_eager = func(*args, **kwargs)
    with ConfigContext(eager_mode=False):
        global_tracing_state.start_tracing()
        try:
            out_traced = func(*args, **kwargs)
        finally:
            global_tracing_state.stop_tracing()
    return (out_eager, out_traced)


def test_cholesky() -> None:
    """Test the cholesky behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    a = Tensor(np.array([[2.0, 0.0], [0.0, 2.0]]), TensorConfig((2, 2), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.cholesky, a)
    assert ensure_eager_tensor(e).shape == (2, 2)
    assert ensure_eager_tensor(t).shape == (2, 2)


def test_svd() -> None:
    """Test the svd behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.svd, a)
    assert ensure_eager_tensor(e)[0].shape == (2, 2)
    assert ensure_eager_tensor(t)[0].shape == (2, 2)


def test_qr() -> None:
    """Test the qr behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.qr, a)
    assert ensure_eager_tensor(e)[0].shape == (2, 2)
    assert ensure_eager_tensor(t)[0].shape == (2, 2)


def test_inv() -> None:
    """Test the inv behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.inv, a)
    assert ensure_eager_tensor(e).shape == (2, 2)


def test_det() -> None:
    """Test the det behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.det, a)
    assert ensure_eager_tensor(e).shape == ()


def test_slogdet() -> None:
    """Test the slogdet behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.slogdet, a)
    assert ensure_eager_tensor(e)[0].shape == ()
    assert ensure_eager_tensor(e)[1].shape == ()


def test_eigh() -> None:
    """Test the eigh behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    a = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]), TensorConfig((2, 2), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.eigh, a)
    assert ensure_eager_tensor(e)[0].shape == (2,)


def test_eigvalsh() -> None:
    """Test the eigvalsh behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    a = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]), TensorConfig((2, 2), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.eigvalsh, a)
    assert ensure_eager_tensor(e).shape == (2,)


def test_matrix_power() -> None:
    """Test the matrix power behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    a = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]), TensorConfig((2, 2), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.matrix_power, a, 2)
    assert ensure_eager_tensor(e).shape == (2, 2)


def test_solve() -> None:
    """Test the solve behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    b = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.solve, a, b)
    assert ensure_eager_tensor(e).shape == (2,)


def test_tri_inv() -> None:
    """Test the tri inv behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    a = Tensor(np.array([[1.0, 0.0], [2.0, 3.0]]), TensorConfig((2, 2), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.tri_inv, a, lower=True)
    assert ensure_eager_tensor(e).shape == (2, 2)
    assert ensure_eager_tensor(t).shape == (2, 2)


def test_solve_triangular() -> None:
    """Test the solve triangular behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    a = Tensor(np.array([[1.0, 2.0], [0.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    b = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.solve_triangular, a, b)
    assert ensure_eager_tensor(e).shape == (2,)


def test_lu() -> None:
    """Test the lu behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.lu, a)
    assert ensure_eager_tensor(e)[0].shape == (2, 2)


def test_lu_factor_solve() -> None:
    """Test the lu factor solve behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    b = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))
    with ConfigContext(eager_mode=True):
        lu_piv_e = decomp.lu_factor(a)
        out_e = decomp.lu_solve(lu_piv_e, b)
    with ConfigContext(eager_mode=False):
        global_tracing_state.start_tracing()
        try:
            lu_piv_t = decomp.lu_factor(a)
            out_t = decomp.lu_solve(lu_piv_t, b)
        finally:
            global_tracing_state.stop_tracing()
    assert ensure_eager_tensor(out_e).shape == (2,)
    assert ensure_eager_tensor(out_t).shape == (2,)


def test_norm() -> None:
    """Test the norm behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.norm, a)
    assert ensure_eager_tensor(e).shape == ()
    (e, t) = ensure_callable(_test_op)(decomp.norm, a, axis=1)
    assert ensure_eager_tensor(e).shape == (2,)
    (e, t) = ensure_callable(_test_op)(decomp.norm, a, axis=1, keepdims=True)
    assert ensure_eager_tensor(e).shape == (2, 1)


def test_matrix_exponential() -> None:
    """Test the matrix exponential behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.matrix_exponential, a)
    assert ensure_eager_tensor(e).shape == (2, 2)


def test_cross() -> None:
    """Test the cross behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    a = Tensor(np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]), TensorConfig((2, 3), "float32", "cpu"))
    b = Tensor(np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]), TensorConfig((2, 3), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(cross, a, b)
    assert ensure_eager_tensor(e).shape == (2, 3)


def test_opdef_infer_shapes() -> None:
    """Test the opdef infer shapes behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    from ml_switcheroo_compiler.ops.linalg.decompositions.cholesky import Cholesky
    from ml_switcheroo_compiler.ops.linalg.decompositions.det import Det, Slogdet
    from ml_switcheroo_compiler.ops.linalg.decompositions.eig import Eigh, Eigvalsh
    from ml_switcheroo_compiler.ops.linalg.decompositions.inv import Inv, TriInv
    from ml_switcheroo_compiler.ops.linalg.decompositions.lu import LuFactor
    from ml_switcheroo_compiler.ops.linalg.decompositions.qr import Qr
    from ml_switcheroo_compiler.ops.linalg.decompositions.solve import Solve
    from ml_switcheroo_compiler.ops.linalg.decompositions.solvers import Lu, LuSolve, MatrixExponential, Norm, TriangularSolve
    from ml_switcheroo_compiler.ops.linalg.decompositions.svd import Svd

    ops = [Cholesky(), Svd(), Qr(), Inv(), Det(), Slogdet(), Eigh(), Eigvalsh(), Solve(), TriInv(), TriangularSolve(), Lu(), LuFactor(), LuSolve(), Norm(), MatrixExponential()]
    for op in ops:
        assert op.infer_shape() == ()
    assert ensure_callable(MatrixPower)().infer_shape(None) == ()


def test_hessenberg() -> None:
    """Test the hessenberg behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.hessenberg, a)
    assert ensure_eager_tensor(e)[0].shape == (2, 2)
    assert ensure_eager_tensor(e)[1].shape == (2, 2)
    assert ensure_eager_tensor(t)[0].shape == (2, 2)
    assert ensure_eager_tensor(t)[1].shape == (2, 2)


def test_householder_product() -> None:
    """Test the householder product behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    tau = Tensor(np.array([1.0, 1.0]), TensorConfig((2,), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.householder_product, a, tau)
    assert ensure_eager_tensor(e).shape == (2, 2)
    assert ensure_eager_tensor(t).shape == (2, 2)


def test_schur() -> None:
    """Test the schur behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.schur, a)
    assert ensure_eager_tensor(e)[0].shape == (2, 2)
    assert ensure_eager_tensor(e)[1].shape == (2, 2)
    assert ensure_eager_tensor(t)[0].shape == (2, 2)
    assert ensure_eager_tensor(t)[1].shape == (2, 2)


def test_tridiagonal() -> None:
    """Test the tridiagonal behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.tridiagonal, a)
    assert ensure_eager_tensor(e)[0].shape == (2,)
    assert ensure_eager_tensor(e)[1].shape == (1,)
    assert ensure_eager_tensor(e)[2].shape == (2, 2)
    assert ensure_eager_tensor(t)[0].shape == (2,)
    assert ensure_eager_tensor(t)[1].shape == (1,)
    assert ensure_eager_tensor(t)[2].shape == (2, 2)


def test_tridiagonal_solve() -> None:
    """Test the tridiagonal solve behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    dl = Tensor(np.array([1.0]), TensorConfig((1,), "float32", "cpu"))
    d = Tensor(np.array([2.0, 2.0]), TensorConfig((2,), "float32", "cpu"))
    du = Tensor(np.array([1.0]), TensorConfig((1,), "float32", "cpu"))
    b = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.tridiagonal_solve, dl, d, du, b)
    assert ensure_eager_tensor(e).shape == (2,)
    assert ensure_eager_tensor(t).shape == (2,)


def test_lu_pivots_to_permutation() -> None:
    """Test the lu pivots to permutation behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    p = Tensor(np.array([1, 0]), TensorConfig((2,), "int32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.lu_pivots_to_permutation, p, 2)
    assert ensure_eager_tensor(e).shape == (2,)
    assert ensure_eager_tensor(t).shape == (2,)


def test_eig() -> None:
    """Test the eig behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    a = Tensor(np.array([[2.0, 0.0], [0.0, 2.0]]), TensorConfig((2, 2), "float32", "cpu"))
    (e_eager, e_traced) = ensure_callable(_test_op)(decomp.eig, a)
    (w_eager, v_eager) = e_eager
    (w_traced, v_traced) = e_traced
    assert ensure_eager_tensor(w_eager).shape == (2,)
    assert ensure_eager_tensor(w_traced).shape == (2,)


def test_cholesky_ex() -> None:
    """Test function test_cholesky_ex."""
    a = Tensor(np.array([[2.0, 0.0], [0.0, 2.0]]), TensorConfig((2, 2), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.cholesky_ex, a, check_errors=True)
    assert ensure_eager_tensor(e)[0].shape == (2, 2)


def test_eigvals() -> None:
    """Test function test_eigvals."""
    a = Tensor(np.array([[2.0, 0.0], [0.0, 2.0]]), TensorConfig((2, 2), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.eigvals, a)
    assert ensure_eager_tensor(e).shape == (2,)


def test_inv_ex() -> None:
    """Test function test_inv_ex."""
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.inv_ex, a, check_errors=True)
    assert ensure_eager_tensor(e)[0].shape == (2, 2)


def test_pinv() -> None:
    """Test function test_pinv."""
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.pinv, a)
    assert ensure_eager_tensor(e).shape == (2, 2)


def test_polar() -> None:
    """Test function test_polar."""
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.polar, a)
    assert ensure_eager_tensor(e)[0].shape == (2, 2)


def test_power_iteration() -> None:
    """Test function test_power_iteration."""
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    u = Tensor(np.array([1.0, 1.0]), TensorConfig((2,), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.power_iteration, a, num_iters=2, u=u)
    assert ensure_eager_tensor(e)[0].shape == (2,)


def test_power_iteration_no_u() -> None:
    """Test function test_power_iteration_no_u."""
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.power_iteration, a, num_iters=2)
    assert ensure_eager_tensor(e)[0].shape == (2,)


def test_qdwh() -> None:
    """Test function test_qdwh."""
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.qdwh, a)
    assert ensure_eager_tensor(e)[0].shape == (2, 2)


def test_solve_ex() -> None:
    """Test function test_solve_ex."""
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    b = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.solve_ex, a, b, check_errors=True)
    assert ensure_eager_tensor(e)[0].shape == (2,)


def test_svd_full_matrices() -> None:
    """Test function test_svd_full_matrices."""
    a = Tensor(np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]), TensorConfig((2, 3), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.svd, a, full_matrices=True)
    assert ensure_eager_tensor(e)[0].shape == (2, 2)
    (e, t) = ensure_callable(_test_op)(decomp.svd, a, full_matrices=False)
    assert ensure_eager_tensor(e)[0].shape == (2, 2)
    (e, t) = ensure_callable(_test_op)(decomp.svd, a, compute_uv=False)
    assert ensure_eager_tensor(e).shape == (2,)


def test_qr_complete() -> None:
    """Test function test_qr_complete."""
    a = Tensor(np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]), TensorConfig((2, 3), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.qr, a, mode="complete")
    assert ensure_eager_tensor(e)[0].shape == (2, 2)


def test_qr_r() -> None:
    """Test function test_qr_r."""
    a = Tensor(np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]), TensorConfig((2, 3), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(decomp.qr, a, mode="r")
    assert ensure_eager_tensor(e).shape == (2, 3)


def test_matrix_exp() -> None:
    """Test function test_matrix_exp."""
    from ml_switcheroo_compiler.ops.linalg.decompositions.norms import matrix_exp

    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    (e, t) = ensure_callable(_test_op)(matrix_exp, a)
    assert ensure_eager_tensor(e).shape == (2, 2)


def test_opdef_infer_shapes_extra() -> None:
    """Test function test_opdef_infer_shapes_extra."""
    from ml_switcheroo_compiler.ops.linalg.decompositions.qr import Hessenberg, HouseholderProduct, Schur, Tridiagonal
    from ml_switcheroo_compiler.ops.linalg.decompositions.solvers import Polar, PowerIteration, TridiagonalSolve
    from ml_switcheroo_compiler.ops.linalg.decompositions.svd import Svd

    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    assert ensure_callable(Hessenberg)().infer_shape(a) == ((2, 2), (2, 2))
    assert HouseholderProduct().infer_shape() == ()
    assert ensure_callable(Schur)().infer_shape(a) == ((2, 2), (2, 2))
    assert ensure_callable(Tridiagonal)().infer_shape(a) == ((2,), (1,), (2, 2))
    assert ensure_callable(PowerIteration)().infer_shape(a) == (((2,), (2,), ()), (DType.Float32, DType.Float32, DType.Float32))
    assert ensure_callable(Polar)().infer_shape(a) == ((2, 2), (2, 2))
    b = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))
    assert ensure_callable(TridiagonalSolve)().infer_shape(a, a, a, b) == (2,)
    from ml_switcheroo_compiler.ops.linalg.decompositions.qr import Qr

    assert Qr().infer_shape() == ()
    assert ensure_callable(Qr)().infer_shape(a, mode="reduced") == ((2, 2), (2, 2))
    assert ensure_callable(Qr)().infer_shape(a, mode="complete") == ((2, 2), (2, 2))
    assert ensure_callable(Qr)().infer_shape(a, mode="r") == ((2, 2),)
    assert Svd().infer_shape() == ()
    assert ensure_callable(Svd)().infer_shape(a, full_matrices=True, compute_uv=True) == ((2, 2), (2,), (2, 2))
    assert ensure_callable(Svd)().infer_shape(a, full_matrices=False, compute_uv=True) == ((2, 2), (2,), (2, 2))
    assert ensure_callable(Svd)().infer_shape(a, full_matrices=True, compute_uv=False) == ((2,),)
    assert ensure_callable(Svd)().infer_shape(a, full_matrices=False, compute_uv=False) == ((2,),)
    from ml_switcheroo_compiler.ops.linalg.decompositions.cholesky import CholeskyEx
    from ml_switcheroo_compiler.ops.linalg.decompositions.eig import Eig, Eigvals
    from ml_switcheroo_compiler.ops.linalg.decompositions.inv import InvEx
    from ml_switcheroo_compiler.ops.linalg.decompositions.lu import LuPivotsToPermutation
    from ml_switcheroo_compiler.ops.linalg.decompositions.qr import Qdwh
    from ml_switcheroo_compiler.ops.linalg.decompositions.solve import SolveEx
    from ml_switcheroo_compiler.ops.linalg.decompositions.solvers import Cross, MatrixExponential, Norm, TridiagonalMatmul

    assert Eig().infer_shape() == ()
    assert CholeskyEx().infer_shape() == ()
    assert Eigvals().infer_shape() == ()
    assert InvEx().infer_shape() == ()
    assert Norm().infer_shape() == ()
    assert MatrixExponential().infer_shape() == ()
    assert Cross().infer_shape() == ()
    assert ensure_callable(TridiagonalMatmul)().infer_shape(a, a, a, a) == (2, 2)
    assert ensure_callable(Qdwh)().infer_shape(a) == ((2, 2), (2, 2), (), ())
    assert SolveEx().infer_shape() == ()
    assert LuPivotsToPermutation().infer_shape(a, permutation_size=2) == (2, 2)
    assert LuPivotsToPermutation().infer_shape() == ()
