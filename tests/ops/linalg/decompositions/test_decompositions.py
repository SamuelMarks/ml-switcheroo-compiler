from ml_switcheroo_compiler.core.dtype import DType

# ruff: noqa: E501
"""Core abstractions and logic definitions for test_linalg_decompositions.py."""

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg import cross
from ml_switcheroo_compiler.ops.linalg import decompositions as decomp
from ml_switcheroo_compiler.ops.linalg.products import MatrixPower
from ml_switcheroo_compiler.tracing.state import global_tracing_state


def _test_op(func: object, *args: object, **kwargs: object) -> object:
    """Test the op behavior.

    Args:
        func (object): The func parameter.
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The inferred shape or computed result.
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


def test_cholesky() -> object:
    """Test the cholesky behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        a = Tensor(np.array([[2.0, 0.0], [0.0, 2.0]]), TensorConfig((2, 2), "float32", "cpu"))
        (e, t) = _test_op(decomp.cholesky, a)
        assert e.shape == (2, 2)
        assert t.shape == (2, 2)
    except Exception as e:
        raise e


def test_svd() -> object:
    """Test the svd behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
        (e, t) = _test_op(decomp.svd, a)
        assert e[0].shape == (2, 2)
        assert t[0].shape == (2, 2)
    except Exception as e:
        raise e


def test_qr() -> object:
    """Test the qr behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
        (e, t) = _test_op(decomp.qr, a)
        assert e[0].shape == (2, 2)
        assert t[0].shape == (2, 2)
    except Exception as e:
        raise e


def test_inv() -> object:
    """Test the inv behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
        (e, t) = _test_op(decomp.inv, a)
        assert e.shape == (2, 2)
    except Exception as e:
        raise e


def test_det() -> object:
    """Test the det behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
        (e, t) = _test_op(decomp.det, a)
        assert e.shape == ()
    except Exception as e:
        raise e


def test_slogdet() -> object:
    """Test the slogdet behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
        (e, t) = _test_op(decomp.slogdet, a)
        assert e[0].shape == ()
        assert e[1].shape == ()
    except Exception as e:
        raise e


def test_eigh() -> object:
    """Test the eigh behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        a = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]), TensorConfig((2, 2), "float32", "cpu"))
        (e, t) = _test_op(decomp.eigh, a)
        assert e[0].shape == (2,)
    except Exception as e:
        raise e


def test_eigvalsh() -> object:
    """Test the eigvalsh behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        a = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]), TensorConfig((2, 2), "float32", "cpu"))
        (e, t) = _test_op(decomp.eigvalsh, a)
        assert e.shape == (2,)
    except Exception as e:
        raise e


def test_matrix_power() -> object:
    """Test the matrix power behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        a = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]), TensorConfig((2, 2), "float32", "cpu"))
        (e, t) = _test_op(decomp.matrix_power, a, 2)
        assert e.shape == (2, 2)
    except Exception as e:
        raise e


def test_solve() -> object:
    """Test the solve behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
        b = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))
        (e, t) = _test_op(decomp.solve, a, b)
        assert e.shape == (2,)
    except Exception as e:
        raise e


def test_tri_inv() -> object:
    """Test the tri inv behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        a = Tensor(np.array([[1.0, 0.0], [2.0, 3.0]]), TensorConfig((2, 2), "float32", "cpu"))
        (e, t) = _test_op(decomp.tri_inv, a, lower=True)
        assert e.shape == (2, 2)
        assert t.shape == (2, 2)
    except Exception as e:
        raise e


def test_solve_triangular() -> object:
    """Test the solve triangular behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        a = Tensor(np.array([[1.0, 2.0], [0.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
        b = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))
        (e, t) = _test_op(decomp.solve_triangular, a, b)
        assert e.shape == (2,)
    except Exception as e:
        raise e


def test_lu() -> object:
    """Test the lu behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
        (e, t) = _test_op(decomp.lu, a)
        assert e[0].shape == (2, 2)
    except Exception as e:
        raise e


def test_lu_factor_solve() -> object:
    """Test the lu factor solve behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
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
        assert out_e.shape == (2,)
        assert out_t.shape == (2,)
    except Exception as e:
        raise e


def test_norm() -> object:
    """Test the norm behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
        (e, t) = _test_op(decomp.norm, a)
        assert e.shape == ()
        (e, t) = _test_op(decomp.norm, a, axis=1)
        assert e.shape == (2,)
        (e, t) = _test_op(decomp.norm, a, axis=1, keepdims=True)
        assert e.shape == (2, 1)
    except Exception as e:
        raise e


def test_matrix_exponential() -> object:
    """Test the matrix exponential behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
        (e, t) = _test_op(decomp.matrix_exponential, a)
        assert e.shape == (2, 2)
    except Exception as e:
        raise e


def test_cross() -> object:
    """Test the cross behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        a = Tensor(np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]), TensorConfig((2, 3), "float32", "cpu"))
        b = Tensor(np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]), TensorConfig((2, 3), "float32", "cpu"))
        (e, t) = _test_op(cross, a, b)
        assert e.shape == (2, 3)
    except Exception as e:
        raise e


def test_opdef_infer_shapes() -> object:
    """Test the opdef infer shapes behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        from ml_switcheroo_compiler.ops.linalg.decompositions.cholesky import Cholesky
        from ml_switcheroo_compiler.ops.linalg.decompositions.det import Det, Slogdet
        from ml_switcheroo_compiler.ops.linalg.decompositions.eig import Eigh, Eigvalsh
        from ml_switcheroo_compiler.ops.linalg.decompositions.inv import Inv, TriInv
        from ml_switcheroo_compiler.ops.linalg.decompositions.lu import LuFactor
        from ml_switcheroo_compiler.ops.linalg.decompositions.qr import Qr
        from ml_switcheroo_compiler.ops.linalg.decompositions.solve import Solve
        from ml_switcheroo_compiler.ops.linalg.decompositions.solvers import (
            Lu,
            LuSolve,
            MatrixExponential,
            Norm,
            TriangularSolve,
        )
        from ml_switcheroo_compiler.ops.linalg.decompositions.svd import Svd

        ops = [
            Cholesky(),
            Svd(),
            Qr(),
            Inv(),
            Det(),
            Slogdet(),
            Eigh(),
            Eigvalsh(),
            Solve(),
            TriInv(),
            TriangularSolve(),
            Lu(),
            LuFactor(),
            LuSolve(),
            Norm(),
            MatrixExponential(),
        ]
        for op in ops:
            assert op.infer_shape() == ()
        assert MatrixPower().infer_shape(None) == ()
    except Exception as e:
        raise e


def test_hessenberg() -> object:
    """Test the hessenberg behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
        (e, t) = _test_op(decomp.hessenberg, a)
        assert e[0].shape == (2, 2)
        assert e[1].shape == (2, 2)
        assert t[0].shape == (2, 2)
        assert t[1].shape == (2, 2)
    except Exception as e:
        raise e


def test_householder_product() -> object:
    """Test the householder product behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
        tau = Tensor(np.array([1.0, 1.0]), TensorConfig((2,), "float32", "cpu"))
        (e, t) = _test_op(decomp.householder_product, a, tau)
        assert e.shape == (2, 2)
        assert t.shape == (2, 2)
    except Exception as e:
        raise e


def test_schur() -> object:
    """Test the schur behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
        (e, t) = _test_op(decomp.schur, a)
        assert e[0].shape == (2, 2)
        assert e[1].shape == (2, 2)
        assert t[0].shape == (2, 2)
        assert t[1].shape == (2, 2)
    except Exception as e:
        raise e


def test_tridiagonal() -> object:
    """Test the tridiagonal behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
        (e, t) = _test_op(decomp.tridiagonal, a)
        assert e[0].shape == (2,)
        assert e[1].shape == (1,)
        assert e[2].shape == (2, 2)
        assert t[0].shape == (2,)
        assert t[1].shape == (1,)
        assert t[2].shape == (2, 2)
    except Exception as e:
        raise e


def test_tridiagonal_solve() -> object:
    """Test the tridiagonal solve behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        dl = Tensor(np.array([1.0]), TensorConfig((1,), "float32", "cpu"))
        d = Tensor(np.array([2.0, 2.0]), TensorConfig((2,), "float32", "cpu"))
        du = Tensor(np.array([1.0]), TensorConfig((1,), "float32", "cpu"))
        b = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))
        (e, t) = _test_op(decomp.tridiagonal_solve, dl, d, du, b)
        assert e.shape == (2,)
        assert t.shape == (2,)
    except Exception as e:
        raise e


def test_lu_pivots_to_permutation() -> object:
    """Test the lu pivots to permutation behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        p = Tensor(np.array([1, 0]), TensorConfig((2,), "int32", "cpu"))
        (e, t) = _test_op(decomp.lu_pivots_to_permutation, p, 2)
        assert e.shape == (2,)
        assert t.shape == (2,)
    except Exception as e:
        raise e


def test_eig() -> object:
    """Test the eig behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        a = Tensor(np.array([[2.0, 0.0], [0.0, 2.0]]), TensorConfig((2, 2), "float32", "cpu"))
        (e_eager, e_traced) = _test_op(decomp.eig, a)
        w_eager, v_eager = e_eager
        w_traced, v_traced = e_traced
        assert w_eager.shape == (2,)
        assert w_traced.shape == (2,)
    except Exception as e:
        raise e


def test_cholesky_ex() -> object:
    try:
        a = Tensor(np.array([[2.0, 0.0], [0.0, 2.0]]), TensorConfig((2, 2), "float32", "cpu"))
        (e, t) = _test_op(decomp.cholesky_ex, a, check_errors=True)
        assert e[0].shape == (2, 2)
    except Exception as e:
        raise e


def test_eigvals() -> object:
    try:
        a = Tensor(np.array([[2.0, 0.0], [0.0, 2.0]]), TensorConfig((2, 2), "float32", "cpu"))
        (e, t) = _test_op(decomp.eigvals, a)
        assert e.shape == (2,)
    except Exception as e:
        raise e


def test_inv_ex() -> object:
    try:
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
        (e, t) = _test_op(decomp.inv_ex, a, check_errors=True)
        assert e[0].shape == (2, 2)
    except Exception as e:
        raise e


def test_pinv() -> object:
    try:
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
        (e, t) = _test_op(decomp.pinv, a)
        assert e.shape == (2, 2)
    except Exception as e:
        raise e


def test_polar() -> object:
    try:
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
        (e, t) = _test_op(decomp.polar, a)
        assert e[0].shape == (2, 2)
    except Exception as e:
        raise e


def test_power_iteration() -> object:
    try:
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
        u = Tensor(np.array([1.0, 1.0]), TensorConfig((2,), "float32", "cpu"))
        (e, t) = _test_op(decomp.power_iteration, a, num_iters=2, u=u)
        assert e[0].shape == (2,)
    except Exception as e:
        raise e


def test_power_iteration_no_u() -> object:
    try:
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
        (e, t) = _test_op(decomp.power_iteration, a, num_iters=2)
        assert e[0].shape == (2,)
    except Exception as e:
        raise e


def test_qdwh() -> object:
    try:
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
        (e, t) = _test_op(decomp.qdwh, a)
        assert e[0].shape == (2, 2)
    except Exception as e:
        raise e


def test_solve_ex() -> object:
    try:
        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
        b = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))
        (e, t) = _test_op(decomp.solve_ex, a, b, check_errors=True)
        assert e[0].shape == (2,)
    except Exception as e:
        raise e


def test_svd_full_matrices() -> object:
    try:
        a = Tensor(np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]), TensorConfig((2, 3), "float32", "cpu"))
        (e, t) = _test_op(decomp.svd, a, full_matrices=True)
        assert e[0].shape == (2, 2)
        (e, t) = _test_op(decomp.svd, a, full_matrices=False)
        assert e[0].shape == (2, 2)
        (e, t) = _test_op(decomp.svd, a, compute_uv=False)
        assert e.shape == (2,)
    except Exception as e:
        raise e


def test_qr_complete() -> object:
    try:
        a = Tensor(np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]), TensorConfig((2, 3), "float32", "cpu"))
        (e, t) = _test_op(decomp.qr, a, mode="complete")
        assert e[0].shape == (2, 2)
    except Exception as e:
        raise e


def test_qr_r() -> object:
    try:
        a = Tensor(np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]), TensorConfig((2, 3), "float32", "cpu"))
        (e, t) = _test_op(decomp.qr, a, mode="r")
        assert e.shape == (2, 3)
    except Exception as e:
        raise e


def test_matrix_exp() -> object:
    try:
        from ml_switcheroo_compiler.ops.linalg.decompositions.norms import matrix_exp

        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
        (e, t) = _test_op(matrix_exp, a)
        assert e.shape == (2, 2)
    except Exception as e:
        raise e


def test_opdef_infer_shapes_extra() -> object:
    try:
        from ml_switcheroo_compiler.ops.linalg.decompositions.qr import Hessenberg, HouseholderProduct, Schur, Tridiagonal
        from ml_switcheroo_compiler.ops.linalg.decompositions.solvers import Polar, PowerIteration, TridiagonalSolve
        from ml_switcheroo_compiler.ops.linalg.decompositions.svd import Svd

        a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
        assert Hessenberg().infer_shape(a) == ((2, 2), (2, 2))
        assert HouseholderProduct().infer_shape() == ()
        assert Schur().infer_shape(a) == ((2, 2), (2, 2))
        assert Tridiagonal().infer_shape(a) == ((2,), (1,), (2, 2))
        assert PowerIteration().infer_shape(a) == (((2,), (2,), ()), (DType.Float32, DType.Float32, DType.Float32))
        assert Polar().infer_shape(a) == ((2, 2), (2, 2))

        b = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))
        assert TridiagonalSolve().infer_shape(a, a, a, b) == (2,)

        # for qr Qr infer_shape edge cases
        from ml_switcheroo_compiler.ops.linalg.decompositions.qr import Qr

        assert Qr().infer_shape() == ()
        assert Qr().infer_shape(a, mode="reduced") == ((2, 2), (2, 2))
        assert Qr().infer_shape(a, mode="complete") == ((2, 2), (2, 2))
        assert Qr().infer_shape(a, mode="r") == ((2, 2),)

        # for Svd infer_shape edge cases
        assert Svd().infer_shape() == ()
        assert Svd().infer_shape(a, full_matrices=True, compute_uv=True) == ((2, 2), (2,), (2, 2))
        assert Svd().infer_shape(a, full_matrices=False, compute_uv=True) == ((2, 2), (2,), (2, 2))
        assert Svd().infer_shape(a, full_matrices=True, compute_uv=False) == ((2,),)
        assert Svd().infer_shape(a, full_matrices=False, compute_uv=False) == ((2,),)

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
        assert TridiagonalMatmul().infer_shape(a, a, a, a) == (2, 2)
        assert Qdwh().infer_shape(a) == ((2, 2), (2, 2), (), ())
        assert SolveEx().infer_shape() == ()
        assert LuPivotsToPermutation().infer_shape(a, permutation_size=2) == (2, 2)
        assert LuPivotsToPermutation().infer_shape() == ()
    except Exception as e:
        raise e
