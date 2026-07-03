"""Module docstring."""

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg import cross
from ml_switcheroo_compiler.ops.linalg import decompositions as decomp
from ml_switcheroo_compiler.ops.linalg.products import MatrixPower
from ml_switcheroo_compiler.tracing.state import global_tracing_state


def _test_op(func: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    # Eager
    with ConfigContext(eager_mode=True):
        out_eager = func(*args, **kwargs)

    # Tracing
    with ConfigContext(eager_mode=False):
        global_tracing_state.start_tracing()
        try:
            out_traced = func(*args, **kwargs)
        finally:
            global_tracing_state.stop_tracing()

    return out_eager, out_traced


def test_cholesky() -> object:
    """Function docstring."""
    a = Tensor(np.array([[2.0, 0.0], [0.0, 2.0]]), TensorConfig((2, 2), "float32", "cpu"))
    e, t = _test_op(decomp.cholesky, a)
    assert e.shape == (2, 2)
    assert t.shape == (2, 2)


def test_svd() -> object:
    """Function docstring."""
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    e, t = _test_op(decomp.svd, a)
    assert e[0].shape == (2, 2)
    assert t[0].shape == (2, 2)


def test_qr() -> object:
    """Function docstring."""
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    e, t = _test_op(decomp.qr, a)
    assert e[0].shape == (2, 2)
    assert t[0].shape == (2, 2)


def test_inv() -> object:
    """Function docstring."""
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    e, t = _test_op(decomp.inv, a)
    assert e.shape == (2, 2)


def test_det() -> object:
    """Function docstring."""
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    e, t = _test_op(decomp.det, a)
    assert e.shape == ()


def test_slogdet() -> object:
    """Function docstring."""
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    e, t = _test_op(decomp.slogdet, a)
    assert e[0].shape == ()
    assert e[1].shape == ()


def test_eigh() -> object:
    """Function docstring."""
    a = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]), TensorConfig((2, 2), "float32", "cpu"))
    e, t = _test_op(decomp.eigh, a)
    assert e[0].shape == (2,)


def test_eigvalsh() -> object:
    """Function docstring."""
    a = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]), TensorConfig((2, 2), "float32", "cpu"))
    e, t = _test_op(decomp.eigvalsh, a)
    assert e.shape == (2,)


def test_matrix_power() -> object:
    """Function docstring."""
    a = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]), TensorConfig((2, 2), "float32", "cpu"))
    e, t = _test_op(decomp.matrix_power, a, 2)
    assert e.shape == (2, 2)


def test_solve() -> object:
    """Function docstring."""
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    b = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))
    e, t = _test_op(decomp.solve, a, b)
    assert e.shape == (2,)


def test_tri_inv() -> object:
    """Function docstring."""
    a = Tensor(np.array([[1.0, 0.0], [2.0, 3.0]]), TensorConfig((2, 2), "float32", "cpu"))
    e, t = _test_op(decomp.tri_inv, a, lower=True)
    assert e.shape == (2, 2)
    assert t.shape == (2, 2)


def test_solve_triangular() -> object:
    """Function docstring."""
    a = Tensor(np.array([[1.0, 2.0], [0.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    b = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))
    e, t = _test_op(decomp.solve_triangular, a, b)
    assert e.shape == (2,)


def test_lu() -> object:
    """Function docstring."""
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    e, t = _test_op(decomp.lu, a)
    assert e[0].shape == (2, 2)


def test_lu_factor_solve() -> object:
    """Function docstring."""
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


def test_norm() -> object:
    """Function docstring."""
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    e, t = _test_op(decomp.norm, a)
    assert e.shape == ()

    e, t = _test_op(decomp.norm, a, axis=1)
    assert e.shape == (2,)

    e, t = _test_op(decomp.norm, a, axis=1, keepdims=True)
    assert e.shape == (2, 1)


def test_matrix_exponential() -> object:
    """Function docstring."""
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    e, t = _test_op(decomp.matrix_exponential, a)
    assert e.shape == (2, 2)


def test_cross() -> object:
    """Function docstring."""
    a = Tensor(np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]), TensorConfig((2, 3), "float32", "cpu"))
    b = Tensor(np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]), TensorConfig((2, 3), "float32", "cpu"))
    e, t = _test_op(cross, a, b)
    assert e.shape == (2, 3)


def test_opdef_infer_shapes() -> object:
    """Function docstring."""
    ops = [
        decomp.Cholesky(),
        decomp.Svd(),
        decomp.Qr(),
        decomp.Inv(),
        decomp.Det(),
        decomp.Slogdet(),
        decomp.Eigh(),
        decomp.Eigvalsh(),
        decomp.Solve(),
        decomp.TriInv(),
        decomp.TriangularSolve(),
        decomp.Lu(),
        decomp.LuFactor(),
        decomp.LuSolve(),
        decomp.Norm(),
        decomp.MatrixExponential(),
    ]
    for op in ops:
        assert op.infer_shape() == ()

    assert MatrixPower().infer_shape(None) == ()


def test_hessenberg() -> object:
    """Function docstring."""
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    e, t = _test_op(decomp.hessenberg, a)
    assert e[0].shape == (2, 2)
    assert e[1].shape == (2, 2)
    assert t[0].shape == (2, 2)
    assert t[1].shape == (2, 2)


def test_householder_product() -> object:
    """Function docstring."""
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    tau = Tensor(np.array([1.0, 1.0]), TensorConfig((2,), "float32", "cpu"))
    e, t = _test_op(decomp.householder_product, a, tau)
    assert e.shape == (2, 2)
    assert t.shape == (2, 2)


def test_schur() -> object:
    """Function docstring."""
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    e, t = _test_op(decomp.schur, a)
    assert e[0].shape == (2, 2)
    assert e[1].shape == (2, 2)
    assert t[0].shape == (2, 2)
    assert t[1].shape == (2, 2)


def test_tridiagonal() -> object:
    """Function docstring."""
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    e, t = _test_op(decomp.tridiagonal, a)
    assert e[0].shape == (2,)
    assert e[1].shape == (1,)
    assert e[2].shape == (2, 2)
    assert t[0].shape == (2,)
    assert t[1].shape == (1,)
    assert t[2].shape == (2, 2)


def test_tridiagonal_solve() -> object:
    """Function docstring."""
    dl = Tensor(np.array([1.0]), TensorConfig((1,), "float32", "cpu"))
    d = Tensor(np.array([2.0, 2.0]), TensorConfig((2,), "float32", "cpu"))
    du = Tensor(np.array([1.0]), TensorConfig((1,), "float32", "cpu"))
    b = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))
    e, t = _test_op(decomp.tridiagonal_solve, dl, d, du, b)
    assert e.shape == (2,)
    assert t.shape == (2,)


def test_lu_pivots_to_permutation() -> object:
    """Function docstring."""
    p = Tensor(np.array([1, 0]), TensorConfig((2,), "int32", "cpu"))
    e, t = _test_op(decomp.lu_pivots_to_permutation, p, 2)
    assert e.shape == (2,)
    assert t.shape == (2,)
