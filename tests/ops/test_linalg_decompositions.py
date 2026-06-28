import numpy as np

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.tracing.tracer import _tracer
from ml_switcheroo_compiler.ops.linalg import decompositions as decomp
from ml_switcheroo_compiler.ops.linalg import basic
from ml_switcheroo_compiler.ops.linalg import cross


def _test_op(func, *args, **kwargs):
    # Eager
    with ConfigContext(eager_mode=True):
        out_eager = func(*args, **kwargs)

    # Tracing
    with ConfigContext(eager_mode=False):
        _tracer.start_tracing()
        try:
            out_traced = func(*args, **kwargs)
        finally:
            _tracer.stop_tracing()

    return out_eager, out_traced


def test_cholesky():
    a = Tensor(np.array([[2.0, 0.0], [0.0, 2.0]]), TensorConfig((2, 2), "float32", "cpu"))
    e, t = _test_op(decomp.cholesky, a)
    assert e.shape == (2, 2)
    assert t.shape == (2, 2)


def test_svd():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    e, t = _test_op(decomp.svd, a)
    assert e[0].shape == (2, 2)
    assert t[0].shape == (2, 2)


def test_qr():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    e, t = _test_op(decomp.qr, a)
    assert e[0].shape == (2, 2)
    assert t[0].shape == (2, 2)


def test_inv():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    e, t = _test_op(decomp.inv, a)
    assert e.shape == (2, 2)


def test_det():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    e, t = _test_op(decomp.det, a)
    assert e.shape == ()


def test_slogdet():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    e, t = _test_op(decomp.slogdet, a)
    assert e[0].shape == ()
    assert e[1].shape == ()


def test_eigh():
    a = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]), TensorConfig((2, 2), "float32", "cpu"))
    e, t = _test_op(decomp.eigh, a)
    assert e[0].shape == (2,)


def test_eigvalsh():
    a = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]), TensorConfig((2, 2), "float32", "cpu"))
    e, t = _test_op(decomp.eigvalsh, a)
    assert e.shape == (2,)


def test_matrix_power():
    a = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]), TensorConfig((2, 2), "float32", "cpu"))
    e, t = _test_op(decomp.matrix_power, a, 2)
    assert e.shape == (2, 2)


def test_solve():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    b = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))
    e, t = _test_op(decomp.solve, a, b)
    assert e.shape == (2,)


def test_tri_inv():
    a = Tensor(np.array([[1.0, 0.0], [2.0, 3.0]]), TensorConfig((2, 2), "float32", "cpu"))
    e, t = _test_op(decomp.tri_inv, a, lower=True)
    assert e.shape == (2, 2)
    assert t.shape == (2, 2)


def test_solve_triangular():
    a = Tensor(np.array([[1.0, 2.0], [0.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    b = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))
    e, t = _test_op(decomp.solve_triangular, a, b)
    assert e.shape == (2,)


def test_lu():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    e, t = _test_op(decomp.lu, a)
    assert e[0].shape == (2, 2)


def test_lu_factor_solve():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    b = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))

    with ConfigContext(eager_mode=True):
        lu_piv_e = decomp.lu_factor(a)
        out_e = decomp.lu_solve(lu_piv_e, b)

    with ConfigContext(eager_mode=False):
        _tracer.start_tracing()
        try:
            lu_piv_t = decomp.lu_factor(a)
            out_t = decomp.lu_solve(lu_piv_t, b)
        finally:
            _tracer.stop_tracing()

    assert out_e.shape == (2,)
    assert out_t.shape == (2,)


def test_norm():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    e, t = _test_op(decomp.norm, a)
    assert e.shape == ()

    e, t = _test_op(decomp.norm, a, axis=1)
    assert e.shape == (2,)

    e, t = _test_op(decomp.norm, a, axis=1, keepdims=True)
    assert e.shape == (2, 1)


def test_matrix_exponential():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    e, t = _test_op(decomp.matrix_exponential, a)
    assert e.shape == (2, 2)


def test_cross():
    a = Tensor(np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]), TensorConfig((2, 3), "float32", "cpu"))
    b = Tensor(np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]), TensorConfig((2, 3), "float32", "cpu"))
    e, t = _test_op(cross, a, b)
    assert e.shape == (2, 3)


def test_opdef_infer_shapes():
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

    assert basic.MatrixPower().infer_shape(None) == ()


def test_hessenberg():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    e, t = _test_op(decomp.hessenberg, a)
    assert e[0].shape == (2, 2)
    assert e[1].shape == (2, 2)
    assert t[0].shape == (2, 2)
    assert t[1].shape == (2, 2)


def test_householder_product():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    tau = Tensor(np.array([1.0, 1.0]), TensorConfig((2,), "float32", "cpu"))
    e, t = _test_op(decomp.householder_product, a, tau)
    assert e.shape == (2, 2)
    assert t.shape == (2, 2)


def test_schur():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    e, t = _test_op(decomp.schur, a)
    assert e[0].shape == (2, 2)
    assert e[1].shape == (2, 2)
    assert t[0].shape == (2, 2)
    assert t[1].shape == (2, 2)


def test_tridiagonal():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    e, t = _test_op(decomp.tridiagonal, a)
    assert e[0].shape == (2,)
    assert e[1].shape == (1,)
    assert e[2].shape == (2, 2)
    assert t[0].shape == (2,)
    assert t[1].shape == (1,)
    assert t[2].shape == (2, 2)


def test_tridiagonal_solve():
    dl = Tensor(np.array([1.0]), TensorConfig((1,), "float32", "cpu"))
    d = Tensor(np.array([2.0, 2.0]), TensorConfig((2,), "float32", "cpu"))
    du = Tensor(np.array([1.0]), TensorConfig((1,), "float32", "cpu"))
    b = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))
    e, t = _test_op(decomp.tridiagonal_solve, dl, d, du, b)
    assert e.shape == (2,)
    assert t.shape == (2,)


def test_lu_pivots_to_permutation():
    p = Tensor(np.array([1, 0]), TensorConfig((2,), "int32", "cpu"))
    e, t = _test_op(decomp.lu_pivots_to_permutation, p, 2)
    assert e.shape == (2,)
    assert t.shape == (2,)
