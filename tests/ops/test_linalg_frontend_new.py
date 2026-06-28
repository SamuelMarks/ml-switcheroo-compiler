import numpy as np

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.tracing.tracer import _tracer
from ml_switcheroo_compiler.ops.linalg import frontend
from ml_switcheroo_compiler.ops.linalg import basic


def _test_op(func, *args, **kwargs):
    with ConfigContext(eager_mode=True):
        out_eager = func(*args, **kwargs)
    with ConfigContext(eager_mode=False):
        _tracer.start_tracing()
        try:
            out_traced = func(*args, **kwargs)
        finally:
            _tracer.stop_tracing()
    return out_eager, out_traced


def test_frontend_ops():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    b = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))

    e, t = _test_op(frontend.matrix_norm, a)
    assert e.shape == ()

    e, t = _test_op(frontend.vector_norm, b)
    assert e.shape == ()

    e, t = _test_op(frontend.svdvals, a)
    assert e.shape == (2,)

    e, t = _test_op(frontend.tensorinv, a, ind=1)
    assert e.shape == (2, 2)

    e, t = _test_op(frontend.tensorsolve, a, b)
    assert e.shape == (2,)

    e, t = _test_op(frontend.diagonal, a)
    assert e.shape == (2,)

    e, t = _test_op(frontend.multi_dot, [a, a])
    assert e.shape == (2, 2)

    e, t = _test_op(frontend.vecdot, b, b)
    assert e.shape == ()


def test_opdefs_infer_shapes():
    ops = [basic.MatrixNorm(), basic.VectorNorm(), basic.Diagonal(), basic.MultiDot()]
    for op in ops:
        assert op.infer_shape() == ()

    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    b = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))
    assert basic.Svdvals().infer_shape(a) == (2,)
    assert basic.Tensorinv().infer_shape(a) == (2, 2)
    assert basic.Tensorsolve().infer_shape(a, b) == (2,)

    assert basic.ConvTranspose().infer_shape(a) == (2, 2)
    assert basic.ConvTranspose().infer_shape() == ()

    from ml_switcheroo_compiler.ops.linalg import CustomLinearSolve, Vecdot, CustomRoot

    assert CustomLinearSolve().infer_shape(a) is a
    assert CustomLinearSolve().infer_shape() == ()
    assert Vecdot().infer_shape(a) is a
    assert Vecdot().infer_shape() == ()
    assert CustomRoot().infer_shape(a) is a
    assert CustomRoot().infer_shape() == ()

    from ml_switcheroo_compiler.ops.linalg.decompositions import LuPivotsToPermutation

    p = Tensor(np.array([1, 0]), TensorConfig((2,), "int32", "cpu"))
    assert LuPivotsToPermutation().infer_shape(p, permutation_size=2) == (2,)
