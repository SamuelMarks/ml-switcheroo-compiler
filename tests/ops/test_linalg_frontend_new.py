"""Module docstring."""

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg import CustomLinearSolve, CustomRoot, Vecdot, multi_dot, solvers, vecdot
from ml_switcheroo_compiler.ops.linalg import matrix_ops as frontend
from ml_switcheroo_compiler.ops.linalg.conv_ops import ConvTranspose
from ml_switcheroo_compiler.ops.linalg.decompositions import LuPivotsToPermutation
from ml_switcheroo_compiler.ops.linalg.products import Diagonal, MultiDot
from ml_switcheroo_compiler.ops.linalg.solvers import (
    MatrixNorm,
    Svdvals,
    Tensorinv,
    Tensorsolve,
    VectorNorm,
)
from ml_switcheroo_compiler.tracing.state import global_tracing_state


def _test_op(func: object, *args: object, **kwargs: object) -> object:
    """Function docstring."""
    with ConfigContext(eager_mode=True):
        out_eager = func(*args, **kwargs)
    with ConfigContext(eager_mode=False):
        global_tracing_state.start_tracing()
        try:
            out_traced = func(*args, **kwargs)
        finally:
            global_tracing_state.stop_tracing()
    return out_eager, out_traced


def test_frontend_ops() -> object:
    """Function docstring."""
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    b = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))

    e, t = _test_op(frontend.matrix_norm, a)
    assert e.shape == ()

    e, t = _test_op(frontend.vector_norm, b)
    assert e.shape == ()

    e, t = _test_op(frontend.svdvals, a)
    assert e.shape == (2,)

    e, t = _test_op(solvers.tensorinv, a, ind=1)
    assert e.shape == (2, 2)

    e, t = _test_op(solvers.tensorsolve, a, b)
    assert e.shape == (2,)

    e, t = _test_op(frontend.diagonal, a)
    assert e.shape == (2,)

    e, t = _test_op(multi_dot, [a, a])
    assert e.shape == (2, 2)

    e, t = _test_op(vecdot, b, b)
    assert e.shape == ()


def test_opdefs_infer_shapes() -> object:
    """Function docstring."""
    ops = [MatrixNorm(), VectorNorm(), Diagonal(), MultiDot()]
    for op in ops:
        assert op.infer_shape() == ()

    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), TensorConfig((2, 2), "float32", "cpu"))
    b = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))
    assert Svdvals().infer_shape(a) == (2,)
    assert Tensorinv().infer_shape(a) == (2, 2)
    assert Tensorsolve().infer_shape(a, b) == (2,)

    assert ConvTranspose().infer_shape(a) == (2, 2)
    assert ConvTranspose().infer_shape() == ()

    assert CustomLinearSolve().infer_shape(a) is a
    assert CustomLinearSolve().infer_shape() == ()
    assert Vecdot().infer_shape(a) is a
    assert Vecdot().infer_shape() == ()
    assert CustomRoot().infer_shape(a) is a
    assert CustomRoot().infer_shape() == ()

    p = Tensor(np.array([1, 0]), TensorConfig((2,), "int32", "cpu"))
    assert LuPivotsToPermutation().infer_shape(p, permutation_size=2) == (2,)
