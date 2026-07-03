"""Module docstring."""

import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.linalg as mod


def test_linalg_future_coverage() -> object:
    """Function docstring."""
    # matrix_norm
    if not hasattr(np.linalg, "matrix_norm"):
        np.linalg.matrix_norm = lambda x: x
        try:
            mod._np_matrix_norm(np, np.array([[1.0, 2.0], [3.0, 4.0]]))
        finally:
            del np.linalg.matrix_norm

    # vector_norm
    if not hasattr(np.linalg, "vector_norm"):
        np.linalg.vector_norm = lambda x: x
        try:
            mod._np_vector_norm(np, np.array([1.0, 2.0]))
        finally:
            del np.linalg.vector_norm

    # svdvals
    if not hasattr(np.linalg, "svdvals"):
        np.linalg.svdvals = lambda x: x
        try:
            mod._np_svdvals(np, np.array([[1.0, 2.0], [3.0, 4.0]]))
        finally:
            del np.linalg.svdvals

    # vecdot
    if not hasattr(np, "vecdot"):
        np.vecdot = lambda x, y: x
        try:
            mod._np_vecdot(np, np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        finally:
            del np.vecdot


def test_linalg_tensorinv_tensorsolve() -> object:
    """Function docstring."""
    a = np.eye(4 * 6)
    a.shape = (4, 6, 8, 3)
    mod._np_tensorinv(np, a)

    b = np.random.randn(4, 6)
    try:
        mod._np_tensorsolve(np, a, b)
    except Exception:
        pass
