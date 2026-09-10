"""Parity, chunk-aware shape propagation, and lazy task graph tests for Dask backend."""

import dask.array as da
import pytest

from ml_switcheroo_compiler.backends.dask.eager import (
    _verify_dask_task_graph,
    dask_avgpool2d,
    dask_conv2d,
    dask_maxpool2d,
    dask_resize_bilinear,
    execute_op,
    propagate_chunk_shapes,
)


def test_dask_chunk_propagation() -> None:
    """Test chunk-aware shape propagation across multidimensional distributed arrays."""
    arr = da.zeros((10, 20, 30), chunks=(5, 10, 15))
    chunks = propagate_chunk_shapes(arr)
    assert chunks == ((5, 5), (10, 10), (15, 15))

    # From sequence
    seq_chunks = propagate_chunk_shapes([1, 2, 3, 4], chunks=2)
    assert len(seq_chunks) > 0

    # Non-dask object without shape
    assert propagate_chunk_shapes(None) == ()


def test_dask_lazy_reductions_parity() -> None:
    """Test reduction operations creating lazy Dask task graphs without eager compute."""
    arr = da.ones((4, 4), chunks=(2, 2))

    # ArgMax
    res_argmax = execute_op(None, "ArgMax", arr, dim=0)
    assert hasattr(res_argmax, "dask")
    _verify_dask_task_graph(res_argmax)

    # ArgMin
    res_argmin = execute_op(None, "ArgMin", arr, dim=1)
    assert hasattr(res_argmin, "dask")
    _verify_dask_task_graph(res_argmin)

    # ReduceL2
    res_l2 = execute_op(None, "ReduceL2", arr, axis=0)
    assert hasattr(res_l2, "dask")
    _verify_dask_task_graph(res_l2)

    # CumSum
    res_cumsum = execute_op(None, "CumSum", arr, dim=0)
    assert hasattr(res_cumsum, "dask")
    _verify_dask_task_graph(res_cumsum)

    # CumProd
    res_cumprod = execute_op(None, "CumProd", arr, dim=1)
    assert hasattr(res_cumprod, "dask")
    _verify_dask_task_graph(res_cumprod)


def test_dask_linalg_parity() -> None:
    """Test linear algebra operations on Dask constructing lazy task graphs."""
    mat_a = da.eye(4, chunks=4)
    mat_b = da.ones((4, 2), chunks=(2, 2))

    # Cholesky
    res_chol = execute_op(None, "Cholesky", mat_a)
    assert hasattr(res_chol, "dask")

    # QR
    q, r = execute_op(None, "QR", mat_a)
    assert hasattr(q, "dask")
    assert hasattr(r, "dask")

    # SVD
    u, s, vh = execute_op(None, "SVD", mat_a)
    assert hasattr(u, "dask")
    assert hasattr(s, "dask")
    assert hasattr(vh, "dask")

    # Solve
    res_solve = execute_op(None, "Solve", mat_a, mat_b)
    assert hasattr(res_solve, "dask")


def test_dask_vision_nn_parity() -> None:
    """Test vision and neural network ops on Dask executing as lazy task graphs."""
    x = da.ones((2, 3, 8, 8), chunks=(1, 3, 4, 4))
    w = da.ones((4, 3, 3, 3), chunks=(4, 3, 3, 3))

    # Conv2D
    res_conv = execute_op(None, "Conv2D", x, w, stride=1, padding="SAME")
    assert hasattr(res_conv, "dask")
    _verify_dask_task_graph(res_conv)

    # MaxPool2D
    res_mp = execute_op(None, "MaxPool2D", x, kernel_size=2, stride=2)
    assert hasattr(res_mp, "dask")
    _verify_dask_task_graph(res_mp)

    # AvgPool2D
    res_ap = execute_op(None, "AvgPool2D", x, kernel_size=2, stride=2)
    assert hasattr(res_ap, "dask")
    _verify_dask_task_graph(res_ap)

    # ResizeBilinear
    res_resize = execute_op(None, "ResizeBilinear", x, size=(16, 16))
    assert hasattr(res_resize, "dask")
    _verify_dask_task_graph(res_resize)


def test_dask_direct_vision_kernels_and_fallbacks() -> None:
    """Directly test native Dask lazy vision functions."""
    x = da.ones((2, 3, 8, 8), chunks=(1, 3, 4, 4))
    w = da.ones((4, 3, 3, 3), chunks=(4, 3, 3, 3))

    conv = dask_conv2d(x, w, stride=1, padding="SAME")
    assert hasattr(conv, "dask")

    # Valid padding and non-4D fallback
    conv_invalid = dask_conv2d(da.ones((4, 4)), da.ones((2, 2)))
    assert conv_invalid.ndim == 2

    # MaxPool2D
    mp = dask_maxpool2d(x, kernel_size=2, stride=2)
    assert hasattr(mp, "dask")
    assert dask_maxpool2d(da.ones((2, 2))).ndim == 2

    # AvgPool2D
    ap = dask_avgpool2d(x, kernel_size=2, stride=2)
    assert hasattr(ap, "dask")
    assert dask_avgpool2d(da.ones((2, 2))).ndim == 2

    # ResizeBilinear
    rb = dask_resize_bilinear(x, size=(8, 8))
    assert rb.shape == x.shape
    assert dask_resize_bilinear(x, size=None).shape == x.shape


def test_dask_task_graph_verification_failure() -> None:
    """Test that missing task graph raises RuntimeError."""
    from unittest.mock import patch

    class FakeDaskArray:
        pass

    fake = FakeDaskArray()
    with patch("dask.array.Array", FakeDaskArray):
        with pytest.raises(RuntimeError, match="Dask array is missing underlying task graph"):
            _verify_dask_task_graph(fake)
