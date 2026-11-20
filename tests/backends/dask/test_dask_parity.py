"""Parity, chunk-aware shape propagation, and lazy task graph tests for Dask backend."""

from __future__ import annotations

import typing

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
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.testing_utils import (
    MockTestStub,
    adapt_backend_module,
    coerce_operand_list,
    extract_test_shape,
    unpack_pair,
)

if typing.TYPE_CHECKING:
    import dask.array as da

pytest.importorskip("dask")
da = pytest.importorskip("dask.array")


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
    assert adapt_backend_module(res_argmax).has_symbol("dask")
    _verify_dask_task_graph(res_argmax)

    # ArgMin
    res_argmin = execute_op(None, "ArgMin", arr, dim=1)
    assert adapt_backend_module(res_argmin).has_symbol("dask")
    _verify_dask_task_graph(res_argmin)

    # ReduceL2
    res_l2 = execute_op(None, "ReduceL2", arr, axis=0)
    assert adapt_backend_module(res_l2).has_symbol("dask")
    _verify_dask_task_graph(res_l2)

    # CumSum
    res_cumsum = execute_op(None, "CumSum", arr, dim=0)
    assert adapt_backend_module(res_cumsum).has_symbol("dask")
    _verify_dask_task_graph(res_cumsum)

    # CumProd
    res_cumprod = execute_op(None, "CumProd", arr, dim=1)
    assert adapt_backend_module(res_cumprod).has_symbol("dask")
    _verify_dask_task_graph(res_cumprod)


def test_dask_linalg_parity() -> None:
    """Test linear algebra operations on Dask constructing lazy task graphs."""
    mat_a = da.eye(4, chunks=4)
    mat_b = da.ones((4, 2), chunks=(2, 2))

    # Cholesky
    res_chol = execute_op(None, "Cholesky", mat_a)
    assert adapt_backend_module(res_chol).has_symbol("dask")

    # QR
    q, r = unpack_pair(execute_op(None, "QR", mat_a))
    assert adapt_backend_module(q).has_symbol("dask")
    assert adapt_backend_module(r).has_symbol("dask")

    # SVD
    svd_out = coerce_operand_list(execute_op(None, "SVD", mat_a))
    assert len(svd_out) == 3
    u, s, vh = svd_out[0], svd_out[1], svd_out[2]
    assert adapt_backend_module(u).has_symbol("dask")
    assert adapt_backend_module(s).has_symbol("dask")
    assert adapt_backend_module(vh).has_symbol("dask")

    # Solve
    res_solve = execute_op(None, "Solve", mat_a, mat_b)
    assert adapt_backend_module(res_solve).has_symbol("dask")


def test_dask_vision_nn_parity() -> None:
    """Test vision and neural network ops on Dask executing as lazy task graphs."""
    x = da.ones((2, 3, 8, 8), chunks=(1, 3, 4, 4))
    w = da.ones((4, 3, 3, 3), chunks=(4, 3, 3, 3))

    # Conv2D
    res_conv = execute_op(None, "Conv2D", x, w, stride=1, padding="SAME")
    assert adapt_backend_module(res_conv).has_symbol("dask")
    _verify_dask_task_graph(res_conv)

    # MaxPool2D
    res_mp = execute_op(None, "MaxPool2D", x, kernel_size=2, stride=2)
    assert adapt_backend_module(res_mp).has_symbol("dask")
    _verify_dask_task_graph(res_mp)

    # AvgPool2D
    res_ap = execute_op(None, "AvgPool2D", x, kernel_size=2, stride=2)
    assert adapt_backend_module(res_ap).has_symbol("dask")
    _verify_dask_task_graph(res_ap)

    # ResizeBilinear
    res_resize = execute_op(None, "ResizeBilinear", x, size=(16, 16))
    assert adapt_backend_module(res_resize).has_symbol("dask")
    _verify_dask_task_graph(res_resize)


def test_dask_direct_vision_kernels_and_fallbacks() -> None:
    """Directly test native Dask lazy vision functions."""
    x = da.ones((2, 3, 8, 8), chunks=(1, 3, 4, 4))
    w = da.ones((4, 3, 3, 3), chunks=(4, 3, 3, 3))

    conv = dask_conv2d(x, w, stride=1, padding="SAME")
    assert adapt_backend_module(conv).has_symbol("dask")

    # Valid padding and non-4D fallback
    conv_invalid = dask_conv2d(
        da.ones((4, 4)),
        da.ones((2, 2)),
    )
    assert ensure_ndarray(conv_invalid).ndim == 2

    # MaxPool2D
    mp = dask_maxpool2d(x, kernel_size=2, stride=2)
    assert adapt_backend_module(mp).has_symbol("dask")
    assert ensure_ndarray(dask_maxpool2d(da.ones((2, 2)))).ndim == 2

    # AvgPool2D
    ap = dask_avgpool2d(x, kernel_size=2, stride=2)
    assert adapt_backend_module(ap).has_symbol("dask")
    assert ensure_ndarray(dask_avgpool2d(da.ones((2, 2)))).ndim == 2

    # ResizeBilinear
    rb = dask_resize_bilinear(x, size=(8, 8))
    assert extract_test_shape(rb) == extract_test_shape(x)
    assert extract_test_shape(dask_resize_bilinear(x, size=None)) == extract_test_shape(x)


def test_dask_task_graph_verification_failure() -> None:
    """Test that missing task graph raises RuntimeError."""
    from unittest.mock import patch

    class FakeDaskArray(MockTestStub):
        """Test class FakeDaskArray."""

        pass

    fake = FakeDaskArray()
    with patch("dask.array.Array", FakeDaskArray):
        with pytest.raises(RuntimeError, match="Dask array is missing underlying task graph"):
            _verify_dask_task_graph(fake)
