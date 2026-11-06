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
from tests.testing_utils import ensure_callable, ensure_eager_tensor, has_backend_member

if typing.TYPE_CHECKING:
    import dask.array as da

pytest.importorskip("dask")
da = pytest.importorskip("dask.array")


def test_dask_chunk_propagation() -> None:
    """Test chunk-aware shape propagation across multidimensional distributed arrays."""
    arr = ensure_callable(ensure_eager_tensor(ensure_eager_tensor(da).zeros))((10, 20, 30), chunks=(5, 10, 15))
    chunks = propagate_chunk_shapes(arr)
    assert chunks == ((5, 5), (10, 10), (15, 15))

    # From sequence
    seq_chunks = propagate_chunk_shapes([1, 2, 3, 4], chunks=2)
    assert len(seq_chunks) > 0

    # Non-dask object without shape
    assert propagate_chunk_shapes(None) == ()


def test_dask_lazy_reductions_parity() -> None:
    """Test reduction operations creating lazy Dask task graphs without eager compute."""
    arr = ensure_callable(ensure_eager_tensor(ensure_eager_tensor(da).ones))((4, 4), chunks=(2, 2))

    # ArgMax
    res_argmax = ensure_callable(execute_op)(None, "ArgMax", arr, dim=0)
    assert has_backend_member(res_argmax, "dask")
    _verify_dask_task_graph(res_argmax)

    # ArgMin
    res_argmin = ensure_callable(execute_op)(None, "ArgMin", arr, dim=1)
    assert has_backend_member(res_argmin, "dask")
    _verify_dask_task_graph(res_argmin)

    # ReduceL2
    res_l2 = ensure_callable(execute_op)(None, "ReduceL2", arr, axis=0)
    assert has_backend_member(res_l2, "dask")
    _verify_dask_task_graph(res_l2)

    # CumSum
    res_cumsum = ensure_callable(execute_op)(None, "CumSum", arr, dim=0)
    assert has_backend_member(res_cumsum, "dask")
    _verify_dask_task_graph(res_cumsum)

    # CumProd
    res_cumprod = ensure_callable(execute_op)(None, "CumProd", arr, dim=1)
    assert has_backend_member(res_cumprod, "dask")
    _verify_dask_task_graph(res_cumprod)


def test_dask_linalg_parity() -> None:
    """Test linear algebra operations on Dask constructing lazy task graphs."""
    mat_a = ensure_callable(ensure_eager_tensor(ensure_eager_tensor(da).eye))(4, chunks=4)
    mat_b = ensure_callable(ensure_eager_tensor(ensure_eager_tensor(da).ones))((4, 2), chunks=(2, 2))

    # Cholesky
    res_chol = ensure_callable(execute_op)(None, "Cholesky", mat_a)
    assert has_backend_member(res_chol, "dask")

    # QR
    q, r = ensure_callable(execute_op)(None, "QR", mat_a)
    assert has_backend_member(q, "dask")
    assert has_backend_member(r, "dask")

    # SVD
    u, s, vh = ensure_callable(execute_op)(None, "SVD", mat_a)
    assert has_backend_member(u, "dask")
    assert has_backend_member(s, "dask")
    assert has_backend_member(vh, "dask")

    # Solve
    res_solve = ensure_callable(execute_op)(None, "Solve", mat_a, mat_b)
    assert has_backend_member(res_solve, "dask")


def test_dask_vision_nn_parity() -> None:
    """Test vision and neural network ops on Dask executing as lazy task graphs."""
    x = ensure_callable(ensure_eager_tensor(ensure_eager_tensor(da).ones))((2, 3, 8, 8), chunks=(1, 3, 4, 4))
    w = ensure_callable(ensure_eager_tensor(ensure_eager_tensor(da).ones))((4, 3, 3, 3), chunks=(4, 3, 3, 3))

    # Conv2D
    res_conv = ensure_callable(execute_op)(None, "Conv2D", x, w, stride=1, padding="SAME")
    assert has_backend_member(res_conv, "dask")
    _verify_dask_task_graph(res_conv)

    # MaxPool2D
    res_mp = ensure_callable(execute_op)(None, "MaxPool2D", x, kernel_size=2, stride=2)
    assert has_backend_member(res_mp, "dask")
    _verify_dask_task_graph(res_mp)

    # AvgPool2D
    res_ap = ensure_callable(execute_op)(None, "AvgPool2D", x, kernel_size=2, stride=2)
    assert has_backend_member(res_ap, "dask")
    _verify_dask_task_graph(res_ap)

    # ResizeBilinear
    res_resize = ensure_callable(execute_op)(None, "ResizeBilinear", x, size=(16, 16))
    assert has_backend_member(res_resize, "dask")
    _verify_dask_task_graph(res_resize)


def test_dask_direct_vision_kernels_and_fallbacks() -> None:
    """Directly test native Dask lazy vision functions."""
    x = ensure_callable(ensure_eager_tensor(ensure_eager_tensor(da).ones))((2, 3, 8, 8), chunks=(1, 3, 4, 4))
    w = ensure_callable(ensure_eager_tensor(ensure_eager_tensor(da).ones))((4, 3, 3, 3), chunks=(4, 3, 3, 3))

    conv = dask_conv2d(x, w, stride=1, padding="SAME")
    assert has_backend_member(conv, "dask")

    # Valid padding and non-4D fallback
    conv_invalid = ensure_callable(dask_conv2d)(ensure_callable(ensure_eager_tensor(ensure_eager_tensor(da).ones))((4, 4)), ensure_callable(ensure_eager_tensor(ensure_eager_tensor(da).ones))((2, 2)))
    assert ensure_eager_tensor(ensure_eager_tensor(conv_invalid).ndim) == 2

    # MaxPool2D
    mp = dask_maxpool2d(x, kernel_size=2, stride=2)
    assert has_backend_member(mp, "dask")
    assert ensure_eager_tensor(ensure_eager_tensor(ensure_callable(dask_maxpool2d)(ensure_callable(ensure_eager_tensor(ensure_eager_tensor(da).ones))((2, 2)))).ndim) == 2

    # AvgPool2D
    ap = dask_avgpool2d(x, kernel_size=2, stride=2)
    assert has_backend_member(ap, "dask")
    assert ensure_eager_tensor(ensure_eager_tensor(ensure_callable(dask_avgpool2d)(ensure_callable(ensure_eager_tensor(ensure_eager_tensor(da).ones))((2, 2)))).ndim) == 2

    # ResizeBilinear
    rb = dask_resize_bilinear(x, size=(8, 8))
    assert ensure_eager_tensor(rb).shape == ensure_eager_tensor(x).shape
    assert ensure_eager_tensor(dask_resize_bilinear(x, size=None)).shape == ensure_eager_tensor(x).shape


def test_dask_task_graph_verification_failure() -> None:
    """Test that missing task graph raises RuntimeError."""
    from unittest.mock import patch

    class FakeDaskArray:
        """Test class FakeDaskArray."""

        pass

    fake = FakeDaskArray()
    with patch("dask.array.Array", FakeDaskArray):
        with pytest.raises(RuntimeError, match="Dask array is missing underlying task graph"):
            ensure_callable(_verify_dask_task_graph)(fake)
