"""Tests for backend profilers, unresolved mappings, and distributed collectives."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

try:
    import jax
    import jax.numpy as jnp
except Exception:
    jax = MagicMock()
    jnp = MagicMock()

import numpy as np

try:
    import torch
except Exception:
    torch = MagicMock()

from ml_switcheroo_compiler.backends.jax.distributed_collectives import (
    jax_all_gather,
    jax_all_reduce,
    jax_broadcast,
)
from ml_switcheroo_compiler.backends.jax.eager import execute_op as jax_execute_op
from ml_switcheroo_compiler.backends.jax.profiler import JAXProfiler, _sync_jax_result
from ml_switcheroo_compiler.backends.jax.profiler import _get_process_memory_mb as jax_get_mem
from ml_switcheroo_compiler.backends.mlx.profiler import (
    MLXProfiler,
    _get_mlx_peak_memory_mb,
    _prepare_mlx_inputs,
    _sync_mlx_result,
)
from ml_switcheroo_compiler.backends.mlx.profiler import (
    _get_process_memory_mb as mlx_get_mem,
)
from ml_switcheroo_compiler.backends.numpy.profiler import (
    NumpyProfiler,
)
from ml_switcheroo_compiler.backends.numpy.profiler import (
    _get_process_memory_mb as np_get_mem,
)
from ml_switcheroo_compiler.backends.pytorch.distributed_collectives import (
    pytorch_all_gather,
    pytorch_all_reduce,
    pytorch_broadcast,
    pytorch_reduce_scatter,
)
from ml_switcheroo_compiler.backends.pytorch.eager import execute_op as torch_execute_op
from ml_switcheroo_compiler.backends.pytorch.profiler import (
    PyTorchProfiler,
)
from ml_switcheroo_compiler.backends.pytorch.profiler import (
    _get_process_memory_mb as torch_get_mem,
)
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_numpy_profiler_branches() -> None:
    """Test branch coverage in numpy/profiler.py."""
    # 1. Non-darwin platform for memory check (lines 20-22)
    with patch("sys.platform", "linux"):
        mem = np_get_mem()
        assert mem > 0.0

    # 2. profile_graph with non-ndarray input list (lines 52-55)
    g = IRGraph()
    g.nodes["x"] = IRNode(id="x", op_type="Input")
    profiler = NumpyProfiler()
    res = profiler.profile_graph(g, inputs={"x": [1.0, 2.0]}, num_iters=2, warmup_iters=1)
    assert "latency_ms" in res
    assert "peak_memory_mb" in res

    # 3. profile_graph with has_ops=True (lines 68-71)
    g_ops = IRGraph()
    g_ops.nodes["x"] = IRNode(id="x", op_type="Input")
    g_ops.nodes["add"] = IRNode(id="add", op_type="Add", inputs=["x", "x"])
    g_ops.outputs = ["add"]
    with patch("ml_switcheroo_compiler.interpreter.evaluator.evaluate_graph", return_value={"add": np.array([2.0])}):
        res_ops = profiler.profile_graph(g_ops, inputs={"x": np.array([1.0])}, num_iters=1, warmup_iters=1)
        assert "latency_ms" in res_ops

    # 4. profile_graph with empty inputs list (line 70)
    res_empty = profiler.profile_graph(g, inputs={}, num_iters=1, warmup_iters=1)
    assert "latency_ms" in res_empty


def test_jax_profiler_branches() -> None:
    """Test branch coverage in jax/profiler.py."""
    # 1. Non-darwin platform for memory check (lines 21-23)
    with patch("sys.platform", "linux"):
        mem = jax_get_mem()
        assert mem > 0.0

    # 2. _sync_jax_result with dictionary whose value has no block_until_ready (line 53-54), single array (line 55), and non-block object (line 56->exit)
    _sync_jax_result({"res": 123.0})
    _sync_jax_result(123.0)
    mock_arr = MagicMock()
    _sync_jax_result(mock_arr)
    mock_arr.block_until_ready.assert_called_once()

    # 3. JAXProfiler with non-array input
    g = IRGraph()
    g.nodes["x"] = IRNode(id="x", op_type="Input")
    profiler = JAXProfiler()
    res = profiler.profile_graph(g, inputs={"x": [1.0, 2.0]}, num_iters=2, warmup_iters=1)
    assert "latency_ms" in res

    # 4. JAXProfiler with has_ops=True (lines 95-98)
    g_ops = IRGraph()
    g_ops.nodes["x"] = IRNode(id="x", op_type="Input")
    g_ops.nodes["add"] = IRNode(id="add", op_type="Add", inputs=["x", "x"])
    g_ops.outputs = ["add"]
    with patch("ml_switcheroo_compiler.interpreter.evaluator.evaluate_graph", return_value={"add": jnp.array([2.0])}):
        res_ops = profiler.profile_graph(g_ops, inputs={"x": [1.0]}, num_iters=1, warmup_iters=1)
        assert "latency_ms" in res_ops


def test_mlx_profiler_branches() -> None:
    """Test branch coverage in mlx/profiler.py."""
    # 1. Non-darwin platform (line 23-25)
    with patch("sys.platform", "linux"):
        mem = mlx_get_mem()
        assert mem > 0.0

    # 2. _get_mlx_peak_memory_mb with metal memory > 0 and <= 0
    mock_metal = MagicMock()
    mock_metal.get_peak_memory.return_value = 1024 * 1024 * 50
    mock_mx = MagicMock(metal=mock_metal)
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_mx):
        assert _get_mlx_peak_memory_mb() == 50.0

    mock_metal.get_peak_memory.return_value = 0
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_mx):
        assert _get_mlx_peak_memory_mb() > 0.0

    # 3. mx is None branches (lines 10-11, 51-52, 87-88)
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", None):
        assert _get_mlx_peak_memory_mb() > 0.0
        assert _prepare_mlx_inputs({"a": [1.0]}) == []
        _sync_mlx_result({"a": 1.0})

    # 4. _prepare_mlx_inputs with mock mx.array, duck-typed array (line 75), and non-dict _sync_mlx_result
    mock_array_cls = type("MockArray", (), {"__module__": "mlx.core", "__init__": lambda self, *a, **kw: None})
    mock_arr_inst = mock_array_cls()
    duck_array_cls = type("array", (), {"__module__": "mlx.custom"})
    duck_arr_inst = duck_array_cls()
    mock_mx_arr = MagicMock()
    mock_mx_arr.array = mock_array_cls
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_mx_arr):
        inps = _prepare_mlx_inputs({"a": mock_arr_inst, "b": duck_arr_inst, "c": [1.0, 2.0]})
        assert len(inps) == 3
        _sync_mlx_result(mock_arr_inst)
        mock_mx_arr.eval.assert_called_once_with(mock_arr_inst)

    # 5. MLXProfiler.profile_graph with has_ops=True (lines 144-147)
    profiler = MLXProfiler()
    g_ops = IRGraph()
    g_ops.nodes["x"] = IRNode(id="x", op_type="Input")
    g_ops.nodes["add"] = IRNode(id="add", op_type="Add", inputs=["x", "x"])
    g_ops.outputs = ["add"]
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_mx_arr):
        with patch("ml_switcheroo_compiler.interpreter.evaluator.evaluate_graph", return_value={"add": [2.0]}):
            res = profiler.profile_graph(g_ops, inputs={"x": [1.0]}, num_iters=1, warmup_iters=1)
            assert "latency_ms" in res


def test_pytorch_profiler_branches() -> None:
    """Test branch coverage in pytorch/profiler.py."""
    # 1. Non-darwin platform (lines 20-22)
    with patch("sys.platform", "linux"):
        mem = torch_get_mem()
        assert mem > 0.0

    profiler = PyTorchProfiler()

    # 2. _resolve_device with unavailable cuda and mps (lines 38-41)
    with patch("torch.cuda.is_available", return_value=False):
        dev = profiler._resolve_device("cuda")
        assert dev.type == "cpu"

    with patch.object(torch.backends, "mps", MagicMock(is_available=lambda: False), create=True):
        dev_mps = profiler._resolve_device("mps")
        assert dev_mps.type == "cpu"

    # 3. _prepare_inputs with torch.Tensor input (line 60-61)
    t = torch.tensor([1.0, 2.0])
    inps = profiler._prepare_inputs({"t": t, "list": [3.0]}, torch.device("cpu"))
    assert len(inps) == 2

    # 4. _sync with cuda and mps (lines 72-75)
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.synchronize") as mock_sync:
            profiler._sync(torch.device("cuda"))
            mock_sync.assert_called_once()

    with patch.object(torch, "mps", MagicMock(is_available=lambda: True), create=True):
        profiler._sync(torch.device("mps"))

    # 5. profile_graph with empty inputs list (lines 124-125)
    g = IRGraph()
    res = profiler.profile_graph(g, inputs={}, num_iters=2, warmup_iters=1)
    assert "latency_ms" in res

    # 6. profile_graph with has_ops=True (lines 116-119)
    g_ops = IRGraph()
    g_ops.nodes["x"] = IRNode(id="x", op_type="Input")
    g_ops.nodes["add"] = IRNode(id="add", op_type="Add", inputs=["x", "x"])
    g_ops.outputs = ["add"]
    with patch("ml_switcheroo_compiler.interpreter.evaluator.evaluate_graph", return_value={"add": torch.tensor([2.0])}):
        res_ops = profiler.profile_graph(g_ops, inputs={"x": [1.0]}, num_iters=1, warmup_iters=1)
        assert "latency_ms" in res_ops


def test_pytorch_eager_unresolved_mapping_branch() -> None:
    """Test pytorch/eager.py branch 474->476 when resolve_target_api returns None."""
    with patch("ml_switcheroo_compiler.backends.mapping_loader.resolve_target_api", return_value=None):
        out = torch_execute_op(None, "Relu", torch.tensor([-1.0, 2.0]))
        assert isinstance(out, torch.Tensor)


def test_jax_eager_unresolved_mapping_branch() -> None:
    """Test jax/eager.py branch 391->393 when resolve_target_api returns None."""
    with patch("ml_switcheroo_compiler.backends.mapping_loader.resolve_target_api", return_value=None):
        out = jax_execute_op(None, "Sin", jnp.array([0.0]))
        assert isinstance(out, jax.Array)


def test_pytorch_distributed_collectives() -> None:
    """Test native PyTorch distributed collective dispatchers."""
    t = torch.tensor([1.0, 2.0])

    # Uninitialized dist fallback
    assert pytorch_all_reduce(t) is t
    assert pytorch_all_gather(t) is t
    assert pytorch_reduce_scatter(t) is t
    assert pytorch_broadcast(t) is t

    # Initialized mock dist using patch.object on torch.distributed
    with (
        patch.object(torch.distributed, "is_available", return_value=True),
        patch.object(torch.distributed, "is_initialized", return_value=True),
        patch.object(torch.distributed, "all_reduce"),
        patch.object(torch.distributed, "get_world_size", return_value=2),
        patch.object(torch.distributed, "all_gather"),
        patch.object(torch.distributed, "reduce_scatter"),
        patch.object(torch.distributed, "broadcast"),
    ):
        # all_reduce ops
        for op in ("SUM", "PRODUCT", "MAX", "MIN"):
            res_ar = pytorch_all_reduce(t, op=op)
            assert res_ar is not None

        # all_gather
        res_ag = pytorch_all_gather(t, axis=0)
        assert res_ag is not None

        # reduce_scatter ops
        t_4 = torch.tensor([1.0, 2.0, 3.0, 4.0])
        res_rs1 = pytorch_reduce_scatter(t_4, op="SUM", scatter_dim=0)
        res_rs2 = pytorch_reduce_scatter(t_4, op="PRODUCT", scatter_dim=0)
        assert res_rs1 is not None and res_rs2 is not None

        # broadcast
        res_bc = pytorch_broadcast(t, src=0)
        assert res_bc is not None

    # ImportError branch
    with patch.dict("sys.modules", {"torch.distributed": None}):
        assert pytorch_all_reduce(t) is t
        assert pytorch_all_gather(t) is t
        assert pytorch_reduce_scatter(t) is t
        assert pytorch_broadcast(t) is t


def test_jax_distributed_collectives() -> None:
    """Test native JAX distributed collective dispatchers."""
    arr = jnp.array([1.0, 2.0])

    # Mock jax.lax collectives
    mock_lax = MagicMock()
    mock_lax.psum.return_value = arr * 2
    mock_lax.pmax.return_value = arr
    mock_lax.pmin.return_value = arr
    mock_lax.all_gather.return_value = jnp.stack([arr, arr])

    with patch.dict("sys.modules", {"jax.lax": mock_lax}):
        assert jax_all_reduce(arr, op="SUM") is not None
        assert jax_all_reduce(arr, op="MAX") is not None
        assert jax_all_reduce(arr, op="MIN") is not None
        # Unknown op falls through
        assert jax_all_reduce(arr, op="UNKNOWN") is arr

        # all_gather
        assert jax_all_gather(arr, axis_name="mesh") is not None

        # broadcast
        assert jax_broadcast(arr, src=0) is arr

    # ImportError branch
    with patch.dict("sys.modules", {"jax.lax": None}):
        assert jax_all_reduce(arr, op="SUM") is arr
        assert jax_all_gather(arr) is arr


def test_backends_jax_distributed_collectives_branches() -> None:
    """Test jax_all_reduce branches (MAX, MIN, and import error)."""
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.backends.jax.distributed_collectives import jax_all_reduce

    mock_jax = MagicMock()
    mock_jax.lax.psum.return_value = "sum_reduced"
    mock_jax.lax.pmax.return_value = "max_reduced"
    mock_jax.lax.pmin.return_value = "min_reduced"
    with patch.dict("sys.modules", {"jax": mock_jax, "jax.lax": mock_jax.lax}):
        assert jax_all_reduce("tensor", op="MAX", axis_name="x") == "max_reduced"
        assert jax_all_reduce("tensor", op="MIN", axis_name="x") == "min_reduced"

    with patch.dict("sys.modules", {"jax": None, "jax.lax": None}):
        assert jax_all_reduce("fallback_tensor", op="MAX", axis_name="x") == "fallback_tensor"


def test_backends_pytorch_distributed_collectives_branches() -> None:
    """Test pytorch distributed collectives when dist is initialized and for various ops."""
    from unittest.mock import patch

    import torch.distributed as dist

    from ml_switcheroo_compiler.backends.pytorch.distributed_collectives import (
        pytorch_all_gather,
        pytorch_all_reduce,
        pytorch_broadcast,
        pytorch_reduce_scatter,
    )

    if torch is not None and dist is not None:
        t = torch.tensor([1.0, 2.0])
        with patch.object(dist, "is_available", return_value=True):
            with patch.object(dist, "is_initialized", return_value=True):
                with patch.object(dist, "all_reduce", return_value=None):
                    with patch.object(dist, "all_gather", return_value=None):
                        with patch.object(dist, "reduce_scatter", return_value=None):
                            with patch.object(dist, "broadcast", return_value=None):
                                with patch.object(dist, "get_world_size", return_value=2):
                                    res_prod = pytorch_all_reduce(t, op="PRODUCT")
                                    assert res_prod is not None
                                    res_max = pytorch_all_reduce(t, op="MAX")
                                    assert res_max is not None
                                    res_min = pytorch_all_reduce(t, op="MIN")
                                    assert res_min is not None
                                    res_sum = pytorch_all_reduce(t, op="SUM")
                                    assert res_sum is not None
                                    res_gather = pytorch_all_gather(t, axis=0)
                                    assert res_gather is not None
                                    res_scatter_sum = pytorch_reduce_scatter(t, op="SUM", scatter_dim=0)
                                    assert res_scatter_sum is not None
                                    res_scatter_prod = pytorch_reduce_scatter(t, op="PROD", scatter_dim=0)
                                    assert res_scatter_prod is not None
                                    res_bcast = pytorch_broadcast(t, src=0)
                                    assert res_bcast is not None

        with patch.object(dist, "is_available", return_value=False):
            assert pytorch_all_reduce("raw_tensor") == "raw_tensor"
            assert pytorch_all_gather("raw_tensor") == "raw_tensor"
            assert pytorch_reduce_scatter("raw_tensor") == "raw_tensor"
            assert pytorch_broadcast("raw_tensor") == "raw_tensor"

    import builtins

    orig_import = builtins.__import__

    def mock_import(name: str, *args: object, **kwargs: object) -> object:
        if name in ("torch", "torch.distributed"):
            raise ImportError("torch missing")
        return orig_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        assert pytorch_all_reduce("raw_tensor") == "raw_tensor"
        assert pytorch_all_gather("raw_tensor") == "raw_tensor"
        assert pytorch_reduce_scatter("raw_tensor") == "raw_tensor"
        assert pytorch_broadcast("raw_tensor") == "raw_tensor"
