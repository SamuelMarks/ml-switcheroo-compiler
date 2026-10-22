"""Tests for backend profilers, unresolved mappings, and distributed collectives."""

from __future__ import annotations

import typing
from unittest.mock import MagicMock, patch

import numpy as np
from ml_switcheroo_ir import NodeDict

from ml_switcheroo_compiler.backends.eager_registry import EagerValue
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
from tests.testing_utils import as_callable, as_tensor, install_profiler_backend_stubs

try:
    import jax
    import jax.numpy as jnp
except Exception:
    jax = MagicMock()
    jnp = MagicMock()


try:
    import torch
except Exception:
    torch = MagicMock()

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


import pytest


@pytest.fixture(autouse=True)
def _ensure_profiler_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure optional profiler backend stubs are installed when packages are missing."""
    import sys

    install_profiler_backend_stubs(monkeypatch)
    monkeypatch.setattr(sys.modules[__name__], "torch", sys.modules["torch"])
    monkeypatch.setattr(sys.modules[__name__], "jax", sys.modules["jax"])
    monkeypatch.setattr(sys.modules[__name__], "jnp", sys.modules["jax.numpy"])


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
        out = as_callable(torch_execute_op)(None, "Relu", as_callable(torch.tensor)([-1.0, 2.0]))
        assert isinstance(as_tensor(out), torch.Tensor)


def test_jax_eager_unresolved_mapping_branch() -> None:
    """Test jax/eager.py branch 391->393 when resolve_target_api returns None."""
    with patch("ml_switcheroo_compiler.backends.mapping_loader.resolve_target_api", return_value=None):
        out = as_callable(jax_execute_op)(None, "Sin", as_callable(jnp.array)([0.0]))
        assert isinstance(as_tensor(out), jax.Array)


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

    if as_tensor(torch) is not None and dist is not None:
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

    def mock_import(name: str, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
        """Test function mock_import.

        Args:
            name: Test parameter `name`.
            args: Test parameter `args`.
            kwargs: Test parameter `kwargs`.
        """
        if name in ("torch", "torch.distributed"):
            raise ImportError("torch missing")
        return as_callable(orig_import)(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        assert pytorch_all_reduce("raw_tensor") == "raw_tensor"
        assert pytorch_all_gather("raw_tensor") == "raw_tensor"
        assert pytorch_reduce_scatter("raw_tensor") == "raw_tensor"
        assert pytorch_broadcast("raw_tensor") == "raw_tensor"


def test_edge_profiler_branches() -> None:
    """Test full line and branch coverage for EdgeProfiler."""
    from ml_switcheroo_compiler.backends.edge.profiler import (
        EdgeProfiler,
        _calculate_edge_memory_bytes,
        _get_process_memory_mb,
    )

    # 1. Platform checks for _get_process_memory_mb
    with patch("sys.platform", "darwin"):
        assert _get_process_memory_mb() > 0.0
    with patch("sys.platform", "linux"):
        mem = _get_process_memory_mb()
        assert mem > 0.0

    # 2. _calculate_edge_memory_bytes with empty graph or no nodes
    empty_graph = IRGraph()
    empty_graph.nodes = NodeDict({})
    assert _calculate_edge_memory_bytes(empty_graph) == 65536
    empty_graph_no_attr = IRGraph()
    del empty_graph_no_attr.nodes
    assert _calculate_edge_memory_bytes(empty_graph_no_attr) == 65536

    # 3. _calculate_edge_memory_bytes with nodes having shape_metadata / shape including non-positive or non-int dim
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Input")
    n1.shape_metadata = (10, -1, "dynamic", 2)
    n2 = IRNode(id="n2", op_type="Relu")
    n2.shape = (4, 4)
    n3 = IRNode(id="n3", op_type="Other")
    n3.shape = None
    g.nodes = NodeDict({"n1": n1, "n2": n2, "n3": n3})
    bytes_calc = _calculate_edge_memory_bytes(g)
    assert bytes_calc >= 65536

    # 4. _prepare_inputs when input_keys match and when they do not match
    profiler = EdgeProfiler()
    g_inputs = IRGraph()
    g_inputs.nodes["x"] = IRNode(id="x", op_type="Input")
    inputs_payload = {"x": [1.0, 2.0]}
    prep = as_callable(profiler._prepare_inputs)(g_inputs, inputs_payload)
    assert prep == [[1.0, 2.0]]

    # _prepare_inputs when graph has no nodes or input_keys do not match
    assert profiler._prepare_inputs(empty_graph, {"a": [1.0]}) == [[1.0]]

    # 5. profile_graph with ops where evaluate_graph succeeds
    g_ops = IRGraph()
    g_ops.nodes["x"] = IRNode(id="x", op_type="Input")
    g_ops.nodes["add"] = IRNode(id="add", op_type="Add", inputs=["x", "x"])
    g_ops.outputs = ["add"]
    with patch("ml_switcheroo_compiler.backends.edge.profiler.evaluate_graph", return_value={"add": [2.0]}):
        res = profiler.profile_graph(g_ops, inputs={"x": [1.0]}, num_iters=2, warmup_iters=1)
        assert "latency_ms" in res
        assert "peak_memory_mb" in res

    # 6. profile_graph with ops where evaluate_graph raises Exception
    with patch("ml_switcheroo_compiler.backends.edge.profiler.evaluate_graph", side_effect=RuntimeError("Eval failed")):
        res_fail = profiler.profile_graph(g_ops, inputs={"x": [1.0]}, num_iters=1, warmup_iters=1)
        assert "latency_ms" in res_fail

    # 7. profile_graph with no ops (has_ops is False)
    res_no_ops = profiler.profile_graph(g_inputs, inputs={"x": [1.0, 2.0]}, num_iters=1, warmup_iters=1)
    assert "latency_ms" in res_no_ops


def _test_numba_profiler_branches_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_numba_profiler_branches.

    Args:
        _state: Shared test state dictionary.
    """
    import importlib

    from ml_switcheroo_compiler.backends.numba import profiler as nb_prof_mod
    from ml_switcheroo_compiler.backends.numba.profiler import (
        NumbaProfiler,
        _get_process_memory_mb,
    )

    # 1. Platform checks
    with patch("sys.platform", "darwin"):
        assert _get_process_memory_mb() > 0.0
    with patch("sys.platform", "linux"):
        mem = _get_process_memory_mb()
        assert mem > 0.0

    # 2. ImportError branch when numba is not installed
    with patch.dict("sys.modules", {"numba": None}):
        importlib.reload(nb_prof_mod)
        assert as_tensor(getattr(nb_prof_mod, "numba")) is None
    importlib.reload(nb_prof_mod)

    profiler = NumbaProfiler()

    # 3. _prepare_inputs with np.ndarray and non-ndarray values
    g = IRGraph()
    g.nodes["x"] = IRNode(id="x", op_type="Input")
    arr_in = np.array([1.0, 2.0], dtype=np.float32)
    prep = profiler._prepare_inputs(g, {"x": arr_in})
    assert len(prep) == 1
    assert isinstance(prep[0], np.ndarray)

    g_both = IRGraph()
    g_both.nodes["x"] = IRNode(id="x", op_type="Input")
    g_both.nodes["y"] = IRNode(id="y", op_type="Input")
    getattr(_state, "update")({"NumbaProfiler": NumbaProfiler, "_get_process_memory_mb": _get_process_memory_mb, "arr_in": arr_in, "g": g})
    getattr(_state, "update")({"g_both": g_both, "importlib": importlib, "mem": mem, "nb_prof_mod": nb_prof_mod})
    getattr(_state, "update")({"prep": prep, "profiler": profiler})
    _state.update(locals())


def _test_numba_profiler_branches_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_numba_profiler_branches.

    Args:
        _state: Shared test state dictionary.
    """
    arr_in, g, g_both, profiler = _state["arr_in"], _state["g"], _state["g_both"], _state["profiler"]
    prep_both = profiler._prepare_inputs(g_both, {"x": arr_in, "y": [3.0, 4.0]})
    assert as_tensor(len(prep_both)) == 2
    assert isinstance(as_tensor(prep_both[0]), np.ndarray)
    assert isinstance(as_tensor(prep_both[1]), np.ndarray)

    # _prepare_inputs fallback when input keys don't match or graph has no nodes
    empty_graph = IRGraph()
    empty_graph.nodes = NodeDict({})
    prep_empty = profiler._prepare_inputs(empty_graph, {"other": [5.0]})
    assert len(prep_empty) == 1

    prep_fallback = profiler._prepare_inputs(g_both, {"other": arr_in})
    assert len(prep_fallback) == 1

    # 4. _compile_graph returning callable and returning None
    with patch("ml_switcheroo_compiler.backends.numba.generator.NumbaGenerator.generate", return_value="def evaluate(args): return args[0]"):
        fn = profiler._compile_graph(g)
        assert callable(fn)

    with patch("ml_switcheroo_compiler.backends.numba.generator.NumbaGenerator.generate", return_value="def not_evaluate(): pass"):
        fn_none = profiler._compile_graph(g)
        assert as_tensor(fn_none) is None

    # 5. profile_graph with compiled function execution
    g_ops = IRGraph()
    g_ops.nodes["x"] = IRNode(id="x", op_type="Input")
    g_ops.nodes["add"] = IRNode(id="add", op_type="Add", inputs=["x", "x"])
    with patch.object(profiler, "_compile_graph", return_value=lambda inputs: inputs[0] * 2):
        res_comp = profiler.profile_graph(g_ops, inputs={"x": np.array([2.0])}, num_iters=2, warmup_iters=1)
        assert "mean_latency_ms" in res_comp
    getattr(_state, "update")({"empty_graph": empty_graph, "fn": fn, "fn_none": fn_none, "g_ops": g_ops})
    getattr(_state, "update")({"prep_both": prep_both, "prep_empty": prep_empty, "prep_fallback": prep_fallback, "res_comp": res_comp})
    _state.update(locals())


def _test_numba_profiler_branches_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_numba_profiler_branches.

    Args:
        _state: Shared test state dictionary.
    """
    g_ops, profiler = _state["g_ops"], _state["profiler"]
    with patch.object(profiler, "_compile_graph", side_effect=RuntimeError("Compilation error")):
        res_err = profiler.profile_graph(g_ops, inputs={"x": np.array([2.0])}, num_iters=1, warmup_iters=1)
        assert "mean_latency_ms" in res_err

    # 7. profile_graph with empty inputs (numpy zeros branch)
    g_empty = IRGraph()
    res_empty = profiler.profile_graph(g_empty, inputs={}, num_iters=1, warmup_iters=1)
    assert "mean_latency_ms" in res_empty
    getattr(_state, "update")({"g_empty": g_empty, "res_empty": res_empty, "res_err": res_err})


def test_numba_profiler_branches() -> None:
    """Test full line and branch coverage for NumbaProfiler."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_numba_profiler_branches_part_1(_state)
    _test_numba_profiler_branches_part_2(_state)
    _test_numba_profiler_branches_part_3(_state)
    assert len(_state) > 0


def _test_sparse_profiler_branches_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_sparse_profiler_branches.

    Args:
        _state: Shared test state dictionary.
    """
    import importlib

    from ml_switcheroo_compiler.backends.sparse import profiler as sp_prof_mod
    from ml_switcheroo_compiler.backends.sparse.profiler import (
        SparseProfiler,
        _get_process_memory_mb,
    )

    # 1. Platform checks
    with patch("sys.platform", "darwin"):
        assert _get_process_memory_mb() > 0.0
    with patch("sys.platform", "linux"):
        mem = _get_process_memory_mb()
        assert mem > 0.0

    # 2. ImportError branch when sparse is not installed
    with patch.dict("sys.modules", {"sparse": None}):
        importlib.reload(sp_prof_mod)
        assert as_tensor(getattr(sp_prof_mod, "sparse")) is None
    importlib.reload(sp_prof_mod)

    profiler = SparseProfiler()

    # 3. _prepare_inputs with sparse module present
    import sparse as sp_mod

    coo = sp_mod.COO.from_numpy(np.array([1.0, 0.0]))
    gcxs = sp_mod.GCXS.from_numpy(np.array([0.0, 2.0]))
    g = IRGraph()
    g.nodes["x"] = IRNode(id="x", op_type="Input")
    prep_sp = profiler._prepare_inputs(g, {"x": coo})
    assert len(prep_sp) == 1

    # Testing all branches of _prepare_inputs: GCXS, np.ndarray, and sequence/scalar
    prep_multi = profiler._prepare_inputs(IRGraph(), {"a": gcxs, "b": np.array([1.0]), "c": [2.0]})
    assert len(prep_multi) == 3
    getattr(_state, "update")({"SparseProfiler": SparseProfiler, "_get_process_memory_mb": _get_process_memory_mb, "coo": coo, "g": g})
    getattr(_state, "update")({"gcxs": gcxs, "importlib": importlib, "mem": mem, "prep_multi": prep_multi})
    getattr(_state, "update")({"prep_sp": prep_sp, "profiler": profiler, "sp_mod": sp_mod, "sp_prof_mod": sp_prof_mod})
    _state.update(locals())


def _test_sparse_profiler_branches_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_sparse_profiler_branches.

    Args:
        _state: Shared test state dictionary.
    """
    g, profiler, sp_mod = _state["g"], _state["profiler"], _state["sp_mod"]

    class _MockSparseNoAttrs:
        """Test class _MockSparseNoAttrs."""

        def __init__(self) -> None:
            """Test function __init__."""
            pass

        @property
        def todense(self) -> EagerValue:
            """Test function todense."""
            raise AttributeError("no todense")

    mock_sparse_val = _MockSparseNoAttrs()
    prep_sparse_fallback = as_callable(profiler._prepare_inputs)(IRGraph(), {"s": mock_sparse_val})
    assert prep_sparse_fallback == [mock_sparse_val]

    # When sparse is None
    with patch("ml_switcheroo_compiler.backends.sparse.profiler.sparse", None):
        prep_no_sp = profiler._prepare_inputs(IRGraph(), {"x": np.array([1.0]), "y": [2.0]})
        assert len(prep_no_sp) == 2

    # Topologically sorted input keys branch
    g_two = IRGraph()
    g_two.nodes["x"] = IRNode(id="x", op_type="Input")
    g_two.nodes["y"] = IRNode(id="y", op_type="Input")
    prep_matched = profiler._prepare_inputs(g_two, {"x": [1.0], "y": [2.0]})
    assert len(prep_matched) == 2

    # 4. _compile_graph returning callable and returning None
    with patch("ml_switcheroo_compiler.backends.sparse.generator.SparseGenerator.generate", return_value="def evaluate(args): return args[0]"):
        fn = profiler._compile_graph(g)
        assert callable(fn)
    getattr(_state, "update")({"_MockSparseNoAttrs": _MockSparseNoAttrs, "fn": fn, "g_two": g_two, "mock_sparse_val": mock_sparse_val})
    getattr(_state, "update")({"prep_matched": prep_matched, "prep_no_sp": prep_no_sp, "prep_sparse_fallback": prep_sparse_fallback})
    _state.update(locals())


def _test_sparse_profiler_branches_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_sparse_profiler_branches.

    Args:
        _state: Shared test state dictionary.
    """
    g, profiler = _state["g"], _state["profiler"]
    with patch("ml_switcheroo_compiler.backends.sparse.generator.SparseGenerator.generate", return_value="def other(): pass"):
        fn_none = profiler._compile_graph(g)
        assert as_tensor(fn_none) is None
    g_ops = IRGraph()
    g_ops.nodes["x"] = IRNode(id="x", op_type="Input")
    g_ops.nodes["add"] = IRNode(id="add", op_type="Add", inputs=["x", "x"])
    with patch.object(profiler, "_compile_graph", return_value=lambda inputs: inputs[0]):
        res_comp = profiler.profile_graph(g_ops, inputs={"x": np.array([2.0])}, num_iters=2, warmup_iters=1)
        assert "mean_latency_ms" in res_comp

    # 6. profile_graph with compilation exception fallback and add accumulation
    with patch.object(profiler, "_compile_graph", side_effect=RuntimeError("Compilation error")):
        with patch.object(profiler, "_prepare_inputs", return_value=[np.array([1.0]), np.array([2.0])]):
            res_accum = profiler.profile_graph(g_ops, inputs={"x": np.array([2.0])}, num_iters=1, warmup_iters=1)
            assert "mean_latency_ms" in res_accum

    # 7. profile_graph fallback with non-addable objects
    class NoAdd:
        """Dummy object without addition support."""

    getattr(_state, "update")({"NoAdd": NoAdd, "fn_none": fn_none, "g_ops": g_ops, "res_accum": res_accum})
    getattr(_state, "update")({"res_comp": res_comp})
    _state.update(locals())


def _test_sparse_profiler_branches_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_sparse_profiler_branches.

    Args:
        _state: Shared test state dictionary.
    """
    NoAdd, g_ops, profiler = _state["NoAdd"], _state["g_ops"], _state["profiler"]
    with patch.object(profiler, "_compile_graph", side_effect=RuntimeError("Compilation error")):
        with patch.object(profiler, "_prepare_inputs", return_value=[NoAdd(), NoAdd()]):
            res_no_add = profiler.profile_graph(g_ops, inputs={"x": np.array([1.0])}, num_iters=1, warmup_iters=1)
            assert "mean_latency_ms" in res_no_add

    # 8. profile_graph with empty inputs
    g_empty = IRGraph()
    res_empty = profiler.profile_graph(g_empty, inputs={}, num_iters=1, warmup_iters=1)
    assert "mean_latency_ms" in res_empty
    getattr(_state, "update")({"g_empty": g_empty, "res_empty": res_empty, "res_no_add": res_no_add})


def test_sparse_profiler_branches() -> None:
    """Test full line and branch coverage for SparseProfiler."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_sparse_profiler_branches_part_1(_state)
    _test_sparse_profiler_branches_part_2(_state)
    _test_sparse_profiler_branches_part_3(_state)
    _test_sparse_profiler_branches_part_4(_state)
    assert len(_state) > 0
