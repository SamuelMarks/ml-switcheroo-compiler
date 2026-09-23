"""Unit tests for backend profilers execution, latency measurement, and memory tracking."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml_switcheroo_compiler.ir.core import IRGraph


def test_numpy_profiler_runtime():
    """Verify execution latency and non-darwin OS memory tracking in NumpyProfiler."""
    from ml_switcheroo_compiler.backends.numpy.profiler import NumpyProfiler

    profiler = NumpyProfiler()

    # 1. Non-darwin platform branch
    with patch("sys.platform", "linux"):
        res_linux = profiler.profile_graph(IRGraph(), {}, num_iters=1, warmup_iters=1)
        assert res_linux["peak_memory_mb"] > 0.0

    # 2. Empty inputs branch (line 55)
    res_empty = profiler.profile_graph(IRGraph(), {}, num_iters=1, warmup_iters=1)
    assert res_empty["latency_ms"] >= 0.0
    assert len(res_empty["latencies"]) == 1

    # 3. Non-empty inputs with list conversion
    inp = {"a": [1.0, 2.0], "b": np.array([3.0, 4.0], dtype=np.float32)}
    res_full = profiler.profile_graph(IRGraph(), inp, num_iters=2, warmup_iters=1)
    assert len(res_full["latencies"]) == 2


def test_jax_profiler_runtime():
    """Verify execution latency and non-darwin OS memory tracking in JAXProfiler."""
    try:
        from ml_switcheroo_compiler.backends.jax.profiler import JAXProfiler
    except (ValueError, ImportError, AttributeError) as e:
        if "RegisterCanCast" in str(e) and "ml_switcheroo_compiler.backends.jax.profiler" in sys.modules:
            JAXProfiler = sys.modules["ml_switcheroo_compiler.backends.jax.profiler"].JAXProfiler
        else:
            pytest.skip(f"JAX not available: {e}")

    profiler = JAXProfiler()

    # 1. Non-darwin platform branch
    with patch("sys.platform", "linux"):
        res_linux = profiler.profile_graph(IRGraph(), {}, num_iters=1, warmup_iters=1)
        assert res_linux["peak_memory_mb"] > 0.0

    # 2. Empty inputs branch
    res_empty = profiler.profile_graph(IRGraph(), {}, num_iters=1, warmup_iters=1)
    assert res_empty["latency_ms"] >= 0.0

    # 3. Already jax array input branch
    import jax.numpy as jnp

    arr = jnp.array([1.0, 2.0], dtype=jnp.float32)
    res_jax = profiler.profile_graph(IRGraph(), {"x": arr, "y": [3.0, 4.0]}, num_iters=2, warmup_iters=1)
    assert len(res_jax["latencies"]) == 2


def test_pytorch_profiler_runtime():
    """Verify execution latency and hardware synchronization paths in PyTorchProfiler."""
    if "torch" in sys.modules and hasattr(sys.modules["torch"], "Tensor"):
        torch = sys.modules["torch"]
    else:
        try:
            import torch
        except Exception as e:
            pytest.skip(f"PyTorch not available: {e}")

    from ml_switcheroo_compiler.backends.pytorch.profiler import PyTorchProfiler

    profiler = PyTorchProfiler()

    # 1. Non-darwin platform branch
    with patch("sys.platform", "linux"):
        res_linux = profiler.profile_graph(IRGraph(), {}, num_iters=1, warmup_iters=1)
        assert res_linux["peak_memory_mb"] > 0.0

    # 2. cuda device fallback when unavailable
    with patch("torch.cuda.is_available", return_value=False):
        dev = profiler._resolve_device("cuda")
        assert dev.type == "cpu"

    # 3. mps device fallback when unavailable
    with patch.object(torch.backends, "mps", create=True) as mock_mps:
        mock_mps.is_available.return_value = False
        dev = profiler._resolve_device("mps")
        assert dev.type == "cpu"

    # 4. Existing torch.Tensor input branch
    t_in = torch.tensor([1.0, 2.0], dtype=torch.float32)
    res_tensor = profiler.profile_graph(IRGraph(), {"t": t_in, "list": [3.0, 4.0]}, num_iters=2, warmup_iters=1)
    assert len(res_tensor["latencies"]) == 2

    # 5. Empty inputs branch
    res_empty = profiler.profile_graph(IRGraph(), {}, num_iters=1, warmup_iters=1)
    assert res_empty["latency_ms"] >= 0.0

    # 6. Simulated CUDA execution and memory peak
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.synchronize") as mock_cuda_sync:
            with patch("torch.cuda.max_memory_allocated", return_value=2097152):  # 2MB
                mock_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                profiler._sync(mock_dev)
                mock_cuda_sync.assert_called()

    # 7. Simulated MPS synchronization
    with patch.object(torch, "mps", create=True) as mock_mps_module:
        mock_mps_module.is_available.return_value = True
        mock_mps_dev = MagicMock()
        mock_mps_dev.type = "mps"
        profiler._sync(mock_mps_dev)
        mock_mps_module.synchronize.assert_called_once()

    # 8. Full profile_graph with CUDA device path
    with patch.object(profiler, "_resolve_device") as mock_res_dev:
        mock_cuda_target = MagicMock()
        mock_cuda_target.type = "cuda"
        mock_res_dev.return_value = mock_cuda_target
        with patch("torch.zeros", return_value=torch.tensor([0.0])):
            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.cuda.synchronize"):
                    with patch("torch.cuda.max_memory_allocated", return_value=4194304):
                        res_cuda = profiler.profile_graph(IRGraph(), {}, device="cuda", num_iters=1, warmup_iters=1)
                        assert res_cuda["peak_memory_mb"] == 4.0

    # 9. Graph ops, fallback when compiled is base_module and input_list is empty, and jit.trace success
    from ml_switcheroo_compiler.ir.core import IRNode

    g_ops = IRGraph()
    g_ops.nodes["x"] = IRNode(id="x", op_type="Input", shape_metadata=(2,))
    g_ops.nodes["y"] = IRNode(id="y", op_type="Add", inputs=["x", "x"], shape_metadata=(2,))
    g_ops.outputs = ["y"]

    with patch("torch.compile", side_effect=RuntimeError("compile fail")):
        # Empty input list with compile fallback
        compiled_empty = profiler._compile_module(g_ops, torch.device("cpu"), input_list=[])
        assert compiled_empty is not None

        # Successful jit.trace
        mock_traced = MagicMock()
        with patch.object(torch.jit, "trace", return_value=mock_traced):
            res_traced = profiler.profile_graph(g_ops, {"x": [1.0, 2.0]}, num_iters=1, warmup_iters=1)
            assert len(res_traced["latencies"]) == 1


def test_mlx_profiler_runtime():
    """Verify execution latency and Apple Silicon Metal peak memory metrics in MLXProfiler."""
    from ml_switcheroo_compiler.backends.mlx.profiler import (
        MLXProfiler,
        _get_mlx_peak_memory_mb,
        _prepare_mlx_inputs,
    )

    profiler = MLXProfiler()

    # 1. Non-darwin platform branch
    with patch("sys.platform", "linux"):
        res_linux = profiler.profile_graph(IRGraph(), {}, num_iters=1, warmup_iters=1)
        assert res_linux["peak_memory_mb"] > 0.0

    # 2. mx is None branch
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", None):
        res_none = profiler.profile_graph(IRGraph(), {"a": [1.0]}, num_iters=2)
        assert res_none["latency_ms"] == 0.0
        assert _prepare_mlx_inputs({"a": [1.0]}) == []

    # 3. Peak memory with metal.get_peak_memory > 0
    mock_mx = MagicMock()
    mock_mx.metal.get_peak_memory.return_value = 8388608  # 8MB
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_mx):
        res_metal = profiler.profile_graph(IRGraph(), {}, num_iters=1, warmup_iters=1)
        assert res_metal["peak_memory_mb"] == 8.0

    # 3b. Peak memory with metal.get_peak_memory == 0 (line 39 fallback)
    mock_mx_zero = MagicMock()
    mock_mx_zero.metal.get_peak_memory.return_value = 0.0
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_mx_zero):
        mem_fallback = _get_mlx_peak_memory_mb()
        assert mem_fallback > 0.0

    # 3c. Peak memory when mx has no metal attribute (branch 35->39)
    mock_no_metal = MagicMock(spec=[])
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_no_metal):
        mem_no_metal = _get_mlx_peak_memory_mb()
        assert mem_no_metal > 0.0

    # 4. Empty inputs branch
    res_empty = profiler.profile_graph(IRGraph(), {}, num_iters=1, warmup_iters=1)
    assert res_empty["latency_ms"] >= 0.0

    # 5. Input as native mx.array
    try:
        import mlx.core as mx

        arr = mx.array([1.0, 2.0], dtype=mx.float32)
        res_mlx = profiler.profile_graph(IRGraph(), {"arr": arr, "list": [3.0, 4.0]}, num_iters=2, warmup_iters=1)
        assert len(res_mlx["latencies"]) == 2
    except ImportError:
        pass

    # 6. Test ImportError on mlx import
    import importlib
    import sys

    import ml_switcheroo_compiler.backends.mlx as real_mlx_pkg
    import ml_switcheroo_compiler.backends.mlx.profiler as mlx_prof

    sys.modules["ml_switcheroo_compiler.backends.mlx"] = real_mlx_pkg

    with patch.dict(sys.modules, {"mlx": None, "mlx.core": None}):
        importlib.reload(mlx_prof)
        assert mlx_prof.mx is None
    importlib.reload(mlx_prof)

    # 7. Additional coverage for _sync_mlx_result and _prepare_mlx_inputs and has_ops compilation
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.backends.mlx.profiler import (
        _prepare_mlx_inputs,
        _sync_mlx_result,
    )

    # Line 49: _sync_mlx_result with None or missing eval
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", None):
        assert _sync_mlx_result("res") is None
    mock_no_eval = MagicMock(spec=[])
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_no_eval):
        assert _sync_mlx_result("res") is None

    # Lines 51-52: _sync_mlx_result with dictionary result
    mock_eval = MagicMock()
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_eval):
        _sync_mlx_result({"k1": 1, "k2": 2})
        assert mock_eval.eval.call_count == 2

    # Line 75: _prepare_mlx_inputs with custom mock array
    class MockMlxArray:
        __module__ = "mlx.core"

    MockMlxArray.__name__ = "array"
    m_arr = MockMlxArray()
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", MagicMock()):
        assert _prepare_mlx_inputs({"custom": m_arr}) == [m_arr]

    # Lines 119-140, 149: has_ops = True with compile, failed compile, and no compile
    g_ops = IRGraph()
    g_ops.nodes["relu"] = LogicalNode("relu", "Relu", ["in1"])

    mock_mlx_ops = MagicMock()
    mock_mlx_ops.compile.side_effect = lambda fn: fn
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_mlx_ops):
        with patch("ml_switcheroo_compiler.interpreter.evaluator.evaluate_graph", return_value={"relu": [1.0]}):
            res_ops = profiler.profile_graph(g_ops, {"in1": [1.0]}, num_iters=1, warmup_iters=1)
            assert res_ops["latency_ms"] >= 0.0

    mock_mlx_fail_compile = MagicMock()
    mock_mlx_fail_compile.compile.side_effect = RuntimeError("compile failed")
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_mlx_fail_compile):
        with patch("ml_switcheroo_compiler.interpreter.evaluator.evaluate_graph", return_value={"relu": [1.0]}):
            res_fail = profiler.profile_graph(g_ops, {"in1": [1.0]}, num_iters=1, warmup_iters=1)
            assert res_fail["latency_ms"] >= 0.0

    mock_mlx_no_compile = MagicMock(spec=["eval", "array", "metal", "zeros", "zeros_like", "add", "float32"])
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_mlx_no_compile):
        with patch("ml_switcheroo_compiler.interpreter.evaluator.evaluate_graph", return_value={"relu": [1.0]}):
            res_no_comp = profiler.profile_graph(g_ops, {"in1": [1.0]}, num_iters=1, warmup_iters=1)
            assert res_no_comp["latency_ms"] >= 0.0


def test_cupy_profiler_runtime():
    """Verify execution simulation and memory peak tracking in CupyProfiler."""
    from ml_switcheroo_compiler.backends.cupy.profiler import CupyProfiler

    profiler = CupyProfiler()

    # 1. Non-darwin platform branch
    with patch("sys.platform", "linux"):
        res_linux = profiler.profile_graph(IRGraph(), {}, num_iters=1, warmup_iters=1)
        assert res_linux["peak_memory_mb"] > 0.0

    # 2. cp is None branch
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", None):
        res_none = profiler.profile_graph(IRGraph(), {"a": [1.0]}, num_iters=2)
        assert res_none["latency_ms"] == 0.0
        assert len(res_none["latencies"]) == 2

    # 3. Simulated CuPy execution with mock array and pool
    mock_cp = MagicMock()
    dummy_arr = MagicMock()
    mock_cp.ndarray = type("DummyArray", (), {})
    mock_cp.asarray.return_value = dummy_arr
    mock_cp.zeros_like.return_value = dummy_arr
    mock_cp.add.return_value = dummy_arr
    mock_cp.zeros.return_value = dummy_arr
    mock_cp.cuda.get_elapsed_time.return_value = 1.5
    mock_cp.get_default_memory_pool().used_bytes.return_value = 5242880  # 5MB

    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mock_cp):
        # Empty inputs
        res_empty = profiler.profile_graph(IRGraph(), {}, num_iters=1, warmup_iters=1)
        assert res_empty["peak_memory_mb"] == 5.0

        # Non-empty inputs with mock_cp.ndarray and list
        inst = mock_cp.ndarray()
        res_exec = profiler.profile_graph(IRGraph(), {"arr": inst, "lst": [1.0, 2.0]}, num_iters=2, warmup_iters=1)
        assert res_exec["peak_memory_mb"] == 5.0
        assert len(res_exec["latencies"]) == 2
        mock_cp.cuda.Device().synchronize.assert_called()

        # Graph with ops branch (has_ops = True)
        from ml_switcheroo_compiler.ir.core import IRNode

        graph_with_ops = IRGraph()
        graph_with_ops.nodes["relu1"] = IRNode("Relu", "relu1", ["arr"])
        with patch("ml_switcheroo_compiler.interpreter.evaluator.evaluate_graph", return_value={"relu1": inst}) as mock_eval:
            res_ops = profiler.profile_graph(graph_with_ops, {"arr": inst}, num_iters=1, warmup_iters=1)
            assert mock_eval.called
            assert res_ops["peak_memory_mb"] == 5.0

        # Fallback when cuda.Event raises an exception
        mock_cp.cuda.Event.side_effect = RuntimeError("No CUDA")
        res_fallback = profiler.profile_graph(IRGraph(), {}, num_iters=1)
        assert len(res_fallback["latencies"]) == 1
        mock_cp.cuda.Event.side_effect = None

        # Fallback when cp does not have cuda events (has_cuda_events is False)
        mock_cp.cuda = None
        res_no_events = profiler.profile_graph(IRGraph(), {}, num_iters=1)
        assert len(res_no_events["latencies"]) == 1

    # 4. Test ImportError on cupy import
    import importlib
    import sys

    import ml_switcheroo_compiler.backends.cupy.profiler as cupy_prof

    with patch.dict(sys.modules, {"cupy": None}):
        importlib.reload(cupy_prof)
        assert cupy_prof.cp is None
    importlib.reload(cupy_prof)

    # 5. Peak memory CuPy branches (_get_cupy_peak_memory_mb)
    from ml_switcheroo_compiler.backends.cupy.profiler import _get_cupy_peak_memory_mb

    # Line 35: cp is None
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", None):
        assert _get_cupy_peak_memory_mb() > 0.0

    # Lines 37-44: Device().mem_info = (free, total)
    mock_cp_mem = MagicMock()
    mock_cp_mem.cuda.Device().mem_info = (1024 * 1024 * 10, 1024 * 1024 * 30)
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mock_cp_mem):
        assert _get_cupy_peak_memory_mb() == 20.0

    # Branch 42->47: mem_info contains non-numbers
    mock_cp_str_mem = MagicMock()
    mock_cp_str_mem.cuda.Device().mem_info = ("str1", "str2")
    mock_cp_str_mem.get_default_memory_pool().used_bytes.return_value = 1024 * 1024 * 4
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mock_cp_str_mem):
        assert _get_cupy_peak_memory_mb() == 4.0

    # Lines 45-46: Device().mem_info raises exception
    mock_cp_mem_err = MagicMock()
    mock_dev_err = MagicMock()
    mock_dev_err.mem_info = property(lambda self: (_ for _ in ()).throw(RuntimeError("err")))
    mock_cp_mem_err.cuda.Device.side_effect = RuntimeError("device error")
    mock_cp_mem_err.get_default_memory_pool().used_bytes.return_value = 1024 * 1024 * 5
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mock_cp_mem_err):
        assert _get_cupy_peak_memory_mb() == 5.0

    # Line 52: used_bytes raises exception
    mock_cp_pool_err = MagicMock(spec=["get_default_memory_pool"])
    mock_cp_pool_err.get_default_memory_pool().used_bytes.side_effect = RuntimeError("pool err")
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mock_cp_pool_err):
        assert _get_cupy_peak_memory_mb() > 0.0

    # Line 54: no get_default_memory_pool
    mock_cp_no_pool = MagicMock(spec=[])
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mock_cp_no_pool):
        assert _get_cupy_peak_memory_mb() > 0.0

    # 6. _sync branches
    # Line 64: cp is None
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", None):
        assert profiler._sync() is None

    # Lines 65-69: Stream.null.synchronize succeeds and fails
    mock_cp_sync = MagicMock()
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mock_cp_sync):
        profiler._sync()
        mock_cp_sync.cuda.Stream.null.synchronize.assert_called_once()

    mock_cp_sync_err = MagicMock()
    mock_cp_sync_err.cuda.Stream.null.synchronize.side_effect = RuntimeError("stream fail")
    mock_cp_sync_err.cuda.Device().synchronize.side_effect = RuntimeError("device fail")
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mock_cp_sync_err):
        profiler._sync()  # does not raise

    # Branches 65->70 and 70->exit: cp without Stream or Device
    mock_cp_bare = MagicMock(spec=["cuda"])
    mock_cp_bare.cuda = MagicMock(spec=[])
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mock_cp_bare):
        profiler._sync()

    # 7. _is_cupy_available branches
    from ml_switcheroo_compiler.backends.cupy.profiler import _is_cupy_available

    class CustomMod:
        """Dummy module container for non-MagicMock availability testing."""

        pass

    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", None):
        assert _is_cupy_available() is False

    mod_no_cuda = CustomMod()
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mod_no_cuda):
        assert _is_cupy_available() is True

    mod_cuda_unavail = CustomMod()
    mod_cuda_unavail.cuda = CustomMod()
    mod_cuda_unavail.cuda.is_available = lambda: False
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mod_cuda_unavail):
        assert _is_cupy_available() is False

    mod_cuda_avail = CustomMod()
    mod_cuda_avail.cuda = CustomMod()
    mod_cuda_avail.cuda.is_available = lambda: True
    mod_cuda_avail.cuda.runtime = CustomMod()
    mod_cuda_avail.cuda.runtime.getDeviceCount = lambda: 1
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mod_cuda_avail):
        assert _is_cupy_available() is True

    mod_cuda_no_runtime = CustomMod()
    mod_cuda_no_runtime.cuda = CustomMod()
    mod_cuda_no_runtime.cuda.is_available = lambda: True
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mod_cuda_no_runtime):
        assert _is_cupy_available() is True

    mod_cuda_zero = CustomMod()
    mod_cuda_zero.cuda = CustomMod()
    mod_cuda_zero.cuda.is_available = lambda: True
    mod_cuda_zero.cuda.runtime = CustomMod()
    mod_cuda_zero.cuda.runtime.getDeviceCount = lambda: 0
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mod_cuda_zero):
        assert _is_cupy_available() is False

    mod_cuda_err = CustomMod()
    mod_cuda_err.cuda = CustomMod()

    def _raise_cuda_error() -> int:
        raise RuntimeError("boom")

    mod_cuda_err.cuda.runtime = CustomMod()
    mod_cuda_err.cuda.runtime.getDeviceCount = _raise_cuda_error
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mod_cuda_err):
        assert _is_cupy_available() is False
