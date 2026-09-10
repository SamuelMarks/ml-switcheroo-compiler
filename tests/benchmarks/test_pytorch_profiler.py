"""Exhaustive unit tests for PyTorchProfiler."""

import sys
from unittest.mock import MagicMock, patch

import pytest

if "torch" in sys.modules and hasattr(sys.modules["torch"], "Tensor"):
    torch = sys.modules["torch"]
else:
    try:
        import torch
    except (ImportError, RuntimeError, TypeError, ValueError) as e:
        pytest.skip(f"PyTorch not available: {e}", allow_module_level=True)

try:
    from ml_switcheroo_compiler.backends.pytorch.profiler import PyTorchProfiler, TorchGraphModule
except (ImportError, RuntimeError, TypeError, ValueError) as e:
    pytest.skip(f"PyTorch profiler not available: {e}", allow_module_level=True)
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_pytorch_profiler_exhaustive() -> None:
    """Verify all branches and hardware synchronization paths in PyTorchProfiler."""
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

    # 6. Synchronize MPS hardware branch
    with patch.object(torch, "mps", create=True) as mock_mps_module:
        mock_mps_module.is_available.return_value = True
        mock_mps_dev = MagicMock()
        mock_mps_dev.type = "mps"
        profiler._sync(mock_mps_dev)
        assert mock_mps_module.synchronize.called

    # 7. Synchronize CUDA hardware branch
    with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.synchronize") as mock_cuda_sync:
        mock_cuda_dev = MagicMock()
        mock_cuda_dev.type = "cuda"
        profiler._sync(mock_cuda_dev)
        assert mock_cuda_sync.called

    # 8. Reset CUDA stats branch
    with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.reset_peak_memory_stats") as mock_reset:
        mock_cuda_dev = MagicMock()
        mock_cuda_dev.type = "cuda"
        profiler._reset_cuda_stats(mock_cuda_dev)
        assert mock_reset.called

    # 9. Peak memory with CUDA device
    with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.max_memory_allocated", return_value=4194304):
        mock_cuda_dev = MagicMock()
        mock_cuda_dev.type = "cuda"
        with patch.object(profiler, "_resolve_device", return_value=mock_cuda_dev):
            with patch.object(profiler, "_sync"), patch.object(profiler, "_reset_cuda_stats"):
                with patch("torch.compile", side_effect=lambda fn: fn):
                    with patch("torch.zeros", return_value=torch.tensor([0.0])):
                        res_cuda = profiler.profile_graph(IRGraph(), {}, num_iters=1, warmup_iters=1)
                        assert res_cuda["peak_memory_mb"] == 4.0

    # 10. Graph with operations to exercise TorchGraphModule and lines 150-166
    g_ops = IRGraph()
    g_ops.nodes["x"] = IRNode(id="x", op_type="Input", shape_metadata=(2,))
    g_ops.nodes["y"] = IRNode(id="y", op_type="Add", inputs=["x", "x"], shape_metadata=(2,))
    g_ops.outputs = ["y"]
    res_ops = profiler.profile_graph(g_ops, {"x": torch.tensor([1.0, 2.0])}, num_iters=2, warmup_iters=1)
    assert len(res_ops["latencies"]) == 2

    # Direct call to TorchGraphModule.forward
    tgm = TorchGraphModule(g_ops)
    out_fwd = tgm.forward({"x": torch.tensor([1.0, 2.0])})
    assert out_fwd is not None

    # TorchGraphModule.forward with non-matching input keys
    out_unmatched = tgm.forward({"z": torch.tensor([1.0, 2.0])})
    assert out_unmatched is not None

    # TorchGraphModule.forward with no graph outputs
    g_no_out = IRGraph()
    g_no_out.nodes["x"] = IRNode(id="x", op_type="Input", shape_metadata=(2,))
    g_no_out.nodes["y"] = IRNode(id="y", op_type="Add", inputs=["x", "x"], shape_metadata=(2,))
    g_no_out.outputs = []
    tgm_no_out = TorchGraphModule(g_no_out)
    out_no_out = tgm_no_out.forward({"x": torch.tensor([1.0, 2.0])})
    assert out_no_out is None

    # TorchGraphModule TypeError when generator doesn't produce Module
    with patch("ml_switcheroo_compiler.backends.pytorch.generator.PyTorchCodeGenerator.generate", return_value="# empty"):
        with pytest.raises(TypeError, match="Failed to generate valid torch.nn.Module"):
            TorchGraphModule(g_ops)

    # Fallback when torch has non-callable compile attribute
    with patch.object(torch, "compile", None):
        res_no_compile = profiler.profile_graph(g_ops, {"x": [1.0, 2.0]}, num_iters=1, warmup_iters=1)
        assert res_no_compile["latency_ms"] >= 0.0

    # Fallback when torch.compile fails with Exception and jit.trace fails with Exception
    with patch("torch.compile", side_effect=RuntimeError("compile fail")):
        with patch.object(torch.jit, "trace", side_effect=RuntimeError("trace fail")):
            res_fail = profiler.profile_graph(g_ops, {"x": [1.0, 2.0]}, num_iters=1, warmup_iters=1)
            assert len(res_fail["latencies"]) == 1

    # Fallback when compiled is base_module and input_list is empty
    with patch("torch.compile", side_effect=RuntimeError("compile fail")):
        compiled_empty = profiler._compile_module(g_ops, torch.device("cpu"), input_list=[])
        assert compiled_empty is not None

    # When torch.compile fails but torch.jit.trace succeeds
    mock_traced = MagicMock()
    with patch("torch.compile", side_effect=RuntimeError("compile fail")):
        with patch.object(torch.jit, "trace", return_value=mock_traced):
            res_traced = profiler.profile_graph(g_ops, {"x": [1.0, 2.0]}, num_iters=1, warmup_iters=1)
            assert len(res_traced["latencies"]) == 1

    # Call _prepare_inputs directly
    prepared = profiler._prepare_inputs({"a": torch.tensor([1.0]), "b": [2.0]}, torch.device("cpu"))
    assert len(prepared) == 2

    # Call _reset_cuda_stats with exception
    with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.reset_peak_memory_stats", side_effect=RuntimeError):
        cuda_dev = MagicMock()
        cuda_dev.type = "cuda"
        profiler._reset_cuda_stats(cuda_dev)

    # CUDA Event timing and exception branch in _measure_latencies
    mock_cuda_device = MagicMock()
    mock_cuda_device.type = "cuda"
    mock_ev1 = MagicMock()
    mock_ev2 = MagicMock()
    mock_ev1.elapsed_time.return_value = 1.25

    with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.synchronize"):
        with patch("torch.cuda.Event", side_effect=[mock_ev1, mock_ev2]):
            lats_cuda = profiler._measure_latencies(mock_cuda_device, lambda: None, num_iters=1)
            assert lats_cuda == [1.25]

        with patch("torch.cuda.Event", side_effect=RuntimeError("event fail")):
            lats_cuda_err = profiler._measure_latencies(mock_cuda_device, lambda: None, num_iters=1)
            assert len(lats_cuda_err) == 1

    # Empty latencies handling in profile_graph
    with patch.object(profiler, "_measure_latencies", return_value=[]):
        res_empty_lats = profiler.profile_graph(IRGraph(), {}, num_iters=0, warmup_iters=0)
        assert res_empty_lats["latency_ms"] == 0.0

    # MPS peak memory tests
    mps_device = MagicMock()
    mps_device.type = "mps"

    mock_mps_mod = MagicMock()
    mock_mps_mod.current_allocated_memory.return_value = 2097152.0  # 2MB
    with patch.object(torch, "mps", mock_mps_mod):
        assert profiler._extract_peak_memory_mb(mps_device) == 2.0

        # mps returning 0.0
        mock_mps_mod.current_allocated_memory.return_value = 0.0
        assert profiler._extract_peak_memory_mb(mps_device) > 0.0

        # mps raising exception
        mock_mps_mod.current_allocated_memory.side_effect = RuntimeError("mps error")
        assert profiler._extract_peak_memory_mb(mps_device) > 0.0

    # mps without current_allocated_memory
    mock_mps_bare = MagicMock(spec=[])
    with patch.object(torch, "mps", mock_mps_bare):
        assert profiler._extract_peak_memory_mb(mps_device) > 0.0
