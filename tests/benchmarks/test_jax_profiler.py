"""Exhaustive unit tests for JAXProfiler."""

import sys
from unittest.mock import MagicMock, patch

import pytest

if "jax" in sys.modules:
    jax = sys.modules["jax"]
    import jax.numpy as jnp
else:
    try:
        import jax
        import jax.numpy as jnp
    except (ValueError, ImportError, AttributeError, RuntimeError) as e:
        pytest.skip(f"JAX not available: {e}", allow_module_level=True)

if "ml_switcheroo_compiler.backends.jax.profiler" in sys.modules:
    prof_mod = sys.modules["ml_switcheroo_compiler.backends.jax.profiler"]
    JAXProfiler = prof_mod.JAXProfiler
    _extract_jax_peak_memory_mb = prof_mod._extract_jax_peak_memory_mb
    _sync_jax_result = prof_mod._sync_jax_result
else:
    try:
        from ml_switcheroo_compiler.backends.jax.profiler import (
            JAXProfiler,
            _extract_jax_peak_memory_mb,
            _sync_jax_result,
        )
    except (ValueError, ImportError, AttributeError, RuntimeError) as e:
        if "ml_switcheroo_compiler.backends.jax.profiler" in sys.modules:
            prof_mod = sys.modules["ml_switcheroo_compiler.backends.jax.profiler"]
            JAXProfiler = prof_mod.JAXProfiler
            _extract_jax_peak_memory_mb = prof_mod._extract_jax_peak_memory_mb
            _sync_jax_result = prof_mod._sync_jax_result
        else:
            pytest.skip(f"JAX profiler not importable: {e}", allow_module_level=True)

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_jax_profiler_exhaustive() -> None:
    """Verify all branches and non-darwin OS memory tracking in JAXProfiler."""
    profiler = JAXProfiler()

    # 1. Non-darwin platform branch
    with patch("sys.platform", "linux"):
        res_linux = profiler.profile_graph(IRGraph(), {}, num_iters=1, warmup_iters=1)
        assert res_linux["peak_memory_mb"] > 0.0

    # 2. Empty inputs branch
    res_empty = profiler.profile_graph(IRGraph(), {}, num_iters=1, warmup_iters=1)
    assert res_empty["latency_ms"] >= 0.0

    # 3. Already jax array input branch
    arr = jnp.array([1.0, 2.0], dtype=jnp.float32)
    res_jax = profiler.profile_graph(IRGraph(), {"x": arr, "y": [3.0, 4.0]}, num_iters=2, warmup_iters=1)
    assert len(res_jax["latencies"]) == 2

    # 4. Graph with operations to exercise _raw_forward_step and jax.jit (lines 89-118)
    g_ops = IRGraph()
    g_ops.nodes["x"] = IRNode(id="x", op_type="Input", shape_metadata=(2,))
    g_ops.nodes["y"] = IRNode(id="y", op_type="Add", inputs=["x", "x"], shape_metadata=(2,))
    g_ops.outputs = ["y"]
    res_ops = profiler.profile_graph(g_ops, {"x": jnp.array([1.0, 2.0])}, num_iters=2, warmup_iters=1)
    assert len(res_ops["latencies"]) == 2

    # 5. Fallback when jax.jit raises Exception
    with patch("jax.jit", side_effect=RuntimeError("jit fail")):
        res_jit_fail = profiler.profile_graph(g_ops, {"x": [1.0, 2.0]}, num_iters=1, warmup_iters=1)
        assert len(res_jit_fail["latencies"]) == 1

    # 6. Failure when JAXCodeGenerator doesn't produce callable apply_model
    with patch("ml_switcheroo_compiler.backends.jax.generator.JAXCodeGenerator.generate", return_value="# empty"):
        with pytest.raises(TypeError, match="Failed to generate valid apply_model function"):
            profiler.profile_graph(g_ops, {"x": [1.0, 2.0]}, num_iters=1, warmup_iters=1)

    # 7. Test _sync_jax_result with dict and single array
    mock_val = MagicMock()
    mock_val.block_until_ready = MagicMock()
    _sync_jax_result({"out": mock_val})
    assert mock_val.block_until_ready.called

    mock_single = MagicMock()
    mock_single.block_until_ready = MagicMock()
    _sync_jax_result(mock_single)
    assert mock_single.block_until_ready.called

    # Object without block_until_ready
    _sync_jax_result({"out": 42})
    _sync_jax_result(100)

    # 8. Test _sync_jax_result fallback branches when tree_util is unavailable or raises
    with patch.object(jax, "tree_util", None):
        mock_d = MagicMock()
        _sync_jax_result({"key": mock_d})
        assert mock_d.block_until_ready.called

        mock_seq = MagicMock()
        _sync_jax_result([mock_seq, (mock_seq,)])
        assert mock_seq.block_until_ready.called

        mock_leaf = MagicMock()
        _sync_jax_result(mock_leaf)
        assert mock_leaf.block_until_ready.called

        _sync_jax_result(42)

    with patch.object(jax.tree_util, "tree_leaves", side_effect=RuntimeError("tree fail")):
        mock_leaf2 = MagicMock()
        _sync_jax_result(mock_leaf2)
        assert mock_leaf2.block_until_ready.called

    # 9. Test _extract_jax_peak_memory_mb with GPU/TPU accelerator backends
    mock_dev_peak = MagicMock()
    mock_dev_peak.memory_stats.return_value = {"peak_bytes_in_use": 4194304}
    with patch("jax.default_backend", return_value="gpu"), patch("jax.devices", return_value=[mock_dev_peak]):
        assert _extract_jax_peak_memory_mb() == 4.0

    mock_dev_bytes = MagicMock()
    mock_dev_bytes.memory_stats.return_value = {"bytes_in_use": 2097152}
    with patch("jax.default_backend", return_value="gpu"), patch("jax.devices", return_value=[mock_dev_bytes]):
        assert _extract_jax_peak_memory_mb() == 2.0

    mock_dev_none = MagicMock()
    mock_dev_none.memory_stats.return_value = None
    with patch("jax.default_backend", return_value="gpu"), patch("jax.devices", return_value=[mock_dev_none]):
        assert _extract_jax_peak_memory_mb() > 0.0

    with patch("jax.default_backend", return_value="gpu"), patch("jax.devices", return_value=[]):
        assert _extract_jax_peak_memory_mb() > 0.0

    mock_dev_nostats = MagicMock(spec=[])
    with patch("jax.default_backend", return_value="tpu"), patch("jax.devices", return_value=[mock_dev_nostats]):
        assert _extract_jax_peak_memory_mb() > 0.0

    with patch("jax.default_backend", side_effect=RuntimeError("backend fail")):
        assert _extract_jax_peak_memory_mb() > 0.0

    # 10. Empty latencies handling in profile_graph
    with patch("builtins.range", return_value=[]):
        res_empty_lats = profiler.profile_graph(IRGraph(), {}, num_iters=0, warmup_iters=0)
        assert res_empty_lats["latency_ms"] == 0.0
