"""Exhaustive unit tests for MLXProfiler."""

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

import ml_switcheroo_compiler.backends.mlx  # noqa: F401
import ml_switcheroo_compiler.backends.mlx.profiler as mlx_prof_module
from ml_switcheroo_compiler.backends.mlx.profiler import (
    MLXProfiler,
    _get_mlx_peak_memory_mb,
    _get_process_memory_mb,
    _prepare_mlx_inputs,
    _reset_mlx_peak_memory,
    _sync_mlx_result,
)
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_mlx_import_error() -> None:
    """Verify fallback when mlx cannot be imported."""
    import ml_switcheroo_compiler.backends.mlx as mlx_parent

    sys.modules["ml_switcheroo_compiler.backends.mlx"] = mlx_parent
    with patch.dict(sys.modules, {"mlx": None, "mlx.core": None}):
        importlib.reload(mlx_prof_module)
        assert mlx_prof_module.mx is None

    # Reload back to normal state
    sys.modules["ml_switcheroo_compiler.backends.mlx"] = mlx_parent
    importlib.reload(mlx_prof_module)


def test_process_memory_branches() -> None:
    """Verify host process memory calculation on darwin and non-darwin systems."""
    with patch("sys.platform", "darwin"):
        mem_darwin = _get_process_memory_mb()
        assert mem_darwin > 0.0

    with patch("sys.platform", "linux"):
        mem_linux = _get_process_memory_mb()
        assert mem_linux > 0.0


def test_get_mlx_peak_memory_branches() -> None:
    """Verify all branches of _get_mlx_peak_memory_mb and _reset_mlx_peak_memory."""
    # 1. mx is None
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", None):
        assert _get_mlx_peak_memory_mb() > 0.0
        _reset_mlx_peak_memory()

    # 2. mx with metal.get_peak_memory and metal.reset_peak_memory
    mock_mx_metal = MagicMock()
    mock_mx_metal.metal.get_peak_memory.return_value = 2097152.0  # 2MB
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_mx_metal):
        assert _get_mlx_peak_memory_mb() == 2.0
        _reset_mlx_peak_memory()
        mock_mx_metal.metal.reset_peak_memory.assert_called_once()

    # 3. mx with metal.get_peak_memory returning 0.0 and get_active_memory returning > 0
    mock_mx_metal_active = MagicMock()
    mock_mx_metal_active.metal.get_peak_memory.return_value = 0.0
    mock_mx_metal_active.metal.get_active_memory.return_value = 3145728.0  # 3MB
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_mx_metal_active):
        assert _get_mlx_peak_memory_mb() == 3.0

    # 3b. mx with metal.get_peak_memory returning 0.0 and get_active_memory returning 0.0
    mock_mx_metal_active_zero = MagicMock()
    mock_mx_metal_active_zero.metal.get_peak_memory.return_value = 0.0
    mock_mx_metal_active_zero.metal.get_active_memory.return_value = 0.0
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_mx_metal_active_zero):
        assert _get_mlx_peak_memory_mb() > 0.0

    # 3c. mx with metal.get_peak_memory returning 0.0 and get_active_memory raising exception
    mock_mx_metal_active_err = MagicMock()
    mock_mx_metal_active_err.metal.get_peak_memory.return_value = 0.0
    mock_mx_metal_active_err.metal.get_active_memory.side_effect = RuntimeError("active memory error")
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_mx_metal_active_err):
        assert _get_mlx_peak_memory_mb() > 0.0

    # 4. mx with metal.get_peak_memory raising exception
    mock_mx_err = MagicMock()
    mock_mx_err.metal.get_peak_memory.side_effect = RuntimeError("metal error")
    mock_mx_err.metal.reset_peak_memory.side_effect = RuntimeError("reset error")
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_mx_err):
        assert _get_mlx_peak_memory_mb() > 0.0
        _reset_mlx_peak_memory()

    # 5. mx with direct get_peak_memory and reset_peak_memory (no metal module)
    mock_mx_direct = MagicMock(spec=["get_peak_memory", "reset_peak_memory"])
    mock_mx_direct.get_peak_memory.return_value = 4194304.0  # 4MB
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_mx_direct):
        assert _get_mlx_peak_memory_mb() == 4.0
        _reset_mlx_peak_memory()
        mock_mx_direct.reset_peak_memory.assert_called_once()

    # 6. mx with direct get_peak_memory returning 0.0
    mock_mx_direct_zero = MagicMock(spec=["get_peak_memory", "reset_peak_memory"])
    mock_mx_direct_zero.get_peak_memory.return_value = 0.0
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_mx_direct_zero):
        assert _get_mlx_peak_memory_mb() > 0.0

    # 7. mx without metal and without get_peak_memory/reset_peak_memory (covers 40->42, 42->49, 58->60, 60->exit)
    mock_mx_bare = MagicMock(spec=[])
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_mx_bare):
        assert _get_mlx_peak_memory_mb() > 0.0
        _reset_mlx_peak_memory()


def test_sync_mlx_result_exhaustive() -> None:
    """Verify all collection and single array synchronization branches in _sync_mlx_result."""
    # mx is None
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", None):
        _sync_mlx_result(123)

    # mx without eval
    mock_no_eval = MagicMock(spec=[])
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_no_eval):
        _sync_mlx_result(123)

    # dict, list, tuple, and scalar/single object
    mock_mx = MagicMock()
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_mx):
        _sync_mlx_result({"a": 1, "b": 2})
        assert mock_mx.eval.call_count == 2
        mock_mx.eval.reset_mock()

        # Non-empty and empty lists
        _sync_mlx_result([3, 4])
        assert mock_mx.eval.call_count == 2
        mock_mx.eval.reset_mock()

        _sync_mlx_result([])
        assert mock_mx.eval.call_count == 0

        # Non-empty and empty tuples
        _sync_mlx_result((5, 6))
        assert mock_mx.eval.call_count == 2
        mock_mx.eval.reset_mock()

        _sync_mlx_result(())
        assert mock_mx.eval.call_count == 0

        _sync_mlx_result(7)
        mock_mx.eval.assert_called_once_with(7)


def test_prepare_mlx_inputs_exhaustive() -> None:
    """Verify input preparation with real arrays, mock array types, and raw lists."""
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", None):
        assert _prepare_mlx_inputs({"a": [1.0]}) == []

    class DummyArrayType:
        """Dummy array class for type check."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialize dummy array."""
            pass

    class CustomMlxArray:
        """Custom MLX array simulator."""

        __module__ = "mlx.core"

    CustomMlxArray.__name__ = "array"

    mock_mx = MagicMock()
    mock_mx.array = DummyArrayType
    mock_mx.float32 = "float32"

    dummy_inst = DummyArrayType()
    custom_inst = CustomMlxArray()

    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_mx):
        res = _prepare_mlx_inputs(
            {
                "is_inst": dummy_inst,
                "custom": custom_inst,
                "raw": [1.0, 2.0],
            }
        )
        assert res[0] is dummy_inst
        assert res[1] is custom_inst


def test_prepare_ordered_inputs_branches() -> None:
    """Verify MLXProfiler._prepare_ordered_inputs branch logic."""
    profiler = MLXProfiler()

    class DummyArrayType:
        """Dummy array class for type check."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialize dummy array."""
            pass

    class CustomMlxArray:
        """Custom MLX array simulator."""

        __module__ = "mlx.core"

    CustomMlxArray.__name__ = "array"

    mock_mx = MagicMock()
    mock_mx.array = DummyArrayType
    mock_mx.float32 = "float32"

    dummy_inst = DummyArrayType()
    custom_inst = CustomMlxArray()

    g = IRGraph()
    g.nodes["k1"] = IRNode(id="k1", op_type="Input")
    g.nodes["k2"] = IRNode(id="k2", op_type="Input")

    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_mx):
        res = profiler._prepare_ordered_inputs(g, {"k1": dummy_inst, "k2": custom_inst})
        assert res[0] is dummy_inst
        assert res[1] is custom_inst


def test_mlx_profiler_compile_and_execute() -> None:
    """Verify MLXCodeGenerator compilation and step execution fallbacks."""
    profiler = MLXProfiler()

    # When mx is None
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", None):
        res_none = profiler.profile_graph(IRGraph(), {}, num_iters=2)
        assert res_none["latency_ms"] == 0.0

    class CustomMlxArray:
        """Custom MLX array simulator."""

        __module__ = "mlx.core"

    CustomMlxArray.__name__ = "array"
    custom_inst = CustomMlxArray()

    g_ops = IRGraph()
    g_ops.nodes["x"] = IRNode(id="x", op_type="Input", shape_metadata=(2,))
    g_ops.nodes["y"] = IRNode(id="y", op_type="Add", inputs=["x", "x"], shape_metadata=(2,))
    g_ops.outputs = ["y"]

    mock_mx = MagicMock()
    dummy_arr = MagicMock()
    mock_mx.zeros.return_value = dummy_arr
    mock_mx.zeros_like.return_value = dummy_arr
    mock_mx.add.return_value = dummy_arr
    mock_mx.float32 = "float32"
    mock_compiled_fn = MagicMock(return_value=dummy_arr)
    mock_mx.compile.return_value = mock_compiled_fn

    # Normal profile_graph with ops
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_mx):
        res = profiler.profile_graph(g_ops, {"x": [1.0, 2.0]}, num_iters=2, warmup_iters=1)
        assert len(res["latencies"]) == 2

    # Fallback when _prepare_ordered_inputs has unordered or unmatched keys
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_mx):
        res_unmatched = profiler.profile_graph(g_ops, {"unmatched": [1.0, 2.0]}, num_iters=1, warmup_iters=1)
        assert len(res_unmatched["latencies"]) == 1

    # Fallback when mx.compile fails with Exception
    mock_mx_fail_comp = MagicMock()
    mock_mx_fail_comp.compile.side_effect = RuntimeError("compile failure")
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_mx_fail_comp):
        res_fail_comp = profiler.profile_graph(g_ops, {"x": [1.0, 2.0]}, num_iters=1, warmup_iters=1)
        assert len(res_fail_comp["latencies"]) == 1

    # Fallback when mx does not have compile
    mock_mx_no_comp = MagicMock(spec=["eval", "array", "metal", "zeros", "zeros_like", "add", "float32"])
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_mx_no_comp):
        res_no_comp = profiler.profile_graph(g_ops, {"x": [1.0, 2.0]}, num_iters=1, warmup_iters=1)
        assert len(res_no_comp["latencies"]) == 1

    # TypeError when generator doesn't provide CompiledModel
    with patch("ml_switcheroo_compiler.backends.mlx.generator.MLXCodeGenerator.generate", return_value="# empty"):
        with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_mx):
            with pytest.raises(TypeError, match="Failed to generate valid CompiledModel"):
                profiler.profile_graph(g_ops, {"x": [1.0, 2.0]})

    # Profile graph without ops and empty inputs
    g_empty = IRGraph()
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_mx):
        res_empty = profiler.profile_graph(g_empty, {}, num_iters=1, warmup_iters=1)
        assert len(res_empty["latencies"]) == 1

    # Execution exception during step execution fallback
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_mx):
        with patch.object(profiler, "_compile_model", return_value=MagicMock(side_effect=RuntimeError("eval error"))):
            res_exec_err = profiler.profile_graph(g_ops, {"x": [1.0, 2.0]}, num_iters=1, warmup_iters=1)
            assert len(res_exec_err["latencies"]) == 1

    # Empty latencies return
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_mx):
        with patch.object(profiler, "_measure_latencies", return_value=[]):
            res_no_lats = profiler.profile_graph(g_empty, {}, num_iters=0, warmup_iters=0)
            assert res_no_lats["latency_ms"] == 0.0
