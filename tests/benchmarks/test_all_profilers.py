"""Comprehensive tests for new backend profilers and orchestrator integration."""

import multiprocessing
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.dask.profiler import (
    DaskProfiler,
    _sync_dask_result,
)
from ml_switcheroo_compiler.backends.edge.profiler import EdgeProfiler
from ml_switcheroo_compiler.backends.keras.profiler import (
    KerasProfiler,
    _get_keras_peak_memory_mb,
    _sync_keras_result,
)
from ml_switcheroo_compiler.backends.llvm_cpp.profiler import CppProfiler
from ml_switcheroo_compiler.backends.numba.profiler import NumbaProfiler
from ml_switcheroo_compiler.backends.sparse.profiler import SparseProfiler
from ml_switcheroo_compiler.backends.tensorflow.profiler import (
    TensorFlowProfiler,
    _get_tf_peak_memory_mb,
    _sync_tf_result,
)
from ml_switcheroo_compiler.benchmarks.config_models import BenchmarkPlan, BenchmarkTarget
from ml_switcheroo_compiler.benchmarks.orchestrator import BenchmarkOrchestrator, _isolated_worker
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode

# ==============================================================================
# TensorFlowProfiler Tests
# ==============================================================================


def test_tf_profiler_none():
    """Test TensorFlowProfiler behavior when tensorflow is None."""
    with patch("ml_switcheroo_compiler.backends.tensorflow.profiler.tf", None):
        profiler = TensorFlowProfiler()
        res = profiler.profile_graph(IRGraph(), {}, num_iters=2, warmup_iters=1)
        assert res["latency_ms"] == 0.0
        assert len(res["latencies"]) == 2
        assert profiler._prepare_inputs(IRGraph(), {"x": [1.0]}) == []
        assert _get_tf_peak_memory_mb() > 0.0
        _sync_tf_result([1.0])


def test_tf_profiler_memory():
    """Test TensorFlow peak memory extraction across physical devices and exceptions."""
    mock_tf = MagicMock()
    mock_tf.config.list_physical_devices.return_value = ["GPU:0"]
    mock_tf.config.experimental.get_memory_info.return_value = {"peak": 1024 * 1024 * 16}
    with patch("ml_switcheroo_compiler.backends.tensorflow.profiler.tf", mock_tf):
        assert _get_tf_peak_memory_mb() == 16.0

    mock_tf.config.experimental.get_memory_info.return_value = {"peak": 0, "current": 1024 * 1024 * 8}
    with patch("ml_switcheroo_compiler.backends.tensorflow.profiler.tf", mock_tf):
        assert _get_tf_peak_memory_mb() == 8.0

    mock_tf.config.experimental.get_memory_info.side_effect = RuntimeError("GPU fail")
    with patch("ml_switcheroo_compiler.backends.tensorflow.profiler.tf", mock_tf):
        assert _get_tf_peak_memory_mb() > 0.0


def test_tf_sync_result():
    """Test TensorFlow result synchronization across containers."""
    mock_val = MagicMock()
    _sync_tf_result({"a": mock_val, "b": [mock_val, (mock_val,)]})
    assert mock_val.numpy.call_count == 3

    mock_err_val = MagicMock()
    mock_err_val.numpy.side_effect = RuntimeError("fail")
    _sync_tf_result(mock_err_val)


def test_tf_profiler_execution():
    """Test TensorFlowProfiler graph compilation and execution."""
    profiler = TensorFlowProfiler()
    graph = IRGraph()
    n_in = LogicalNode(id="in_0", op_type="Input")
    n_add = LogicalNode(id="add_0", op_type="Add", inputs=["in_0", "in_0"])
    graph.nodes = {"in_0": n_in, "add_0": n_add}
    graph.inputs = ["in_0"]

    inputs = {"in_0": [1.0, 2.0]}
    res = profiler.profile_graph(graph, inputs, num_iters=3, warmup_iters=1)
    assert len(res["latencies"]) == 3
    assert res["peak_memory_mb"] > 0.0

    # Test empty inputs branch
    res_empty = profiler.profile_graph(IRGraph(), {}, num_iters=2, warmup_iters=1)
    assert len(res_empty["latencies"]) == 2

    # Test compile failure fallback
    with patch.object(profiler, "_compile_graph", side_effect=Exception("comp fail")):
        res_fallback = profiler.profile_graph(graph, inputs, num_iters=2, warmup_iters=1)
        assert len(res_fallback["latencies"]) == 2

    # Test compile graph invalid apply_model
    with patch("ml_switcheroo_compiler.backends.tensorflow.generator.TensorFlowCodeGenerator.generate", return_value="x = 1"):
        with pytest.raises(TypeError, match="Failed to generate valid apply_model"):
            profiler._compile_graph(graph)


# ==============================================================================
# KerasProfiler Tests
# ==============================================================================


def test_keras_profiler_none():
    """Test KerasProfiler behavior when keras is None."""
    with patch("ml_switcheroo_compiler.backends.keras.profiler.keras", None):
        profiler = KerasProfiler()
        res = profiler.profile_graph(IRGraph(), {}, num_iters=2, warmup_iters=1)
        assert res["latency_ms"] == 0.0
        assert len(res["latencies"]) == 2
        assert profiler._prepare_inputs(IRGraph(), {"x": [1.0]}) == []
        assert _get_keras_peak_memory_mb() > 0.0
        _sync_keras_result([1.0])


def test_keras_profiler_memory_and_sync():
    """Test Keras peak memory and synchronization across backends."""
    mock_keras = MagicMock()
    mock_keras.backend.backend.return_value = "torch"
    with patch("ml_switcheroo_compiler.backends.keras.profiler.keras", mock_keras):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.max_memory_allocated", return_value=1024 * 1024 * 32):
                assert _get_keras_peak_memory_mb() == 32.0

    mock_keras.backend.backend.return_value = "tensorflow"
    with patch("ml_switcheroo_compiler.backends.keras.profiler.keras", mock_keras):
        with patch("tensorflow.config.experimental.get_memory_info", return_value={"peak": 1024 * 1024 * 12}):
            assert _get_keras_peak_memory_mb() == 12.0

    mock_val = MagicMock()
    mock_keras.ops.convert_to_numpy.return_value = np.array([1.0])
    with patch("ml_switcheroo_compiler.backends.keras.profiler.keras", mock_keras):
        _sync_keras_result({"res": [mock_val]})
        mock_keras.ops.convert_to_numpy.assert_called()

    # Fallback numpy() on val
    mock_keras_no_ops = MagicMock(spec=["backend"])
    with patch("ml_switcheroo_compiler.backends.keras.profiler.keras", mock_keras_no_ops):
        val_np = MagicMock()
        _sync_keras_result(val_np)
        val_np.numpy.assert_called_once()

        # Exception in numpy() fallback
        val_np_err = MagicMock()
        val_np_err.numpy.side_effect = RuntimeError("numpy fail")
        _sync_keras_result(val_np_err)

    # Exception in ops.convert_to_numpy
    mock_keras.ops.convert_to_numpy.side_effect = RuntimeError("ops fail")
    with patch("ml_switcheroo_compiler.backends.keras.profiler.keras", mock_keras):
        _sync_keras_result(mock_val)


def test_keras_profiler_execution():
    """Test KerasProfiler execution and fallback branches."""
    profiler = KerasProfiler()
    graph = IRGraph()
    n_in = LogicalNode(id="in_0", op_type="Input")
    graph.nodes = {"in_0": n_in}
    graph.inputs = ["in_0"]

    inputs = {"in_0": [1.0, 2.0]}
    res = profiler.profile_graph(graph, inputs, num_iters=2, warmup_iters=1)
    assert len(res["latencies"]) == 2

    # Graph with compiled model mock
    mock_model = MagicMock()
    mock_model.return_value = np.array([2.0, 4.0])
    with patch.object(profiler, "_compile_model", return_value=mock_model):
        n_add = LogicalNode(id="add_0", op_type="Add", inputs=["in_0", "in_0"])
        graph.nodes["add_0"] = n_add
        res_compiled = profiler.profile_graph(graph, inputs, num_iters=2, warmup_iters=1)
        assert len(res_compiled["latencies"]) == 2

    # Test compile model type error
    with patch("ml_switcheroo_compiler.backends.keras.generator.KerasCodeGenerator.generate", return_value="a = 1"):
        with pytest.raises(TypeError, match="Failed to generate valid get_model"):
            profiler._compile_model(graph)

    # Test empty inputs with ops.zeros
    mock_keras_ops = MagicMock()
    mock_keras_ops.ops.zeros.return_value = 0.0
    with patch("ml_switcheroo_compiler.backends.keras.profiler.keras", mock_keras_ops):
        fn_zeros = profiler._build_step_fn(False, None, [])
        assert fn_zeros() == 0.0

    # Test prepare_inputs when keras has no ops (line 136)
    mock_k_no_ops = MagicMock(spec=["backend"])
    with patch("ml_switcheroo_compiler.backends.keras.profiler.keras", mock_k_no_ops):
        res_no_ops = profiler._prepare_inputs(graph, {"in_0": 1.0})
        assert len(res_no_ops) == 1


# ==============================================================================
# DaskProfiler Tests
# ==============================================================================


def test_dask_profiler_none():
    """Test DaskProfiler when dask.array is None."""
    with patch("ml_switcheroo_compiler.backends.dask.profiler.da", None):
        profiler = DaskProfiler()
        res = profiler.profile_graph(IRGraph(), {}, num_iters=2, warmup_iters=1)
        assert res["latency_ms"] == 0.0
        assert len(res["latencies"]) == 2
        assert profiler._prepare_inputs(IRGraph(), {"x": [1.0]}) == []
        assert _sync_dask_result(123) == 123


def test_dask_profiler_execution():
    """Test DaskProfiler graph execution with compute synchronization."""
    profiler = DaskProfiler()
    graph = IRGraph()
    n_in = LogicalNode(id="in_0", op_type="Input")
    graph.nodes = {"in_0": n_in}
    graph.inputs = ["in_0"]

    inputs = {"in_0": np.array([1.0, 2.0], dtype=np.float32)}
    res = profiler.profile_graph(graph, inputs, num_iters=3, warmup_iters=1)
    assert len(res["latencies"]) == 3
    assert res["peak_memory_mb"] > 0.0

    # Test with compiled evaluate function
    mock_fn = MagicMock()
    mock_fn.return_value = np.array([3.0])
    with patch.object(profiler, "_compile_graph", return_value=mock_fn):
        n_op = LogicalNode(id="op_1", op_type="Add", inputs=["in_0", "in_0"])
        graph.nodes["op_1"] = n_op
        res_comp = profiler.profile_graph(graph, inputs, num_iters=2, warmup_iters=1)
        assert len(res_comp["latencies"]) == 2

    # Test _sync_dask_result compute success and failure
    mock_compute_ok = MagicMock()
    mock_compute_ok.compute.return_value = 100
    assert _sync_dask_result(mock_compute_ok) == 100

    mock_compute_err = MagicMock()
    mock_compute_err.compute.side_effect = RuntimeError("compute failed")
    assert _sync_dask_result(mock_compute_err) is mock_compute_err

    # Test _sync_dask_result list and tuple branches (lines 40-42)
    assert _sync_dask_result([mock_compute_ok]) == [100]
    assert _sync_dask_result((mock_compute_ok,)) == (100,)

    # Test compile graph type error
    with patch("ml_switcheroo_compiler.backends.dask.generator.DaskGenerator.generate", return_value="z = 0"):
        with pytest.raises(TypeError, match="Failed to generate valid evaluate"):
            profiler._compile_graph(graph)


# ==============================================================================
# CppProfiler Tests
# ==============================================================================


def test_cpp_profiler_execution():
    """Test CppProfiler execution and fallback handling."""
    profiler = CppProfiler()
    graph = IRGraph()
    n_in = LogicalNode(id="in_0", op_type="Input")
    graph.nodes = {"in_0": n_in}
    graph.inputs = ["in_0"]

    res_empty = profiler.profile_graph(graph, {}, num_iters=2, warmup_iters=1)
    assert len(res_empty["latencies"]) == 2

    mock_exec = MagicMock()
    with patch.object(profiler, "_compile_graph", return_value=mock_exec):
        n_add = LogicalNode(id="add_0", op_type="Add", inputs=["in_0", "in_0"])
        graph.nodes["add_0"] = n_add
        res = profiler.profile_graph(graph, {}, num_iters=3, warmup_iters=1)
        assert len(res["latencies"]) == 3
        assert mock_exec.call_count >= 4

    with patch("ml_switcheroo_compiler.backends.llvm_cpp.generator.CppGenerator._compile_aot_impl", side_effect=RuntimeError("gcc missing")):
        assert profiler._compile_graph(graph) is None

    # Test linux platform memory branch
    from ml_switcheroo_compiler.backends.llvm_cpp.profiler import _get_process_memory_mb

    with patch("sys.platform", "linux"):
        assert _get_process_memory_mb() > 0.0


def test_cpp_provider_edge_cases():
    """Test CppProvider edge cases."""
    from ml_switcheroo_compiler.backends.llvm_cpp.config_models import CppTemplatesConfig
    from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_operation, get_cpp_prelude, get_cpp_template

    assert get_cpp_template("non_existent_tmpl_xyz") == {}
    assert get_cpp_operation("non_existent_op_xyz") is None

    with patch("ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider._CPP_CONFIG", CppTemplatesConfig(templates={}, operations={}, prelude=None)):
        assert get_cpp_prelude() == ""


# ==============================================================================
# Numba, Sparse, and Edge Profiler Tests
# ==============================================================================


def test_numba_sparse_edge_profilers():
    """Test Numba, Sparse, and Edge profiler execution."""
    graph = IRGraph()
    n_in = LogicalNode(id="in_0", op_type="Input")
    graph.nodes = {"in_0": n_in}
    graph.inputs = ["in_0"]
    inputs = {"in_0": [1.0, 2.0]}

    numba_p = NumbaProfiler()
    res_numba = numba_p.profile_graph(graph, inputs, num_iters=2, warmup_iters=1)
    assert len(res_numba["latencies"]) == 2

    sparse_p = SparseProfiler()
    res_sparse = sparse_p.profile_graph(graph, inputs, num_iters=2, warmup_iters=1)
    assert len(res_sparse["latencies"]) == 2

    edge_p = EdgeProfiler()
    res_edge = edge_p.profile_graph(graph, inputs, num_iters=2, warmup_iters=1)
    assert len(res_edge["latencies"]) == 2


# ==============================================================================
# BenchmarkOrchestrator Resolution and Process Isolation Tests
# ==============================================================================


def test_orchestrator_dynamic_resolution_all_13_backends():
    """Assert BenchmarkOrchestrator._get_profiler dynamically resolves all registered backends."""
    plan = BenchmarkPlan(name="test_plan", models=["m"], batch_sizes=[1], targets=[])
    orchestrator = BenchmarkOrchestrator(plan)

    registered_backends = [
        "numpy",
        "pytorch",
        "torch",
        "jax",
        "mlx",
        "cupy",
        "tensorflow",
        "tf",
        "keras",
        "dask",
        "llvm_cpp",
        "numba",
        "sparse",
        "sparse_coo",
        "pure_python",
        "edge",
        "edge_wgsl",
        "edge_wasm_simd",
    ]

    for backend in registered_backends:
        profiler = orchestrator._get_profiler(backend)
        assert profiler is not None
        assert hasattr(profiler, "profile_graph")

    with pytest.raises(BackendNotSupportedError, match="does not have a supported profiler"):
        orchestrator._get_profiler("unknown_backend_123")


def test_orchestrator_isolated_worker():
    """Test isolated worker process function."""
    queue = multiprocessing.Queue()
    graph = IRGraph()
    spec = ("ml_switcheroo_compiler.backends.numpy.profiler", "NumpyProfiler")
    _isolated_worker(spec, graph, {}, "cpu", 2, 1, queue)
    status, result = queue.get(timeout=5)
    assert status == "ok"
    assert "latencies" in result

    # Test error reporting in isolated worker
    bad_spec = ("non_existent_module_xyz", "BadCls")
    _isolated_worker(bad_spec, graph, {}, "cpu", 2, 1, queue)
    status, err = queue.get(timeout=5)
    assert status == "error"
    assert "No module named" in err


def test_orchestrator_isolated_execution():
    """Test BenchmarkOrchestrator isolated execution out-of-process."""
    plan = BenchmarkPlan(
        name="test_iso",
        models=["m"],
        batch_sizes=[1],
        targets=[BenchmarkTarget(backend="numpy")],
        isolate_process=True,
    )
    orchestrator = BenchmarkOrchestrator(plan)
    graph = IRGraph()
    n_in = LogicalNode(id="in_0", op_type="Input")
    graph.nodes = {"in_0": n_in}
    graph.inputs = ["in_0"]

    res = orchestrator._run_single(graph, "numpy", 1, 2, 1, isolate=True)
    assert len(res["latencies"]) == 2
    assert res["peak_memory_mb"] > 0.0


def test_profiler_linux_memory_branches():
    """Test Linux platform memory extraction branches across all backend profilers."""
    from ml_switcheroo_compiler.backends.dask.profiler import _get_process_memory_mb as dask_mem
    from ml_switcheroo_compiler.backends.edge.profiler import _get_process_memory_mb as edge_mem
    from ml_switcheroo_compiler.backends.keras.profiler import _get_process_memory_mb as keras_mem
    from ml_switcheroo_compiler.backends.numba.profiler import _get_process_memory_mb as numba_mem
    from ml_switcheroo_compiler.backends.sparse.profiler import _get_process_memory_mb as sparse_mem
    from ml_switcheroo_compiler.backends.tensorflow.profiler import _get_process_memory_mb as tf_mem

    with patch("sys.platform", "linux"):
        assert dask_mem() > 0.0
        assert edge_mem() > 0.0
        assert keras_mem() > 0.0
        assert numba_mem() > 0.0
        assert sparse_mem() > 0.0
        assert tf_mem() > 0.0


def test_orchestrator_isolated_error_conditions():
    """Test out-of-process isolation timeout, empty queue, and worker failure."""
    plan = BenchmarkPlan(
        name="test_iso_err",
        models=["m"],
        batch_sizes=[1],
        targets=[BenchmarkTarget(backend="numpy")],
        isolate_process=True,
    )
    orchestrator = BenchmarkOrchestrator(plan)
    graph = IRGraph()

    # 1. Process timeout branch
    mock_proc_timeout = MagicMock()
    mock_proc_timeout.is_alive.return_value = True
    with patch("multiprocessing.get_context") as mock_ctx:
        mock_ctx().Process.return_value = mock_proc_timeout
        with pytest.raises(RuntimeError, match="timed out"):
            orchestrator._run_isolated(graph, "numpy", {})

    # 2. Empty queue branch
    mock_proc_empty = MagicMock()
    mock_proc_empty.is_alive.return_value = False
    mock_q_empty = MagicMock()
    mock_q_empty.empty.return_value = True
    with patch("multiprocessing.get_context") as mock_ctx:
        mock_ctx().Process.return_value = mock_proc_empty
        mock_ctx().Queue.return_value = mock_q_empty
        with pytest.raises(RuntimeError, match="terminated unexpectedly"):
            orchestrator._run_isolated(graph, "numpy", {})

    # 3. Worker reported error branch
    mock_q_err = MagicMock()
    mock_q_err.empty.return_value = False
    mock_q_err.get.return_value = ("error", "Crash inside worker")
    with patch("multiprocessing.get_context") as mock_ctx:
        mock_ctx().Process.return_value = mock_proc_empty
        mock_ctx().Queue.return_value = mock_q_err
        with pytest.raises(RuntimeError, match="Isolated execution failed: Crash inside worker"):
            orchestrator._run_isolated(graph, "numpy", {})

    # 4. Manifest path not found and exception
    with patch("pathlib.Path.exists", return_value=False):
        assert BenchmarkOrchestrator._load_manifest_profiles() == {}

    with patch("builtins.open", side_effect=OSError("read error")):
        assert BenchmarkOrchestrator._load_manifest_profiles() == {}
