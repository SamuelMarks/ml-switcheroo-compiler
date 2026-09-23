import sys
from unittest import mock

from ml_switcheroo_ir import LogicalGraph, LogicalNode

"""Comprehensive tests for new backend profilers and orchestrator integration."""

import multiprocessing
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.dask.profiler import (
    DaskProfiler,
    _sync_dask_result,
)
from ml_switcheroo_compiler.backends.edge.profiler import (
    EdgeProfiler,
    _calculate_edge_memory_bytes,
    _get_process_memory_mb,
)
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
from ml_switcheroo_compiler.ir.core import IRGraph

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

    mock_tf_no_gpu = MagicMock()
    mock_tf_no_gpu.config.list_physical_devices.return_value = []
    with patch("ml_switcheroo_compiler.backends.tensorflow.profiler.tf", mock_tf_no_gpu):
        assert _get_tf_peak_memory_mb() > 0.0

    mock_tf_zero = MagicMock()
    mock_tf_zero.config.list_physical_devices.return_value = ["GPU:0"]
    mock_tf_zero.config.experimental.get_memory_info.return_value = {"peak": 0, "current": 0}
    with patch("ml_switcheroo_compiler.backends.tensorflow.profiler.tf", mock_tf_zero):
        assert _get_tf_peak_memory_mb() > 0.0

    import importlib

    import ml_switcheroo_compiler.backends.tensorflow.profiler as tf_prof

    with patch.dict("sys.modules", {"tensorflow": None}):
        importlib.reload(tf_prof)
        assert tf_prof.tf is None
    importlib.reload(tf_prof)

    with patch.object(tf_prof.sys, "platform", "linux"):
        assert tf_prof._get_process_memory_mb() > 0.0


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
    from ml_switcheroo_compiler.backends.llvm_cpp.profiler import (
        _flatten_input_values,
        _get_process_memory_mb,
    )

    # Test _flatten_input_values branches
    assert _flatten_input_values(42) == [42.0]
    assert _flatten_input_values(3.14) == [3.14]
    assert _flatten_input_values([1.0, (2.0, [3.0])]) == [1.0, 2.0, 3.0]
    assert _flatten_input_values(np.array([4.0, 5.0])) == [4.0, 5.0]
    assert _flatten_input_values(None) == []

    profiler = CppProfiler()
    graph = IRGraph()
    n_in = LogicalNode(id="in_0", op_type="Input")
    graph.nodes = {"in_0": n_in}
    graph.inputs = ["in_0"]

    res_empty = profiler.profile_graph(graph, {}, num_iters=2, warmup_iters=1)
    assert len(res_empty["latencies"]) == 2

    mock_exec = MagicMock()
    with patch.object(profiler, "_compile_graph", return_value=mock_exec):
        n_add = LogicalNode(id="add_0", op_type="Add", inputs=["in_0", "in_0"], shape_metadata=(2, -1, 3))
        graph.nodes["add_0"] = n_add
        graph.outputs = ["add_0"]
        inputs = {"in_0": [1.0, 2.0]}
        res = profiler.profile_graph(graph, inputs, num_iters=3, warmup_iters=1)
        assert len(res["latencies"]) == 3
        assert mock_exec.call_count >= 4

    with patch("ml_switcheroo_compiler.backends.llvm_cpp.generator.CppGenerator._compile_aot_impl", side_effect=RuntimeError("gcc missing")):
        assert profiler._compile_graph(graph) is None

    # Test darwin and linux platform memory branches
    with patch("sys.platform", "linux"):
        assert _get_process_memory_mb() > 0.0

    with patch("sys.platform", "darwin"):
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

    # 4. Manifest path not found, normal manifest loading, and fallback branches
    with patch("pathlib.Path.exists", return_value=False):
        assert BenchmarkOrchestrator._load_manifest_profiles() == {}

    with patch("builtins.open", side_effect=OSError("read error")):
        assert BenchmarkOrchestrator._load_manifest_profiles() == {}

    # Normal successful loading of manifest profiles (lines 97-101)
    specs = BenchmarkOrchestrator._load_manifest_profiles()
    assert "numpy" in specs

    # Manifest profile without profiler_module or profiler_class (branch 99->98)
    mock_incomplete_profile = MagicMock(profiler_module=None, profiler_class="SomeClass")
    with patch("ml_switcheroo_compiler.benchmarks.orchestrator.load_backend_profiles", return_value={"inc": mock_incomplete_profile}):
        assert BenchmarkOrchestrator._load_manifest_profiles() == {}

    # Isolated execution with unknown backend fallback to NumpyProfiler (line 162)
    mock_proc_ok = MagicMock()
    mock_proc_ok.is_alive.return_value = False
    mock_q_ok = MagicMock()
    mock_q_ok.empty.return_value = False
    mock_q_ok.get.return_value = ("ok", {"latencies": [1.0], "peak_memory_mb": 10.0})
    with patch("multiprocessing.get_context") as mock_ctx:
        mock_ctx().Process.return_value = mock_proc_ok
        mock_ctx().Queue.return_value = mock_q_ok
        metrics = orchestrator._run_isolated(graph, "unknown_backend_fallback_xyz", {})
        assert metrics["latencies"] == [1.0]


def test_tf_profiler_memory_and_sync_branches() -> None:
    """Test TensorFlow profiler memory retrieval and sync branches."""
    from ml_switcheroo_compiler.backends.tensorflow.profiler import TensorFlowProfiler

    _sync_tf_result(42)
    mock_tf_no_mem = mock.MagicMock()
    mock_tf_no_mem.config.list_physical_devices.return_value = ["GPU:0"]
    del mock_tf_no_mem.config.experimental.get_memory_info
    with mock.patch("ml_switcheroo_compiler.backends.tensorflow.profiler.tf", mock_tf_no_mem):
        mem_fallback = _get_tf_peak_memory_mb()
        assert mem_fallback > 0.0
    mock_tf = mock.MagicMock()
    mock_tf.config.list_physical_devices.return_value = ["GPU:0"]
    mock_tf.config.experimental.get_memory_info.return_value = {"peak": 1024 * 1024 * 5, "current": 1024 * 1024 * 2}
    with mock.patch("ml_switcheroo_compiler.backends.tensorflow.profiler.tf", mock_tf):
        mem = _get_tf_peak_memory_mb()
        assert mem >= 5.0
    profiler = TensorFlowProfiler()
    tensor_input = mock.MagicMock()
    tensor_input.shape = (2, 2)
    tensor_input.dtype = "float32"
    graph = LogicalGraph(name="test_tf_graph")
    graph.nodes["inp"] = LogicalNode(id="inp", op_type="Input")
    graph.inputs = ["inp"]
    prepared = profiler._prepare_inputs(graph, {"inp": tensor_input})
    assert len(prepared) == 1
    step_fn = profiler._build_step_fn(has_ops=True, compiled_fn=lambda x: x, tf_inputs=[tensor_input], device="CPU")
    with mock.patch("ml_switcheroo_compiler.backends.tensorflow.profiler._sync_tf_result"):
        step_fn()


def test_keras_profiler_branches() -> None:
    """Test Keras profiler sync and memory tracking branches."""
    from ml_switcheroo_compiler.backends.keras.profiler import _get_keras_peak_memory_mb

    mock_keras_no_ops = mock.MagicMock()
    del mock_keras_no_ops.ops
    with mock.patch("ml_switcheroo_compiler.backends.keras.profiler.keras", mock_keras_no_ops):
        _sync_keras_result(99)

    class MockTensor:
        """Mock tensor with numpy export."""

        def numpy(self) -> np.ndarray:
            """Return numpy array."""
            return np.array([1.0])

    _sync_keras_result(MockTensor())
    mock_tf_no_cfg = mock.MagicMock()
    del mock_tf_no_cfg.config
    with mock.patch.dict(sys.modules, {"tensorflow": mock_tf_no_cfg}):
        m = _get_keras_peak_memory_mb()
        assert m > 0.0
    profiler = KerasProfiler()
    mock_keras = mock.MagicMock()
    del mock_keras.ops
    with mock.patch("ml_switcheroo_compiler.backends.keras.profiler.keras", mock_keras):
        step_fn = profiler._build_step_fn(has_ops=False, compiled_model=None, keras_inputs=[])
        res = step_fn()
        assert res == 0.0
    mock_keras_with_ops = mock.MagicMock()
    mock_keras_with_ops.ops.zeros.return_value = np.zeros((1,))
    with mock.patch("ml_switcheroo_compiler.backends.keras.profiler.keras", mock_keras_with_ops):
        step_fn2 = profiler._build_step_fn(has_ops=False, compiled_model=None, keras_inputs=[])
        res2 = step_fn2()
        assert len(res2) == 1
    mock_keras_no_add = mock.MagicMock()
    del mock_keras_no_add.ops.add
    with mock.patch("ml_switcheroo_compiler.backends.keras.profiler.keras", mock_keras_no_add):
        step_fn3 = profiler._build_step_fn(has_ops=False, compiled_model=None, keras_inputs=[np.array([1.0])])
        res3 = step_fn3()
        assert res3 is not None
    step_fn4 = profiler._build_step_fn(has_ops=True, compiled_model=lambda x: x, keras_inputs=[np.array([1.0])])
    with mock.patch("ml_switcheroo_compiler.backends.keras.profiler._sync_keras_result"):
        step_fn4()


def test_edge_profiler_prepare_inputs_fallback_and_empty_step() -> None:
    """Test EdgeProfiler._prepare_inputs key mismatch and empty inputs step.

    Returns:
        None
    """
    profiler = EdgeProfiler()
    graph = IRGraph(name="edge_test")
    node = LogicalNode(id="in_expected", op_type="Input")
    graph.nodes = {"in_expected": node}
    res_inputs = profiler._prepare_inputs(graph, {"in_other": [1.0, 2.0]})
    assert res_inputs == [[1.0, 2.0]]
    res_empty = profiler.profile_graph(IRGraph(), {}, num_iters=2, warmup_iters=1)
    assert res_empty["latency_ms"] >= 0.0


def test_edge_profiler_memory_and_evaluation_branches() -> None:
    """Test EdgeProfiler memory calculation and evaluation exception/success branches.

    Returns:
        None
    """
    # Test darwin process memory calculation
    with mock.patch("sys.platform", "darwin"):
        assert _get_process_memory_mb() > 0.0

    # Test _calculate_edge_memory_bytes with multiple node configurations
    graph_empty = IRGraph(name="empty_graph")
    del graph_empty.nodes
    assert _calculate_edge_memory_bytes(graph_empty) == 65536

    graph_ops = IRGraph(name="edge_ops")
    graph_ops.nodes = {
        "n1": LogicalNode(id="n1", op_type="Add", shape_metadata=(200, 200)),
        "n2": LogicalNode(id="n2", op_type="Sub", shape_metadata=(-1, "invalid")),
        "n3": LogicalNode(id="n3", op_type="Mul", shape_metadata=None),
        "n4": LogicalNode(id="n4", op_type="Div", shape_metadata="not_a_tuple"),
    }
    # 200*200*4 = 160,000 bytes, aligned and padded to WASM page size (65536) -> 196,608
    mem_bytes = _calculate_edge_memory_bytes(graph_ops)
    assert mem_bytes == 196608

    profiler = EdgeProfiler()
    # Test execution when evaluate_graph succeeds
    with mock.patch("ml_switcheroo_compiler.backends.edge.profiler.evaluate_graph", return_value={"out": [1.0]}):
        res_succ = profiler.profile_graph(graph_ops, {"x": [1.0]}, num_iters=1, warmup_iters=1)
        assert res_succ["latency_ms"] >= 0.0
        assert res_succ["peak_memory_mb"] > 0.0

    # Test execution when evaluate_graph raises Exception
    with mock.patch("ml_switcheroo_compiler.backends.edge.profiler.evaluate_graph", side_effect=RuntimeError("eval error")):
        res_fail = profiler.profile_graph(graph_ops, {"x": [1.0]}, num_iters=1, warmup_iters=1)
        assert res_fail["latency_ms"] >= 0.0


def test_numba_profiler_branches() -> None:
    """Test NumbaProfiler _prepare_inputs, _compile_graph, and profile_graph branches.

    Returns:
        None
    """
    profiler = NumbaProfiler()
    graph = IRGraph(name="numba_test")
    node_in = LogicalNode(id="in_0", op_type="Input")
    node_op = LogicalNode(id="add_0", op_type="Add", inputs=["in_0", "in_0"])
    graph.nodes = {"in_0": node_in, "add_0": node_op}
    graph.inputs = ["in_0"]
    arr = np.array([1.0, 2.0], dtype=np.float32)
    inputs1 = {"in_0": arr}
    assert len(profiler._prepare_inputs(graph, inputs1)) == 1
    inputs2 = {"other_key": arr}
    assert len(profiler._prepare_inputs(graph, inputs2)) == 1
    mock_code = "def evaluate(inputs):\n    return inputs[0]\n"
    with patch.dict("sys.modules", {"numba": MagicMock()}):
        with patch("ml_switcheroo_compiler.backends.numba.generator.NumbaGenerator.generate", return_value=mock_code):
            fn = profiler._compile_graph(graph)
            assert callable(fn)
            res_compiled = profiler.profile_graph(graph, inputs1, num_iters=2, warmup_iters=1)
            assert len(res_compiled["latencies"]) == 2
        with patch("ml_switcheroo_compiler.backends.numba.generator.NumbaGenerator.generate", return_value="x = 1\n"):
            fn_invalid = profiler._compile_graph(graph)
            assert fn_invalid is None
        res_empty = profiler.profile_graph(IRGraph(), {}, num_iters=2, warmup_iters=1)
        assert len(res_empty["latencies"]) == 2


def test_sparse_profiler_branches() -> None:
    """Test SparseProfiler branches: COO input, ndarray input, mismatched keys, and compiled execution.

    Returns:
        None
    """
    profiler = SparseProfiler()
    graph = IRGraph(name="sparse_test")
    node_in1 = LogicalNode(id="in_0", op_type="Input")
    node_in2 = LogicalNode(id="in_1", op_type="Input")
    node_add = LogicalNode(id="add_0", op_type="Add", inputs=["in_0", "in_1"])
    graph.nodes = {"in_0": node_in1, "in_1": node_in2, "add_0": node_add}
    graph.inputs = ["in_0", "in_1"]
    arr1 = np.array([1.0, 2.0], dtype=np.float32)
    arr2 = np.array([3.0, 4.0], dtype=np.float32)

    class MockCOO:
        """Mock COO array."""

        def __init__(self, arr: np.ndarray) -> None:
            self.arr = arr

        def todense(self) -> np.ndarray:
            return self.arr

        @classmethod
        def from_numpy(cls, arr: np.ndarray) -> "MockCOO":
            return cls(arr)

    mock_sparse_module = MagicMock()
    mock_sparse_module.COO = MockCOO
    mock_sparse_module.GCXS = MockCOO
    with patch("ml_switcheroo_compiler.backends.sparse.profiler.sparse", mock_sparse_module):
        inputs1 = {"in_0": arr1, "in_1": MockCOO(arr2)}
        prep1 = profiler._prepare_inputs(graph, inputs1)
        assert len(prep1) == 2
        inputs2 = {"other_0": arr1, "other_1": arr2}
        prep2 = profiler._prepare_inputs(graph, inputs2)
        assert len(prep2) == 2
        mock_code = "def evaluate(inputs):\n    return inputs[0]\n"
        with patch.dict("sys.modules", {"sparse": mock_sparse_module}):
            with patch("ml_switcheroo_compiler.backends.sparse.generator.SparseGenerator.generate", return_value=mock_code):
                fn = profiler._compile_graph(graph)
                assert callable(fn)
                res_compiled = profiler.profile_graph(graph, inputs1, num_iters=2, warmup_iters=1)
                assert len(res_compiled["latencies"]) == 2
        with patch("ml_switcheroo_compiler.backends.sparse.generator.SparseGenerator.generate", return_value="x = 1\n"):
            fn_invalid = profiler._compile_graph(graph)
            assert fn_invalid is None
        res_empty = profiler.profile_graph(IRGraph(), {}, num_iters=2, warmup_iters=1)
        assert len(res_empty["latencies"]) == 2
        with patch.object(profiler, "_compile_graph", return_value=None):
            res_multi = profiler.profile_graph(graph, inputs1, num_iters=2, warmup_iters=1)
            assert len(res_multi["latencies"]) == 2

            class SimpleCOO(MockCOO):
                """COO class without __add__."""

            res_no_add = profiler.profile_graph(graph, {"in_0": SimpleCOO(arr1), "in_1": SimpleCOO(arr2)}, num_iters=1, warmup_iters=1)
            assert len(res_no_add["latencies"]) == 1


def test_dask_profiler_branches() -> None:
    """Test DaskProfiler branches: da.Array input, mismatched keys, _compile_graph, and compiled execution.

    Returns:
        None
    """
    import importlib

    import ml_switcheroo_compiler.backends.dask.profiler as dask_prof

    profiler = DaskProfiler()
    graph = IRGraph(name="dask_test")
    node_in1 = LogicalNode(id="in_0", op_type="Input")
    node_in2 = LogicalNode(id="in_1", op_type="Input")
    node_add = LogicalNode(id="add_0", op_type="Add", inputs=["in_0", "in_1"])
    graph.nodes = {"in_0": node_in1, "in_1": node_in2, "add_0": node_add}
    graph.inputs = ["in_0", "in_1"]
    arr = np.array([1.0, 2.0], dtype=np.float32)
    mock_compute_val = MagicMock()
    mock_compute_val.compute.return_value = arr
    inputs_mismatch = {"other_key": mock_compute_val}
    prep_mismatch = profiler._prepare_inputs(graph, inputs_mismatch)
    assert len(prep_mismatch) == 1
    mock_code = "def evaluate(inputs):\n    return inputs[0]\n"
    with patch.dict("sys.modules", {"dask": MagicMock(), "dask.array": MagicMock()}):
        with patch("ml_switcheroo_compiler.backends.dask.generator.DaskGenerator.generate", return_value=mock_code):
            fn = profiler._compile_graph(graph)
            assert callable(fn)
            res_comp = profiler.profile_graph(graph, {"in_0": arr, "in_1": arr}, num_iters=2, warmup_iters=1)
            assert len(res_comp["latencies"]) == 2
        with patch("ml_switcheroo_compiler.backends.dask.generator.DaskGenerator.generate", return_value="x = 1\n"):
            with pytest.raises(TypeError, match="Failed to generate valid evaluate function"):
                profiler._compile_graph(graph)
    mock_da = MagicMock()
    mock_da.compute.return_value = arr
    _sync_dask_result(mock_da)
    mock_da.compute.assert_called_once()
    assert isinstance(dask_prof._sync_dask_result({"a": mock_da}), dict)
    assert isinstance(dask_prof._sync_dask_result([mock_da, (mock_da,)]), list)
    mock_err_da = MagicMock()
    mock_err_da.compute.side_effect = RuntimeError("compute fail")
    _sync_dask_result(mock_err_da)
    res_empty = profiler.profile_graph(IRGraph(), {}, num_iters=2, warmup_iters=1)
    assert len(res_empty["latencies"]) == 2
    with patch.object(profiler, "_compile_graph", side_effect=RuntimeError("compile failed")):
        res_comp_err = profiler.profile_graph(graph, {"in_0": arr, "in_1": arr}, num_iters=2, warmup_iters=1)
        assert len(res_comp_err["latencies"]) == 2
    with patch.object(profiler, "_compile_graph", return_value=None):
        res_fb = profiler.profile_graph(graph, {"in_0": arr, "in_1": arr}, num_iters=2, warmup_iters=1)
        assert len(res_fb["latencies"]) == 2

    with patch.dict("sys.modules", {"dask.array": None}):
        importlib.reload(dask_prof)
        assert dask_prof.da is None
    importlib.reload(dask_prof)

    with patch.object(dask_prof.sys, "platform", "linux"):
        assert dask_prof._get_process_memory_mb() > 0.0

    mock_err_da = MagicMock()
    mock_err_da.compute.side_effect = RuntimeError("compute fail")
    assert dask_prof._sync_dask_result(mock_err_da) is mock_err_da

    class NonCallableCompute:
        compute = "not_callable"

    assert dask_prof._sync_dask_result(NonCallableCompute()) is not None


def test_keras_profiler_additional_branches() -> None:
    """Test KerasProfiler mismatched keys, _compile_graph, empty inputs, and memory/sync branches.

    Returns:
        None
    """
    profiler = KerasProfiler()
    graph = IRGraph(name="keras_test")
    node = LogicalNode(id="in_0", op_type="Input")
    graph.nodes = {"in_0": node}
    graph.inputs = ["in_0"]
    arr = np.array([1.0, 2.0], dtype=np.float32)
    prep = profiler._prepare_inputs(graph, {"other_key": arr})
    assert len(prep) == 1
    mock_code = "def get_model():\n    return lambda inputs: inputs[0]\n"
    with patch("ml_switcheroo_compiler.backends.keras.generator.KerasCodeGenerator.generate", return_value=mock_code):
        fn = profiler._compile_model(graph)
        assert callable(fn)
        res_comp = profiler.profile_graph(graph, {"in_0": arr}, num_iters=2, warmup_iters=1)
        assert len(res_comp["latencies"]) == 2
    with patch("ml_switcheroo_compiler.backends.keras.generator.KerasCodeGenerator.generate", return_value="x = 1\n"):
        with pytest.raises(TypeError, match="Failed to generate valid get_model function"):
            profiler._compile_model(graph)
    res_empty = profiler.profile_graph(IRGraph(), {}, num_iters=2, warmup_iters=1)
    assert len(res_empty["latencies"]) == 2
    with patch("ml_switcheroo_compiler.backends.keras.profiler.keras", None):
        assert _get_keras_peak_memory_mb() > 0.0
    mock_val = MagicMock()
    mock_val.numpy.side_effect = RuntimeError("numpy fail")
    with patch("ml_switcheroo_compiler.backends.keras.profiler.keras", None):
        _sync_keras_result(mock_val)

    import importlib

    import ml_switcheroo_compiler.backends.keras.profiler as k_prof

    with patch.dict("sys.modules", {"keras": None}):
        importlib.reload(k_prof)
        assert k_prof.keras is None
    importlib.reload(k_prof)

    with patch.object(k_prof.sys, "platform", "linux"):
        assert k_prof._get_process_memory_mb() > 0.0

    mock_torch_cuda = MagicMock()
    mock_torch_cuda.cuda.is_available.return_value = True
    mock_torch_cuda.cuda.max_memory_allocated.return_value = 1024 * 1024 * 32
    with patch.dict(sys.modules, {"torch": mock_torch_cuda}):
        with patch.object(k_prof, "keras", MagicMock(backend=MagicMock())):
            assert k_prof._get_keras_peak_memory_mb() == 32.0

    mock_torch_err = MagicMock()
    mock_torch_err.cuda.is_available.return_value = True
    mock_torch_err.cuda.max_memory_allocated.side_effect = RuntimeError("cuda fail")
    with patch.dict(sys.modules, {"torch": mock_torch_err}):
        with patch.object(k_prof, "keras", MagicMock(backend=MagicMock())):
            assert k_prof._get_keras_peak_memory_mb() > 0.0

    mock_tf_mod = MagicMock()
    mock_tf_mod.config.experimental.get_memory_info.return_value = {"peak": 0}
    with patch.dict(sys.modules, {"torch": None, "tensorflow": mock_tf_mod}):
        with patch.object(k_prof, "keras", MagicMock(backend=MagicMock())):
            assert k_prof._get_keras_peak_memory_mb() > 0.0

    step_fn_multi = profiler._build_step_fn(True, lambda inps: inps[0], [arr, arr])
    assert step_fn_multi() is not None

    with patch.object(profiler, "_compile_model", side_effect=RuntimeError("compile fail")):
        n_add = LogicalNode(id="add_0", op_type="Add", inputs=["in_0", "in_0"])
        graph_err = IRGraph()
        graph_err.nodes = {"in_0": node, "add_0": n_add}
        graph_err.inputs = ["in_0"]
        res_comp_err = profiler.profile_graph(graph_err, {"in_0": arr}, num_iters=2, warmup_iters=1)
        assert len(res_comp_err["latencies"]) == 2


def test_tensorflow_profiler_additional_branches() -> None:
    """Test TensorFlowProfiler mismatched keys, _compile_graph, empty inputs, and memory/sync branches.

    Returns:
        None
    """
    profiler = TensorFlowProfiler()
    graph = IRGraph(name="tf_test")
    node = LogicalNode(id="in_0", op_type="Input")
    graph.nodes = {"in_0": node}
    graph.inputs = ["in_0"]
    arr = np.array([1.0, 2.0], dtype=np.float32)
    prep = profiler._prepare_inputs(graph, {"other_key": arr})
    assert len(prep) == 1
    mock_code = "def apply_model(inputs):\n    return inputs[0]\n"
    with patch("ml_switcheroo_compiler.backends.tensorflow.generator.TensorFlowCodeGenerator.generate", return_value=mock_code):
        fn = profiler._compile_graph(graph)
        assert callable(fn)
        res_comp = profiler.profile_graph(graph, {"in_0": arr}, num_iters=2, warmup_iters=1)
        assert len(res_comp["latencies"]) == 2
    with patch("ml_switcheroo_compiler.backends.tensorflow.generator.TensorFlowCodeGenerator.generate", return_value="x = 1\n"):
        with pytest.raises(TypeError, match="Failed to generate valid apply_model function"):
            profiler._compile_graph(graph)
    res_empty = profiler.profile_graph(IRGraph(), {}, num_iters=2, warmup_iters=1)
    assert len(res_empty["latencies"]) == 2
    with patch("ml_switcheroo_compiler.backends.tensorflow.profiler.tf", None):
        assert _get_tf_peak_memory_mb() > 0.0
    mock_val = MagicMock()
    mock_val.numpy.side_effect = RuntimeError("numpy fail")
    with patch("ml_switcheroo_compiler.backends.tensorflow.profiler.tf", None):
        _sync_tf_result(mock_val)


def test_profilers_compilation_failure_fallback() -> None:
    """Test all profilers falling back gracefully when compilation fails.

    Returns:
        None
    """
    graph = IRGraph(name="test_fail")
    node_in = LogicalNode(id="in_0", op_type="Input")
    node_op = LogicalNode(id="add_0", op_type="Add", inputs=["in_0", "in_0"])
    graph.nodes = {"in_0": node_in, "add_0": node_op}
    graph.inputs = ["in_0"]
    inputs = {"in_0": [1.0, 2.0]}
    p_numba = NumbaProfiler()
    with patch.object(p_numba, "_compile_graph", side_effect=RuntimeError("comp fail")):
        res_numba = p_numba.profile_graph(graph, inputs, num_iters=1, warmup_iters=1)
        assert len(res_numba["latencies"]) == 1
    p_sparse = SparseProfiler()
    with patch.object(p_sparse, "_compile_graph", side_effect=RuntimeError("comp fail")):
        res_sparse = p_sparse.profile_graph(graph, inputs, num_iters=1, warmup_iters=1)
        assert len(res_sparse["latencies"]) == 1
    p_dask = DaskProfiler()
    with patch.object(p_dask, "_compile_graph", side_effect=RuntimeError("comp fail")):
        res_dask = p_dask.profile_graph(graph, inputs, num_iters=1, warmup_iters=1)
        assert len(res_dask["latencies"]) == 1
    p_keras = KerasProfiler()
    with patch.object(p_keras, "_compile_model", side_effect=RuntimeError("comp fail")):
        res_keras = p_keras.profile_graph(graph, inputs, num_iters=1, warmup_iters=1)
        assert len(res_keras["latencies"]) == 1
    p_tf = TensorFlowProfiler()
    with patch.object(p_tf, "_compile_graph", side_effect=RuntimeError("comp fail")):
        res_tf = p_tf.profile_graph(graph, inputs, num_iters=1, warmup_iters=1)
        assert len(res_tf["latencies"]) == 1


def test_backends_sparse_profiler_multi_input_accumulation() -> None:
    """Test SparseProfiler multi-input accumulation loop.

    Returns:
        None
    """
    p = SparseProfiler()
    graph = IRGraph(name="test_sparse_multi")
    graph.nodes = {"in_0": LogicalNode(id="in_0", op_type="Input"), "in_1": LogicalNode(id="in_1", op_type="Input")}
    graph.inputs = ["in_0", "in_1"]
    inputs = {"in_0": np.array([1.0, 2.0]), "in_1": np.array([3.0, 4.0])}
    with patch.object(p, "_compile_graph", return_value=None):
        res = p.profile_graph(graph, inputs, num_iters=1, warmup_iters=1)
        assert len(res["latencies"]) == 1


def test_backends_dask_and_tf_and_keras_profiler_edge_cases() -> None:
    """Test profiler edge cases for Dask, TensorFlow, and Keras.

    Returns:
        None
    """
    import importlib
    import sys

    import ml_switcheroo_compiler.backends.dask as dask_pkg
    import ml_switcheroo_compiler.backends.dask.profiler as dp
    import ml_switcheroo_compiler.backends.keras as keras_pkg
    import ml_switcheroo_compiler.backends.keras.profiler as kp
    import ml_switcheroo_compiler.backends.tensorflow as tf_pkg
    import ml_switcheroo_compiler.backends.tensorflow.profiler as tp

    sys.modules["ml_switcheroo_compiler.backends.dask"] = dask_pkg
    with patch.dict(sys.modules, {"dask": None, "dask.array": None}):
        importlib.reload(dp)
        assert dp.da is None
    sys.modules["ml_switcheroo_compiler.backends.dask"] = dask_pkg
    importlib.reload(dp)
    sys.modules["ml_switcheroo_compiler.backends.keras"] = keras_pkg
    with patch.dict(sys.modules, {"keras": None}):
        importlib.reload(kp)
        assert kp.keras is None
    sys.modules["ml_switcheroo_compiler.backends.keras"] = keras_pkg
    importlib.reload(kp)
    sys.modules["ml_switcheroo_compiler.backends.tensorflow"] = tf_pkg
    with patch.dict(sys.modules, {"tensorflow": None}):
        importlib.reload(tp)
        assert tp.tf is None
    sys.modules["ml_switcheroo_compiler.backends.tensorflow"] = tf_pkg
    importlib.reload(tp)
    mock_dask_lazy = MagicMock()
    mock_dask_lazy.compute.side_effect = RuntimeError("compute failed")
    assert dp._sync_dask_result(mock_dask_lazy) is mock_dask_lazy
    mock_dask_success = MagicMock()
    mock_dask_success.compute.return_value = 42
    assert dp._sync_dask_result(mock_dask_success) == 42
    assert dp._sync_dask_result([mock_dask_success]) == [42]
    assert dp._sync_dask_result((mock_dask_success,)) == (42,)
    assert dp._sync_dask_result({"res": mock_dask_success}) == {"res": 42}
    old_da = dp.da
    try:
        dp.da = None
        assert dp._sync_dask_result(mock_dask_success) is mock_dask_success
    finally:
        dp.da = old_da
    p_tf = TensorFlowProfiler()
    graph = IRGraph(name="test_tf_dev")
    graph.nodes = {"in_0": LogicalNode(id="in_0", op_type="Input"), "add_0": LogicalNode(id="add_0", op_type="Add", inputs=["in_0", "in_0"])}
    graph.inputs = ["in_0"]
    inputs = {"in_0": [1.0, 2.0]}
    with patch.object(p_tf, "_compile_graph", return_value=lambda x: x):
        res_tf = p_tf.profile_graph(graph, inputs, num_iters=1, warmup_iters=1, device="/cpu:0")
        assert len(res_tf["latencies"]) == 1
    with patch("sys.platform", "linux"):
        mem_linux = kp._get_process_memory_mb()
        assert mem_linux > 0.0
    mock_torch_err = MagicMock()
    mock_torch_err.cuda.is_available.side_effect = RuntimeError("torch cuda fail")
    with patch.dict("sys.modules", {"torch": mock_torch_err}):
        _ = _get_keras_peak_memory_mb()
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.max_memory_allocated.return_value = 10485760
    with patch.dict("sys.modules", {"torch": mock_torch}):
        mem = _get_keras_peak_memory_mb()
        assert mem > 0.0
    mock_torch_no_cuda = MagicMock()
    mock_torch_no_cuda.cuda.is_available.return_value = False
    mock_tf_err = MagicMock()
    mock_tf_err.config.experimental.get_memory_info.side_effect = RuntimeError("tf fail")
    with patch.dict("sys.modules", {"torch": mock_torch_no_cuda, "tensorflow": mock_tf_err}):
        _ = _get_keras_peak_memory_mb()
    mock_tf = MagicMock()
    mock_tf.config.experimental.get_memory_info.return_value = {"peak": 20971520}
    with patch.dict("sys.modules", {"torch": mock_torch_no_cuda, "tensorflow": mock_tf}):
        mem_tf = _get_keras_peak_memory_mb()
        assert mem_tf > 0.0
    _sync_keras_result({"a": [1.0, (2.0,)]})
    mock_result_err = MagicMock()
    mock_result_err.numpy.side_effect = RuntimeError("numpy fail")
    _sync_keras_result(mock_result_err)
    mock_k_no_ops = MagicMock(spec=["backend"])
    mock_success = MagicMock()
    mock_success.numpy.return_value = 1.0
    mock_fail = MagicMock()
    mock_fail.numpy.side_effect = RuntimeError("numpy fail")
    with patch.object(kp, "keras", mock_k_no_ops):
        kp._sync_keras_result(mock_success)
        kp._sync_keras_result(mock_fail)
    mock_k_ops = MagicMock()
    mock_k_ops.ops.convert_to_numpy.side_effect = [1.0, RuntimeError("ops fail")]
    with patch.object(kp, "keras", mock_k_ops):
        kp._sync_keras_result(mock_success)
        kp._sync_keras_result(mock_fail)
    p_k = KerasProfiler()
    g = IRGraph(name="test_inputs")
    g.nodes = {"in_0": LogicalNode(id="in_0", op_type="Input")}
    g.inputs = ["in_0"]
    r1 = p_k._prepare_inputs(g, {"in_0": [1.0]})
    assert len(r1) == 1
    r2 = p_k._prepare_inputs(g, {"mismatch": [1.0]})
    assert len(r2) == 1
    mock_k_zeros = MagicMock()
    mock_k_zeros.ops.zeros.return_value = 0.0
    with patch.object(kp, "keras", mock_k_zeros):
        fn1 = p_k._build_step_fn(False, None, [])
        assert fn1() == 0.0
    with patch.object(kp, "keras", mock_k_no_ops):
        fn2 = p_k._build_step_fn(False, None, [])
        assert fn2() == 0.0
    graph_single = IRGraph(name="test_single")
    graph_single.nodes = {"in_0": LogicalNode(id="in_0", op_type="Input"), "add_0": LogicalNode(id="add_0", op_type="Add", inputs=["in_0"])}
    graph_single.inputs = ["in_0"]
    with patch.object(p_k, "_compile_model", return_value=lambda x: x):
        res_single = p_k.profile_graph(graph_single, {"in_0": [1.0]}, num_iters=1, warmup_iters=1)
        assert len(res_single["latencies"]) == 1
    with patch.object(p_k, "_compile_model", side_effect=RuntimeError("compile fail")):
        res_comp_err = p_k.profile_graph(graph_single, {"in_0": [1.0]}, num_iters=1, warmup_iters=1)
        assert len(res_comp_err["latencies"]) == 1
    old_k = kp.keras
    try:
        kp.keras = None
        assert p_k._prepare_inputs(graph_single, {}) == []
        res_none = p_k.profile_graph(graph_single, {}, num_iters=1, warmup_iters=1)
        assert len(res_none["latencies"]) == 1
    finally:
        kp.keras = old_k
    graph_multi = IRGraph(name="test_keras_multi")
    graph_multi.nodes = {"in_0": LogicalNode(id="in_0", op_type="Input"), "in_1": LogicalNode(id="in_1", op_type="Input"), "add_0": LogicalNode(id="add_0", op_type="Add", inputs=["in_0", "in_1"])}
    graph_multi.inputs = ["in_0", "in_1"]
    inputs_multi = {"in_0": [1.0, 2.0], "in_1": [3.0, 4.0]}
    with patch.object(p_k, "_compile_model", return_value=lambda xs: xs[0]):
        res_multi = p_k.profile_graph(graph_multi, inputs_multi, num_iters=1, warmup_iters=1)
        assert len(res_multi["latencies"]) == 1
