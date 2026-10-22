"""Unit tests for backend profilers execution, latency measurement, and memory tracking."""

from __future__ import annotations

import sys
import typing
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from tests.testing_utils import as_callable, as_tensor, install_profiler_backend_stubs

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


@pytest.fixture(autouse=True)
def _ensure_profiler_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure optional profiler backend stubs are installed when packages are missing."""
    install_profiler_backend_stubs(monkeypatch)


def test_numpy_profiler_runtime() -> None:
    """Verify execution latency and non-darwin OS memory tracking in NumpyProfiler."""
    from ml_switcheroo_compiler.backends.numpy.profiler import NumpyProfiler

    profiler = NumpyProfiler()
    with patch("sys.platform", "linux"):
        res_linux = profiler.profile_graph(IRGraph(), {}, num_iters=1, warmup_iters=1)
        assert getattr(res_linux, "__getitem__")("peak_memory_mb") > 0.0
    res_empty = profiler.profile_graph(IRGraph(), {}, num_iters=1, warmup_iters=1)
    assert getattr(res_empty, "__getitem__")("latency_ms") >= 0.0
    assert as_callable(len)(res_empty["latencies"]) == 1
    inp = {"a": [1.0, 2.0], "b": np.array([3.0, 4.0], dtype=np.float32)}
    res_full = as_callable(profiler.profile_graph)(as_callable(IRGraph)(), inp, num_iters=2, warmup_iters=1)
    assert as_callable(len)(res_full["latencies"]) == 2


def test_jax_profiler_runtime() -> None:
    """Verify execution latency and non-darwin OS memory tracking in JAXProfiler."""
    try:
        from ml_switcheroo_compiler.backends.jax.profiler import JAXProfiler
    except (ValueError, ImportError, AttributeError) as e:
        if "RegisterCanCast" in str(e) and "ml_switcheroo_compiler.backends.jax.profiler" in sys.modules:
            JAXProfiler_reloaded = sys.modules["ml_switcheroo_compiler.backends.jax.profiler"].JAXProfiler
            assert JAXProfiler_reloaded is not None
        else:
            pytest.skip(f"JAX not available: {e}")
    profiler = JAXProfiler()
    with patch("sys.platform", "linux"):
        res_linux = profiler.profile_graph(IRGraph(), {}, num_iters=1, warmup_iters=1)
        assert getattr(res_linux, "__getitem__")("peak_memory_mb") > 0.0
    res_empty = profiler.profile_graph(IRGraph(), {}, num_iters=1, warmup_iters=1)
    assert getattr(res_empty, "__getitem__")("latency_ms") >= 0.0
    import jax.numpy as jnp

    arr = jnp.array([1.0, 2.0], dtype=jnp.float32)
    res_jax = profiler.profile_graph(IRGraph(), {"x": arr, "y": [3.0, 4.0]}, num_iters=2, warmup_iters=1)
    assert as_callable(len)(res_jax["latencies"]) == 2


def _test_pytorch_profiler_runtime_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_pytorch_profiler_runtime.

    Args:
        _state: Shared test state dictionary.
    """
    if "torch" in sys.modules and hasattr(sys.modules["torch"], "Tensor"):
        torch = sys.modules["torch"]
    else:
        try:
            pass
        except Exception as e:
            pytest.skip(f"PyTorch not available: {e}")
    from ml_switcheroo_compiler.backends.pytorch.profiler import PyTorchProfiler

    profiler = PyTorchProfiler()
    with patch("sys.platform", "linux"):
        res_linux = profiler.profile_graph(IRGraph(), {}, num_iters=1, warmup_iters=1)
        assert getattr(res_linux, "__getitem__")("peak_memory_mb") > 0.0
    with patch("torch.cuda.is_available", return_value=False):
        dev = profiler._resolve_device("cuda")
        assert dev.type == "cpu"
    getattr(_state, "update")({"PyTorchProfiler": PyTorchProfiler, "dev": dev, "profiler": profiler, "res_linux": res_linux})
    _state.update(locals())


def _test_pytorch_profiler_runtime_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_pytorch_profiler_runtime.

    Args:
        _state: Shared test state dictionary.
    """
    (profiler, torch) = (_state["profiler"], _state["torch"])
    with patch.object(torch.backends, "mps", create=True) as mock_mps:
        mock_mps.is_available.return_value = False
        dev = profiler._resolve_device("mps")
        assert dev.type == "cpu"
    t_in = torch.tensor([1.0, 2.0], dtype=torch.float32)
    res_tensor = profiler.profile_graph(IRGraph(), {"t": t_in, "list": [3.0, 4.0]}, num_iters=2, warmup_iters=1)
    assert len(res_tensor["latencies"]) == 2
    res_empty = profiler.profile_graph(IRGraph(), {}, num_iters=1, warmup_iters=1)
    assert res_empty["latency_ms"] >= 0.0
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.synchronize") as mock_cuda_sync:
            with patch("torch.cuda.max_memory_allocated", return_value=2097152):
                mock_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                profiler._sync(mock_dev)
                mock_cuda_sync.assert_called()
    with patch.object(torch, "mps", create=True) as mock_mps_module:
        mock_mps_module.is_available.return_value = True
        mock_mps_dev = MagicMock()
        mock_mps_dev.type = "mps"
        profiler._sync(mock_mps_dev)
        mock_mps_module.synchronize.assert_called_once()
    getattr(_state, "update")({"dev": dev, "mock_cuda_sync": mock_cuda_sync, "mock_dev": mock_dev, "mock_mps": mock_mps})
    getattr(_state, "update")({"mock_mps_dev": mock_mps_dev, "mock_mps_module": mock_mps_module, "res_empty": res_empty, "res_tensor": res_tensor})
    getattr(_state, "update")({"t_in": t_in})
    _state.update(locals())


def _test_pytorch_profiler_runtime_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_pytorch_profiler_runtime.

    Args:
        _state: Shared test state dictionary.
    """
    (profiler, torch) = (_state["profiler"], _state["torch"])
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
    from ml_switcheroo_compiler.ir.core import IRNode

    g_ops = IRGraph()
    g_ops.nodes["x"] = IRNode(id="x", op_type="Input", shape_metadata=(2,))
    g_ops.nodes["y"] = IRNode(id="y", op_type="Add", inputs=["x", "x"], shape_metadata=(2,))
    g_ops.outputs = ["y"]
    getattr(_state, "update")({"IRNode": IRNode, "g_ops": g_ops, "mock_cuda_target": mock_cuda_target, "mock_res_dev": mock_res_dev})
    getattr(_state, "update")({"res_cuda": res_cuda})
    _state.update(locals())


def _test_pytorch_profiler_runtime_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_pytorch_profiler_runtime.

    Args:
        _state: Shared test state dictionary.
    """
    (g_ops, profiler, torch) = (_state["g_ops"], _state["profiler"], _state["torch"])
    with patch("torch.compile", side_effect=RuntimeError("compile fail")):
        compiled_empty = profiler._compile_module(g_ops, torch.device("cpu"), input_list=[])
        assert compiled_empty is not None
        mock_traced = MagicMock()
        with patch.object(torch.jit, "trace", return_value=mock_traced):
            res_traced = profiler.profile_graph(g_ops, {"x": [1.0, 2.0]}, num_iters=1, warmup_iters=1)
            assert len(res_traced["latencies"]) == 1
    getattr(_state, "update")({"compiled_empty": compiled_empty, "mock_traced": mock_traced, "res_traced": res_traced})


def test_pytorch_profiler_runtime() -> None:
    """Verify execution latency and hardware synchronization paths in PyTorchProfiler."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_pytorch_profiler_runtime_part_1(_state)
    _test_pytorch_profiler_runtime_part_2(_state)
    _test_pytorch_profiler_runtime_part_3(_state)
    _test_pytorch_profiler_runtime_part_4(_state)
    assert len(_state) > 0


def _test_mlx_profiler_runtime_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_mlx_profiler_runtime.

    Args:
        _state: Shared test state dictionary.
    """
    from ml_switcheroo_compiler.backends.mlx.profiler import MLXProfiler, _get_mlx_peak_memory_mb, _prepare_mlx_inputs

    profiler = MLXProfiler()
    with patch("sys.platform", "linux"):
        res_linux = profiler.profile_graph(IRGraph(), {}, num_iters=1, warmup_iters=1)
        assert getattr(res_linux, "__getitem__")("peak_memory_mb") > 0.0
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", None):
        res_none = profiler.profile_graph(IRGraph(), {"a": [1.0]}, num_iters=2)
        assert res_none["latency_ms"] == 0.0
        assert _prepare_mlx_inputs({"a": [1.0]}) == []
    mock_mx = MagicMock()
    mock_mx.metal.get_peak_memory.return_value = 8388608
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_mx):
        res_metal = profiler.profile_graph(IRGraph(), {}, num_iters=1, warmup_iters=1)
        assert res_metal["peak_memory_mb"] == 8.0
    mock_mx_zero = MagicMock()
    mock_mx_zero.metal.get_peak_memory.return_value = 0.0
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_mx_zero):
        mem_fallback = _get_mlx_peak_memory_mb()
        assert mem_fallback > 0.0
    mock_no_metal = MagicMock(spec=[])
    getattr(_state, "update")({"MLXProfiler": MLXProfiler, "_get_mlx_peak_memory_mb": _get_mlx_peak_memory_mb, "_prepare_mlx_inputs": _prepare_mlx_inputs, "mem_fallback": mem_fallback})
    getattr(_state, "update")({"mock_mx": mock_mx, "mock_mx_zero": mock_mx_zero, "mock_no_metal": mock_no_metal, "profiler": profiler})
    getattr(_state, "update")({"res_linux": res_linux, "res_metal": res_metal, "res_none": res_none})
    _state.update(locals())


def _test_mlx_profiler_runtime_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_mlx_profiler_runtime.

    Args:
        _state: Shared test state dictionary.
    """
    (_get_mlx_peak_memory_mb, mock_no_metal, profiler) = (_state["_get_mlx_peak_memory_mb"], _state["mock_no_metal"], _state["profiler"])
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_no_metal):
        mem_no_metal = _get_mlx_peak_memory_mb()
        assert mem_no_metal > 0.0
    res_empty = profiler.profile_graph(IRGraph(), {}, num_iters=1, warmup_iters=1)
    assert res_empty["latency_ms"] >= 0.0
    import mlx.core as mx

    arr = mx.array([1.0, 2.0], dtype=mx.float32)
    res_mlx = profiler.profile_graph(IRGraph(), {"arr": arr, "list": [3.0, 4.0]}, num_iters=2, warmup_iters=1)
    assert len(res_mlx["latencies"]) == 2
    import importlib
    import sys

    import ml_switcheroo_compiler.backends.mlx as real_mlx_pkg
    import ml_switcheroo_compiler.backends.mlx.profiler as mlx_prof

    sys.modules["ml_switcheroo_compiler.backends.mlx"] = real_mlx_pkg
    with patch.dict(sys.modules, {"mlx": None, "mlx.core": None}):
        importlib.reload(mlx_prof)
        assert as_tensor(getattr(mlx_prof, "mx")) is None
    importlib.reload(mlx_prof)
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.backends.mlx.profiler import _prepare_mlx_inputs, _sync_mlx_result

    getattr(_state, "update")({"LogicalNode": LogicalNode, "_prepare_mlx_inputs": _prepare_mlx_inputs, "_sync_mlx_result": _sync_mlx_result, "importlib": importlib})
    getattr(_state, "update")({"mem_no_metal": mem_no_metal, "mlx_prof": mlx_prof, "real_mlx_pkg": real_mlx_pkg, "res_empty": res_empty})
    getattr(_state, "update")({"sys": sys})
    _state.update(locals())


def _test_mlx_profiler_runtime_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_mlx_profiler_runtime.

    Args:
        _state: Shared test state dictionary.
    """
    (LogicalNode, _prepare_mlx_inputs, _sync_mlx_result) = (_state["LogicalNode"], _state["_prepare_mlx_inputs"], _state["_sync_mlx_result"])
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", None):
        assert _sync_mlx_result("res") is None
    mock_no_eval = MagicMock(spec=[])
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_no_eval):
        assert _sync_mlx_result("res") is None
    mock_eval = MagicMock()
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_eval):
        _sync_mlx_result({"k1": 1, "k2": 2})
        assert mock_eval.eval.call_count == 2

    class MockMlxArray:
        """Test class MockMlxArray."""

        __module__ = "mlx.core"

    MockMlxArray.__name__ = "array"
    m_arr = MockMlxArray()
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", MagicMock()):
        assert as_callable(_prepare_mlx_inputs)({"custom": m_arr}) == [m_arr]
    g_ops = IRGraph()
    g_ops.nodes["relu"] = IRNode(id="relu", op_type="Relu", inputs=[])
    mock_mlx_ops = MagicMock()
    mock_mlx_ops.compile.side_effect = lambda fn: fn
    getattr(_state, "update")({"MockMlxArray": MockMlxArray, "g_ops": g_ops, "m_arr": m_arr, "mock_eval": mock_eval})
    getattr(_state, "update")({"mock_mlx_ops": mock_mlx_ops, "mock_no_eval": mock_no_eval})
    _state.update(locals())


def _test_mlx_profiler_runtime_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_mlx_profiler_runtime.

    Args:
        _state: Shared test state dictionary.
    """
    (g_ops, mock_mlx_ops, profiler) = (_state["g_ops"], _state["mock_mlx_ops"], _state["profiler"])
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
    getattr(_state, "update")({"mock_mlx_fail_compile": mock_mlx_fail_compile, "mock_mlx_no_compile": mock_mlx_no_compile, "res_fail": res_fail, "res_ops": res_ops})
    _state.update(locals())


def _test_mlx_profiler_runtime_part_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 5 of test_mlx_profiler_runtime.

    Args:
        _state: Shared test state dictionary.
    """
    (g_ops, mock_mlx_no_compile, profiler) = (_state["g_ops"], _state["mock_mlx_no_compile"], _state["profiler"])
    with patch("ml_switcheroo_compiler.backends.mlx.profiler.mx", mock_mlx_no_compile):
        with patch("ml_switcheroo_compiler.interpreter.evaluator.evaluate_graph", return_value={"relu": [1.0]}):
            res_no_comp = profiler.profile_graph(g_ops, {"in1": [1.0]}, num_iters=1, warmup_iters=1)
            assert res_no_comp["latency_ms"] >= 0.0
    getattr(_state, "update")({"res_no_comp": res_no_comp})


def test_mlx_profiler_runtime() -> None:
    """Verify execution latency and Apple Silicon Metal peak memory metrics in MLXProfiler."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_mlx_profiler_runtime_part_1(_state)
    _test_mlx_profiler_runtime_part_2(_state)
    _test_mlx_profiler_runtime_part_3(_state)
    _test_mlx_profiler_runtime_part_4(_state)
    _test_mlx_profiler_runtime_part_5(_state)
    assert len(_state) > 0


def _test_cupy_profiler_runtime_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_cupy_profiler_runtime.

    Args:
        _state: Shared test state dictionary.
    """
    from ml_switcheroo_compiler.backends.cupy.profiler import CupyProfiler

    profiler = CupyProfiler()
    with patch("sys.platform", "linux"):
        res_linux = profiler.profile_graph(IRGraph(), {}, num_iters=1, warmup_iters=1)
        assert getattr(res_linux, "__getitem__")("peak_memory_mb") > 0.0
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", None):
        res_none = profiler.profile_graph(IRGraph(), {"a": [1.0]}, num_iters=2)
        assert res_none["latency_ms"] == 0.0
        assert as_callable(len)(res_none["latencies"]) == 2
    mock_cp = MagicMock()
    dummy_arr = MagicMock()
    mock_cp.ndarray = type("DummyArray", (), {})
    mock_cp.asarray.return_value = dummy_arr
    mock_cp.zeros_like.return_value = dummy_arr
    mock_cp.add.return_value = dummy_arr
    mock_cp.zeros.return_value = dummy_arr
    mock_cp.cuda.get_elapsed_time.return_value = 1.5
    mock_cp.get_default_memory_pool().used_bytes.return_value = 5242880
    getattr(_state, "update")({"CupyProfiler": CupyProfiler, "dummy_arr": dummy_arr, "mock_cp": mock_cp, "profiler": profiler})
    getattr(_state, "update")({"res_linux": res_linux, "res_none": res_none})
    _state.update(locals())


def _test_cupy_profiler_runtime_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_cupy_profiler_runtime.

    Args:
        _state: Shared test state dictionary.
    """
    (mock_cp, profiler) = (_state["mock_cp"], _state["profiler"])
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mock_cp):
        res_empty = profiler.profile_graph(IRGraph(), {}, num_iters=1, warmup_iters=1)
        assert res_empty["peak_memory_mb"] == 5.0
        inst = mock_cp.ndarray()
        res_exec = profiler.profile_graph(IRGraph(), {"arr": inst, "lst": [1.0, 2.0]}, num_iters=2, warmup_iters=1)
        assert res_exec["peak_memory_mb"] == 5.0
        assert len(res_exec["latencies"]) == 2
        mock_cp.cuda.Device().synchronize.assert_called()
        from ml_switcheroo_compiler.ir.core import IRNode

        graph_with_ops = IRGraph()
        graph_with_ops.nodes["relu1"] = IRNode("Relu", "relu1", inputs=["arr"])
        with patch("ml_switcheroo_compiler.interpreter.evaluator.evaluate_graph", return_value={"relu1": inst}) as mock_eval:
            res_ops = profiler.profile_graph(graph_with_ops, {"arr": inst}, num_iters=1, warmup_iters=1)
            assert mock_eval.called
            assert res_ops["peak_memory_mb"] == 5.0
        setattr(mock_cp.cuda.Event, "side_effect", RuntimeError("No CUDA"))
        res_fallback = profiler.profile_graph(IRGraph(), {}, num_iters=1)
        assert len(res_fallback["latencies"]) == 1
        setattr(mock_cp.cuda.Event, "side_effect", None)
        setattr(mock_cp, "cuda", None)
        res_no_events = profiler.profile_graph(IRGraph(), {}, num_iters=1)
        assert len(res_no_events["latencies"]) == 1
    getattr(_state, "update")({"IRNode": IRNode, "graph_with_ops": graph_with_ops, "inst": inst, "mock_eval": mock_eval})
    getattr(_state, "update")({"res_empty": res_empty, "res_exec": res_exec, "res_fallback": res_fallback, "res_no_events": res_no_events})
    getattr(_state, "update")({"res_ops": res_ops})
    _state.update(locals())


def _test_cupy_profiler_runtime_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_cupy_profiler_runtime.

    Args:
        _state: Shared test state dictionary.
    """
    import importlib
    import sys

    import ml_switcheroo_compiler.backends.cupy.profiler as cupy_prof

    with patch.dict(sys.modules, {"cupy": None}):
        importlib.reload(cupy_prof)
        assert as_tensor(getattr(cupy_prof, "cp")) is None
    importlib.reload(cupy_prof)
    from ml_switcheroo_compiler.backends.cupy.profiler import _get_cupy_peak_memory_mb

    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", None):
        assert _get_cupy_peak_memory_mb() > 0.0
    mock_cp_mem = MagicMock()
    mock_cp_mem.cuda.Device().mem_info = (1024 * 1024 * 10, 1024 * 1024 * 30)
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mock_cp_mem):
        assert _get_cupy_peak_memory_mb() == 20.0
    mock_cp_str_mem = MagicMock()
    mock_cp_str_mem.cuda.Device().mem_info = ("str1", "str2")
    mock_cp_str_mem.get_default_memory_pool().used_bytes.return_value = 1024 * 1024 * 4
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mock_cp_str_mem):
        assert _get_cupy_peak_memory_mb() == 4.0
    mock_cp_mem_err = MagicMock()
    mock_dev_err = MagicMock()
    mock_dev_err.mem_info = property(lambda self: (_ for _ in ()).throw(RuntimeError("err")))
    getattr(_state, "update")({"_get_cupy_peak_memory_mb": _get_cupy_peak_memory_mb, "cupy_prof": cupy_prof, "importlib": importlib, "mock_cp_mem": mock_cp_mem})
    getattr(_state, "update")({"mock_cp_mem_err": mock_cp_mem_err, "mock_cp_str_mem": mock_cp_str_mem, "mock_dev_err": mock_dev_err, "sys": sys})
    _state.update(locals())


def _test_cupy_profiler_runtime_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_cupy_profiler_runtime.

    Args:
        _state: Shared test state dictionary.
    """
    (_get_cupy_peak_memory_mb, mock_cp_mem_err, profiler) = (_state["_get_cupy_peak_memory_mb"], _state["mock_cp_mem_err"], _state["profiler"])
    setattr(mock_cp_mem_err.cuda.Device, "side_effect", RuntimeError("device error"))
    setattr(mock_cp_mem_err.get_default_memory_pool().used_bytes, "return_value", 1024 * 1024 * 5)
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mock_cp_mem_err):
        assert _get_cupy_peak_memory_mb() == 5.0
    mock_cp_pool_err = MagicMock(spec=["get_default_memory_pool"])
    mock_cp_pool_err.get_default_memory_pool().used_bytes.side_effect = RuntimeError("pool err")
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mock_cp_pool_err):
        assert _get_cupy_peak_memory_mb() > 0.0
    mock_cp_no_pool = MagicMock(spec=[])
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mock_cp_no_pool):
        assert _get_cupy_peak_memory_mb() > 0.0
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", None):
        assert profiler._sync() is None
    mock_cp_sync = MagicMock()
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mock_cp_sync):
        profiler._sync()
        mock_cp_sync.cuda.Stream.null.synchronize.assert_called_once()
    mock_cp_sync_err = MagicMock()
    mock_cp_sync_err.cuda.Stream.null.synchronize.side_effect = RuntimeError("stream fail")
    mock_cp_sync_err.cuda.Device().synchronize.side_effect = RuntimeError("device fail")
    getattr(_state, "update")({"mock_cp_no_pool": mock_cp_no_pool, "mock_cp_pool_err": mock_cp_pool_err, "mock_cp_sync": mock_cp_sync, "mock_cp_sync_err": mock_cp_sync_err})
    _state.update(locals())


def _test_cupy_profiler_runtime_part_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 5 of test_cupy_profiler_runtime.

    Args:
        _state: Shared test state dictionary.
    """
    (mock_cp_sync_err, profiler) = (_state["mock_cp_sync_err"], _state["profiler"])
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mock_cp_sync_err):
        profiler._sync()
    mock_cp_bare = MagicMock(spec=["cuda"])
    mock_cp_bare.cuda = MagicMock(spec=[])
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mock_cp_bare):
        profiler._sync()
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
    setattr(mod_cuda_unavail, "cuda", CustomMod())
    setattr(as_tensor(getattr(mod_cuda_unavail, "cuda")), "is_available", lambda: False)
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mod_cuda_unavail):
        assert _is_cupy_available() is False
    mod_cuda_avail = CustomMod()
    setattr(mod_cuda_avail, "cuda", CustomMod())
    getattr(_state, "update")({"CustomMod": CustomMod, "_is_cupy_available": _is_cupy_available, "mock_cp_bare": mock_cp_bare, "mod_cuda_avail": mod_cuda_avail})
    getattr(_state, "update")({"mod_cuda_unavail": mod_cuda_unavail, "mod_no_cuda": mod_no_cuda})
    _state.update(locals())


def _test_cupy_profiler_runtime_part_6(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 6 of test_cupy_profiler_runtime.

    Args:
        _state: Shared test state dictionary.
    """
    (CustomMod, _is_cupy_available, mod_cuda_avail) = (_state["CustomMod"], _state["_is_cupy_available"], _state["mod_cuda_avail"])
    setattr(mod_cuda_avail.cuda, "is_available", lambda: True)
    setattr(mod_cuda_avail.cuda, "runtime", CustomMod())
    setattr(mod_cuda_avail.cuda.runtime, "getDeviceCount", lambda: 1)
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mod_cuda_avail):
        assert bool(_is_cupy_available()) is True
    mod_cuda_no_runtime = CustomMod()
    setattr(mod_cuda_no_runtime, "cuda", CustomMod())
    setattr(mod_cuda_no_runtime.cuda, "is_available", lambda: True)
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mod_cuda_no_runtime):
        assert bool(_is_cupy_available()) is True
    mod_cuda_zero = CustomMod()
    setattr(mod_cuda_zero, "cuda", CustomMod())
    setattr(mod_cuda_zero.cuda, "is_available", lambda: True)
    setattr(mod_cuda_zero.cuda, "runtime", as_callable(CustomMod)())
    setattr(getattr(mod_cuda_zero.cuda, "runtime"), "getDeviceCount", lambda: 0)
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mod_cuda_zero):
        assert bool(_is_cupy_available()) is False
    mod_cuda_err = CustomMod()
    setattr(mod_cuda_err, "cuda", CustomMod())

    def _raise_cuda_error() -> int:
        """Test function _raise_cuda_error."""
        raise RuntimeError("boom")

    getattr(_state, "update")({"_raise_cuda_error": _raise_cuda_error, "mod_cuda_err": mod_cuda_err, "mod_cuda_no_runtime": mod_cuda_no_runtime, "mod_cuda_zero": mod_cuda_zero})
    _state.update(locals())


def _test_cupy_profiler_runtime_part_7(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 7 of test_cupy_profiler_runtime.

    Args:
        _state: Shared test state dictionary.
    """
    (CustomMod, _is_cupy_available, _raise_cuda_error, mod_cuda_err) = (_state["CustomMod"], _state["_is_cupy_available"], _state["_raise_cuda_error"], _state["mod_cuda_err"])
    setattr(mod_cuda_err.cuda, "runtime", CustomMod())
    setattr(mod_cuda_err.cuda.runtime, "getDeviceCount", _raise_cuda_error)
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mod_cuda_err):
        assert bool(_is_cupy_available()) is False


def test_cupy_profiler_runtime() -> None:
    """Verify execution simulation and memory peak tracking in CupyProfiler."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_cupy_profiler_runtime_part_1(_state)
    _test_cupy_profiler_runtime_part_2(_state)
    _test_cupy_profiler_runtime_part_3(_state)
    _test_cupy_profiler_runtime_part_4(_state)
    _test_cupy_profiler_runtime_part_5(_state)
    _test_cupy_profiler_runtime_part_6(_state)
    _test_cupy_profiler_runtime_part_7(_state)
    assert len(_state) > 0
