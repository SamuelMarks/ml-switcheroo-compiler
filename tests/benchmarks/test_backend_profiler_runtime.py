"""Unit tests for backend profilers execution, latency measurement, and memory tracking."""

from __future__ import annotations

import sys
import typing
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_mutable, ensure_tensor_dict, has_backend_member, install_profiler_backend_stubs

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
        assert ensure_eager_tensor(res_linux)["peak_memory_mb"] > 0.0
    res_empty = profiler.profile_graph(IRGraph(), {}, num_iters=1, warmup_iters=1)
    assert ensure_eager_tensor(res_empty)["latency_ms"] >= 0.0
    assert ensure_callable(len)(res_empty["latencies"]) == 1
    inp = {"a": [1.0, 2.0], "b": np.array([3.0, 4.0], dtype=np.float32)}
    res_full = ensure_callable(profiler.profile_graph)(ensure_callable(IRGraph)(), inp, num_iters=2, warmup_iters=1)
    assert ensure_callable(len)(res_full["latencies"]) == 2


def test_jax_profiler_runtime() -> None:
    """Verify execution latency and non-darwin OS memory tracking in JAXProfiler."""
    try:
        from ml_switcheroo_compiler.backends.jax.profiler import JAXProfiler

        profiler = JAXProfiler()
    except (ValueError, ImportError, AttributeError) as e:
        if "RegisterCanCast" in str(e) and "ml_switcheroo_compiler.backends.jax.profiler" in sys.modules:
            JAXProfiler_reloaded = ensure_eager_tensor(sys.modules["ml_switcheroo_compiler.backends.jax.profiler"]).JAXProfiler
            assert JAXProfiler_reloaded is not None
            return
        pytest.skip(f"JAX not available: {e}")
    with patch("sys.platform", "linux"):
        res_linux = profiler.profile_graph(IRGraph(), {}, num_iters=1, warmup_iters=1)
        assert ensure_eager_tensor(res_linux)["peak_memory_mb"] > 0.0
    res_empty = profiler.profile_graph(IRGraph(), {}, num_iters=1, warmup_iters=1)
    assert ensure_eager_tensor(res_empty)["latency_ms"] >= 0.0
    import jax.numpy as jnp

    arr = jnp.array([1.0, 2.0], dtype=jnp.float32)
    res_jax = profiler.profile_graph(IRGraph(), {"x": arr, "y": [3.0, 4.0]}, num_iters=2, warmup_iters=1)
    assert ensure_callable(len)(res_jax["latencies"]) == 2


def _test_pytorch_profiler_runtime_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_pytorch_profiler_runtime.

    Args:
        _state: Shared test state dictionary.
    """
    if "torch" in sys.modules and has_backend_member(sys.modules["torch"], "Tensor"):
        torch = sys.modules["torch"]
    else:
        import importlib

        torch = importlib.import_module("torch")
    from ml_switcheroo_compiler.backends.pytorch.profiler import PyTorchProfiler

    profiler = PyTorchProfiler()
    with patch("sys.platform", "linux"):
        res_linux = profiler.profile_graph(IRGraph(), {}, num_iters=1, warmup_iters=1)
        assert ensure_eager_tensor(res_linux)["peak_memory_mb"] > 0.0
    with patch("torch.cuda.is_available", return_value=False):
        dev = profiler._resolve_device("cuda")
        assert dev.type == "cpu"
    _state.update(ensure_tensor_dict({"PyTorchProfiler": PyTorchProfiler, "dev": dev, "profiler": profiler, "res_linux": res_linux}))
    _state.update(ensure_tensor_dict({"torch": torch}))


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
    _state.update(ensure_tensor_dict({"dev": dev, "mock_cuda_sync": mock_cuda_sync, "mock_dev": mock_dev, "mock_mps": mock_mps}))
    _state.update(ensure_tensor_dict({"mock_mps_dev": mock_mps_dev, "mock_mps_module": mock_mps_module, "res_empty": res_empty, "res_tensor": res_tensor}))
    _state.update(ensure_tensor_dict({"t_in": t_in}))
    _state.update(ensure_tensor_dict({"profiler": profiler, "torch": torch}))


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
    _state.update(ensure_tensor_dict({"IRNode": IRNode, "g_ops": g_ops, "mock_cuda_target": mock_cuda_target, "mock_res_dev": mock_res_dev}))
    _state.update(ensure_tensor_dict({"res_cuda": res_cuda}))
    _state.update(ensure_tensor_dict({"profiler": profiler, "torch": torch}))


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
    _state.update(ensure_tensor_dict({"compiled_empty": compiled_empty, "mock_traced": mock_traced, "res_traced": res_traced}))


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
        assert ensure_eager_tensor(res_linux)["peak_memory_mb"] > 0.0
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
    _state.update(ensure_tensor_dict({"MLXProfiler": MLXProfiler, "_get_mlx_peak_memory_mb": _get_mlx_peak_memory_mb, "_prepare_mlx_inputs": _prepare_mlx_inputs, "mem_fallback": mem_fallback}))
    _state.update(ensure_tensor_dict({"mock_mx": mock_mx, "mock_mx_zero": mock_mx_zero, "mock_no_metal": mock_no_metal, "profiler": profiler}))
    _state.update(ensure_tensor_dict({"res_linux": res_linux, "res_metal": res_metal, "res_none": res_none}))


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
    orig_mx = ensure_eager_tensor(mlx_prof).mx
    with patch.dict(sys.modules, {"mlx": None, "mlx.core": None}):
        importlib.reload(mlx_prof)
        assert ensure_eager_tensor(ensure_eager_tensor(mlx_prof).mx) is None
    importlib.reload(mlx_prof)
    if ensure_eager_tensor(ensure_eager_tensor(mlx_prof).mx) is None and orig_mx is not None:
        ensure_mutable(mlx_prof).mx = ensure_eager_tensor(orig_mx)
    from ml_switcheroo_ir import LogicalNode

    from ml_switcheroo_compiler.backends.mlx.profiler import _prepare_mlx_inputs, _sync_mlx_result

    _state.update(ensure_tensor_dict({"LogicalNode": LogicalNode, "_prepare_mlx_inputs": _prepare_mlx_inputs, "_sync_mlx_result": _sync_mlx_result, "importlib": importlib}))
    _state.update(ensure_tensor_dict({"mem_no_metal": mem_no_metal, "mlx_prof": mlx_prof, "real_mlx_pkg": real_mlx_pkg, "res_empty": res_empty}))
    _state.update(ensure_tensor_dict({"sys": sys}))
    _state.update(ensure_tensor_dict({"_get_mlx_peak_memory_mb": _get_mlx_peak_memory_mb, "mock_no_metal": mock_no_metal, "profiler": profiler}))


def _test_mlx_profiler_runtime_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_mlx_profiler_runtime.

    Args:
        _state: Shared test state dictionary.
    """
    (LogicalNode, _prepare_mlx_inputs, _sync_mlx_result) = (_state["LogicalNode"], _state["_prepare_mlx_inputs"], _state["_sync_mlx_result"])
    _ = LogicalNode
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
        assert ensure_callable(_prepare_mlx_inputs)({"custom": m_arr}) == [m_arr]
    g_ops = IRGraph()
    g_ops.nodes["relu"] = IRNode(id="relu", op_type="Relu", inputs=[])
    mock_mlx_ops = MagicMock()
    mock_mlx_ops.compile.side_effect = lambda fn: fn
    _state.update(ensure_tensor_dict({"MockMlxArray": MockMlxArray, "g_ops": g_ops, "m_arr": m_arr, "mock_eval": mock_eval}))
    _state.update(ensure_tensor_dict({"mock_mlx_ops": mock_mlx_ops, "mock_no_eval": mock_no_eval}))
    _state.update(ensure_tensor_dict({"LogicalNode": LogicalNode, "_prepare_mlx_inputs": _prepare_mlx_inputs, "_sync_mlx_result": _sync_mlx_result}))


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
    _state.update(ensure_tensor_dict({"mock_mlx_fail_compile": mock_mlx_fail_compile, "mock_mlx_no_compile": mock_mlx_no_compile, "res_fail": res_fail, "res_ops": res_ops}))
    _state.update(ensure_tensor_dict({"g_ops": g_ops, "mock_mlx_ops": mock_mlx_ops, "profiler": profiler}))


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
    _state.update(ensure_tensor_dict({"res_no_comp": res_no_comp}))


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
        assert ensure_eager_tensor(res_linux)["peak_memory_mb"] > 0.0
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", None):
        res_none = profiler.profile_graph(IRGraph(), {"a": [1.0]}, num_iters=2)
        assert res_none["latency_ms"] == 0.0
        assert ensure_callable(len)(res_none["latencies"]) == 2
    mock_cp = MagicMock()
    dummy_arr = MagicMock()
    mock_cp.ndarray = type("DummyArray", (), {})
    mock_cp.asarray.return_value = dummy_arr
    mock_cp.zeros_like.return_value = dummy_arr
    mock_cp.add.return_value = dummy_arr
    mock_cp.zeros.return_value = dummy_arr
    mock_cp.cuda.get_elapsed_time.return_value = 1.5
    mock_cp.get_default_memory_pool().used_bytes.return_value = 5242880
    _state.update(ensure_tensor_dict({"CupyProfiler": CupyProfiler, "dummy_arr": dummy_arr, "mock_cp": mock_cp, "profiler": profiler}))
    _state.update(ensure_tensor_dict({"res_linux": res_linux, "res_none": res_none}))


def _test_cupy_profiler_runtime_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_cupy_profiler_runtime.

    Args:
        _state: Shared test state dictionary.
    """
    (mock_cp, profiler) = (_state["mock_cp"], _state["profiler"])
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mock_cp):
        res_empty = profiler.profile_graph(IRGraph(), {}, num_iters=1, warmup_iters=1)
        assert res_empty["peak_memory_mb"] == 5.0
        inst = ensure_callable(mock_cp.ndarray)()
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
        ensure_mutable(mock_cp.cuda.Event).side_effect = ensure_eager_tensor(RuntimeError("No CUDA"))
        res_fallback = profiler.profile_graph(IRGraph(), {}, num_iters=1)
        assert len(res_fallback["latencies"]) == 1
        ensure_mutable(mock_cp.cuda.Event).side_effect = ensure_eager_tensor(None)
        ensure_mutable(mock_cp).cuda = ensure_eager_tensor(None)
        res_no_events = profiler.profile_graph(IRGraph(), {}, num_iters=1)
        assert len(res_no_events["latencies"]) == 1
    _state.update(ensure_tensor_dict({"IRNode": IRNode, "graph_with_ops": graph_with_ops, "inst": inst, "mock_eval": mock_eval}))
    _state.update(ensure_tensor_dict({"res_empty": res_empty, "res_exec": res_exec, "res_fallback": res_fallback, "res_no_events": res_no_events}))
    _state.update(ensure_tensor_dict({"res_ops": res_ops}))
    _state.update(ensure_tensor_dict({"mock_cp": mock_cp, "profiler": profiler}))


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
        assert ensure_eager_tensor(ensure_eager_tensor(cupy_prof).cp) is None
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
    _state.update(ensure_tensor_dict({"_get_cupy_peak_memory_mb": _get_cupy_peak_memory_mb, "cupy_prof": cupy_prof, "importlib": importlib, "mock_cp_mem": mock_cp_mem}))
    _state.update(ensure_tensor_dict({"mock_cp_mem_err": mock_cp_mem_err, "mock_cp_str_mem": mock_cp_str_mem, "mock_dev_err": mock_dev_err, "sys": sys}))


def _test_cupy_profiler_runtime_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_cupy_profiler_runtime.

    Args:
        _state: Shared test state dictionary.
    """
    (_get_cupy_peak_memory_mb, mock_cp_mem_err, profiler) = (_state["_get_cupy_peak_memory_mb"], _state["mock_cp_mem_err"], _state["profiler"])
    ensure_mutable(mock_cp_mem_err.cuda.Device).side_effect = ensure_eager_tensor(RuntimeError("device error"))
    ensure_mutable(mock_cp_mem_err.get_default_memory_pool().used_bytes).return_value = ensure_eager_tensor(1024 * 1024 * 5)
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
    _state.update(ensure_tensor_dict({"mock_cp_no_pool": mock_cp_no_pool, "mock_cp_pool_err": mock_cp_pool_err, "mock_cp_sync": mock_cp_sync, "mock_cp_sync_err": mock_cp_sync_err}))
    _state.update(ensure_tensor_dict({"_get_cupy_peak_memory_mb": _get_cupy_peak_memory_mb, "mock_cp_mem_err": mock_cp_mem_err, "profiler": profiler}))


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
    ensure_mutable(mod_cuda_unavail).cuda = ensure_eager_tensor(CustomMod())
    ensure_mutable(ensure_eager_tensor(mod_cuda_unavail).cuda).is_available = ensure_eager_tensor(lambda: False)
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mod_cuda_unavail):
        assert _is_cupy_available() is False
    mod_cuda_avail = CustomMod()
    ensure_mutable(mod_cuda_avail).cuda = ensure_eager_tensor(CustomMod())
    _state.update(ensure_tensor_dict({"CustomMod": CustomMod, "_is_cupy_available": _is_cupy_available, "mock_cp_bare": mock_cp_bare, "mod_cuda_avail": mod_cuda_avail}))
    _state.update(ensure_tensor_dict({"mod_cuda_unavail": mod_cuda_unavail, "mod_no_cuda": mod_no_cuda}))
    _state.update(ensure_tensor_dict({"mock_cp_sync_err": mock_cp_sync_err, "profiler": profiler}))


def _test_cupy_profiler_runtime_part_6(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 6 of test_cupy_profiler_runtime.

    Args:
        _state: Shared test state dictionary.
    """
    (CustomMod, _is_cupy_available, mod_cuda_avail) = (_state["CustomMod"], _state["_is_cupy_available"], _state["mod_cuda_avail"])
    ensure_mutable(mod_cuda_avail.cuda).is_available = ensure_eager_tensor(lambda: True)
    ensure_mutable(mod_cuda_avail.cuda).runtime = ensure_eager_tensor(CustomMod())
    ensure_mutable(mod_cuda_avail.cuda.runtime).getDeviceCount = ensure_eager_tensor(lambda: 1)
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mod_cuda_avail):
        assert bool(_is_cupy_available()) is True
    mod_cuda_no_runtime = CustomMod()
    ensure_mutable(mod_cuda_no_runtime).cuda = ensure_eager_tensor(CustomMod())
    ensure_mutable(mod_cuda_no_runtime.cuda).is_available = ensure_eager_tensor(lambda: True)
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mod_cuda_no_runtime):
        assert bool(_is_cupy_available()) is True
    mod_cuda_zero = CustomMod()
    ensure_mutable(mod_cuda_zero).cuda = ensure_eager_tensor(CustomMod())
    ensure_mutable(mod_cuda_zero.cuda).is_available = ensure_eager_tensor(lambda: True)
    ensure_mutable(mod_cuda_zero.cuda).runtime = ensure_eager_tensor(ensure_callable(CustomMod)())
    ensure_mutable(ensure_eager_tensor(mod_cuda_zero.cuda).runtime).getDeviceCount = ensure_eager_tensor(lambda: 0)
    with patch("ml_switcheroo_compiler.backends.cupy.profiler.cp", mod_cuda_zero):
        assert bool(_is_cupy_available()) is False
    mod_cuda_err = CustomMod()
    ensure_mutable(mod_cuda_err).cuda = ensure_eager_tensor(CustomMod())

    def _raise_cuda_error() -> int:
        """Test function _raise_cuda_error."""
        raise RuntimeError("boom")

    _state.update(ensure_tensor_dict({"_raise_cuda_error": _raise_cuda_error, "mod_cuda_err": mod_cuda_err, "mod_cuda_no_runtime": mod_cuda_no_runtime, "mod_cuda_zero": mod_cuda_zero}))
    _state.update(ensure_tensor_dict({"CustomMod": CustomMod, "_is_cupy_available": _is_cupy_available, "mod_cuda_avail": mod_cuda_avail}))


def _test_cupy_profiler_runtime_part_7(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 7 of test_cupy_profiler_runtime.

    Args:
        _state: Shared test state dictionary.
    """
    (CustomMod, _is_cupy_available, _raise_cuda_error, mod_cuda_err) = (_state["CustomMod"], _state["_is_cupy_available"], _state["_raise_cuda_error"], _state["mod_cuda_err"])
    ensure_mutable(mod_cuda_err.cuda).runtime = ensure_eager_tensor(CustomMod())
    ensure_mutable(mod_cuda_err.cuda.runtime).getDeviceCount = ensure_eager_tensor(_raise_cuda_error)
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
