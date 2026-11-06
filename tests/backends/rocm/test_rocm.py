"""Tests for ROCm backend components."""

from __future__ import annotations

import ctypes
import typing
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ml_switcheroo_ir import LogicalNode, NodeDict

from ml_switcheroo_compiler.backends.rocm.rocm import RocmCodeGenerator, ROCmRunner, _calculate_node_bytes, _extract_kernel_name
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_mutable, ensure_node_dict, ensure_tensor_dict, ensure_tensor_list

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_rocm_generator() -> None:
    """Test ROCm code generation."""
    graph = IRGraph()
    graph.nodes = NodeDict({"matmul": IRNode(id="matmul", op_type="MatMul", inputs=[])})
    gen = RocmCodeGenerator(graph)
    out = gen.generate()
    assert "matmul" in out


def test_rocm_generator_coverage() -> None:
    """Test ROCm generator coverage."""
    graph = IRGraph()
    graph.nodes = NodeDict({"input": IRNode(id="input", op_type="Input", inputs=[])})
    gen = RocmCodeGenerator(graph)
    out = gen.generate()
    assert "matmul" not in out


def test_rocm_missing_yaml(monkeypatch: EagerTensorProtocol) -> None:
    """Test ROCm generator with missing YAML files.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    import os

    monkeypatch.setattr(os.path, "exists", lambda p: False)
    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    graph = IRGraph()
    gen = RocmCodeGenerator(graph)
    assert gen.config.templates == {}


def test_calculate_node_bytes_branches() -> None:
    """Test all branches of _calculate_node_bytes."""
    node_64 = IRNode(id="n1", op_type="Custom", inputs=[], attributes={"dtype": "float64"})
    node_64.shape_metadata = (2, 3)
    assert _calculate_node_bytes(node_64) == (6, 48)
    node_16 = IRNode(id="n2", op_type="Custom", inputs=[], attributes={"dtype": "float16"})
    node_16.shape_metadata = (4,)
    assert _calculate_node_bytes(node_16) == (4, 8)
    node_8 = IRNode(id="n3", op_type="Custom", inputs=[], attributes={"dtype": "int8"})
    node_8.shape_metadata = (5,)
    assert _calculate_node_bytes(node_8) == (5, 5)
    node_def = IRNode(id="n4", op_type="Custom", inputs=[])
    assert _calculate_node_bytes(node_def) == (1024, 4096)


def test_extract_kernel_name_fallback() -> None:
    """Test kernel name extraction fallback when regex pattern does not match."""
    assert _extract_kernel_name("int invalid_func() {}", "fallback_kernel") == "fallback_kernel"


def test_rocm_init_yaml_edge_cases(monkeypatch: EagerTensorProtocol) -> None:
    """Test non-yaml files and non-dict op_data in yaml_dir, and empty fallback yaml.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    import os

    monkeypatch.setattr(os.path, "isdir", lambda p: True)
    monkeypatch.setattr(os, "listdir", lambda p: ["ignore.txt", "invalid.yaml"])
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_open.return_value.__enter__.return_value = "file"
        with patch("yaml.safe_load", return_value=["not_a_dict"]):
            gen = RocmCodeGenerator(IRGraph())
            assert gen.config.templates == {}
    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    monkeypatch.setattr(os.path, "exists", lambda p: True)
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_open.return_value.__enter__.return_value = "file"
        with patch("yaml.safe_load", return_value={"other_key": 123}):
            gen2 = RocmCodeGenerator(IRGraph())
            assert gen2.config.templates == {}


def test_rocm_generate_missing_template_branches() -> None:
    """Test generate with op_type missing from templates raises BackendNotSupportedError."""
    graph = IRGraph()
    graph.inputs = ["in_0"]
    graph.nodes = NodeDict({"unmapped_op": IRNode(id="unmapped_op", op_type="UnsupportedCustomOp", inputs=["in_0"])})
    gen = RocmCodeGenerator(graph)
    with pytest.raises(BackendNotSupportedError, match="not supported"):
        gen.generate()


def test_rocm_runner_ctypes_no_rocm_lib_branches() -> None:
    """Test ROCmRunner operations when rocm_lib is None and limits_path does not exist."""
    import os

    with patch("ml_switcheroo_compiler.backends.rocm.rocm.cupy", None):
        with patch("ctypes.util.find_library", return_value=None):
            with patch("os.path.exists", side_effect=lambda p: False if "hardware_limits.yaml" in p else os.path.exists(p)):
                runner = ROCmRunner()
                assert runner.rocm_lib is None
                assert runner.limits == {}
                runner.write_buffer(ctypes.c_void_p(123), b"data")
                runner.free_buffer(ctypes.c_void_p(123))
                runner.load_and_dispatch("path.bin", "main", [1, 1, 1], [1, 1, 1])


def test_rocm_marshall_kernel_params() -> None:
    """Test all branches of _marshall_kernel_params for ROCm."""
    from ml_switcheroo_compiler.backends.rocm.rocm import _marshall_kernel_params

    assert _marshall_kernel_params(None) is None
    assert _marshall_kernel_params([]) is None

    class CustomVal:
        """Test class CustomVal."""

        def __init__(self, val: int) -> None:
            """Test function __init__.

            Args:
                val: Test parameter `val`.
            """
            self.value = val

    args = [ctypes.c_void_p(123), 42, 3.14, CustomVal(999), "100"]
    res = ensure_callable(_marshall_kernel_params)(args)
    assert res is not None
    assert len(res) == 5


def test_rocm_compile_hip_to_code() -> None:
    """Test compile_hip_to_code via ctypes HIPRTC library."""
    mock_hiprtc = MagicMock()
    mock_hiprtc.hiprtcCreateProgram.return_value = 0
    mock_hiprtc.hiprtcCompileProgram.return_value = 0
    expected_code = b"compiled_hip_binary_code"

    def get_code_size_side_effect(prog: EagerValue, size_ptr: EagerValue) -> EagerValue:
        """Test function get_code_size_side_effect.

        Args:
            prog: Test parameter `prog`.
            size_ptr: Test parameter `size_ptr`.
        """
        ensure_mutable(ensure_eager_tensor(size_ptr)._obj).value = ensure_eager_tensor(len(expected_code))
        return 0

    mock_hiprtc.hiprtcGetCodeSize.side_effect = get_code_size_side_effect

    def get_code_side_effect(prog: EagerValue, buf: EagerValue) -> EagerValue:
        """Test function get_code_side_effect.

        Args:
            prog: Test parameter `prog`.
            buf: Test parameter `buf`.
        """
        ensure_mutable(buf).raw = ensure_eager_tensor(expected_code)
        return 0

    mock_hiprtc.hiprtcGetCode.side_effect = get_code_side_effect
    with patch("ctypes.cdll.LoadLibrary", return_value=mock_hiprtc):
        runner = ROCmRunner()
        code = runner.compile_hip_to_code("__global__ void k() {}", "k.hip", options=["-O3"])
        assert code == expected_code
    mock_hiprtc.hiprtcCreateProgram.return_value = 1
    with patch("ctypes.cdll.LoadLibrary", return_value=mock_hiprtc):
        runner = ROCmRunner()
        with pytest.raises(RuntimeError, match="hiprtcCreateProgram failed"):
            runner.compile_hip_to_code("src")
    mock_hiprtc.hiprtcCreateProgram.return_value = 0
    mock_hiprtc.hiprtcCompileProgram.return_value = 2

    def get_log_size_side_effect(prog: EagerValue, size_ptr: EagerValue) -> EagerValue:
        """Test function get_log_size_side_effect.

        Args:
            prog: Test parameter `prog`.
            size_ptr: Test parameter `size_ptr`.
        """
        ensure_mutable(ensure_eager_tensor(size_ptr)._obj).value = ensure_eager_tensor(64)
        return 0

    mock_hiprtc.hiprtcGetProgramLogSize.side_effect = get_log_size_side_effect

    def get_log_side_effect(prog: EagerValue, buf: EagerValue) -> EagerValue:
        """Test function get_log_side_effect.

        Args:
            prog: Test parameter `prog`.
            buf: Test parameter `buf`.
        """
        ensure_mutable(buf).value = ensure_eager_tensor(b"syntax error in hip kernel")
        return 0

    mock_hiprtc.hiprtcGetProgramLog.side_effect = get_log_side_effect
    with patch("ctypes.cdll.LoadLibrary", return_value=mock_hiprtc):
        runner = ROCmRunner()
        with pytest.raises(RuntimeError, match="HIPRTC compilation failed"):
            runner.compile_hip_to_code("src")
    with patch("ctypes.cdll.LoadLibrary", side_effect=Exception("hiprtc not found")):
        runner = ROCmRunner()
        with pytest.raises(RuntimeError, match="HIPRTC library unavailable"):
            runner.compile_hip_to_code("src")


def test_rocm_expanded_templates() -> None:
    """Test code generation for newly added ROCm templates."""
    graph = IRGraph()
    graph.inputs = ["x", "w"]
    graph.nodes = NodeDict(
        {
            "conv3d": IRNode(id="conv3d", op_type="Conv3D", inputs=["x", "w"]),
            "layernorm": IRNode(id="layernorm", op_type="LayerNorm", inputs=["x"]),
            "attention": IRNode(id="attention", op_type="Attention", inputs=["x"]),
            "bcast": IRNode(id="bcast", op_type="Broadcast_Elementwise", inputs=["x", "w"]),
        }
    )
    gen = RocmCodeGenerator(graph)
    out = gen.generate()
    assert "conv3d" in out
    assert "layernorm" in out
    assert "attention" in out
    assert "broadcast_elementwise" in out


def _test_rocm_additional_coverage_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_rocm_additional_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    g = IRGraph()
    n = IRNode(id="fused_0", op_type="FusedElementwise", inputs=["ghost_input"], attributes={"scalar_expr": "1.0f"})
    n.shape_metadata = (4,)
    g.nodes = NodeDict({"fused_0": n})
    gen = RocmCodeGenerator(g)
    code = gen.generate()
    assert "kernel_0" in code
    g_empty = IRGraph()
    n_empty = IRNode(id="fused_empty", op_type="FusedElementwise", inputs=[], attributes={"scalar_expr": "1.0f"})
    n_empty.shape_metadata = (4,)
    g_empty.nodes = NodeDict({"fused_empty": n_empty})
    assert "kernel_0" in RocmCodeGenerator(g_empty).generate()
    runner = ROCmRunner()
    runner.mode = "ctypes"
    mock_hip = MagicMock()
    mock_hip.hipMalloc.return_value = 0
    mock_hip.hipMemcpy.return_value = 0
    mock_hip.hipFree.return_value = 0
    mock_hip.hipDeviceSynchronize.return_value = 0
    runner.rocm_lib = mock_hip
    ensure_mutable(runner).is_available = ensure_eager_tensor(lambda: True)
    ensure_mutable(runner).compile_hip_to_code = ensure_eager_tensor(lambda src: None)
    _state.update(ensure_tensor_dict({"code": code, "g": g, "g_empty": g_empty, "gen": gen}))
    _state.update(ensure_tensor_dict({"mock_hip": mock_hip, "n": n, "n_empty": n_empty, "runner": runner}))


def _test_rocm_additional_coverage_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_rocm_additional_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    runner = _state["runner"]
    g_out = IRGraph()
    g_out.outputs = ["missing_output"]
    runner.execute_graph(g_out, {"raw": b"\x00\x00\x00\x00", "mem": memoryview(b"\x00\x00\x00\x00"), "barr": bytearray(b"\x00\x00\x00\x00")})
    ensure_mutable(runner).allocate_buffer = ensure_eager_tensor(lambda size: None)
    g_none = IRGraph()
    g_none.outputs = ["out_0"]
    runner.execute_graph(g_none, {"raw": b"\x00\x00\x00\x00"})
    runner_avail = ROCmRunner()
    mock_cupy = MagicMock()
    type(mock_cupy.cuda).is_hip = property(lambda self: (_ for _ in ()).throw(RuntimeError("CuPy err")))
    _state.update(ensure_tensor_dict({"g_none": g_none, "g_out": g_out, "mock_cupy": mock_cupy, "runner_avail": runner_avail}))
    _state.update(ensure_tensor_dict({"runner": runner}))


def _test_rocm_additional_coverage_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_rocm_additional_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (mock_cupy, runner_avail) = (_state["mock_cupy"], _state["runner_avail"])
    with patch("ml_switcheroo_compiler.backends.rocm.rocm.cupy", mock_cupy):
        with patch("ctypes.util.find_library", return_value="/usr/lib/libamdhip64.so"):
            with patch("ctypes.cdll.LoadLibrary", side_effect=OSError("CDLL load fail")):
                assert bool(runner_avail.is_available()) is False
            mock_init_fail = MagicMock()
            mock_init_fail.hipInit.return_value = 1
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_init_fail):
                assert bool(runner_avail.is_available()) is False
            mock_dev_lib = MagicMock()
            mock_dev_lib.hipInit.return_value = 0
            mock_dev_lib.hipGetDeviceCount.return_value = 1
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_dev_lib):
                assert bool(runner_avail.is_available()) is False
            mock_dev_zero = MagicMock()
            mock_dev_zero.hipInit.return_value = 0
            mock_dev_zero.hipGetDeviceCount.side_effect = lambda ptr: (ensure_callable(setattr)(ensure_eager_tensor(ptr)._obj, "value", 0), 0)[1]
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_dev_zero):
                assert bool(runner_avail.is_available()) is False
    _state.update(ensure_tensor_dict({"mock_dev_lib": mock_dev_lib, "mock_dev_zero": mock_dev_zero, "mock_init_fail": mock_init_fail}))
    _state.update(ensure_tensor_dict({"mock_cupy": mock_cupy, "runner_avail": runner_avail}))


def _test_rocm_additional_coverage_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_rocm_additional_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    runner_compile = ROCmRunner()
    mock_hiprtc = MagicMock(spec=["hiprtcCreateProgram", "hiprtcCompileProgram", "hiprtcGetProgramLogSize", "hiprtcGetProgramLog", "hiprtcDestroyProgram"])
    mock_hiprtc.hiprtcCreateProgram.return_value = 0
    mock_hiprtc.hiprtcCompileProgram.return_value = 2
    mock_hiprtc.hiprtcGetProgramLogSize.side_effect = lambda prog, size_ptr: (ensure_callable(setattr)(ensure_eager_tensor(size_ptr)._obj, "value", 16), 0)[1]
    mock_hiprtc.hiprtcGetProgramLog.side_effect = lambda prog, buf: (ensure_callable(setattr)(buf, "value", b"err"), 0)[1]
    with patch("ctypes.cdll.LoadLibrary", return_value=mock_hiprtc):
        with pytest.raises(RuntimeError, match="HIPRTC compilation failed"):
            runner_compile.compile_hip_to_code("src")
    del mock_hiprtc.hiprtcDestroyProgram
    _state.update(ensure_tensor_dict({"mock_hiprtc": mock_hiprtc, "runner_compile": runner_compile}))


def _test_rocm_additional_coverage_part_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 5 of test_rocm_additional_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (mock_hiprtc, runner_compile) = (_state["mock_hiprtc"], _state["runner_compile"])
    with patch("ctypes.cdll.LoadLibrary", return_value=mock_hiprtc):
        with pytest.raises(RuntimeError, match="HIPRTC compilation failed"):
            runner_compile.compile_hip_to_code("src")
    mock_hiprtc_ok = MagicMock(spec=["hiprtcCreateProgram", "hiprtcCompileProgram", "hiprtcGetCodeSize", "hiprtcGetCode"])
    mock_hiprtc_ok.hiprtcCreateProgram.return_value = 0
    mock_hiprtc_ok.hiprtcCompileProgram.return_value = 0
    mock_hiprtc_ok.hiprtcGetCodeSize.side_effect = lambda prog, size_ptr: (ensure_callable(setattr)(ensure_eager_tensor(size_ptr)._obj, "value", 8), 0)[1]
    mock_hiprtc_ok.hiprtcGetCode.side_effect = lambda prog, buf: (ensure_callable(setattr)(buf, "raw", b"code_bin"), 0)[1]
    with patch("ctypes.cdll.LoadLibrary", return_value=mock_hiprtc_ok):
        assert runner_compile.compile_hip_to_code("src") == b"code_bin"
    runner_sync = ROCmRunner()
    mock_cupy = MagicMock()
    _state.update(ensure_tensor_dict({"mock_cupy": mock_cupy, "mock_hiprtc_ok": mock_hiprtc_ok, "runner_sync": runner_sync}))
    _state.update(ensure_tensor_dict({"mock_hiprtc": mock_hiprtc, "runner_compile": runner_compile}))


def _test_rocm_additional_coverage_part_6(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 6 of test_rocm_additional_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (mock_cupy, runner_sync) = (_state["mock_cupy"], _state["runner_sync"])
    with patch("ml_switcheroo_compiler.backends.rocm.rocm.cupy", mock_cupy):
        ensure_mutable(runner_sync).mode = ensure_eager_tensor("cupy")
        runner_sync.synchronize()
        mock_cupy.cuda.Stream.null.synchronize.assert_called_once()
        ensure_mutable(mock_cupy.cuda.Stream.null.synchronize).side_effect = ensure_eager_tensor(RuntimeError("sync err"))
        runner_sync.synchronize()
    ensure_mutable(runner_sync).mode = ensure_eager_tensor("ctypes")
    ensure_mutable(runner_sync).rocm_lib = ensure_eager_tensor(None)
    runner_sync.synchronize()
    mock_hip_lib = MagicMock()
    mock_hip_lib.hipDeviceSynchronize.return_value = 0
    ensure_mutable(runner_sync).rocm_lib = ensure_eager_tensor(mock_hip_lib)
    runner_sync.synchronize()
    mock_hip_lib.hipDeviceSynchronize.return_value = 1
    with pytest.raises(RuntimeError, match="hipDeviceSynchronize failed"):
        runner_sync.synchronize()
    mock_hip_lib.hipDeviceSynchronize.return_value = 0
    mock_hip_lib.hipMalloc.return_value = 0
    mock_hip_lib.hipModuleLoad.return_value = 0
    mock_hip_lib.hipModuleGetFunction.return_value = 0
    mock_hip_lib.hipModuleLaunchKernel.return_value = 0
    unmapped_node = IRNode("unmapped", "UnsupportedUnknownOp")
    _state.update(ensure_tensor_dict({"mock_hip_lib": mock_hip_lib, "unmapped_node": unmapped_node}))
    _state.update(ensure_tensor_dict({"mock_cupy": mock_cupy, "runner_sync": runner_sync}))


def _test_rocm_additional_coverage_part_7(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 7 of test_rocm_additional_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    (runner_sync, unmapped_node) = (_state["runner_sync"], _state["unmapped_node"])
    gen_rocm = RocmCodeGenerator(IRGraph())
    ensure_callable(runner_sync._dispatch_compute_node)(unmapped_node, 0, gen_rocm, None, {})
    mapped_node = IRNode("mapped", "Relu", inputs=[])
    ensure_mutable(ensure_eager_tensor(ensure_eager_tensor(gen_rocm.config).templates)["relu"]).workgroup_size = ensure_eager_tensor([16, 16, 2])
    ensure_callable(runner_sync._dispatch_compute_node)(mapped_node, 0, gen_rocm, "/tmp/mock.co", {})
    fused_node = IRNode("fused", "FusedElementwise", inputs=[], attributes={"scalar_expr": "1.0f"})
    ensure_callable(runner_sync._dispatch_compute_node)(fused_node, 0, gen_rocm, "/tmp/mock.co", {})
    g_unmapped = IRGraph()
    g_unmapped.nodes = NodeDict(ensure_node_dict({"unmapped": unmapped_node}))
    with patch.object(runner_sync, "compile_hip_to_code", return_value=b"CO"):
        with pytest.raises(BackendNotSupportedError, match="not supported"):
            runner_sync.execute_graph(g_unmapped, {})
    _state.update(ensure_tensor_dict({"fused_node": fused_node, "g_unmapped": g_unmapped, "gen_rocm": gen_rocm, "mapped_node": mapped_node}))


def test_rocm_additional_coverage() -> None:
    """Test remaining coverage branches in ROCm code generator and runner."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_rocm_additional_coverage_part_1(_state)
    _test_rocm_additional_coverage_part_2(_state)
    _test_rocm_additional_coverage_part_3(_state)
    _test_rocm_additional_coverage_part_4(_state)
    _test_rocm_additional_coverage_part_5(_state)
    _test_rocm_additional_coverage_part_6(_state)
    _test_rocm_additional_coverage_part_7(_state)
    assert len(_state) > 0


def _test_rocm_full_coverage_branches_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_rocm_full_coverage_branches.

    Args:
        _state: Shared test state dictionary.
    """
    gen = RocmCodeGenerator(IRGraph())
    lines = ensure_tensor_list([])
    fused_node = IRNode("fused_node", "FusedElementwise", inputs=["x", "y"], attributes={"scalar_expr": "in0 + in1"})
    ensure_callable(gen._emit_node)(fused_node, 0, {"x": "d_x", "y": "d_y"}, {"x": "fused_node", "y": "other"}, outputs_set=set(), inputs_set={"y"}, hip=lines)
    assert any("hipFree(d_x)" in line for line in lines)
    lines.clear()
    relu_node = IRNode("relu_node", "Relu", inputs=["a", "b", "c"])
    ensure_callable(gen._emit_node)(relu_node, 0, {"a": "d_a", "b": "d_b"}, {"a": "relu_node", "b": "relu_node", "c": "relu_node"}, outputs_set={"b"}, inputs_set=set(), hip=lines)
    assert any("hipFree(d_a)" in line for line in lines)

    class _BareLib:
        """Empty dummy library."""

    runner = ROCmRunner()
    runner.mode = "ctypes"
    ensure_mutable(runner).rocm_lib = ensure_eager_tensor(_BareLib())
    runner.synchronize()
    ensure_mutable(runner).allocate_buffer = ensure_eager_tensor(lambda size: ctypes.c_void_p(12345))
    written = []
    ensure_mutable(runner).write_buffer = ensure_eager_tensor(lambda ptr, raw: written.append((ptr, raw)))
    bufs = runner._prepare_device_buffers({"b": b"raw_bytes", "m": memoryview(b"mem_bytes"), "l": [1.0, 2.0, 3.0]})
    assert len(bufs) == 3
    assert len(written) == 3
    ensure_mutable(runner).allocate_buffer = ensure_eager_tensor(lambda size: None)
    runner._dispatch_compute_node(fused_node, 0, gen, None, {})
    runner._dispatch_compute_node(relu_node, 0, gen, "/tmp/mock.co", {})
    _state.update(ensure_tensor_dict({"bufs": bufs, "fused_node": fused_node, "gen": gen, "lines": lines}))
    _state.update(ensure_tensor_dict({"relu_node": relu_node, "runner": runner, "written": written}))


def _test_rocm_full_coverage_branches_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_rocm_full_coverage_branches.

    Args:
        _state: Shared test state dictionary.
    """
    runner_unavail = ROCmRunner()
    runner_unavail.mode = "ctypes"
    runner_unavail.rocm_lib = None
    ensure_mutable(runner_unavail).is_available = ensure_eager_tensor(lambda: False)
    with pytest.raises(BackendNotSupportedError, match="ROCm HIP accelerator hardware or driver is not available"):
        runner_unavail.execute_graph(IRGraph(), {})
    runner_valid = ROCmRunner()
    runner_valid.mode = "ctypes"
    runner_valid.rocm_lib = MagicMock()
    ensure_mutable(runner_valid).is_available = ensure_eager_tensor(lambda: True)
    ensure_mutable(runner_valid).compile_hip_to_code = ensure_eager_tensor(lambda code: b"CO_BYTES")
    ensure_mutable(runner_valid).allocate_buffer = ensure_eager_tensor(lambda size: ctypes.c_void_p(999))
    ensure_mutable(runner_valid).write_buffer = ensure_eager_tensor(lambda ptr, raw: None)
    ensure_mutable(runner_valid).load_and_dispatch = ensure_eager_tensor(lambda *args, **kwargs: None)
    ensure_mutable(runner_valid).synchronize = ensure_eager_tensor(lambda: None)
    ensure_mutable(runner_valid).read_buffer = ensure_eager_tensor(lambda ptr, size: b"result_bytes")
    freed = []
    ensure_mutable(runner_valid).free_buffer = ensure_eager_tensor(lambda ptr: freed.append(ptr))
    g_exec = IRGraph()
    inp_node = IRNode("inp", "Input")
    inp_node.shape_metadata = (4,)
    calc_node = IRNode("calc", "Relu", inputs=["inp"])
    _state.update(ensure_tensor_dict({"calc_node": calc_node, "freed": freed, "g_exec": g_exec, "inp_node": inp_node}))
    _state.update(ensure_tensor_dict({"runner_unavail": runner_unavail, "runner_valid": runner_valid}))


def _test_rocm_full_coverage_branches_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_rocm_full_coverage_branches.

    Args:
        _state: Shared test state dictionary.
    """
    (calc_node, freed, g_exec, inp_node) = (_state["calc_node"], _state["freed"], _state["g_exec"], _state["inp_node"])
    runner_valid = _state["runner_valid"]
    ensure_mutable(calc_node).shape_metadata = ensure_eager_tensor([4])
    ensure_mutable(calc_node).attributes = ensure_eager_tensor({"buffer_offset": 0})
    ensure_mutable(g_exec).nodes = ensure_eager_tensor({"inp": inp_node, "calc": calc_node})
    ensure_mutable(g_exec).inputs = ensure_eager_tensor(["inp"])
    ensure_mutable(g_exec).outputs = ensure_eager_tensor(["calc"])
    with patch.object(runner_valid, "_prepare_device_buffers", return_value={"inp": ctypes.c_void_p(500), "dummy": None}):
        res = runner_valid.execute_graph(g_exec, {"inp": b"\x00" * 16})
        assert "calc" in res
        assert res["calc"] == b"result_bytes"
        assert len(freed) > 0
    _state.update(ensure_tensor_dict({"res": res}))


def test_rocm_full_coverage_branches() -> None:
    """Test all remaining branches and statements in rocm.py for 100% coverage."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_rocm_full_coverage_branches_part_1(_state)
    _test_rocm_full_coverage_branches_part_2(_state)
    _test_rocm_full_coverage_branches_part_3(_state)
    assert len(_state) > 0


def test_rocm_streams() -> None:
    """Test ROCm HIP stream creation, synchronization, and dispatch with shared memory."""
    runner = ROCmRunner()
    runner.mode = "ctypes"
    mock_lib = MagicMock()
    mock_lib.hipStreamCreate.return_value = 0
    mock_lib.hipStreamSynchronize.return_value = 0
    runner.rocm_lib = mock_lib
    stream = runner.create_stream()
    assert stream is not None
    runner.synchronize_stream(stream)
    mock_lib.hipStreamCreate.return_value = 1
    with pytest.raises(RuntimeError, match="hipStreamCreate failed"):
        runner.create_stream()
    mock_lib.hipStreamSynchronize.return_value = 1
    with pytest.raises(RuntimeError, match="hipStreamSynchronize failed"):
        runner.synchronize_stream(stream)
    runner_cupy = ROCmRunner()
    mock_cupy = MagicMock()
    with patch("ml_switcheroo_compiler.backends.rocm.rocm.cupy", mock_cupy):
        runner_cupy.mode = "cupy"
        c_stream = runner_cupy.create_stream()
        assert c_stream is not None
        runner_cupy.synchronize_stream(c_stream)
        runner_cupy.synchronize_stream(None)


def test_rocm_emitter_branches() -> None:
    """Test ROCmRunner stream creation and synchronization without rocm_lib."""
    from ml_switcheroo_ir import LogicalGraph

    emitter = ROCmRunner()
    emitter.mode = "native"
    emitter.rocm_lib = None
    stream = emitter.create_stream()
    assert stream is None
    emitter.synchronize_stream(None)
    g = LogicalGraph(name="g", outputs=["out_node"])
    n = LogicalNode(id="out_node", op_type="NonExistentUnsupportedOp")
    g.nodes["out_node"] = n
    gen = RocmCodeGenerator(g)
    with pytest.raises((RuntimeError, ValueError, TypeError, KeyError, Exception)):
        gen.generate()


def test_rocm_synthesized_kernels_and_fused_elementwise() -> None:
    """Verify BackendNotSupportedError for unknown ops, and FusedElementwise template handling."""
    g = IRGraph()
    n_unary = IRNode(id="n_un", op_type="CustomSynthUnary", inputs=["in0"])
    n_bin = IRNode(id="n_bin", op_type="CustomSynthBinary", inputs=["in0", "in1"])
    n_tri = IRNode(id="n_tri", op_type="CustomSynthTernary", inputs=["in0", "in1", "in2"])
    g.nodes = NodeDict({"n_un": n_unary, "n_bin": n_bin, "n_tri": n_tri})
    gen = RocmCodeGenerator(g)
    tpl_unary = gen._ensure_template(n_unary, 0)
    assert tpl_unary is not None
    assert "customsynthunary_kernel" in ensure_eager_tensor(ensure_eager_tensor(tpl_unary).body)
    tpl_bin = gen._ensure_template(n_bin, 1)
    assert tpl_bin is not None
    assert "customsynthbinary_kernel" in ensure_eager_tensor(ensure_eager_tensor(tpl_bin).body)
    tpl_tri = gen._ensure_template(n_tri, 2)
    assert tpl_tri is not None
    assert "customsynthternary_kernel" in ensure_eager_tensor(ensure_eager_tensor(tpl_tri).body)
    n_unsup = IRNode(id="n_unsup", op_type="UnsupportedCustomOp", inputs=["in0"])
    with pytest.raises(BackendNotSupportedError, match="not supported"):
        gen._ensure_template(n_unsup, 3)
    n_in = IRNode(id="n_in", op_type="Input", inputs=[])
    n_out = IRNode(id="n_out", op_type="Output", inputs=[])
    n_fused = IRNode(id="n_fuse", op_type="FusedElementwise", inputs=["in0"])
    assert gen._ensure_template(n_in, 4) is None
    assert gen._ensure_template(n_out, 5) is None
    assert gen._ensure_template(n_fused, 6) is None
    hip_lines: list[str] = []
    with patch.object(gen, "_ensure_template", return_value=None):
        gen._emit_node(n_unary, 3, {}, {}, set(), set(), hip_lines)
    assert hip_lines == []
    node_multi = IRNode(id="multi_out", op_type="Split", inputs=["in0"])
    ensure_mutable(node_multi).outputs = ensure_eager_tensor(["out0", "out1"])
    multi_lines: list[str] = []
    out_str = gen._allocate_node_outputs(node_multi, 1, {}, 1024, multi_lines)
    assert "d_out_1_0" in out_str
    assert any("d_out_1_0" in ln for ln in multi_lines)
    assert any("d_out_1_1" in ln for ln in multi_lines)
    g_custom = IRGraph()
    n_custom = IRNode(id="n_c", op_type="CustomNoTpl", inputs=[])
    g_custom.nodes = NodeDict({"n_c": n_custom})
    gen_custom = RocmCodeGenerator(g_custom)
    with patch.object(gen_custom, "_ensure_template", return_value=None):
        code = gen_custom.generate()
        assert "evaluate_rocm" in code


def test_rocm_prepare_input_buffers() -> None:
    """Verify ROCm input buffer preparation across array, bytes, data wrapper, and fallback types."""
    from ml_switcheroo_compiler.backends.rocm.rocm import _prepare_rocm_input_buffers

    class _WrapperWithData:
        """Test class _WrapperWithData."""

        def __init__(self, arr: np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> None:
            """Test function __init__.

            Args:
                arr: Test parameter `arr`.
            """
            self.data = arr

    graph = IRGraph()
    graph.inputs = ["x", "y"]
    arr = np.array([1, 2, 3], dtype=np.int32)
    b_raw = b"raw_bytes"
    ba_raw = bytearray(b"bytearray_data")
    mv_raw = memoryview(b"memoryview_data")
    wrapped = _WrapperWithData(np.array([4, 5], dtype=np.int32))
    list_raw = [1, 2, 3]
    buffers = ensure_callable(_prepare_rocm_input_buffers)(graph, (arr, b_raw, ba_raw, mv_raw, wrapped, list_raw))
    assert buffers["x"] == arr.tobytes()
    assert buffers["y"] == b_raw
    assert buffers["in_2"] == bytes(ba_raw)
    assert buffers["in_3"] == bytes(mv_raw)
    assert buffers["in_4"] == wrapped.data.tobytes()
    assert buffers["in_5"] == bytes(list_raw)


def test_rocm_evaluate_fallback() -> None:
    """Verify interpreter fallback execution with single, multi, and empty output graphs."""
    from ml_switcheroo_compiler.backends.rocm.rocm import _evaluate_rocm_fallback

    class _MockTensor:
        """Test class _MockTensor."""

        def __init__(self, data: np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> None:
            """Test function __init__.

            Args:
                data: Test parameter `data`.
            """
            self.data = data

    g_single = IRGraph()
    n_in = IRNode(id="x", op_type="Input", inputs=[], shape_metadata=[2])
    n_exp = IRNode(id="out", op_type="Exp", inputs=["x"], shape_metadata=[2])
    g_single.nodes = NodeDict({"x": n_in, "out": n_exp})
    g_single.inputs = ["x"]
    g_single.outputs = ["out"]
    arr = np.array([1.0, 2.0], dtype=np.float32)
    res_single = _evaluate_rocm_fallback(g_single, (arr,))
    assert res_single is not None
    res_wrapper = ensure_callable(_evaluate_rocm_fallback)(g_single, (ensure_callable(_MockTensor)(arr), "extra_arg"))
    assert res_wrapper is not None
    g_multi = IRGraph()
    n_out1 = IRNode(id="out1", op_type="Exp", inputs=["x"], shape_metadata=[2])
    n_out2 = IRNode(id="out2", op_type="Exp", inputs=["x"], shape_metadata=[2])
    g_multi.nodes = NodeDict({"x": n_in, "out1": n_out1, "out2": n_out2})
    g_multi.inputs = ["x"]
    g_multi.outputs = ["out1", "out2"]
    res_multi = _evaluate_rocm_fallback(g_multi, (arr,))
    assert isinstance(res_multi, tuple)
    assert len(res_multi) == 2
    g_empty = IRGraph()
    g_empty.nodes = NodeDict({"x": n_in})
    g_empty.inputs = ["x"]
    g_empty.outputs = []
    res_empty = _evaluate_rocm_fallback(g_empty, (arr,))
    assert isinstance(res_empty, dict)


def test_rocm_compile_and_runner_dispatch() -> None:
    """Verify RocmCodeGenerator._compile_aot_impl and runner execution paths."""
    from ml_switcheroo_compiler.backends.base_generator import CompiledArtifact
    from ml_switcheroo_compiler.backends.hardware_compilers import HardwareCompilationResult

    graph = IRGraph()
    n_in = IRNode(id="x", op_type="Input", inputs=[], shape_metadata=[2])
    n_exp = IRNode(id="out", op_type="Exp", inputs=["x"], shape_metadata=[2])
    graph.nodes = NodeDict({"x": n_in, "out": n_exp})
    graph.inputs = ["x"]
    graph.outputs = ["out"]
    gen = RocmCodeGenerator(graph)
    mock_comp_res = HardwareCompilationResult(success=True, assembly_or_ptx="hsaco code object", compiler_cli="hipcc")
    with patch("ml_switcheroo_compiler.backends.hardware_compilers.HIPCompiler.compile", return_value=mock_comp_res):
        artifact = gen.compile_aot(graph, compile_options=["-O3"])
        assert isinstance(artifact, CompiledArtifact)
        assert artifact.binary_bytes == b"hsaco code object"
        assert ensure_eager_tensor(artifact.metadata)["compiler_cli"] == "hipcc"
    mock_comp_empty = HardwareCompilationResult(success=False, assembly_or_ptx="", compiler_cli="hipcc")
    with patch("ml_switcheroo_compiler.backends.hardware_compilers.HIPCompiler.compile", return_value=mock_comp_empty):
        artifact_empty = gen.compile_aot(graph, compile_options="not-a-list")
        assert artifact_empty.binary_bytes == b""
    mock_runner = MagicMock()
    mock_runner.execute_graph.return_value = {"out": np.array([42.0])}
    with patch.object(ROCmRunner, "is_available", return_value=True):
        with patch("ml_switcheroo_compiler.backends.rocm.rocm.ROCmRunner", return_value=mock_runner):
            res_avail = ensure_callable(artifact.callable_fn)(ensure_callable(np.array)([1.0, 2.0], dtype=np.float32))
            assert ensure_callable(np.array_equal)(res_avail, ensure_callable(np.array)([42.0]))
    g_multi = IRGraph()
    n_out1 = IRNode(id="out1", op_type="Exp", inputs=["x"], shape_metadata=[2])
    n_out2 = IRNode(id="out2", op_type="Exp", inputs=["x"], shape_metadata=[2])
    g_multi.nodes = NodeDict({"x": n_in, "out1": n_out1, "out2": n_out2})
    g_multi.inputs = ["x"]
    g_multi.outputs = ["out1", "out2"]
    gen_multi = RocmCodeGenerator(g_multi)
    with patch("ml_switcheroo_compiler.backends.hardware_compilers.HIPCompiler.compile", return_value=mock_comp_res):
        art_multi = gen_multi.compile_aot(g_multi)
    mock_runner_multi = MagicMock()
    mock_runner_multi.execute_graph.return_value = {"out1": 100, "out2": 200}
    with patch.object(ROCmRunner, "is_available", return_value=True):
        with patch("ml_switcheroo_compiler.backends.rocm.rocm.ROCmRunner", return_value=mock_runner_multi):
            res_m = ensure_callable(art_multi.callable_fn)(ensure_callable(np.array)([1.0, 2.0], dtype=np.float32))
            assert res_m == (100, 200)
    mock_runner_err = MagicMock()
    mock_runner_err.execute_graph.side_effect = RuntimeError("ROCm crash")
    with patch.object(ROCmRunner, "is_available", return_value=True):
        with patch("ml_switcheroo_compiler.backends.rocm.rocm.ROCmRunner", return_value=mock_runner_err):
            res_err = ensure_callable(artifact.callable_fn)(ensure_callable(np.array)([1.0, 2.0], dtype=np.float32))
            assert res_err is not None
    with patch.object(ROCmRunner, "is_available", return_value=False):
        res_fallback = ensure_callable(artifact.callable_fn)(ensure_callable(np.array)([1.0, 2.0], dtype=np.float32))
        assert res_fallback is not None
