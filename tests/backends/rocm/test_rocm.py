"""Tests for ROCm backend components."""

import ctypes
from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo_compiler.backends.rocm.rocm import (
    RocmCodeGenerator,
    ROCmRunner,
    _calculate_node_bytes,
    _extract_kernel_name,
)
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_rocm_generator():
    """Test ROCm code generation."""
    graph = IRGraph()
    graph.nodes = {"matmul": IRNode(id="matmul", op_type="MatMul", inputs=[])}
    gen = RocmCodeGenerator(graph)
    out = gen.generate()
    assert "matmul" in out


def test_rocm_generator_coverage():
    """Test ROCm generator coverage."""
    graph = IRGraph()
    graph.nodes = {"input": IRNode(id="input", op_type="Input", inputs=[])}
    gen = RocmCodeGenerator(graph)
    out = gen.generate()
    assert "matmul" not in out


def test_rocm_missing_yaml(monkeypatch):
    """Test ROCm generator with missing YAML files."""
    import os

    monkeypatch.setattr(os.path, "exists", lambda p: False)
    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    graph = IRGraph()
    gen = RocmCodeGenerator(graph)
    assert gen.config.templates == {}


def test_calculate_node_bytes_branches():
    """Test all branches of _calculate_node_bytes."""
    # float64
    node_64 = IRNode(id="n1", op_type="Custom", inputs=[], attributes={"dtype": "float64"})
    node_64.shape_metadata = [2, 3]
    assert _calculate_node_bytes(node_64) == (6, 48)

    # float16
    node_16 = IRNode(id="n2", op_type="Custom", inputs=[], attributes={"dtype": "float16"})
    node_16.shape_metadata = [4]
    assert _calculate_node_bytes(node_16) == (4, 8)

    # int8
    node_8 = IRNode(id="n3", op_type="Custom", inputs=[], attributes={"dtype": "int8"})
    node_8.shape_metadata = [5]
    assert _calculate_node_bytes(node_8) == (5, 5)

    # Default shape and dtype
    node_def = IRNode(id="n4", op_type="Custom", inputs=[])
    assert _calculate_node_bytes(node_def) == (1024, 4096)


def test_extract_kernel_name_fallback():
    """Test kernel name extraction fallback when regex pattern does not match."""
    assert _extract_kernel_name("int invalid_func() {}", "fallback_kernel") == "fallback_kernel"


def test_rocm_init_yaml_edge_cases(monkeypatch):
    """Test non-yaml files and non-dict op_data in yaml_dir, and empty fallback yaml."""
    import os

    monkeypatch.setattr(os.path, "isdir", lambda p: True)
    monkeypatch.setattr(os, "listdir", lambda p: ["ignore.txt", "invalid.yaml"])

    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_open.return_value.__enter__.return_value = "file"
        with patch("yaml.safe_load", return_value=["not_a_dict"]):
            gen = RocmCodeGenerator(IRGraph())
            assert gen.config.templates == {}

    # Test yaml_path exists but has no templates
    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    monkeypatch.setattr(os.path, "exists", lambda p: True)
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_open.return_value.__enter__.return_value = "file"
        with patch("yaml.safe_load", return_value={"other_key": 123}):
            gen2 = RocmCodeGenerator(IRGraph())
            assert gen2.config.templates == {}


def test_rocm_generate_missing_template_branches():
    """Test generate with op_type missing from templates and with graph inputs."""
    graph = IRGraph()
    graph.inputs = ["in_0"]
    graph.nodes = {
        "unmapped_op": IRNode(id="unmapped_op", op_type="NonExistentOp", inputs=["in_0"]),
    }
    gen = RocmCodeGenerator(graph)
    out = gen.generate()
    assert "evaluate_rocm" in out
    assert "d_out_0" not in out


def test_rocm_runner_ctypes_no_rocm_lib_branches():
    """Test ROCmRunner operations when rocm_lib is None and limits_path does not exist."""
    import os

    with patch("ml_switcheroo_compiler.backends.rocm.rocm.cupy", None):
        with patch("ctypes.util.find_library", return_value=None):
            with patch("os.path.exists", side_effect=lambda p: False if "hardware_limits.yaml" in p else os.path.exists(p)):
                runner = ROCmRunner()
                assert runner.rocm_lib is None
                assert runner.limits == {}
                # write_buffer with no rocm_lib
                runner.write_buffer(ctypes.c_void_p(123), b"data")
                # free_buffer with no rocm_lib
                runner.free_buffer(ctypes.c_void_p(123))
                # load_and_dispatch with no rocm_lib
                runner.load_and_dispatch("path.bin", "main", [1, 1, 1], [1, 1, 1])


def test_rocm_marshall_kernel_params():
    """Test all branches of _marshall_kernel_params for ROCm."""
    from ml_switcheroo_compiler.backends.rocm.rocm import _marshall_kernel_params

    assert _marshall_kernel_params(None) is None
    assert _marshall_kernel_params([]) is None

    class CustomVal:
        def __init__(self, val: int) -> None:
            self.value = val

    args = [
        ctypes.c_void_p(123),
        42,
        3.14,
        CustomVal(999),
        "100",
    ]
    res = _marshall_kernel_params(args)
    assert res is not None
    assert len(res) == 5


def test_rocm_compile_hip_to_code():
    """Test compile_hip_to_code via ctypes HIPRTC library."""
    mock_hiprtc = MagicMock()
    mock_hiprtc.hiprtcCreateProgram.return_value = 0
    mock_hiprtc.hiprtcCompileProgram.return_value = 0

    expected_code = b"compiled_hip_binary_code"

    def get_code_size_side_effect(prog, size_ptr):
        size_ptr._obj.value = len(expected_code)
        return 0

    mock_hiprtc.hiprtcGetCodeSize.side_effect = get_code_size_side_effect

    def get_code_side_effect(prog, buf):
        buf.raw = expected_code
        return 0

    mock_hiprtc.hiprtcGetCode.side_effect = get_code_side_effect

    with patch("ctypes.cdll.LoadLibrary", return_value=mock_hiprtc):
        runner = ROCmRunner()
        code = runner.compile_hip_to_code("__global__ void k() {}", "k.hip", options=["-O3"])
        assert code == expected_code

    # Create program failure
    mock_hiprtc.hiprtcCreateProgram.return_value = 1
    with patch("ctypes.cdll.LoadLibrary", return_value=mock_hiprtc):
        runner = ROCmRunner()
        with pytest.raises(RuntimeError, match="hiprtcCreateProgram failed"):
            runner.compile_hip_to_code("src")

    # Compile program failure
    mock_hiprtc.hiprtcCreateProgram.return_value = 0
    mock_hiprtc.hiprtcCompileProgram.return_value = 2

    def get_log_size_side_effect(prog, size_ptr):
        size_ptr._obj.value = 64
        return 0

    mock_hiprtc.hiprtcGetProgramLogSize.side_effect = get_log_size_side_effect

    def get_log_side_effect(prog, buf):
        buf.value = b"syntax error in hip kernel"
        return 0

    mock_hiprtc.hiprtcGetProgramLog.side_effect = get_log_side_effect
    with patch("ctypes.cdll.LoadLibrary", return_value=mock_hiprtc):
        runner = ROCmRunner()
        with pytest.raises(RuntimeError, match="HIPRTC compilation failed"):
            runner.compile_hip_to_code("src")

    # HIPRTC unavailable
    with patch("ctypes.cdll.LoadLibrary", side_effect=Exception("hiprtc not found")):
        runner = ROCmRunner()
        with pytest.raises(RuntimeError, match="HIPRTC library unavailable"):
            runner.compile_hip_to_code("src")


def test_rocm_expanded_templates():
    """Test code generation for newly added ROCm templates."""
    graph = IRGraph()
    graph.inputs = ["x", "w"]
    graph.nodes = {
        "conv3d": IRNode(id="conv3d", op_type="Conv3D", inputs=["x", "w"]),
        "layernorm": IRNode(id="layernorm", op_type="LayerNorm", inputs=["x"]),
        "attention": IRNode(id="attention", op_type="Attention", inputs=["x"]),
        "bcast": IRNode(id="bcast", op_type="Broadcast_Elementwise", inputs=["x", "w"]),
    }
    gen = RocmCodeGenerator(graph)
    out = gen.generate()
    assert "conv3d" in out
    assert "layernorm" in out
    assert "attention" in out
    assert "broadcast_elementwise" in out


def test_rocm_additional_coverage():
    """Test remaining coverage branches in ROCm code generator and runner."""
    # 1. FusedElementwise with 0 inputs (covers load_lines empty) and ghost input for buf_to_free
    g = IRGraph()
    n = IRNode(id="fused_0", op_type="FusedElementwise", inputs=["ghost_input"], attributes={"scalar_expr": "1.0f"})
    n.shape_metadata = (4,)
    g.nodes = {"fused_0": n}
    gen = RocmCodeGenerator(g)
    code = gen.generate()
    assert "kernel_0" in code

    # FusedElementwise with empty inputs -> covers branch 200->202
    g_empty = IRGraph()
    n_empty = IRNode(id="fused_empty", op_type="FusedElementwise", inputs=[], attributes={"scalar_expr": "1.0f"})
    n_empty.shape_metadata = (4,)
    g_empty.nodes = {"fused_empty": n_empty}
    assert "kernel_0" in RocmCodeGenerator(g_empty).generate()

    # 2. execute_graph with bytearray data and missing output
    runner = ROCmRunner()
    mock_hip = MagicMock()
    mock_hip.hipMalloc.return_value = 0
    mock_hip.hipMemcpy.return_value = 0
    mock_hip.hipFree.return_value = 0
    mock_hip.hipDeviceSynchronize.return_value = 0
    runner.rocm_lib = mock_hip
    runner.is_available = lambda: True
    runner.compile_hip_to_code = lambda src: None
    g_out = IRGraph()
    g_out.outputs = ["missing_output"]
    runner.execute_graph(g_out, {"raw": b"\x00\x00\x00\x00", "mem": memoryview(b"\x00\x00\x00\x00"), "barr": bytearray(b"\x00\x00\x00\x00")})

    # Test execute_graph when allocate_buffer returns None with an output in outputs_set
    runner.allocate_buffer = lambda size: None
    g_none = IRGraph()
    g_none.outputs = ["out_0"]
    runner.execute_graph(g_none, {"raw": b"\x00\x00\x00\x00"})

    # 3. is_available exception handling and failed device count
    runner_avail = ROCmRunner()
    mock_cupy = MagicMock()
    type(mock_cupy.cuda).is_hip = property(lambda self: (_ for _ in ()).throw(RuntimeError("CuPy err")))
    with patch("ml_switcheroo_compiler.backends.rocm.rocm.cupy", mock_cupy):
        with patch("ctypes.util.find_library", return_value="/usr/lib/libamdhip64.so"):
            with patch("ctypes.cdll.LoadLibrary", side_effect=OSError("CDLL load fail")):
                assert runner_avail.is_available() is False

            # lib has failed hipInit
            mock_init_fail = MagicMock()
            mock_init_fail.hipInit.return_value = 1
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_init_fail):
                assert runner_avail.is_available() is False

            # Device count non-zero return code
            mock_dev_lib = MagicMock()
            mock_dev_lib.hipInit.return_value = 0
            mock_dev_lib.hipGetDeviceCount.return_value = 1  # non-zero means failure
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_dev_lib):
                assert runner_avail.is_available() is False

            # Device count zero
            mock_dev_zero = MagicMock()
            mock_dev_zero.hipInit.return_value = 0
            mock_dev_zero.hipGetDeviceCount.side_effect = lambda ptr: setattr(ptr._obj, "value", 0) or 0
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_dev_zero):
                assert runner_avail.is_available() is False

    # 4. hiprtcDestroyProgram missing and present
    runner_compile = ROCmRunner()
    mock_hiprtc = MagicMock(spec=["hiprtcCreateProgram", "hiprtcCompileProgram", "hiprtcGetProgramLogSize", "hiprtcGetProgramLog", "hiprtcDestroyProgram"])
    mock_hiprtc.hiprtcCreateProgram.return_value = 0
    mock_hiprtc.hiprtcCompileProgram.return_value = 2
    mock_hiprtc.hiprtcGetProgramLogSize.side_effect = lambda prog, size_ptr: setattr(size_ptr._obj, "value", 16) or 0
    mock_hiprtc.hiprtcGetProgramLog.side_effect = lambda prog, buf: setattr(buf, "value", b"err") or 0
    with patch("ctypes.cdll.LoadLibrary", return_value=mock_hiprtc):
        with pytest.raises(RuntimeError, match="HIPRTC compilation failed"):
            runner_compile.compile_hip_to_code("src")

    # Without hiprtcDestroyProgram
    del mock_hiprtc.hiprtcDestroyProgram
    with patch("ctypes.cdll.LoadLibrary", return_value=mock_hiprtc):
        with pytest.raises(RuntimeError, match="HIPRTC compilation failed"):
            runner_compile.compile_hip_to_code("src")

    # Success path without hiprtcDestroyProgram (covers branch 440->442)
    mock_hiprtc_ok = MagicMock(spec=["hiprtcCreateProgram", "hiprtcCompileProgram", "hiprtcGetCodeSize", "hiprtcGetCode"])
    mock_hiprtc_ok.hiprtcCreateProgram.return_value = 0
    mock_hiprtc_ok.hiprtcCompileProgram.return_value = 0
    mock_hiprtc_ok.hiprtcGetCodeSize.side_effect = lambda prog, size_ptr: setattr(size_ptr._obj, "value", 8) or 0
    mock_hiprtc_ok.hiprtcGetCode.side_effect = lambda prog, buf: setattr(buf, "raw", b"code_bin") or 0
    with patch("ctypes.cdll.LoadLibrary", return_value=mock_hiprtc_ok):
        assert runner_compile.compile_hip_to_code("src") == b"code_bin"

    # 5. ROCmRunner.synchronize branches
    runner_sync = ROCmRunner()
    mock_cupy = MagicMock()
    with patch("ml_switcheroo_compiler.backends.rocm.rocm.cupy", mock_cupy):
        runner_sync.mode = "cupy"
        runner_sync.synchronize()
        mock_cupy.cuda.Stream.null.synchronize.assert_called_once()

    runner_sync.mode = "ctypes"
    runner_sync.rocm_lib = None
    runner_sync.synchronize()

    mock_hip_lib = MagicMock()
    mock_hip_lib.hipDeviceSynchronize.return_value = 0
    runner_sync.rocm_lib = mock_hip_lib
    runner_sync.synchronize()

    mock_hip_lib.hipDeviceSynchronize.return_value = 1
    with pytest.raises(RuntimeError, match="hipDeviceSynchronize failed"):
        runner_sync.synchronize()

    # Reset sync return value to 0 for successful execute_graph
    mock_hip_lib.hipDeviceSynchronize.return_value = 0

    # 6. _dispatch_compute_node with unmapped op and 3D workgroup op
    mock_hip_lib.hipMalloc.return_value = 0
    mock_hip_lib.hipModuleLoad.return_value = 0
    mock_hip_lib.hipModuleGetFunction.return_value = 0
    mock_hip_lib.hipModuleLaunchKernel.return_value = 0
    unmapped_node = IRNode("unmapped", "UnmappedUnknownOp")
    gen_rocm = RocmCodeGenerator(IRGraph())
    runner_sync._dispatch_compute_node(unmapped_node, 0, gen_rocm, None, {})

    mapped_node = IRNode("mapped", "Relu", inputs=[])
    gen_rocm.config.templates["relu"].workgroup_size = [16, 16, 2]
    runner_sync._dispatch_compute_node(mapped_node, 0, gen_rocm, "/tmp/mock.co", {})

    # FusedElementwise dispatch
    fused_node = IRNode("fused", "FusedElementwise", inputs=[], attributes={"scalar_expr": "1.0f"})
    runner_sync._dispatch_compute_node(fused_node, 0, gen_rocm, "/tmp/mock.co", {})

    # execute_graph with an unmapped op node
    g_unmapped = IRGraph()
    g_unmapped.nodes = {"unmapped": unmapped_node}
    with patch.object(runner_sync, "compile_hip_to_code", return_value=b"CO"):
        runner_sync.execute_graph(g_unmapped, {})


def test_rocm_full_coverage_branches():
    """Test all remaining branches and statements in rocm.py for 100% coverage."""
    gen = RocmCodeGenerator(IRGraph())
    lines = []

    # 1. _emit_node buffer freeing for FusedElementwise: input 'x' freed, 'y' in inputs_set so skipped
    fused_node = IRNode("fused_node", "FusedElementwise", inputs=["x", "y"], attributes={"scalar_expr": "in0 + in1"})
    gen._emit_node(
        fused_node,
        0,
        {"x": "d_x", "y": "d_y"},
        {"x": "fused_node", "y": "other"},
        outputs_set=set(),
        inputs_set={"y"},
        hip=lines,
    )
    assert any("hipFree(d_x)" in line for line in lines)

    # 2. _emit_node buffer freeing for standard op: 'a' freed, 'b' in outputs_set, 'c' has no buffer
    lines.clear()
    relu_node = IRNode("relu_node", "Relu", inputs=["a", "b", "c"])
    gen._emit_node(
        relu_node,
        0,
        {"a": "d_a", "b": "d_b"},
        {"a": "relu_node", "b": "relu_node", "c": "relu_node"},
        outputs_set={"b"},
        inputs_set=set(),
        hip=lines,
    )
    assert any("hipFree(d_a)" in line for line in lines)

    # 3. synchronize when rocm_lib does not have hipDeviceSynchronize
    runner = ROCmRunner()
    runner.mode = "ctypes"
    runner.rocm_lib = object()
    runner.synchronize()

    # 4. _prepare_device_buffers with non-None buffer allocation and writing
    runner.allocate_buffer = lambda size: ctypes.c_void_p(12345)
    written = []
    runner.write_buffer = lambda ptr, raw: written.append((ptr, raw))
    bufs = runner._prepare_device_buffers(
        {
            "b": b"raw_bytes",
            "m": memoryview(b"mem_bytes"),
            "l": [1.0, 2.0, 3.0],
        }
    )
    assert len(bufs) == 3
    assert len(written) == 3

    # 5. _dispatch_compute_node with out_ptr is None and tmp_code_path is None
    runner.allocate_buffer = lambda size: None
    runner._dispatch_compute_node(fused_node, 0, gen, None, {})
    runner._dispatch_compute_node(relu_node, 0, gen, "/tmp/mock.co", {})

    # 6. execute_graph raising BackendNotSupportedError
    runner_unavail = ROCmRunner()
    runner_unavail.mode = "ctypes"
    runner_unavail.rocm_lib = None
    runner_unavail.is_available = lambda: False
    with pytest.raises(BackendNotSupportedError, match="ROCm HIP accelerator hardware or driver is not available"):
        runner_unavail.execute_graph(IRGraph(), {})

    # 7. execute_graph with Input node, output retrieval, and None pointer in device_buffers
    runner_valid = ROCmRunner()
    runner_valid.mode = "ctypes"
    runner_valid.rocm_lib = MagicMock()
    runner_valid.is_available = lambda: True
    runner_valid.compile_hip_to_code = lambda code: b"CO_BYTES"
    runner_valid.allocate_buffer = lambda size: ctypes.c_void_p(999)
    runner_valid.write_buffer = lambda ptr, raw: None
    runner_valid.load_and_dispatch = lambda *args, **kwargs: None
    runner_valid.synchronize = lambda: None
    runner_valid.read_buffer = lambda ptr, size: b"result_bytes"
    freed = []
    runner_valid.free_buffer = lambda ptr: freed.append(ptr)

    g_exec = IRGraph()
    inp_node = IRNode("inp", "Input")
    inp_node.shape_metadata = [4]
    calc_node = IRNode("calc", "Relu", inputs=["inp"])
    calc_node.shape_metadata = [4]
    calc_node.attributes = {"buffer_offset": 0}
    g_exec.nodes = {"inp": inp_node, "calc": calc_node}
    g_exec.inputs = ["inp"]
    g_exec.outputs = ["calc"]

    # Inject a None buffer to test `if ptr is not None:` when freeing
    with patch.object(runner_valid, "_prepare_device_buffers", return_value={"inp": ctypes.c_void_p(500), "dummy": None}):
        res = runner_valid.execute_graph(g_exec, {"inp": b"\x00" * 16})
        assert "calc" in res
        assert res["calc"] == b"result_bytes"
        assert len(freed) > 0
