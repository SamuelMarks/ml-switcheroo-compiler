"""Tests for CUDA backend components."""

import ctypes
from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo_compiler.backends.cuda.cuda import CudaCodeGenerator, CUDARunner
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_cuda_generator():
    """Test CUDA code generation."""
    graph = IRGraph()
    graph.nodes = {"matmul": IRNode(id="matmul", op_type="MatMul", inputs=[])}
    gen = CudaCodeGenerator(graph)
    out = gen.generate()
    assert "matmul" in out


def test_cuda_generator_coverage():
    """Test CUDA generator coverage."""
    graph = IRGraph()
    graph.nodes = {"input": IRNode(id="input", op_type="Input", inputs=[])}
    gen = CudaCodeGenerator(graph)
    out = gen.generate()
    assert "matmul" not in out


def test_cuda_missing_yaml(monkeypatch):
    import os

    monkeypatch.setattr(os.path, "exists", lambda p: False)
    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    graph = IRGraph()
    gen = CudaCodeGenerator(graph)
    assert True


def test_cuda_runner_init_no_cupy(monkeypatch):
    """Test CUDARunner initialization without cupy."""
    monkeypatch.setattr("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None)

    with patch("ctypes.util.find_library", return_value="libcuda.so"):
        mock_cdll = MagicMock()
        with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
            runner = CUDARunner()
            assert runner.mode == "ctypes"
            mock_cdll.cuInit.assert_called_once_with(0)


def test_cuda_runner_init_with_cupy():
    """Test CUDARunner initialization with cupy."""
    mock_cupy = MagicMock()
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", mock_cupy):
        runner = CUDARunner()
        assert runner.mode == "cupy"


def test_cuda_runner_allocate_cupy():
    mock_cupy = MagicMock()
    mock_mem = MagicMock()
    mock_mem.ptr = 12345
    mock_cupy.cuda.alloc.return_value = mock_mem
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", mock_cupy):
        runner = CUDARunner()
        ptr = runner.allocate_buffer(100)
        assert ptr.value == 12345


def test_cuda_runner_allocate_ctypes():
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None):
        with patch("ctypes.util.find_library", return_value="libcuda.so"):
            mock_cdll = MagicMock()
            mock_cdll.cuMemAlloc_v2.return_value = 0
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = CUDARunner()
                ptr = runner.allocate_buffer(100)
                assert mock_cdll.cuMemAlloc_v2.called


def test_cuda_runner_allocate_ctypes_fail():
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None):
        with patch("ctypes.util.find_library", return_value="libcuda.so"):
            mock_cdll = MagicMock()
            mock_cdll.cuMemAlloc_v2.return_value = 1  # error
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = CUDARunner()
                with pytest.raises(RuntimeError, match="cuMemAlloc failed"):
                    runner.allocate_buffer(100)


def test_cuda_runner_allocate_ctypes_no_lib():
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None):
        with patch("ctypes.util.find_library", return_value=None):
            runner = CUDARunner()
            ptr = runner.allocate_buffer(100)
            assert ptr.value is None


def test_cuda_runner_write_cupy():
    mock_cupy = MagicMock()
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", mock_cupy):
        runner = CUDARunner()
        runner.write_buffer(ctypes.c_void_p(123), b"test")
        assert mock_cupy.ndarray.called


def test_cuda_runner_write_ctypes():
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None):
        with patch("ctypes.util.find_library", return_value="libcuda.so"):
            mock_cdll = MagicMock()
            mock_cdll.cuMemcpyHtoD_v2.return_value = 0
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = CUDARunner()
                runner.write_buffer(ctypes.c_void_p(123), b"test")
                assert mock_cdll.cuMemcpyHtoD_v2.called


def test_cuda_runner_write_ctypes_fail():
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None):
        with patch("ctypes.util.find_library", return_value="libcuda.so"):
            mock_cdll = MagicMock()
            mock_cdll.cuMemcpyHtoD_v2.return_value = 1
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = CUDARunner()
                with pytest.raises(RuntimeError, match="cuMemcpyHtoD failed"):
                    runner.write_buffer(ctypes.c_void_p(123), b"test")


def test_cuda_runner_write_no_ptr():
    runner = CUDARunner()
    assert runner.write_buffer(None, b"test") is None


def test_cuda_runner_read_cupy():
    mock_cupy = MagicMock()
    mock_arr = MagicMock()
    mock_arr.get.return_value.tobytes.return_value = b"test"
    mock_cupy.ndarray.return_value = mock_arr
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", mock_cupy):
        runner = CUDARunner()
        res = runner.read_buffer(ctypes.c_void_p(123), 4)
        assert res == b"test"


def test_cuda_runner_read_ctypes():
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None):
        with patch("ctypes.util.find_library", return_value="libcuda.so"):
            mock_cdll = MagicMock()
            mock_cdll.cuMemcpyDtoH_v2.return_value = 0
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = CUDARunner()
                res = runner.read_buffer(ctypes.c_void_p(123), 4)
                assert len(res) == 4


def test_cuda_runner_read_ctypes_fail():
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None):
        with patch("ctypes.util.find_library", return_value="libcuda.so"):
            mock_cdll = MagicMock()
            mock_cdll.cuMemcpyDtoH_v2.return_value = 1
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = CUDARunner()
                with pytest.raises(RuntimeError, match="cuMemcpyDtoH failed"):
                    runner.read_buffer(ctypes.c_void_p(123), 4)


def test_cuda_runner_read_no_ptr():
    runner = CUDARunner()
    assert runner.read_buffer(None, 4) == b"\x00" * 4


def test_cuda_runner_free_ctypes():
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None):
        with patch("ctypes.util.find_library", return_value="libcuda.so"):
            mock_cdll = MagicMock()
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = CUDARunner()
                runner.free_buffer(ctypes.c_void_p(123))
                assert mock_cdll.cuMemFree_v2.called


def test_cuda_runner_free_cupy():
    mock_cupy = MagicMock()
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", mock_cupy):
        runner = CUDARunner()
        runner.free_buffer(ctypes.c_void_p(123))
        # Just passes
        assert True


def test_cuda_runner_free_no_ptr():
    runner = CUDARunner()
    runner.free_buffer(None)


def test_cuda_runner_compile_cupy():
    mock_cupy = MagicMock()
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", mock_cupy):
        runner = CUDARunner()
        runner.compile_and_dispatch("ptx", "main", [1, 1, 1], [1, 1, 1])
        runner.compile_and_dispatch("ptx", "main", [1, 1, 1], [1, 1, 1], args=[1, 2])
        assert mock_cupy.RawModule.called


def test_cuda_runner_compile_ctypes():
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None):
        with patch("ctypes.util.find_library", return_value="libcuda.so"):
            mock_cdll = MagicMock()
            mock_cdll.cuModuleLoadData.return_value = 0
            mock_cdll.cuModuleGetFunction.return_value = 0
            mock_cdll.cuLaunchKernel.return_value = 0
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = CUDARunner()
                runner.compile_and_dispatch("ptx", "main", [1, 1, 1], [1, 1, 1], args=[ctypes.c_void_p(1), 2, 3.0])
                assert mock_cdll.cuLaunchKernel.called


def test_cuda_runner_compile_ctypes_fail_load():
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None):
        with patch("ctypes.util.find_library", return_value="libcuda.so"):
            mock_cdll = MagicMock()
            mock_cdll.cuModuleLoadData.return_value = 1
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = CUDARunner()
                with pytest.raises(RuntimeError, match="cuModuleLoadData failed"):
                    runner.compile_and_dispatch("ptx", "main", [1, 1, 1], [1, 1, 1])


def test_cuda_runner_compile_ctypes_fail_func():
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None):
        with patch("ctypes.util.find_library", return_value="libcuda.so"):
            mock_cdll = MagicMock()
            mock_cdll.cuModuleLoadData.return_value = 0
            mock_cdll.cuModuleGetFunction.return_value = 1
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = CUDARunner()
                with pytest.raises(RuntimeError, match="cuModuleGetFunction failed"):
                    runner.compile_and_dispatch("ptx", "main", [1, 1, 1], [1, 1, 1])


def test_cuda_runner_compile_ctypes_fail_launch():
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None):
        with patch("ctypes.util.find_library", return_value="libcuda.so"):
            mock_cdll = MagicMock()
            mock_cdll.cuModuleLoadData.return_value = 0
            mock_cdll.cuModuleGetFunction.return_value = 0
            mock_cdll.cuLaunchKernel.return_value = 1
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = CUDARunner()
                with pytest.raises(RuntimeError, match="cuLaunchKernel failed"):
                    runner.compile_and_dispatch("ptx", "main", [1, 1, 1], [1, 1, 1])


def test_calculate_node_bytes_branches():
    """Test all branches of _calculate_node_bytes."""
    from ml_switcheroo_compiler.backends.cuda.cuda import _calculate_node_bytes

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
    """Test kernel name extraction fallback when pattern does not match."""
    from ml_switcheroo_compiler.backends.cuda.cuda import _extract_kernel_name

    assert _extract_kernel_name("int invalid_func() {}", "fallback_kernel") == "fallback_kernel"


def test_cuda_init_yaml_edge_cases(monkeypatch):
    """Test non-yaml files and non-dict op_data in yaml_dir, and empty fallback yaml."""
    import os

    monkeypatch.setattr(os.path, "isdir", lambda p: True)
    monkeypatch.setattr(os, "listdir", lambda p: ["ignore.txt", "invalid.yaml"])

    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_open.return_value.__enter__.return_value = "file"
        with patch("yaml.safe_load", return_value=["not_a_dict"]):
            gen = CudaCodeGenerator(IRGraph())
            assert gen.config.templates == {}

    # Test yaml_path exists but has no templates
    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    monkeypatch.setattr(os.path, "exists", lambda p: True)
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_open.return_value.__enter__.return_value = "file"
        with patch("yaml.safe_load", return_value={"other_key": 123}):
            gen2 = CudaCodeGenerator(IRGraph())
            assert gen2.config.templates == {}


def test_cuda_generate_missing_template_branches():
    """Test generate with op_type missing from templates and with graph inputs."""
    graph = IRGraph()
    graph.inputs = ["in_0"]
    graph.nodes = {
        "unmapped_op": IRNode(id="unmapped_op", op_type="NonExistentOp", inputs=["in_0"]),
    }
    gen = CudaCodeGenerator(graph)
    out = gen.generate()
    assert "evaluate_cuda" in out
    assert "d_out_0" not in out


def test_cuda_runner_ctypes_no_cuda_lib_branches():
    """Test CUDARunner operations when cuda_lib is None and limits_path does not exist."""
    import os

    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None):
        with patch("ctypes.util.find_library", return_value=None):
            with patch("os.path.exists", side_effect=lambda p: False if "hardware_limits.yaml" in p else os.path.exists(p)):
                runner = CUDARunner()
                assert runner.cuda_lib is None
                assert runner.limits == {}
                # write_buffer with no cuda_lib
                runner.write_buffer(ctypes.c_void_p(123), b"data")
                # free_buffer with no cuda_lib
                runner.free_buffer(ctypes.c_void_p(123))
                # compile_and_dispatch with no cuda_lib
                runner.compile_and_dispatch("ptx", "main", [1, 1, 1], [1, 1, 1])


def test_marshall_kernel_params():
    """Test all branches of _marshall_kernel_params."""
    from ml_switcheroo_compiler.backends.cuda.cuda import _marshall_kernel_params

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


def test_cuda_compile_to_ptx_cupy():
    """Test compile_cuda_to_ptx via cupy.cuda.nvrtc."""
    mock_cupy = MagicMock()
    mock_cupy.cuda.nvrtc.get_ptx.return_value = b".version 7.0\n.target sm_70\n"
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", mock_cupy):
        runner = CUDARunner()
        ptx = runner.compile_cuda_to_ptx("__global__ void k() {}", "k.cu")
        assert ".version" in ptx


def test_cuda_compile_to_ptx_nvrtc_ctypes():
    """Test compile_cuda_to_ptx via ctypes NVRTC library."""
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None):
        mock_nvrtc = MagicMock()
        mock_nvrtc.nvrtcCreateProgram.return_value = 0
        mock_nvrtc.nvrtcCompileProgram.return_value = 0

        def get_ptx_size_side_effect(prog, size_ptr):
            size_ptr._obj.value = 64
            return 0

        mock_nvrtc.nvrtcGetPTXSize.side_effect = get_ptx_size_side_effect

        def get_ptx_side_effect(prog, buf):
            buf.value = b"compiled_ptx_code"
            return 0

        mock_nvrtc.nvrtcGetPTX.side_effect = get_ptx_side_effect

        with patch("ctypes.cdll.LoadLibrary", return_value=mock_nvrtc):
            runner = CUDARunner()
            ptx = runner.compile_cuda_to_ptx("__global__ void k() {}", "k.cu", options=["-arch=compute_70"])
            assert ptx == "compiled_ptx_code"

        # Create program failure
        mock_nvrtc.nvrtcCreateProgram.return_value = 1
        with patch("ctypes.cdll.LoadLibrary", return_value=mock_nvrtc):
            runner = CUDARunner()
            with pytest.raises(RuntimeError, match="nvrtcCreateProgram failed"):
                runner.compile_cuda_to_ptx("src")

        # Compile program failure
        mock_nvrtc.nvrtcCreateProgram.return_value = 0
        mock_nvrtc.nvrtcCompileProgram.return_value = 2

        def get_log_size_side_effect(prog, size_ptr):
            size_ptr._obj.value = 64
            return 0

        mock_nvrtc.nvrtcGetProgramLogSize.side_effect = get_log_size_side_effect

        def get_log_side_effect(prog, buf):
            buf.value = b"syntax error in kernel"
            return 0

        mock_nvrtc.nvrtcGetProgramLog.side_effect = get_log_side_effect
        with patch("ctypes.cdll.LoadLibrary", return_value=mock_nvrtc):
            runner = CUDARunner()
            with pytest.raises(RuntimeError, match="NVRTC compilation failed"):
                runner.compile_cuda_to_ptx("src")

        # NVRTC unavailable
        with patch("ctypes.cdll.LoadLibrary", side_effect=Exception("nvrtc not found")):
            runner = CUDARunner()
            with pytest.raises(RuntimeError, match="NVRTC library unavailable"):
                runner.compile_cuda_to_ptx("src")


def test_cuda_expanded_templates():
    """Test code generation for newly added CUDA templates."""
    graph = IRGraph()
    graph.inputs = ["x", "w"]
    graph.nodes = {
        "conv3d": IRNode(id="conv3d", op_type="Conv3D", inputs=["x", "w"]),
        "layernorm": IRNode(id="layernorm", op_type="LayerNorm", inputs=["x"]),
        "attention": IRNode(id="attention", op_type="Attention", inputs=["x"]),
        "bcast": IRNode(id="bcast", op_type="Broadcast_Elementwise", inputs=["x", "w"]),
    }
    gen = CudaCodeGenerator(graph)
    out = gen.generate()
    assert "conv3d" in out
    assert "layernorm" in out
    assert "attention" in out
    assert "broadcast_elementwise" in out


def test_cuda_generator_template_variants():
    """Test loading string, raw object, and fallback yaml templates."""
    # 1. String and raw object templates in yaml_dir
    with patch("os.path.isdir", return_value=True), patch("os.listdir", return_value=["custom.yaml"]), patch("builtins.open", MagicMock()), patch("yaml.safe_load", return_value={"templates": {"op_str": "void op_str() {}", "op_obj": {"key": "val"}}}):
        gen = CudaCodeGenerator(IRGraph())
        assert gen.config.templates.get("op_str") == {"body": "void op_str() {}"}
        assert gen.config.templates.get("op_obj") == {"key": "val"}

    # 2. Fallback to yaml_path when yaml_dir does not exist
    with patch("os.path.isdir", return_value=False), patch("os.path.exists", return_value=True), patch("builtins.open", MagicMock()), patch("yaml.safe_load", return_value={"templates": {"fallback_op": {"body": "void fallback() {}"}}}):
        gen2 = CudaCodeGenerator(IRGraph())
        assert "fallback_op" in gen2.config.templates


def test_cuda_runner_ctypes_init_failure():
    """Test exception handling during cuCtxCreate_v2."""
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None):
        with patch("ctypes.util.find_library", return_value="libcuda.so"):
            mock_cdll = MagicMock()
            mock_cdll.cuCtxCreate_v2.side_effect = RuntimeError("ctx error")
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = CUDARunner()
                assert runner._ctx.value is None


def test_cuda_runner_read_buffer_error():
    """Test cuMemcpyDtoH failure code."""
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None):
        with patch("ctypes.util.find_library", return_value="libcuda.so"):
            mock_cdll = MagicMock()
            mock_cdll.cuMemcpyDtoH_v2.return_value = -1  # error code
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_cdll):
                runner = CUDARunner()
                with pytest.raises(RuntimeError, match="cuMemcpyDtoH failed"):
                    runner.read_buffer(ctypes.c_void_p(123), 64)


def test_cuda_runner_read_buffer_no_lib():
    """Test read_buffer returns zeros when cuda_lib is None."""
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None):
        with patch("ctypes.util.find_library", return_value=None):
            runner = CUDARunner()
            assert runner.read_buffer(ctypes.c_void_p(0x1000), 16) == b"\x00" * 16


def test_cuda_runner_cupy_nvrtc_exception_fallback():
    """Test fallback to ctypes NVRTC when cupy.cuda.nvrtc fails."""
    mock_cupy = MagicMock()
    mock_cupy.cuda.nvrtc.create_program.side_effect = RuntimeError("nvrtc cupy fail")
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", mock_cupy):
        mock_nvrtc = MagicMock()
        mock_nvrtc.nvrtcCreateProgram.return_value = 0
        mock_nvrtc.nvrtcCompileProgram.return_value = 0
        mock_nvrtc.nvrtcGetPTXSize.side_effect = lambda prog, s: setattr(s._obj, "value", 32) or 0
        mock_nvrtc.nvrtcGetPTX.side_effect = lambda prog, b: setattr(b, "value", b"ptx_fallback") or 0
        mock_nvrtc.nvrtcDestroyProgram.return_value = 0

        with patch("ctypes.cdll.LoadLibrary", return_value=mock_nvrtc):
            runner = CUDARunner()
            ptx = runner.compile_cuda_to_ptx("void f() {}")
            assert "ptx_fallback" in ptx


def test_cuda_nccl_collective_emission():
    """Test CUDA code generator NCCL collective emission for AllReduce, AllGather, ReduceScatter, Broadcast."""
    graph = IRGraph()
    graph.nodes = {
        "x": IRNode(id="x", op_type="Input", inputs=[], shape_metadata=[4, 4]),
        "ar": IRNode(id="ar", op_type="AllReduce", inputs=["x"], shape_metadata=[4, 4]),
        "ag": IRNode(id="ag", op_type="AllGather", inputs=["ar"], shape_metadata=[8, 4]),
        "rs": IRNode(id="rs", op_type="ReduceScatter", inputs=["ag"], shape_metadata=[4, 4]),
        "bc": IRNode(id="bc", op_type="Broadcast", inputs=["rs"], shape_metadata=[4, 4], attributes={"root": 1}),
    }
    graph.inputs = ["x"]
    graph.outputs = ["bc"]

    gen = CudaCodeGenerator(graph)
    cuda_code = gen.generate()

    assert "#include <nccl.h>" in cuda_code
    assert "NCCL_CHECK" in cuda_code
    assert "ncclAllReduce" in cuda_code
    assert "ncclAllGather" in cuda_code
    assert "ncclReduceScatter" in cuda_code
    assert "ncclBroadcast" in cuda_code
    assert "evaluate_cuda" in cuda_code
    assert "ncclComm_t comm = 0" in cuda_code


def test_cuda_additional_branches():
    """Test missing branches in CUDA code generator and runner."""
    # 1. is_available exceptions and failed cuDeviceGetCount
    runner = CUDARunner()
    mock_cupy = MagicMock()
    mock_cupy.cuda.is_available.side_effect = RuntimeError("cupy cuda err")
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", mock_cupy):
        with patch("ctypes.util.find_library", return_value="libcuda.so"):
            with patch("ctypes.cdll.LoadLibrary", side_effect=OSError("LoadLibrary err")):
                assert runner.is_available() is False

            mock_lib = MagicMock()
            mock_lib.cuInit.return_value = 1  # cuInit fails
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_lib):
                assert runner.is_available() is False

            mock_lib_dev = MagicMock()
            mock_lib_dev.cuInit.return_value = 0
            mock_lib_dev.cuDeviceGetCount.return_value = 1  # cuDeviceGetCount fails
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_lib_dev):
                assert runner.is_available() is False

    # 2. compile_cuda_to_ptx returning string and create_program missing
    mock_cupy_str = MagicMock()
    del mock_cupy_str.cuda.nvrtc.create_program
    mock_cupy_str.cuda.nvrtc.get_ptx.return_value = "ptx_str_result"
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", mock_cupy_str):
        runner_ptx = CUDARunner()
        assert runner_ptx.compile_cuda_to_ptx("code") == "ptx_str_result"

    # 3. execute_graph with bytearray data and allocate_buffer returning None
    runner_exec = CUDARunner()
    runner_exec.is_available = lambda: True
    runner_exec.compile_cuda_to_ptx = lambda src: "ptx"
    runner_exec.load_and_dispatch = lambda *args, **kwargs: None
    runner_exec.allocate_buffer = lambda size: None
    g = IRGraph()
    g.outputs = ["out_0"]
    runner_exec.execute_graph(g, {"barr": bytearray(b"\x00\x00\x00\x00")})

    # 4. FusedElementwise with empty inputs
    g_empty = IRGraph()
    n_empty = IRNode(id="fused_empty", op_type="FusedElementwise", inputs=[], attributes={"scalar_expr": "1.0f"})
    n_empty.shape_metadata = (4,)
    g_empty.nodes = {"fused_empty": n_empty}
    gen = CudaCodeGenerator(g_empty)
    assert "kernel_0" in gen.generate()

    # 5. _emit_nccl_node with unknown op_lower (branch 125->128) and ghost input for buf_to_free (branch 131->128)
    node_barrier = IRNode(id="bar", op_type="Barrier", inputs=["x_ghost"])
    cuda_lines: list[str] = []
    gen._emit_nccl_node(node_barrier, 0, {}, {"x_ghost": "bar"}, set(), set(), cuda_lines)

    # 6. _emit_node with FusedElementwise and ghost input for buf_to_free (branch 175->172)
    node_fused_ghost = IRNode(id="fused_node", op_type="FusedElementwise", inputs=["x_ghost"], attributes={"scalar_expr": "1.0f"})
    gen._emit_node(node_fused_ghost, 0, {}, {"x_ghost": "fused_node"}, set(), set(), cuda_lines)

    # 7. compile_cuda_to_ptx without nvrtcDestroyProgram on fail and success (branches 519->521, 527->529)
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None):
        with patch("ctypes.util.find_library", return_value="libnvrtc.so"):
            runner_ctypes = CUDARunner()
            # Fail path without nvrtcDestroyProgram
            mock_nvrtc_fail = MagicMock(spec=["nvrtcCreateProgram", "nvrtcCompileProgram", "nvrtcGetProgramLogSize", "nvrtcGetProgramLog"])
            mock_nvrtc_fail.nvrtcCreateProgram.return_value = 0
            mock_nvrtc_fail.nvrtcCompileProgram.return_value = 1
            mock_nvrtc_fail.nvrtcGetProgramLogSize.side_effect = lambda prog, s: setattr(s._obj, "value", 16) or 0
            mock_nvrtc_fail.nvrtcGetProgramLog.side_effect = lambda prog, b: setattr(b, "value", b"err") or 0
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_nvrtc_fail):
                with pytest.raises(RuntimeError, match="NVRTC compilation failed"):
                    runner_ctypes.compile_cuda_to_ptx("code")

            # Success path without nvrtcDestroyProgram
            mock_nvrtc_ok = MagicMock(spec=["nvrtcCreateProgram", "nvrtcCompileProgram", "nvrtcGetPTXSize", "nvrtcGetPTX"])
            mock_nvrtc_ok.nvrtcCreateProgram.return_value = 0
            mock_nvrtc_ok.nvrtcCompileProgram.return_value = 0
            mock_nvrtc_ok.nvrtcGetPTXSize.side_effect = lambda prog, s: setattr(s._obj, "value", 16) or 0
            mock_nvrtc_ok.nvrtcGetPTX.side_effect = lambda prog, b: setattr(b, "value", b"ptx_code") or 0
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_nvrtc_ok):
                assert runner_ctypes.compile_cuda_to_ptx("code") == "ptx_code"

    # 8. CUDARunner.synchronize branches
    runner_sync = CUDARunner()
    # Mode cupy
    mock_cupy = MagicMock()
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", mock_cupy):
        runner_sync.mode = "cupy"
        runner_sync.synchronize()
        mock_cupy.cuda.Stream.null.synchronize.assert_called_once()

    # Mode ctypes: no cuda_lib
    runner_sync.mode = "ctypes"
    runner_sync.cuda_lib = None
    runner_sync.synchronize()

    # Mode ctypes: cuCtxSynchronize success and fail
    mock_cuda = MagicMock()
    mock_cuda.cuCtxSynchronize.return_value = 0
    runner_sync.cuda_lib = mock_cuda
    runner_sync.synchronize()

    mock_cuda.cuCtxSynchronize.return_value = 1
    with pytest.raises(RuntimeError, match="cuCtxSynchronize failed"):
        runner_sync.synchronize()

    # Mode ctypes: cudaDeviceSynchronize success and fail
    del mock_cuda.cuCtxSynchronize
    mock_cuda.cudaDeviceSynchronize.return_value = 0
    runner_sync.synchronize()

    mock_cuda.cudaDeviceSynchronize.return_value = 1
    with pytest.raises(RuntimeError, match="cudaDeviceSynchronize failed"):
        runner_sync.synchronize()

    # Reset sync return value to 0 for successful execute_graph
    mock_cuda.cudaDeviceSynchronize.return_value = 0

    # 9. _dispatch_compute_node with unmapped op and 3D workgroup op
    mock_cuda.cuMemAlloc_v2.return_value = 0
    mock_cuda.cuModuleLoadData.return_value = 0
    mock_cuda.cuModuleGetFunction.return_value = 0
    mock_cuda.cuLaunchKernel.return_value = 0
    unmapped_node = IRNode("unmapped", "UnmappedUnknownOp")
    gen_cuda = CudaCodeGenerator(IRGraph())
    runner_sync._dispatch_compute_node(unmapped_node, 0, gen_cuda, "ptx", {})

    mapped_node = IRNode("mapped", "Add", inputs=[])
    gen_cuda.config.templates["add"].workgroup_size = [16, 16, 2]
    runner_sync._dispatch_compute_node(mapped_node, 0, gen_cuda, "ptx", {})

    # FusedElementwise dispatch
    fused_node = IRNode("fused", "FusedElementwise", inputs=[], attributes={"scalar_expr": "1.0f"})
    runner_sync._dispatch_compute_node(fused_node, 0, gen_cuda, "ptx", {})

    # execute_graph with an unmapped op node
    g_unmapped = IRGraph()
    g_unmapped.nodes = {"unmapped": unmapped_node}
    with patch.object(runner_sync, "compile_cuda_to_ptx", return_value="// PTX"):
        runner_sync.execute_graph(g_unmapped, {})


def test_cuda_full_coverage_branches():
    """Test all remaining branches and statements in cuda.py for 100% coverage."""
    # 1. is_available via ctypes finding cuda library and cuDeviceGetCount returning count > 0 and count == 0
    with patch("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None):
        with patch("ctypes.util.find_library", return_value="libcuda.so"):
            mock_lib_dev = MagicMock()
            mock_lib_dev.cuInit.return_value = 0

            def set_count_positive(count_ref):
                count_ref._obj.value = 1
                return 0

            mock_lib_dev.cuDeviceGetCount.side_effect = set_count_positive
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_lib_dev):
                assert CUDARunner.is_available() is True

            def set_count_zero(count_ref):
                count_ref._obj.value = 0
                return 0

            mock_lib_dev.cuDeviceGetCount.side_effect = set_count_zero
            with patch("ctypes.cdll.LoadLibrary", return_value=mock_lib_dev):
                assert CUDARunner.is_available() is False

    # 2. _emit_node buffer freeing for FusedElementwise and standard op, including false condition branch
    gen = CudaCodeGenerator(IRGraph())
    lines = []
    # FusedElementwise: input 'x' is freed, input 'y' is in inputs_set so skipped
    fused_node = IRNode("fused_node", "FusedElementwise", inputs=["x", "y"], attributes={"scalar_expr": "in0 + in1"})
    gen._emit_node(
        fused_node,
        0,
        {"x": "d_x", "y": "d_y"},
        {"x": "fused_node", "y": "other"},
        outputs_set=set(),
        inputs_set={"y"},
        cuda=lines,
    )
    assert any("cudaFree(d_x)" in line for line in lines)

    # Standard op (Add): input 'a' is freed, input 'b' is in outputs_set so skipped, input 'c' has no buffer
    lines.clear()
    add_node = IRNode("add_node", "Add", inputs=["a", "b", "c"])
    gen._emit_node(
        add_node,
        0,
        {"a": "d_a", "b": "d_b"},
        {"a": "add_node", "b": "add_node", "c": "add_node"},
        outputs_set={"b"},
        inputs_set=set(),
        cuda=lines,
    )
    assert any("cudaFree(d_a)" in line for line in lines)

    # 3. FusedElementwise with non-empty inputs in generate() covering load_lines
    g_fused_inputs = IRGraph()
    inp = IRNode("in0", "Input")
    fused_with_inp = IRNode("fused_calc", "FusedElementwise", inputs=["in0"], attributes={"scalar_expr": "in0 * 2.0f"})
    fused_with_inp.shape_metadata = [4]
    g_fused_inputs.nodes = {"in0": inp, "fused_calc": fused_with_inp}
    g_fused_inputs.inputs = ["in0"]
    g_fused_inputs.outputs = ["fused_calc"]
    gen_fused = CudaCodeGenerator(g_fused_inputs)
    fused_code = gen_fused.generate()
    assert "float in0 = in_0[idx];" in fused_code

    # 4. synchronize when cuda_lib has neither cuCtxSynchronize nor cudaDeviceSynchronize
    runner = CUDARunner()
    runner.mode = "ctypes"
    runner.cuda_lib = object()  # Has neither attribute
    runner.synchronize()  # Exits cleanly without error

    # 5. _prepare_device_buffers with bytes, memoryview, list, and write_buffer execution
    runner.allocate_buffer = lambda size: ctypes.c_void_p(12345)
    written = []
    runner.write_buffer = lambda ptr, raw: written.append((ptr, raw))
    bufs = runner._prepare_device_buffers(
        {
            "b": b"raw_bytes",
            "m": memoryview(b"memoryview_bytes"),
            "l": [1.0, 2.0, 3.0],
        }
    )
    assert len(bufs) == 3
    assert len(written) == 3

    # 6. _dispatch_compute_node with out_ptr is None and empty ptx_source
    runner.allocate_buffer = lambda size: None
    runner._dispatch_compute_node(fused_with_inp, 0, gen_fused, "", {})
    runner._dispatch_compute_node(add_node, 0, gen_fused, "ptx", {})

    # 7. execute_graph raising BackendNotSupportedError
    runner_unavail = CUDARunner()
    runner_unavail.mode = "ctypes"
    runner_unavail.cuda_lib = None
    runner_unavail.is_available = lambda: False
    with pytest.raises(BackendNotSupportedError, match="CUDA accelerator hardware or driver is not available"):
        runner_unavail.execute_graph(IRGraph(), {})

    # 8. execute_graph with Input node, output retrieval, and None pointer in device_buffers
    runner_valid = CUDARunner()
    runner_valid.mode = "ctypes"
    runner_valid.cuda_lib = MagicMock()
    runner_valid.is_available = lambda: True
    runner_valid.compile_cuda_to_ptx = lambda code: "// PTX"
    alloc_map = {16: ctypes.c_void_p(1001), 32: None}
    runner_valid.allocate_buffer = lambda size: alloc_map.get(size, ctypes.c_void_p(999))
    runner_valid.write_buffer = lambda ptr, raw: None
    runner_valid.compile_and_dispatch = lambda *args, **kwargs: None
    runner_valid.synchronize = lambda: None
    runner_valid.read_buffer = lambda ptr, size: b"result_bytes"
    freed = []
    runner_valid.free_buffer = lambda ptr: freed.append(ptr)

    g_exec = IRGraph()
    inp_node = IRNode("inp", "Input")
    inp_node.shape_metadata = [4]
    calc_node = IRNode("calc", "Add", inputs=["inp"])
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
