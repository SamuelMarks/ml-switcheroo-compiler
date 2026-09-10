"""Exhaustive unit tests for generalized hardware backend emitters (CUDA, ROCm, Metal) and runner dispatch."""

import ctypes
from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo_compiler.backends.cuda.cuda import CudaCodeGenerator, CUDARunner
from ml_switcheroo_compiler.backends.hardware_config_models import (
    calculate_hardware_launch_config,
    generate_nd_coordinate_offset_logic,
)
from ml_switcheroo_compiler.backends.metal.metal import MetalCodeGenerator, MetalRunner
from ml_switcheroo_compiler.backends.rocm.rocm import RocmCodeGenerator, ROCmRunner
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_calculate_hardware_launch_config() -> None:
    """Verify launch grid and thread block calculations across 0D, 1D, 2D, 3D and 4D tensor shapes."""
    # 0D
    b0, g0 = calculate_hardware_launch_config(())
    assert b0 == (1, 1, 1)
    assert g0 == (1, 1, 1)

    # 1D
    b1, g1 = calculate_hardware_launch_config((2048,), max_threads_per_block=512)
    assert b1 == (512, 1, 1)
    assert g1 == (4, 1, 1)

    # 2D
    b2, g2 = calculate_hardware_launch_config((64, 128))
    assert b2 == (16, 16, 1)
    assert g2 == (8, 4, 1)

    # 3D
    b3, g3 = calculate_hardware_launch_config((4, 32, 64))
    assert b3[0] == 16
    assert b3[1] == 8
    assert g3[0] == 4
    assert g3[1] == 4
    assert g3[2] >= 1

    # 4D with symbolic / string dim handling
    b4, g4 = calculate_hardware_launch_config(["batch", 3, 32, 32])
    assert b4[0] == 16
    assert g4[0] == 2


def test_generate_nd_coordinate_offset_logic() -> None:
    """Verify N-dimensional coordinate-to-linear-offset index calculation generation for CUDA, ROCm and Metal."""
    # 1D contiguous
    lines_1d = generate_nd_coordinate_offset_logic((100,), language="cuda")
    assert any("offset = idx" in l for l in lines_1d)

    # 1D strided
    lines_1d_s = generate_nd_coordinate_offset_logic((100,), strides=[4], language="metal")
    assert any("offset = idx * 4" in l for l in lines_1d_s)

    # 2D contiguous
    lines_2d = generate_nd_coordinate_offset_logic((10, 20), language="cuda")
    assert any("rem_idx" in l for l in lines_2d)
    assert any("c_idx_1 = rem_idx % 20" in l for l in lines_2d)

    # 3D with explicit strides
    lines_3d = generate_nd_coordinate_offset_logic((2, 4, 8), strides=[32, 8, 1], language="hip")
    assert any("offset = (c_idx_0 * 32) + (c_idx_1 * 8) + c_idx_2" in l for l in lines_3d)

    # Zero stride / broadcast dim handling
    lines_bcast = generate_nd_coordinate_offset_logic((2, 4), strides=[0, 1], language="cuda")
    assert any("offset = c_idx_1" in l for l in lines_bcast)

    # Invalid non-integer shape elements fallback to 1
    lines_invalid = generate_nd_coordinate_offset_logic(("invalid", None), language="cuda")
    assert any("rem_idx" in l for l in lines_invalid)


def test_cuda_generator_emits_nd_index_helpers() -> None:
    """Verify that CudaCodeGenerator emits N-dimensional coordinate offset functions in header."""
    graph = IRGraph(name="test_cuda_nd")
    graph.nodes["in0"] = IRNode(id="in0", op_type="Input")
    graph.nodes["relu0"] = IRNode(id="relu0", op_type="Relu", inputs=["in0"])
    graph.outputs = ["relu0"]

    code = CudaCodeGenerator(graph).generate()
    assert "coords_to_offset_nd" in code
    assert "__device__ inline int coords_to_offset_nd" in code


def test_rocm_generator_emits_nd_index_helpers() -> None:
    """Verify that RocmCodeGenerator emits N-dimensional coordinate offset functions in header."""
    graph = IRGraph(name="test_rocm_nd")
    graph.nodes["in0"] = IRNode(id="in0", op_type="Input")
    graph.nodes["relu0"] = IRNode(id="relu0", op_type="Relu", inputs=["in0"])
    graph.outputs = ["relu0"]

    code = RocmCodeGenerator(graph).generate()
    assert "coords_to_offset_nd" in code
    assert "__device__ inline int coords_to_offset_nd" in code


def test_metal_generator_emits_nd_index_helpers() -> None:
    """Verify that MetalCodeGenerator emits N-dimensional coordinate offset functions in MSL header."""
    graph = IRGraph(name="test_metal_nd")
    graph.nodes["in0"] = IRNode(id="in0", op_type="Input")
    graph.nodes["relu0"] = IRNode(id="relu0", op_type="Relu", inputs=["in0"])
    graph.outputs = ["relu0"]

    code = MetalCodeGenerator(graph).generate()
    assert "coords_to_offset_nd" in code
    assert "inline int coords_to_offset_nd" in code


def _mock_device_count(ptr: object) -> int:
    """Mock device count setter for ctypes pointer.

    Args:
        ptr (object): Pointer object.

    Returns:
        int: Return code 0.
    """
    getattr(ptr, "_obj", ptr).value = 1
    return 0


def test_cuda_runner_execute_graph_and_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CUDARunner.is_available, execute_graph dispatch, and BackendNotSupportedError."""
    # 1. Unavailable -> raises BackendNotSupportedError
    monkeypatch.setattr("ml_switcheroo_compiler.backends.cuda.cuda.cupy", None)
    with patch("ctypes.util.find_library", return_value=None):
        runner = CUDARunner()
        assert runner.is_available() is False
        with pytest.raises(BackendNotSupportedError, match="CUDA accelerator hardware"):
            runner.execute_graph(IRGraph(), {})

    # 2. Mock driver available -> execute_graph succeeds
    mock_lib = MagicMock()
    mock_lib.cuInit.return_value = 0
    mock_lib.cuDeviceGetCount.side_effect = _mock_device_count
    mock_lib.cuMemAlloc_v2.return_value = 0
    mock_lib.cuMemcpyHtoD_v2.return_value = 0
    mock_lib.cuMemcpyDtoH_v2.return_value = 0
    mock_lib.cuMemFree_v2.return_value = 0
    mock_lib.cuModuleLoadData.return_value = 0
    mock_lib.cuModuleGetFunction.return_value = 0
    mock_lib.cuLaunchKernel.return_value = 0
    mock_lib.cuCtxSynchronize.return_value = 0

    with patch("ctypes.util.find_library", return_value="libcuda.so"):
        with patch("ctypes.cdll.LoadLibrary", return_value=mock_lib):
            runner_mock = CUDARunner()
            assert runner_mock.is_available() is True

            graph = IRGraph(name="cuda_multi_node")
            graph.nodes["x"] = IRNode(id="x", op_type="Input", shape_metadata=(4,))
            graph.nodes["add"] = IRNode(id="add", op_type="Add", inputs=["x", "x"], shape_metadata=(4,))
            graph.inputs = ["x"]
            graph.outputs = ["add"]

            with patch.object(runner_mock, "compile_cuda_to_ptx", return_value="// PTX"):
                results = runner_mock.execute_graph(graph, {"x": [1.0, 2.0, 3.0, 4.0]})
                assert "add" in results
                assert isinstance(results["add"], bytes)


def test_rocm_runner_execute_graph_and_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify ROCmRunner.is_available, execute_graph dispatch, and BackendNotSupportedError."""
    # 1. Unavailable -> raises BackendNotSupportedError
    monkeypatch.setattr("ml_switcheroo_compiler.backends.rocm.rocm.cupy", None)
    with patch("ctypes.util.find_library", return_value=None):
        runner = ROCmRunner()
        assert runner.is_available() is False
        with pytest.raises(BackendNotSupportedError, match="ROCm HIP accelerator hardware"):
            runner.execute_graph(IRGraph(), {})

    # 2. Mock driver available -> execute_graph succeeds
    mock_lib = MagicMock()
    mock_lib.hipInit.return_value = 0
    mock_lib.hipGetDeviceCount.side_effect = _mock_device_count
    mock_lib.hipMalloc.return_value = 0
    mock_lib.hipMemcpyHtoD.return_value = 0
    mock_lib.hipMemcpyDtoH.return_value = 0
    mock_lib.hipFree.return_value = 0
    mock_lib.hipModuleLoad.return_value = 0
    mock_lib.hipModuleGetFunction.return_value = 0
    mock_lib.hipModuleLaunchKernel.return_value = 0
    mock_lib.hipDeviceSynchronize.return_value = 0

    with patch("ctypes.util.find_library", return_value="libamdhip64.so"):
        with patch("ctypes.cdll.LoadLibrary", return_value=mock_lib):
            runner_mock = ROCmRunner()
            assert runner_mock.is_available() is True

            graph = IRGraph(name="rocm_multi_node")
            graph.nodes["x"] = IRNode(id="x", op_type="Input", shape_metadata=(4,))
            graph.nodes["relu"] = IRNode(id="relu", op_type="Relu", inputs=["x"], shape_metadata=(4,))
            graph.inputs = ["x"]
            graph.outputs = ["relu"]

            with patch.object(runner_mock, "compile_hip_to_code", return_value=b"ELF_BINARY"):
                results = runner_mock.execute_graph(graph, {"x": [1.0, 2.0, 3.0, 4.0]})
                assert "relu" in results
                assert isinstance(results["relu"], bytes)


def test_metal_runner_execute_graph_and_availability() -> None:
    """Verify MetalRunner.is_available, execute_graph multi-node dispatch, and BackendNotSupportedError."""
    # 1. Unavailable -> raises BackendNotSupportedError
    with patch.object(MetalRunner, "is_available", return_value=False):
        runner = MetalRunner()
        runner.device = None
        with pytest.raises(BackendNotSupportedError, match="Apple Silicon Metal GPU runtime"):
            runner.execute_graph(IRGraph(), {})

    # 2. Mock Metal framework available -> execute_graph multi-node pipeline succeeds
    mock_metal = MagicMock()
    mock_dev = MagicMock()
    mock_lib = MagicMock()
    mock_func = MagicMock()
    mock_pipe = MagicMock()
    mock_queue = MagicMock()
    mock_cmd = MagicMock()
    mock_enc = MagicMock()
    mock_out_buf = MagicMock()

    dummy_data = (ctypes.c_float * 4)(1.0, 2.0, 3.0, 4.0)
    mock_out_buf.contents.return_value = ctypes.addressof(dummy_data)

    mock_metal.MTLCreateSystemDefaultDevice.return_value = mock_dev
    mock_dev.newLibraryWithSource_options_error_.return_value = (mock_lib, None)
    mock_lib.newFunctionWithName_.return_value = mock_func
    mock_dev.newComputePipelineStateWithFunction_error_.return_value = (mock_pipe, None)
    mock_dev.newCommandQueue.return_value = mock_queue
    mock_queue.commandBuffer.return_value = mock_cmd
    mock_cmd.computeCommandEncoder.return_value = mock_enc
    mock_dev.newBufferWithBytes_length_options_.return_value = MagicMock()
    mock_dev.newBufferWithLength_options_.return_value = mock_out_buf

    with patch.object(MetalRunner, "is_available", return_value=True):
        runner = MetalRunner()
        runner.device = mock_dev

        graph = IRGraph(name="metal_multi_node")
        graph.nodes["x"] = IRNode(id="x", op_type="Input", shape_metadata=(4,))
        graph.nodes["relu"] = IRNode(id="relu", op_type="Relu", inputs=["x"], shape_metadata=(4,))
        graph.inputs = ["x"]
        graph.outputs = ["relu"]

        import sys

        with patch.dict(sys.modules, {"Metal": mock_metal}):
            results = runner.execute_graph(graph, {"x": [1.0, 2.0, 3.0, 4.0]})
            assert "relu" in results
            assert len(results["relu"]) == 16


def test_hardware_emitters_fused_elementwise_code_generation() -> None:
    """Verify code generation and buffer lifecycle tracking for FusedElementwise across backends."""
    graph = IRGraph(name="fused_emitter_test")
    graph.nodes["in0"] = IRNode(id="in0", op_type="Input", shape_metadata=(16,))
    graph.nodes["in1"] = IRNode(id="in1", op_type="Input", shape_metadata=(16,))
    graph.nodes["fused0"] = IRNode(
        id="fused0",
        op_type="FusedElementwise",
        inputs=["in0", "in1"],
        shape_metadata=(16,),
        attributes={"fused_ops": ["Add", "Relu"], "scalar_expr": "fmaxf(0.0f, (in0 + in1))"},
    )
    graph.nodes["fused1"] = IRNode(
        id="fused1",
        op_type="FusedElementwise",
        inputs=["fused0"],
        shape_metadata=(16,),
        attributes={"fused_ops": ["Neg"], "scalar_expr": "(-in0)"},
    )
    graph.inputs = ["in0", "in1"]
    graph.outputs = ["fused1"]

    # CUDA
    cuda_gen = CudaCodeGenerator(graph).generate()
    assert "fused_elementwise_kernel" in cuda_gen
    assert "cudaFree" in cuda_gen

    # ROCm
    rocm_gen = RocmCodeGenerator(graph).generate()
    assert "fused_elementwise_kernel" in rocm_gen
    assert "hipFree" in rocm_gen

    # Metal
    metal_gen = MetalCodeGenerator(graph).generate()
    assert "fused_elementwise" in metal_gen
    assert "= None" in metal_gen


def test_runner_execute_graph_input_types() -> None:
    """Verify execute_graph handles bytes, memoryview, and numpy arrays correctly across runners."""
    graph = IRGraph(name="test_types")
    graph.nodes["a"] = IRNode(id="a", op_type="Input", shape_metadata=(2,))
    graph.nodes["out"] = IRNode(id="out", op_type="Relu", inputs=["a"], shape_metadata=(2,))
    graph.inputs = ["a"]
    graph.outputs = ["out"]

    raw_bytes = b"\x00\x00\x80?\x00\x00\x00@"  # [1.0, 2.0] in float32
    mem_view = memoryview(raw_bytes)

    # 1. CUDA runner with bytes and memoryview
    mock_lib = MagicMock()
    mock_lib.cuInit.return_value = 0
    mock_lib.cuDeviceGetCount.side_effect = _mock_device_count
    mock_lib.cuMemAlloc_v2.return_value = 0
    mock_lib.cuMemcpyHtoD_v2.return_value = 0
    mock_lib.cuMemcpyDtoH_v2.return_value = 0
    mock_lib.cuMemFree_v2.return_value = 0
    mock_lib.cuModuleLoadData.return_value = 0
    mock_lib.cuModuleGetFunction.return_value = 0
    mock_lib.cuLaunchKernel.return_value = 0
    mock_lib.cuCtxSynchronize.return_value = 0
    with patch("ctypes.util.find_library", return_value="libcuda.so"), patch("ctypes.cdll.LoadLibrary", return_value=mock_lib):
        c_runner = CUDARunner()
        with patch.object(c_runner, "compile_cuda_to_ptx", return_value="// PTX"):
            res_bytes = c_runner.execute_graph(graph, {"a": raw_bytes})
            assert "out" in res_bytes
            res_mv = c_runner.execute_graph(graph, {"a": mem_view})
            assert "out" in res_mv

    # 2. Metal runner with bytes, memoryview and missing func
    mock_metal = MagicMock()
    mock_dev = MagicMock()
    mock_lib_m = MagicMock()
    mock_metal.MTLCreateSystemDefaultDevice.return_value = mock_dev
    mock_dev.newLibraryWithSource_options_error_.return_value = (mock_lib_m, None)
    # func returns None to test safe skip
    mock_lib_m.newFunctionWithName_.return_value = None

    import sys

    with patch.object(MetalRunner, "is_available", return_value=True):
        m_runner = MetalRunner()
        m_runner.device = mock_dev
        with patch.dict(sys.modules, {"Metal": mock_metal}):
            res_m = m_runner.execute_graph(graph, {"a": raw_bytes})
            assert res_m == {}


def test_runner_availability_via_cupy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify is_available works with cupy runtime detection for CUDA and ROCm."""
    mock_cupy = MagicMock()
    mock_cupy.cuda.is_available.return_value = True
    mock_cupy.cuda.runtime.getDeviceCount.return_value = 1
    mock_cupy.cuda.is_hip = True

    # CUDA
    monkeypatch.setattr("ml_switcheroo_compiler.backends.cuda.cuda.cupy", mock_cupy)
    assert CUDARunner.is_available() is True

    mock_cupy_no_dev = MagicMock()
    mock_cupy_no_dev.cuda.is_available.return_value = False
    monkeypatch.setattr("ml_switcheroo_compiler.backends.cuda.cuda.cupy", mock_cupy_no_dev)
    with patch("ctypes.util.find_library", return_value=None):
        assert CUDARunner.is_available() is False

    # ROCm
    monkeypatch.setattr("ml_switcheroo_compiler.backends.rocm.rocm.cupy", mock_cupy)
    assert ROCmRunner.is_available() is True
