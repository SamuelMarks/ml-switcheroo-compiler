"""Tests for Metal backend generator and runtime runner."""

import ctypes
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.metal.metal import (
    MetalCodeGenerator,
    MetalRunner,
    _calculate_node_bytes,
)
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_metal_generator():
    """Test Metal code generation."""
    graph = IRGraph()
    graph.inputs = ["in_0"]
    graph.nodes = {"matmul": IRNode(id="matmul", op_type="MatMul", inputs=["in_0"])}
    gen = MetalCodeGenerator(graph)
    out = gen.generate()
    assert "matmul" in out
    assert "setBuffer_offset_atIndex_" in out


def test_metal_generator_coverage():
    """Test Metal generator coverage."""
    graph = IRGraph()
    graph.nodes = {"input": IRNode(id="input", op_type="Input", inputs=[])}
    gen = MetalCodeGenerator(graph)
    out = gen.generate()
    assert "matmul" not in out


def test_metal_missing_yaml(monkeypatch):
    """Test Metal generator when YAML directories do not exist."""
    import os

    monkeypatch.setattr(os.path, "exists", lambda p: False)
    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    graph = IRGraph()
    gen = MetalCodeGenerator(graph)
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


def test_metal_init_yaml_edge_cases(monkeypatch):
    """Test non-yaml files and non-dict op_data in yaml_dir, and empty fallback yaml."""
    import os

    monkeypatch.setattr(os.path, "isdir", lambda p: True)
    monkeypatch.setattr(os, "listdir", lambda p: ["ignore.txt", "invalid.yaml"])

    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_open.return_value.__enter__.return_value = "file"
        with patch("yaml.safe_load", return_value=["not_a_dict"]):
            gen = MetalCodeGenerator(IRGraph())
            assert gen.config.templates == {}

    # Test yaml_path exists but has no templates
    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    monkeypatch.setattr(os.path, "exists", lambda p: True)
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_open.return_value.__enter__.return_value = "file"
        with patch("yaml.safe_load", return_value={"other_key": 123}):
            gen2 = MetalCodeGenerator(IRGraph())
            assert gen2.config.templates == {}


def test_metal_generate_missing_template_branches():
    """Test generate with op_type missing from templates and with graph inputs."""
    graph = IRGraph()
    graph.inputs = ["in_0"]
    graph.nodes = {
        "unmapped_op": IRNode(id="unmapped_op", op_type="NonExistentOp", inputs=["in_0"]),
    }
    gen = MetalCodeGenerator(graph)
    out = gen.generate()
    assert "evaluate_metal" in out
    assert "buffer_out_0" not in out


def test_metal_runner_init_with_metal():
    """Test MetalRunner.__init__ when Metal module is present."""
    mock_metal = MagicMock()
    mock_dev = MagicMock()
    mock_metal.MTLCreateSystemDefaultDevice.return_value = mock_dev
    with patch.dict(sys.modules, {"Metal": mock_metal}):
        runner = MetalRunner()
        assert runner.device is mock_dev


def test_metal_runner_is_available():
    """Test MetalRunner.is_available branches."""
    mock_metal = MagicMock()
    mock_metal.MTLCreateSystemDefaultDevice.return_value = "mock_device"
    with patch.dict(sys.modules, {"Metal": mock_metal}):
        assert MetalRunner.is_available() is True

        mock_metal.MTLCreateSystemDefaultDevice.return_value = None
        with patch("ctypes.cdll.LoadLibrary", side_effect=Exception("error")):
            assert MetalRunner.is_available() is False

        mock_metal.MTLCreateSystemDefaultDevice.side_effect = Exception("error")
        with patch("ctypes.cdll.LoadLibrary", side_effect=Exception("error")):
            assert MetalRunner.is_available() is False


def test_metal_runner_allocate_and_buffer_rw():
    """Test MetalRunner buffer allocation and read/write."""
    runner = MetalRunner()
    mock_dev = MagicMock()
    mock_buf = MagicMock()
    # Create an actual memory buffer
    raw_mem = ctypes.create_string_buffer(64)
    mem_addr = ctypes.addressof(raw_mem)
    mock_buf.contents.return_value = mem_addr
    mock_dev.newBufferWithLength_options_.return_value = mock_buf
    runner.device = mock_dev

    ptr = runner.allocate_buffer(64)
    assert ptr is not None and ptr.value == mem_addr

    # Write and read
    runner.write_buffer(ptr, b"hello_metal_test")
    read_back = runner.read_buffer(ptr, 16)
    assert read_back == b"hello_metal_test"

    # Edge cases: buffer with value=None or 0
    runner.write_buffer(ctypes.c_void_p(0), b"test")
    assert runner.read_buffer(ctypes.c_void_p(0), 10) == b""


def test_metal_runner_execute_kernel_failures():
    """Test error handling in MetalRunner.execute_kernel."""
    runner = MetalRunner()
    runner.device = None

    mock_metal = MagicMock()
    mock_metal.MTLCreateSystemDefaultDevice.return_value = None
    with patch.dict(sys.modules, {"Metal": mock_metal}):
        with pytest.raises(RuntimeError, match="No Metal device available."):
            runner.execute_kernel("src", "main")

    mock_dev = MagicMock()
    mock_metal.MTLCreateSystemDefaultDevice.return_value = mock_dev

    # Compilation failed
    mock_dev.newLibraryWithSource_options_error_.return_value = (None, "comp_err")
    with patch.dict(sys.modules, {"Metal": mock_metal}):
        with pytest.raises(RuntimeError, match="Metal compilation failed"):
            runner.execute_kernel("src", "main")

    # Entry point not found
    mock_lib = MagicMock()
    mock_dev.newLibraryWithSource_options_error_.return_value = (mock_lib, None)
    mock_lib.newFunctionWithName_.return_value = None
    with patch.dict(sys.modules, {"Metal": mock_metal}):
        with pytest.raises(RuntimeError, match="not found in library"):
            runner.execute_kernel("src", "main")

    # Pipeline state failure
    mock_func = MagicMock()
    mock_lib.newFunctionWithName_.return_value = mock_func
    mock_dev.newComputePipelineStateWithFunction_error_.return_value = (None, "pipe_err")
    with patch.dict(sys.modules, {"Metal": mock_metal}):
        with pytest.raises(RuntimeError, match="Failed to create pipeline state"):
            runner.execute_kernel("src", "main")


def test_metal_runner_execute_kernel_success():
    """Test successful execution of Metal compute kernel."""
    runner = MetalRunner()
    mock_metal = MagicMock()
    mock_dev = MagicMock()
    mock_lib = MagicMock()
    mock_func = MagicMock()
    mock_pipe = MagicMock()
    mock_queue = MagicMock()
    mock_cmd = MagicMock()
    mock_enc = MagicMock()

    mock_metal.MTLCreateSystemDefaultDevice.return_value = mock_dev
    mock_dev.newLibraryWithSource_options_error_.return_value = (mock_lib, None)
    mock_lib.newFunctionWithName_.return_value = mock_func
    mock_dev.newComputePipelineStateWithFunction_error_.return_value = (mock_pipe, None)
    mock_dev.newCommandQueue.return_value = mock_queue
    mock_queue.commandBuffer.return_value = mock_cmd
    mock_cmd.computeCommandEncoder.return_value = mock_enc
    mock_pipe.threadExecutionWidth.return_value = 32

    # Return buffer with float data
    out_array = np.array([42.0], dtype=np.float32)
    mock_out_buf = MagicMock()
    mock_out_buf.contents.return_value = out_array.ctypes.data
    mock_dev.newBufferWithLength_options_.return_value = mock_out_buf

    with patch.dict(sys.modules, {"Metal": mock_metal}):
        # Case 1: with inputs
        res = runner.execute_kernel("src", "main", np.array([1.0], dtype=np.float32))
        assert np.isclose(res[0], 42.0)

        # Case 2: without inputs
        res2 = runner.execute_kernel("src", "main")
        assert np.isclose(res2[0], 42.0)


def test_metal_runner_launch_kernel():
    """Test MetalRunner.launch_kernel all branches and error handling."""
    runner = MetalRunner()

    # 1. Device is None -> returns immediately
    runner.device = None
    runner.compile_and_dispatch("kernel_src", "main", [16, 1, 1])

    # Setup mock Metal and device
    mock_metal = MagicMock()
    mock_dev = MagicMock()
    mock_lib = MagicMock()
    mock_func = MagicMock()
    mock_pipe = MagicMock()
    mock_queue = MagicMock()
    mock_cmd = MagicMock()
    mock_enc = MagicMock()

    mock_metal.MTLCreateSystemDefaultDevice.return_value = mock_dev
    mock_dev.newLibraryWithSource_options_error_.return_value = (mock_lib, None)
    mock_lib.newFunctionWithName_.return_value = mock_func
    mock_dev.newComputePipelineStateWithFunction_error_.return_value = (mock_pipe, None)
    mock_dev.newCommandQueue.return_value = mock_queue
    mock_queue.commandBuffer.return_value = mock_cmd
    mock_cmd.computeCommandEncoder.return_value = mock_enc

    runner.device = mock_dev

    with patch.dict(sys.modules, {"Metal": mock_metal}):
        # 2. Library is None (compilation failed)
        mock_dev.newLibraryWithSource_options_error_.return_value = (None, "comp_error")
        with pytest.raises(RuntimeError, match="Metal compilation failed"):
            runner.compile_and_dispatch("bad_src", "main", [16, 1, 1])

        # 3. Func is None (entry point not found)
        mock_dev.newLibraryWithSource_options_error_.return_value = (mock_lib, None)
        mock_lib.newFunctionWithName_.return_value = None
        with pytest.raises(RuntimeError, match="not found in library"):
            runner.compile_and_dispatch("src", "missing_entry", [16, 1, 1])

        # 4. Pipeline state is None
        mock_lib.newFunctionWithName_.return_value = mock_func
        mock_dev.newComputePipelineStateWithFunction_error_.return_value = (None, "pipe_error")
        with pytest.raises(RuntimeError, match="Failed to create pipeline state"):
            runner.compile_and_dispatch("src", "main", [16, 1, 1])

        # 5. Success with buffers and 3D grid_size
        mock_dev.newComputePipelineStateWithFunction_error_.return_value = (mock_pipe, None)
        valid_buf = ctypes.c_void_p(123456)
        null_buf = ctypes.c_void_p(0)
        none_buf = None
        buffers = [valid_buf, null_buf, none_buf]

        runner.compile_and_dispatch(
            msl_source="valid_msl",
            entry_point="main",
            workgroup_size=[16, 2, 2],
            buffers=buffers,
            grid_size=[64, 4, 2],
        )
        assert mock_enc.dispatchThreads_threadsPerThreadgroup_.called
        assert mock_cmd.commit.called
        assert mock_cmd.waitUntilCompleted.called

        # 6. Success without buffers and without grid_size (grid_size defaults to workgroup_size[0], 1, 1)
        mock_enc.reset_mock()
        runner.compile_and_dispatch(
            msl_source="valid_msl",
            entry_point="main",
            workgroup_size=[32, 1, 1],
            buffers=None,
            grid_size=None,
        )
        assert mock_enc.dispatchThreads_threadsPerThreadgroup_.called

        # 7. grid_size with length 1 (covering gy=1, gz=1 defaults)
        runner.compile_and_dispatch(
            msl_source="valid_msl",
            entry_point="main",
            workgroup_size=[32, 1, 1],
            buffers=None,
            grid_size=[32],
        )

        # 8. grid_size with length 2 (covering gz=1 default)
        runner.compile_and_dispatch(
            msl_source="valid_msl",
            entry_point="main",
            workgroup_size=[32, 1, 1],
            buffers=None,
            grid_size=[32, 2],
        )


def test_metal_coverage_additional_branches():
    """Verify missing branches in Metal code generation and runner."""
    # 1. _calculate_node_bytes with None node
    assert _calculate_node_bytes(None) == (1024, 4096)

    # 2. fused_elementwise with 0 inputs
    graph = IRGraph()
    node = IRNode(id="fused_0", op_type="FusedElementwise", inputs=[], attributes={"scalar_expr": "1.0f"})
    node.shape_metadata = (4,)
    graph.nodes = {"fused_0": node}
    gen = MetalCodeGenerator(graph)
    code = gen.generate()
    assert "kernel void fused_elementwise_0(" in code
    assert "out[idx] = 1.0f;" in code

    # 3. Runner branches: numpy array, pipeline_state is None, library is None
    mock_metal = MagicMock()
    mock_dev = MagicMock()
    mock_metal.MTLCreateSystemDefaultDevice.return_value = mock_dev
    with patch.dict(sys.modules, {"Metal": mock_metal}):
        runner = MetalRunner()
        runner.device = mock_dev

        # numpy array input in _prepare_device_buffers
        np_arr = np.array([1.0, 2.0], dtype=np.float32)
        runner._prepare_device_buffers({"arr": np_arr})
        assert mock_dev.newBufferWithBytes_length_options_.called

        # pipeline_state is None in _dispatch_compute_node
        mock_func = MagicMock()
        mock_lib = MagicMock()
        mock_lib.newFunctionWithName_.return_value = mock_func
        mock_dev.newComputePipelineStateWithFunction_error_.return_value = (None, "pipeline error")
        dummy_node = IRNode(id="n1", op_type="Add", inputs=["missing_input"])
        encoder = MagicMock()
        runner._dispatch_compute_node(dummy_node, 0, mock_lib, encoder, {})
        assert not encoder.setComputePipelineState_.called

        # pipeline_state valid but buffer missing in device_buffers -> covers branch 611->609
        mock_dev.newComputePipelineStateWithFunction_error_.return_value = (MagicMock(), None)
        runner._dispatch_compute_node(dummy_node, 0, mock_lib, encoder, {})

        # library is None in execute_graph -> raises RuntimeError
        runner.is_available = lambda: True
        mock_dev.newLibraryWithSource_options_error_.return_value = (None, "compilation failed")
        with pytest.raises(RuntimeError, match="Metal compilation failed"):
            runner.execute_graph(IRGraph(), {})
