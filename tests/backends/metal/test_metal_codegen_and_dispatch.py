"""Unit tests for Metal code generation, memory buffer lifecycle, and kernel execution dispatch."""

from __future__ import annotations

import ctypes
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.metal.metal import (
    MetalCodeGenerator,
    MetalRunner,
    TensorLike,
    _calculate_node_bytes,
    _resolve_metal_grid_sizes,
)
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


class _MockTensor:
    """Mock tensor object implementing TensorLike Protocol."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize mock tensor.

        Args:
            shape (tuple[int, ...]): Shape tuple.
        """
        self._shape = shape

    @property
    def shape(self) -> tuple[int, ...]:
        """Get shape tuple.

        Returns:
            tuple[int, ...]: Shape tuple.
        """
        return self._shape


def test_tensor_like_protocol() -> None:
    """Verify TensorLike protocol conformance."""
    tensor = _MockTensor((2, 4))
    assert isinstance(tensor, TensorLike)
    assert tensor.shape == (2, 4)


def test_calculate_node_bytes_dtypes() -> None:
    """Verify all dtype and shape branches in _calculate_node_bytes."""
    # None node fallback
    assert _calculate_node_bytes(None) == (1024, 4096)

    # Node with empty shape metadata
    node_no_shape = IRNode(id="n0", op_type="Custom", inputs=[], shape_metadata=[])
    assert _calculate_node_bytes(node_no_shape) == (1024, 4096)

    # double (8 bytes)
    node_double = IRNode(id="n1", op_type="Custom", inputs=[], shape_metadata=[2, 4], attributes={"dtype": "double"})
    assert _calculate_node_bytes(node_double) == (8, 64)

    # int64 (8 bytes)
    node_int64 = IRNode(id="n2", op_type="Custom", inputs=[], shape_metadata=[3], attributes={"dtype": "int64"})
    assert _calculate_node_bytes(node_int64) == (3, 24)

    # int16 (2 bytes)
    node_int16 = IRNode(id="n3", op_type="Custom", inputs=[], shape_metadata=[4], attributes={"dtype": "int16"})
    assert _calculate_node_bytes(node_int16) == (4, 8)

    # bfloat16 (2 bytes)
    node_bf16 = IRNode(id="n4", op_type="Custom", inputs=[], shape_metadata=[5], attributes={"dtype": "bfloat16"})
    assert _calculate_node_bytes(node_bf16) == (5, 10)

    # uint8 (1 byte)
    node_u8 = IRNode(id="n5", op_type="Custom", inputs=[], shape_metadata=[6], attributes={"dtype": "uint8"})
    assert _calculate_node_bytes(node_u8) == (6, 6)

    # bool (1 byte)
    node_bool = IRNode(id="n6", op_type="Custom", inputs=[], shape_metadata=[7], attributes={"dtype": "bool"})
    assert _calculate_node_bytes(node_bool) == (7, 7)


def test_resolve_metal_grid_sizes_all_patterns() -> None:
    """Verify 3D grid size resolution across matmul, batchmatmul, conv2d, and defaults."""
    graph = IRGraph()

    # Matmul with missing inputs
    node_mm_no_inp = IRNode(id="mm_none", op_type="MatMul", inputs=[])
    graph.nodes["mm_none"] = node_mm_no_inp
    assert _resolve_metal_grid_sizes(node_mm_no_inp, 16, graph) == ("1", "1", "1")

    # Matmul with 1D input shapes
    in0 = IRNode(id="in0", op_type="Input", shape_metadata=[10])
    in1 = IRNode(id="in1", op_type="Input", shape_metadata=[])
    node_mm_1d = IRNode(id="mm_1d", op_type="dot", inputs=["in0", "in1"])
    graph.nodes["in0"] = in0
    graph.nodes["in1"] = in1
    graph.nodes["mm_1d"] = node_mm_1d
    assert _resolve_metal_grid_sizes(node_mm_1d, 10, graph) == ("1", "1", "1")

    # Batchmatmul with 0 inputs
    node_bmm_empty = IRNode(id="bmm_empty", op_type="BatchMatMul", inputs=[])
    assert _resolve_metal_grid_sizes(node_bmm_empty, 100, graph) == ("1", "1", "1")

    # Batchmatmul with 2D inputs (len(in0_shape) <= 2)
    in_b0 = IRNode(id="in_b0", op_type="Input", shape_metadata=[8, 16])
    in_b1 = IRNode(id="in_b1", op_type="Input", shape_metadata=[16, 32])
    node_bmm_2d = IRNode(id="bmm_2d", op_type="batchmatmul", inputs=["in_b0", "in_b1"])
    graph.nodes["in_b0"] = in_b0
    graph.nodes["in_b1"] = in_b1
    graph.nodes["bmm_2d"] = node_bmm_2d
    assert _resolve_metal_grid_sizes(node_bmm_2d, 8 * 32, graph) == ("32", "8", "1")

    # Batchmatmul with 3D inputs (len(in0_shape) > 2)
    in_b3d_0 = IRNode(id="in_b3d_0", op_type="Input", shape_metadata=[4, 8, 16])
    in_b3d_1 = IRNode(id="in_b3d_1", op_type="Input", shape_metadata=[4, 16, 32])
    node_bmm_3d = IRNode(id="bmm_3d", op_type="batchmatmul", inputs=["in_b3d_0", "in_b3d_1"])
    graph.nodes["in_b3d_0"] = in_b3d_0
    graph.nodes["in_b3d_1"] = in_b3d_1
    graph.nodes["bmm_3d"] = node_bmm_3d
    assert _resolve_metal_grid_sizes(node_bmm_3d, 4 * 8 * 32, graph) == ("32", "8", "4")

    # Conv2D with 4D input and output shapes
    conv_in = IRNode(id="conv_in", op_type="Input", shape_metadata=[2, 3, 32, 32])
    conv_node = IRNode(id="conv_node", op_type="conv2d", inputs=["conv_in"], shape_metadata=[2, 64, 16, 16])
    graph.nodes["conv_in"] = conv_in
    graph.nodes["conv_node"] = conv_node
    gx, gy, gz = _resolve_metal_grid_sizes(conv_node, 2 * 64 * 16 * 16, graph)
    assert (gx, gy, gz) == ("256", "64", "2")

    # Conv2D with empty inputs and empty shapes
    conv_empty = IRNode(id="conv_empty", op_type="conv2d", inputs=[], shape_metadata=[])
    assert _resolve_metal_grid_sizes(conv_empty, 1, graph) == ("1", "1", "1")

    # Default elementwise op
    default_node = IRNode(id="elem_node", op_type="Relu", inputs=[], shape_metadata=[512])
    assert _resolve_metal_grid_sizes(default_node, 512, graph) == ("512", "1", "1")


def test_free_dead_buffers_lifecycle() -> None:
    """Verify dead buffer deallocation filtering for inputs, outputs, and missing buffers."""
    graph = IRGraph()
    gen = MetalCodeGenerator(graph)

    node = IRNode(id="node_a", op_type="Custom", inputs=["in_dead", "in_out", "in_inp", "in_missing"])
    node_buffers = {"in_dead": "buf_dead", "in_out": "buf_out", "in_inp": "buf_inp"}
    last_consumer = {
        "in_dead": "node_a",
        "in_out": "node_a",
        "in_inp": "node_a",
        "in_missing": "node_a",
    }
    outputs_set = {"in_out"}
    inputs_set = {"in_inp"}
    msl: list[str] = []

    gen._free_dead_buffers(node, node_buffers, last_consumer, outputs_set, inputs_set, msl)
    assert msl == ["    buf_dead = None"]


def test_metal_generator_synthesize_template_more_than_two_inputs() -> None:
    """Verify template synthesis for operations with 2 inputs and more than 2 inputs."""
    graph = IRGraph()
    n_in0 = IRNode(id="in0", op_type="Input")
    n_in1 = IRNode(id="in1", op_type="Input")
    n_in2 = IRNode(id="in2", op_type="Input")
    # 2 inputs unmapped op
    n_binary = IRNode(id="binary", op_type="BinaryCustomOp", inputs=["in0", "in1"], shape_metadata=[128])
    # 3 inputs unmapped op
    n_ternary = IRNode(id="ternary", op_type="TernaryCustomOp", inputs=["in0", "in1", "in2"], shape_metadata=[128])
    graph.nodes = {"in0": n_in0, "in1": n_in1, "in2": n_in2, "binary": n_binary, "ternary": n_ternary}
    graph.inputs = ["in0", "in1", "in2"]
    graph.outputs = ["binary", "ternary"]

    gen = MetalCodeGenerator(graph)
    msl = gen.generate()
    assert "kernel void binarycustomop_kernel(" in msl
    assert "kernel void ternarycustomop_kernel(" in msl
    assert "const device float* in_0 [[buffer(0)]]" in msl
    assert "const device float* in_1 [[buffer(1)]]" in msl
    assert "const device float* in_2 [[buffer(2)]]" in msl
    assert "device float* C [[buffer(3)]]" in msl
    assert "constant uint& N [[buffer(4)]]" in msl


def test_metal_generator_fused_elementwise_codegen() -> None:
    """Verify code generation and MSL definitions for FusedElementwise with 0 inputs and 2 inputs."""
    # 0 inputs
    graph0 = IRGraph()
    fused_node0 = IRNode(id="fused_0", op_type="FusedElementwise", inputs=[], attributes={"scalar_expr": "1.0f"}, shape_metadata=[64])
    graph0.nodes = {"fused_0": fused_node0}
    graph0.outputs = ["fused_0"]

    gen0 = MetalCodeGenerator(graph0)
    msl0 = gen0.generate()
    assert "kernel void fused_elementwise_0(" in msl0
    assert "out[idx] = 1.0f;" in msl0
    assert "grid_size_0 = MTLSize(64, 1, 1)" in msl0

    # 2 inputs (covers lines 183-184)
    graph2 = IRGraph()
    in_a = IRNode(id="in_a", op_type="Input", shape_metadata=[64])
    in_b = IRNode(id="in_b", op_type="Input", shape_metadata=[64])
    fused_node2 = IRNode(
        id="fused_2",
        op_type="FusedElementwise",
        inputs=["in_a", "in_b"],
        attributes={"scalar_expr": "in0 + in1"},
        shape_metadata=[64],
    )
    graph2.nodes = {"in_a": in_a, "in_b": in_b, "fused_2": fused_node2}
    graph2.inputs = ["in_a", "in_b"]
    graph2.outputs = ["fused_2"]

    gen2 = MetalCodeGenerator(graph2)
    msl2 = gen2.generate()
    assert "encoder.setBuffer_offset_atIndex_(" in msl2
    assert "pipeline_fused_elementwise_0" in msl2


def test_metal_generator_emit_node_skipped_for_none_template() -> None:
    """Verify _emit_node skips emission when _ensure_template returns None."""
    graph = IRGraph()
    gen = MetalCodeGenerator(graph)
    node_input = IRNode(id="in_node", op_type="Input")
    msl: list[str] = []
    gen._emit_node(node_input, 0, {}, {}, set(), set(), msl)
    assert msl == []

    # And in _emit_kernel_definitions when template is None
    with patch.object(gen, "_ensure_template", return_value=None):
        gen.graph.nodes = {"n": IRNode(id="n", op_type="SomeOp")}
        gen._emit_kernel_definitions(msl)
        assert msl == []


def test_metal_runner_init_and_availability_exceptions() -> None:
    """Verify MetalRunner __init__ and is_available exception fallbacks."""
    with patch.dict(sys.modules, {"Metal": None}):
        with patch("ctypes.util.find_library", side_effect=OSError("find failed")):
            runner = MetalRunner()
            assert runner.device is None
            assert runner.objc is None
            assert runner.metal is None

        with patch("ctypes.cdll.LoadLibrary", side_effect=OSError("load failed")):
            assert MetalRunner.is_available() is False


def test_metal_runner_compile_and_dispatch_branches() -> None:
    """Verify all branches and error handling in compile_and_dispatch."""
    runner = MetalRunner()

    # Device is None: no-op
    runner.device = None
    runner.compile_and_dispatch("kernel void k() {}", "k", [256, 1, 1])

    # Setup mock Metal runtime
    mock_metal = MagicMock()
    mock_dev = MagicMock()
    mock_lib = MagicMock()
    mock_func = MagicMock()
    mock_pipe = MagicMock()
    mock_queue = MagicMock()
    mock_cmd = MagicMock()
    mock_enc = MagicMock()

    runner.device = mock_dev
    mock_dev.newCommandQueue.return_value = mock_queue
    mock_queue.commandBuffer.return_value = mock_cmd
    mock_cmd.computeCommandEncoder.return_value = mock_enc

    with patch.dict(sys.modules, {"Metal": mock_metal}):
        # Compilation error
        mock_dev.newLibraryWithSource_options_error_.return_value = (None, "compilation syntax error")
        with pytest.raises(RuntimeError, match="Metal compilation failed"):
            runner.compile_and_dispatch("invalid", "k", [256, 1, 1])

        # Entry point not found
        mock_dev.newLibraryWithSource_options_error_.return_value = (mock_lib, None)
        mock_lib.newFunctionWithName_.return_value = None
        with pytest.raises(RuntimeError, match="Metal entry point 'missing_k' not found"):
            runner.compile_and_dispatch("valid", "missing_k", [256, 1, 1])

        # Pipeline state error
        mock_lib.newFunctionWithName_.return_value = mock_func
        mock_dev.newComputePipelineStateWithFunction_error_.return_value = (None, "pipeline error")
        with pytest.raises(RuntimeError, match="Failed to create pipeline state"):
            runner.compile_and_dispatch("valid", "k", [256, 1, 1])

        # Successful dispatch with buffers and custom grid_sizes
        mock_dev.newComputePipelineStateWithFunction_error_.return_value = (mock_pipe, None)
        ptr_valid = ctypes.c_void_p(0x1000)
        ptr_null = ctypes.c_void_p(0)
        buffers = [ptr_valid, None, ptr_null]

        # With 3D grid_size
        runner.compile_and_dispatch("valid", "k", [64, 1, 1], buffers=buffers, grid_size=[128, 2, 4])
        assert mock_enc.setBuffer_offset_atIndex_.called
        assert mock_enc.dispatchThreads_threadsPerThreadgroup_.called
        assert mock_cmd.commit.called

        # With empty grid_size and 1D/2D grid_sizes
        runner.compile_and_dispatch("valid", "k", [64, 1, 1], buffers=None, grid_size=None)
        runner.compile_and_dispatch("valid", "k", [64, 1, 1], buffers=None, grid_size=[128])
        runner.compile_and_dispatch("valid", "k", [64, 1, 1], buffers=None, grid_size=[128, 2])


def test_metal_runner_allocate_buffer_all_branches() -> None:
    """Verify allocate_buffer branches with PyObjC device, ctypes device, and fallback creation."""
    runner = MetalRunner()

    # Branch 1a: Device has newBufferWithLength_options_ and returns buffer with contents
    mock_pyobjc_dev = MagicMock()
    mock_buf = MagicMock()
    mock_buf.contents.return_value = 0x9999
    mock_pyobjc_dev.newBufferWithLength_options_.return_value = mock_buf
    runner.device = mock_pyobjc_dev
    runner.metal = None
    runner.objc = None
    ptr_pyobjc = runner.allocate_buffer(64)
    assert ptr_pyobjc is not None and ptr_pyobjc.value == 0x9999

    # Branch 1b: Device has newBufferWithLength_options_ but returns None (and metal is None)
    mock_pyobjc_dev.newBufferWithLength_options_.return_value = None
    assert runner.allocate_buffer(64) is None

    # Branch 1c: Device has newBufferWithLength_options_ but buffer has no contents attribute
    mock_bad_buf = object()
    mock_pyobjc_dev.newBufferWithLength_options_.return_value = mock_bad_buf
    assert runner.allocate_buffer(64) is None

    # Branch 2a: Device has no newBufferWithLength_options_ and objc is None (530->546)
    runner.device = object()
    assert runner.allocate_buffer(64) is None

    # Branch 2b: Device is an integer pointer with objc runtime
    mock_objc = MagicMock()
    mock_objc.sel_registerName.return_value = 0x1111
    mock_objc.objc_msgSend.side_effect = [0x2222, 0x3333]  # buf_ptr, contents_ptr
    runner.device = 0x1234
    runner.objc = mock_objc
    ptr = runner.allocate_buffer(128)
    assert ptr is not None and ptr.value == 0x3333

    # Branch 2c: Device is ctypes.c_void_p and buf_ptr is 0 (539->546)
    mock_objc.objc_msgSend.side_effect = [0]  # buf_ptr = 0
    runner.device = ctypes.c_void_p(0x1234)
    assert runner.allocate_buffer(128) is None

    # Branch 3a: Device is None, metal and objc available, func is None (548->567)
    mock_metal = MagicMock(spec=[])
    runner.device = None
    runner.metal = mock_metal
    runner.objc = mock_objc
    assert runner.allocate_buffer(256) is None

    # Branch 3b: Device is None, func returns dev_ptr = 0 (551->567)
    mock_metal = MagicMock()
    mock_func = MagicMock(return_value=0)
    mock_metal.MTLCreateSystemDefaultDevice = mock_func
    runner.metal = mock_metal
    assert runner.allocate_buffer(256) is None

    # Branch 3c: Device is None, dev_ptr != 0, but buf_ptr is 0 (560->567)
    mock_func.return_value = 0x5555
    mock_objc.objc_msgSend.side_effect = [0]  # buf_ptr = 0
    assert runner.allocate_buffer(256) is None

    # Branch 3d: Device is None, dev_ptr != 0, and buf_ptr != 0
    runner.device = None
    mock_objc.objc_msgSend.side_effect = [0x6666, 0x7777]
    ptr3 = runner.allocate_buffer(256)
    assert ptr3 is not None and ptr3.value == 0x7777

    # Branch 4: Fallback when both are None
    runner.device = None
    runner.metal = None
    runner.objc = None
    assert runner.allocate_buffer(256) is None


def test_metal_runner_read_write_free_buffer() -> None:
    """Verify buffer read, write, and free operations and null-safety."""
    runner = MetalRunner()

    # Free None or valid pointer
    runner.free_buffer(None)
    runner.free_buffer(ctypes.c_void_p(0x1000))

    # Write to null pointer
    runner.write_buffer(None, b"test")
    runner.write_buffer(ctypes.c_void_p(0), b"test")

    # Read from null pointer
    assert runner.read_buffer(None, 4) == b""
    assert runner.read_buffer(ctypes.c_void_p(0), 4) == b""

    # Read and write with actual memory buffer
    raw_mem = ctypes.create_string_buffer(16)
    buf_ptr = ctypes.c_void_p(ctypes.addressof(raw_mem))
    runner.write_buffer(buf_ptr, b"12345678")
    assert runner.read_buffer(buf_ptr, 8) == b"12345678"


def test_metal_runner_execute_kernel_error_branches() -> None:
    """Verify error branches in execute_kernel."""
    runner = MetalRunner()
    mock_metal = MagicMock()

    # No device available
    mock_metal.MTLCreateSystemDefaultDevice.return_value = None
    runner.device = None
    with patch.dict(sys.modules, {"Metal": mock_metal}):
        with pytest.raises(RuntimeError, match="No Metal device available"):
            runner.execute_kernel("kernel", "k")

    # Compilation error
    mock_dev = MagicMock()
    mock_metal.MTLCreateSystemDefaultDevice.return_value = mock_dev
    mock_dev.newLibraryWithSource_options_error_.return_value = (None, "compile error")
    with patch.dict(sys.modules, {"Metal": mock_metal}):
        with pytest.raises(RuntimeError, match="Metal compilation failed"):
            runner.execute_kernel("kernel", "k")

    # Missing entry point
    mock_lib = MagicMock()
    mock_dev.newLibraryWithSource_options_error_.return_value = (mock_lib, None)
    mock_lib.newFunctionWithName_.return_value = None
    with patch.dict(sys.modules, {"Metal": mock_metal}):
        with pytest.raises(RuntimeError, match="Metal entry point 'missing' not found"):
            runner.execute_kernel("kernel", "missing")

    # Pipeline error
    mock_func = MagicMock()
    mock_lib.newFunctionWithName_.return_value = mock_func
    mock_dev.newComputePipelineStateWithFunction_error_.return_value = (None, "pipe error")
    with patch.dict(sys.modules, {"Metal": mock_metal}):
        with pytest.raises(RuntimeError, match="Failed to create pipeline state"):
            runner.execute_kernel("kernel", "k")


def test_metal_runner_execute_graph() -> None:
    """Verify execute_graph across valid execution and error conditions."""
    runner = MetalRunner()

    graph = IRGraph()
    in_node = IRNode(id="in_0", op_type="Input", shape_metadata=[4])
    add_node = IRNode(id="n_add", op_type="Add", inputs=["in_0"], shape_metadata=[4])
    fused_node = IRNode(id="n_fused", op_type="FusedElementwise", inputs=["n_add"], shape_metadata=[4], attributes={"scalar_expr": "in0 * 2.0f"})
    graph.nodes = {"in_0": in_node, "n_add": add_node, "n_fused": fused_node}
    graph.inputs = ["in_0"]
    graph.outputs = ["n_fused", "unallocated_output"]

    # Backend not available
    with patch.object(runner, "is_available", return_value=False):
        with pytest.raises(BackendNotSupportedError):
            runner.execute_graph(graph, {})

    # Metal compilation failure
    mock_metal = MagicMock()
    mock_dev = MagicMock()
    mock_dev.newLibraryWithSource_options_error_.return_value = (None, "compilation error")
    runner.device = mock_dev
    with patch.object(runner, "is_available", return_value=True):
        with patch.dict(sys.modules, {"Metal": mock_metal}):
            with pytest.raises(RuntimeError, match="Metal compilation failed"):
                runner.execute_graph(graph, {})

    # Successful execution with bytes, memoryview, list, and array inputs
    mock_lib = MagicMock()
    mock_dev.newLibraryWithSource_options_error_.return_value = (mock_lib, None)
    mock_queue = MagicMock()
    mock_cmd = MagicMock()
    mock_enc = MagicMock()
    mock_dev.newCommandQueue.return_value = mock_queue
    mock_queue.commandBuffer.return_value = mock_cmd
    mock_cmd.computeCommandEncoder.return_value = mock_enc

    # Output buffer setup
    out_data = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)
    mock_out_buf = MagicMock()
    mock_out_buf.contents.return_value = out_data.ctypes.data
    mock_dev.newBufferWithLength_options_.return_value = mock_out_buf

    # Test library returns function for nodes
    mock_pipe = MagicMock()
    mock_func = MagicMock()
    mock_lib.newFunctionWithName_.return_value = mock_func
    mock_dev.newComputePipelineStateWithFunction_error_.return_value = (mock_pipe, None)

    inputs = {
        "bytes_in": b"\x00\x00\x80?\x00\x00\x00@",
        "memview_in": memoryview(b"\x00\x00\x80?"),
        "list_in": [1.0, 2.0, 3.0],
        "numpy_in": np.array([1.0, 2.0], dtype=np.float32),
    }

    with patch.object(runner, "is_available", return_value=True):
        with patch.dict(sys.modules, {"Metal": mock_metal}):
            res = runner.execute_graph(graph, inputs)
            assert "n_fused" in res
            assert len(res["n_fused"]) == 16  # 4 floats * 4 bytes
            assert mock_enc.dispatchThreads_threadsPerThreadgroup_.called
            assert mock_cmd.commit.called

            # Also verify node dispatch early exits when func is None or pipeline_state is None
            mock_lib.newFunctionWithName_.return_value = None
            res2 = runner.execute_graph(graph, inputs)
            assert isinstance(res2, dict)

            mock_lib.newFunctionWithName_.return_value = mock_func
            mock_dev.newComputePipelineStateWithFunction_error_.return_value = (None, None)
            res3 = runner.execute_graph(graph, inputs)
            assert isinstance(res3, dict)
