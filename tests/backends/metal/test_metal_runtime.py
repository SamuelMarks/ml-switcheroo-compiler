"""Exhaustive unit and runtime tests for Metal backend and MSL templates."""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import yaml

from ml_switcheroo_compiler.backends.hardware_config_models import HardwareTemplatesConfig
from ml_switcheroo_compiler.backends.metal.metal import MetalCodeGenerator, MetalRunner
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_msl_templates_consolidation() -> None:
    """Test that msl_templates.yaml covers all core math, reduction, and vision operations."""
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "ml_switcheroo_compiler", "backends", "metal", "msl_templates.yaml")
    assert os.path.exists(yaml_path)

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    cfg = HardwareTemplatesConfig(**data)
    templates = cfg.templates

    # Unary ops
    for op in ("abs", "cos", "exp", "gelu", "log", "neg", "relu", "sigmoid", "silu", "sin", "sqrt", "tanh"):
        assert op in templates
        assert "kernel void" in templates[op].body
        assert len(templates[op].workgroup_size) == 3

    # Binary ops
    for op in ("add", "sub", "mul", "div", "pow", "maximum", "minimum"):
        assert op in templates
        assert "kernel void" in templates[op].body

    # Reductions
    for op in ("reducesum", "reducemean", "reducemax", "reducemin", "sum"):
        assert op in templates
        assert "kernel void" in templates[op].body

    # Matrix multiplication and vision
    for op in ("matmul", "batchmatmul", "conv2d", "depthwise_conv2d", "maxpool2d", "avgpool2d", "multihead_attention"):
        assert op in templates
        assert "kernel void" in templates[op].body


def test_metal_code_generator_comprehensive() -> None:
    """Test MetalCodeGenerator code generation for multi-node graph using consolidated templates."""
    graph = IRGraph()
    graph.inputs = ["in_a", "in_b"]
    n_add = IRNode(id="n_add", op_type="Add", inputs=["in_a", "in_b"], shape_metadata=[1024])
    n_relu = IRNode(id="n_relu", op_type="Relu", inputs=["n_add"], shape_metadata=[1024])
    n_conv = IRNode(id="n_conv", op_type="Conv2D", inputs=["n_relu"], shape_metadata=[1024])
    graph.nodes = {"n_add": n_add, "n_relu": n_relu, "n_conv": n_conv}

    gen = MetalCodeGenerator(graph)
    msl = gen.generate()

    assert "kernel void add" in msl
    assert "kernel void relu" in msl
    assert "kernel void conv2d" in msl
    assert "evaluate_metal" in msl
    assert "command_buffer.commit()" in msl


def test_metal_runner_ctypes_availability() -> None:
    """Test MetalRunner.is_available via ctypes fallback when PyObjC is not installed."""
    mock_metal_lib = MagicMock()
    mock_metal_lib.MTLCreateSystemDefaultDevice.return_value = 0x123456

    with patch.dict(sys.modules, {"Metal": None}):
        with patch("ctypes.cdll.LoadLibrary", return_value=mock_metal_lib):
            assert MetalRunner.is_available() is True

        mock_empty_lib = MagicMock(spec=[])
        with patch("ctypes.cdll.LoadLibrary", return_value=mock_empty_lib):
            assert MetalRunner.is_available() is False

        with patch("ctypes.cdll.LoadLibrary", side_effect=Exception("load failed")):
            assert MetalRunner.is_available() is False


def test_metal_runner_ctypes_buffer_and_execution() -> None:
    """Test MetalRunner execution harness using ctypes pointers and mocked command buffer."""
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
    mock_pipe.threadExecutionWidth.return_value = 64

    # Setup output buffer
    out_data = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    mock_out_buf = MagicMock()
    mock_out_buf.contents.return_value = out_data.ctypes.data
    mock_dev.newBufferWithLength_options_.return_value = mock_out_buf

    with patch.dict(sys.modules, {"Metal": mock_metal}):
        in_a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        in_b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        res = runner.execute_kernel("kernel_source", "add_kernel", in_a, in_b)

        assert len(res) == 3
        assert np.allclose(res, [10.0, 20.0, 30.0])
        assert mock_enc.dispatchThreads_threadsPerThreadgroup_.called
        assert mock_cmd.commit.called
        assert mock_cmd.waitUntilCompleted.called
