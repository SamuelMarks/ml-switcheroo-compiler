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


def test_metal_reduce_sum_parallel_tree_reduction() -> None:
    """Verify MetalRunner.reduce_sum with SIMD and multi-pass parallel tree reduction."""
    runner = MetalRunner()

    # Empty array
    assert runner.reduce_sum(np.array([], dtype=np.float32)) == 0.0

    # Single element
    assert runner.reduce_sum(np.array([42.5], dtype=np.float32)) == 42.5

    # Small array (N = 10^3)
    arr_1e3 = np.ones(1000, dtype=np.float32) * 1.5
    res_1e3 = runner.reduce_sum(arr_1e3)
    assert np.isclose(res_1e3, float(np.sum(arr_1e3)), rtol=1e-5)

    # Medium array (N = 10^5)
    arr_1e5 = np.linspace(0.0, 10.0, 100000, dtype=np.float32)
    res_1e5 = runner.reduce_sum(arr_1e5)
    assert np.isclose(res_1e5, float(np.sum(arr_1e5)), rtol=1e-4)

    # Large array (N = 10^6 - 10^7)
    arr_large = np.ones(1000000, dtype=np.float32)
    res_large = runner.reduce_sum(arr_large)
    assert np.isclose(res_large, float(np.sum(arr_large)), rtol=1e-4)


def test_metal_reduce_sum_mock_multipass_dispatch() -> None:
    """Test Metal multi-pass execution loop with mocked PyObjC Metal pipeline."""
    import ctypes

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

    # Output buffer containing scalar 500.0
    out_val = (ctypes.c_float * 1)(500.0)
    mock_out_buf = MagicMock()
    mock_out_buf.contents.return_value = ctypes.addressof(out_val)
    mock_dev.newBufferWithLength_options_.return_value = mock_out_buf

    with patch.dict(sys.modules, {"Metal": mock_metal}):
        with patch.object(runner, "is_available", return_value=True):
            runner.device = mock_dev
            arr = np.ones(256, dtype=np.float32)
            val = runner._execute_metal_reduction_passes(arr, fallback=0.0)
            assert val == 500.0
            assert mock_enc.dispatchThreads_threadsPerThreadgroup_.called


def test_metal_generator_unsupported_op_error() -> None:
    """Verify MetalCodeGenerator raises BackendNotSupportedError on unsupported op_type."""
    import pytest

    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    g = IRGraph()
    bad_node = IRNode(id="n_bad", op_type="UnsupportedMetalOp", inputs=[])
    g.nodes = {"n_bad": bad_node}
    gen = MetalCodeGenerator(g)
    with pytest.raises(BackendNotSupportedError, match="not supported by metal backend"):
        gen.generate()


def test_metal_reduction_fallbacks_and_multipass() -> None:
    """Verify MetalRunner reduction fallbacks on library/func/pipeline failure, multi-pass dispatch, and exceptions."""
    import ctypes

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
    mock_dev.newCommandQueue.return_value = mock_queue
    mock_queue.commandBuffer.return_value = mock_cmd
    mock_cmd.computeCommandEncoder.return_value = mock_enc

    out_val = (ctypes.c_float * 1)(42.0)
    mock_out_buf = MagicMock()
    mock_out_buf.contents.return_value = ctypes.addressof(out_val)
    mock_dev.newBufferWithLength_options_.return_value = mock_out_buf

    with patch.dict(sys.modules, {"Metal": mock_metal}):
        runner.device = mock_dev

        # 1. library is None (line 859)
        mock_dev.newLibraryWithSource_options_error_.return_value = (None, "error")
        res1 = runner._execute_metal_reduction_passes(np.ones(10, dtype=np.float32), fallback=99.0)
        assert res1 == 99.0

        # 2. func is None (line 863)
        mock_dev.newLibraryWithSource_options_error_.return_value = (mock_lib, None)
        mock_lib.newFunctionWithName_.return_value = None
        res2 = runner._execute_metal_reduction_passes(np.ones(10, dtype=np.float32), fallback=88.0)
        assert res2 == 88.0

        # 3. pipeline_state is None (line 867)
        mock_lib.newFunctionWithName_.return_value = mock_func
        mock_dev.newComputePipelineStateWithFunction_error_.return_value = (None, "error")
        res3 = runner._execute_metal_reduction_passes(np.ones(10, dtype=np.float32), fallback=77.0)
        assert res3 == 77.0

        # 4. Multi-pass loop (lines 911-912: num_tgs > 1 on pass 1, then num_tgs == 1 on pass 2)
        mock_dev.newComputePipelineStateWithFunction_error_.return_value = (mock_pipe, None)
        # 2048 elements: items_per_tg = 1024 -> pass 1 has num_tgs = 2 > 1, pass 2 has num_tgs = 1
        arr_2048 = np.ones(2048, dtype=np.float32)
        res_mp = runner._execute_metal_reduction_passes(arr_2048, fallback=0.0)
        assert res_mp == 42.0

        # 5. reduce_sum exception fallback (lines 937-940)
        with patch.object(runner, "is_available", return_value=True):
            with patch.object(runner, "_execute_metal_reduction_passes", side_effect=RuntimeError("Metal failure")):
                arr_sum = np.array([1.0, 2.0, 3.0], dtype=np.float32)
                res_exc = runner.reduce_sum(arr_sum)
                assert res_exc == 6.0
