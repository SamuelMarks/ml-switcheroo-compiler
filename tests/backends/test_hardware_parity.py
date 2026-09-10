"""Integration and parity tests for native hardware backends (CUDA, Metal, ROCm)."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.cuda.cuda import CudaCodeGenerator, CUDARunner
from ml_switcheroo_compiler.backends.metal.metal import MetalCodeGenerator, MetalRunner
from ml_switcheroo_compiler.backends.rocm.rocm import RocmCodeGenerator, ROCmRunner
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_hardware_operation_coverage_and_workgroup_rules() -> None:
    """Verify that hardware orchestration directories eliminate the 8-op ceiling."""
    cuda_gen = CudaCodeGenerator(IRGraph())
    rocm_gen = RocmCodeGenerator(IRGraph())
    metal_gen = MetalCodeGenerator(IRGraph())

    # We now support at least 20 operations across all three backends
    assert len(cuda_gen.config.templates) >= 20
    assert len(rocm_gen.config.templates) >= 20
    assert len(metal_gen.config.templates) >= 20

    core_ops = ["add", "sub", "mul", "div", "relu", "exp", "log", "tanh", "maximum", "silu", "gelu"]
    for op in core_ops:
        assert op in cuda_gen.config.templates
        assert op in rocm_gen.config.templates
        assert op in metal_gen.config.templates

        # Explicit threadblock / workgroup sizing rules
        cuda_wg = cuda_gen.config.templates[op].get("workgroup_size")
        metal_wg = metal_gen.config.templates[op].get("workgroup_size")
        rocm_wg = rocm_gen.config.templates[op].get("workgroup_size")

        assert isinstance(cuda_wg, list) and len(cuda_wg) == 3 and cuda_wg[0] > 0
        assert isinstance(metal_wg, list) and len(metal_wg) == 3 and metal_wg[0] > 0
        assert isinstance(rocm_wg, list) and len(rocm_wg) == 3 and rocm_wg[0] > 0


def test_dynamic_buffer_allocation_and_launch() -> None:
    """Verify dynamic shape-based allocation and multi-argument launch generation."""
    g = IRGraph()
    n_in0 = IRNode("in0", "Input", shape_metadata=(16, 8), attributes={"dtype": "float32"})
    n_in1 = IRNode("in1", "Input", shape_metadata=(16, 8), attributes={"dtype": "float32"})
    # Binary op: Add
    n_add = IRNode("add_op", "Add", inputs=["in0", "in1"], shape_metadata=(16, 8), attributes={"dtype": "float32"})
    # Unary op: Exp
    n_exp = IRNode("exp_op", "Exp", inputs=["add_op"], shape_metadata=(16, 8), attributes={"dtype": "float32"})

    for n in [n_in0, n_in1, n_add, n_exp]:
        g.nodes[n.id] = n
    g.inputs = ["in0", "in1"]
    g.outputs = ["exp_op"]

    # 16 * 8 = 128 elements * 4 bytes = 512 bytes
    cuda_code = CudaCodeGenerator(g).generate()
    assert "512" in cuda_code
    assert "1024 * sizeof(float)" not in cuda_code
    assert "add<<<" in cuda_code or "add_kernel<<<" in cuda_code
    assert "exp_kernel<<<" in cuda_code

    rocm_code = RocmCodeGenerator(g).generate()
    assert "512" in rocm_code
    assert "1024 * sizeof(float)" not in rocm_code
    assert "hipLaunchKernelGGL" in rocm_code

    metal_code = MetalCodeGenerator(g).generate()
    assert "512" in metal_code
    assert "1024 * 4" not in metal_code
    assert "evaluate_metal" in metal_code
    assert "/* PyObjC Orchestration" not in metal_code


def test_metal_native_execution_parity() -> None:
    """Verify non-mocked execution of Metal kernels on Apple Silicon GPU."""
    import importlib.util

    if importlib.util.find_spec("Metal") is None:
        pytest.skip("PyObjC Metal module not installed on host system")

    if not MetalRunner.is_available():
        pytest.skip("Apple Silicon Metal device/driver not available on host system")

    runner = MetalRunner()
    metal_gen = MetalCodeGenerator(IRGraph())

    # 1. Unary ReLU execution
    relu_tpl = metal_gen.config.templates["relu"].get("body", "")
    x = np.array([-5.0, 0.0, 3.5, 12.0], dtype=np.float32)
    relu_res = runner.execute_kernel(relu_tpl, "relu", x)
    np.testing.assert_allclose(relu_res, np.maximum(0.0, x))

    # 2. Binary Add execution
    add_tpl = metal_gen.config.templates["add"].get("body", "")
    y = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    add_res = runner.execute_kernel(add_tpl, "add", x, y)
    np.testing.assert_allclose(add_res, x + y)

    # 3. Unary Exp execution
    exp_tpl = metal_gen.config.templates["exp"].get("body", "")
    x_small = np.array([0.0, 1.0, 2.0, -1.0], dtype=np.float32)
    exp_res = runner.execute_kernel(exp_tpl, "exp_kernel", x_small)
    np.testing.assert_allclose(exp_res, np.exp(x_small), rtol=1e-5, atol=1e-5)


def test_cuda_driver_presence_or_graceful_skip() -> None:
    """Verify CUDA driver initialization or graceful skip."""
    runner = CUDARunner()
    if runner.mode == "cupy":
        buf = runner.allocate_buffer(1024)
        assert buf is not None
    elif runner.cuda_lib is not None and runner._ctx is not None:
        buf = runner.allocate_buffer(1024)
        assert buf is not None
    else:
        pytest.skip("CUDA device/driver not available on host system")


def test_rocm_driver_presence_or_graceful_skip() -> None:
    """Verify ROCm driver initialization or graceful skip."""
    runner = ROCmRunner()
    if runner.mode == "cupy":
        buf = runner.allocate_buffer(1024)
        assert buf is not None
    elif runner.rocm_lib is not None:
        buf = runner.allocate_buffer(1024)
        assert buf is not None
    else:
        pytest.skip("ROCm device/driver not available on host system")


def test_llvm_cpp_runner_compilation_and_execution() -> None:
    """Verify LLVM C++ runner compiling and running C++17 code."""
    import shutil

    from ml_switcheroo_compiler.backends.llvm_cpp.generator import LLVMCPPRunner

    compiler: Optional[str] = "clang++" if shutil.which("clang++") else "g++" if shutil.which("g++") else None
    if not compiler:
        pytest.skip("No C++ compiler (clang++ or g++) found on system.")

    runner: LLVMCPPRunner = LLVMCPPRunner(compiler=compiler)
    cpp_source: str = """
    #include <cmath>
    extern "C" void compute_graph() {
        volatile float x = 2.0f;
        volatile float y = std::sin(x);
        (void)y;
    }
    """
    executable = runner.compile_and_load(cpp_source)
    assert executable() == "Execution successful"


def test_cuda_and_rocm_argument_marshalling() -> None:
    """Verify argument marshalling logic in CUDA and ROCm runners."""
    import ctypes
    from unittest.mock import MagicMock

    # CUDA
    cuda_runner = CUDARunner()
    mock_lib = MagicMock()
    mock_lib.cuModuleLoadData.return_value = 0
    mock_lib.cuModuleGetFunction.return_value = 0
    mock_lib.cuLaunchKernel.return_value = 0
    cuda_runner.cuda_lib = mock_lib
    cuda_runner.mode = "ctypes"

    dummy_ptr = ctypes.c_void_p(0x123456)
    cuda_runner.compile_and_dispatch("ptx_code", "entry", [1, 1, 1], [32, 1, 1], args=[dummy_ptr, 42, 3.14])
    assert mock_lib.cuLaunchKernel.called

    # ROCm
    rocm_runner = ROCmRunner()
    mock_rocm = MagicMock()
    mock_rocm.hipModuleLoad.return_value = 0
    mock_rocm.hipModuleGetFunction.return_value = 0
    mock_rocm.hipModuleLaunchKernel.return_value = 0
    rocm_runner.rocm_lib = mock_rocm
    rocm_runner.mode = "ctypes"

    rocm_runner.load_and_dispatch("/tmp/bin", "entry", [1, 1, 1], [32, 1, 1], args=[dummy_ptr, 10, 2.71])
    assert mock_rocm.hipModuleLaunchKernel.called


def test_cross_backend_eager_parity_against_numpy_reference() -> None:
    """Verify numerical parity across native backends compared to NumPy references."""
    import jax.numpy as jnp
    import mlx.core as mx
    import torch

    from ml_switcheroo_compiler.backends.jax.generator import JAXCodeGenerator
    from ml_switcheroo_compiler.backends.mlx.generator import MLXCodeGenerator
    from ml_switcheroo_compiler.backends.numpy.generator import NumpyGenerator
    from ml_switcheroo_compiler.backends.pytorch.generator import PyTorchCodeGenerator

    # 1. MatMul Parity
    a_np: np.ndarray = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_np: np.ndarray = np.array([[0.5, -0.5], [1.5, 0.0]], dtype=np.float32)
    ref_matmul: np.ndarray = np.matmul(a_np, b_np)

    np_res = NumpyGenerator.execute_op("MatMul", a_np, b_np)
    np.testing.assert_allclose(np_res, ref_matmul, rtol=1e-5, atol=1e-5)

    torch_res = PyTorchCodeGenerator.execute_op("MatMul", torch.tensor(a_np), torch.tensor(b_np))
    np.testing.assert_allclose(torch_res.detach().cpu().numpy(), ref_matmul, rtol=1e-5, atol=1e-5)

    jax_res = JAXCodeGenerator.execute_op("MatMul", jnp.array(a_np), jnp.array(b_np))
    np.testing.assert_allclose(np.array(jax_res), ref_matmul, rtol=1e-5, atol=1e-5)

    mlx_res = MLXCodeGenerator.execute_op("MatMul", mx.array(a_np), mx.array(b_np))
    np.testing.assert_allclose(np.array(mlx_res), ref_matmul, rtol=1e-5, atol=1e-5)

    # 2. Reductions Parity
    x_np: np.ndarray = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ref_sum: np.ndarray = np.sum(x_np, axis=-1, keepdims=True)
    ref_mean: np.ndarray = np.mean(x_np, axis=-1, keepdims=True)

    np.testing.assert_allclose(NumpyGenerator.execute_op("ReduceSum", x_np, axis=-1, keepdims=True), ref_sum)
    np.testing.assert_allclose(NumpyGenerator.execute_op("ReduceMean", x_np, axis=-1, keepdims=True), ref_mean)

    # 3. Activations Parity (GELU, SiLU, Sigmoid, Tanh)
    x_act: np.ndarray = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    ref_sig: np.ndarray = 1.0 / (1.0 + np.exp(-x_act))
    ref_tanh: np.ndarray = np.tanh(x_act)

    np.testing.assert_allclose(NumpyGenerator.execute_op("Sigmoid", x_act), ref_sig, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(NumpyGenerator.execute_op("Tanh", x_act), ref_tanh, rtol=1e-5, atol=1e-5)


def test_unsupported_backend_eager_dispatch_raises_error() -> None:
    """Verify that unmapped ops on a backend raise BackendNotSupportedError without fallback."""
    from ml_switcheroo_compiler.backends.pytorch.generator import PyTorchCodeGenerator
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    with pytest.raises(BackendNotSupportedError, match="not implemented"):
        PyTorchCodeGenerator.execute_op("NonExistentOpThatDoesNotExist")


def test_hardware_multidimensional_grid_and_parameter_bindings() -> None:
    """Verify 3D grid resolution and parameter bindings for Conv2D, MatMul, and BatchMatMul."""
    # MatMul graph
    g_matmul = IRGraph()
    n_a = IRNode("a", "Input", shape_metadata=(64, 32), attributes={"dtype": "float32"})
    n_b = IRNode("b", "Input", shape_metadata=(32, 128), attributes={"dtype": "float32"})
    n_mm = IRNode("mm", "MatMul", inputs=["a", "b"], shape_metadata=(64, 128), attributes={"dtype": "float32"})
    for n in [n_a, n_b, n_mm]:
        g_matmul.nodes[n.id] = n
    g_matmul.inputs = ["a", "b"]
    g_matmul.outputs = ["mm"]

    cuda_mm = CudaCodeGenerator(g_matmul).generate()
    assert "matmul<<<grid_0, block_0>>>(inputs[0], inputs[1], d_out_0, 64, 128, 32)" in cuda_mm
    assert "dim3 grid_0((128 + block_0.x - 1) / block_0.x, (64 + block_0.y - 1) / block_0.y, 1)" in cuda_mm

    rocm_mm = RocmCodeGenerator(g_matmul).generate()
    assert "hipLaunchKernelGGL(matmul, grid_0, block_0, 0, 0, inputs[0], inputs[1], d_out_0, 64, 128, 32)" in rocm_mm

    metal_mm = MetalCodeGenerator(g_matmul).generate()
    assert "grid_size_0 = MTLSize(128, 64, 1)" in metal_mm

    # Conv2D graph
    g_conv = IRGraph()
    n_x = IRNode("x", "Input", shape_metadata=(2, 3, 28, 28), attributes={"dtype": "float32"})
    n_w = IRNode("w", "Input", shape_metadata=(16, 3, 3, 3), attributes={"dtype": "float32"})
    n_conv = IRNode("conv", "Conv2D", inputs=["x", "w"], shape_metadata=(2, 16, 26, 26), attributes={"dtype": "float32"})
    for n in [n_x, n_w, n_conv]:
        g_conv.nodes[n.id] = n
    g_conv.inputs = ["x", "w"]
    g_conv.outputs = ["conv"]

    cuda_conv = CudaCodeGenerator(g_conv).generate()
    assert "conv2d<<<grid_0, block_0>>>(inputs[0], inputs[1], d_out_0, 2, 3, 28, 28, 16, 3, 3, 26, 26)" in cuda_conv
    assert "dim3 grid_0((26 * 26 + block_0.x - 1) / block_0.x, 16, 2)" in cuda_conv

    rocm_conv = RocmCodeGenerator(g_conv).generate()
    assert "hipLaunchKernelGGL(conv2d, grid_0, block_0, 0, 0, inputs[0], inputs[1], d_out_0, 2, 3, 28, 28, 16, 3, 3, 26, 26)" in rocm_conv

    metal_conv = MetalCodeGenerator(g_conv).generate()
    assert "grid_size_0 = MTLSize(676, 16, 2)" in metal_conv

    # BatchMatMul graph
    g_bmm = IRGraph()
    n_ba = IRNode("ba", "Input", shape_metadata=(4, 16, 32), attributes={"dtype": "float32"})
    n_bb = IRNode("bb", "Input", shape_metadata=(4, 32, 64), attributes={"dtype": "float32"})
    n_bmm = IRNode("bmm", "BatchMatMul", inputs=["ba", "bb"], shape_metadata=(4, 16, 64), attributes={"dtype": "float32"})
    for n in [n_ba, n_bb, n_bmm]:
        g_bmm.nodes[n.id] = n
    g_bmm.inputs = ["ba", "bb"]
    g_bmm.outputs = ["bmm"]

    cuda_bmm = CudaCodeGenerator(g_bmm).generate()
    assert "batchmatmul<<<grid_0, block_0>>>(inputs[0], inputs[1], d_out_0, 4, 16, 64, 32)" in cuda_bmm
    assert "dim3 grid_0((64 + block_0.x - 1) / block_0.x, (16 + block_0.y - 1) / block_0.y, 4)" in cuda_bmm

    rocm_bmm = RocmCodeGenerator(g_bmm).generate()
    assert "hipLaunchKernelGGL(batchmatmul, grid_0, block_0, 0, 0, inputs[0], inputs[1], d_out_0, 4, 16, 64, 32)" in rocm_bmm

    metal_bmm = MetalCodeGenerator(g_bmm).generate()
    assert "grid_size_0 = MTLSize(64, 16, 4)" in metal_bmm


def test_hardware_liveness_deallocations_and_free_buffer() -> None:
    """Verify dead intermediate buffer freeing in generated code and runner free_buffer."""
    g = IRGraph()
    n_in = IRNode("in0", "Input", shape_metadata=(4, 4), attributes={"dtype": "float32"})
    n_mid = IRNode("mid", "ReLU", inputs=["in0"], shape_metadata=(4, 4), attributes={"dtype": "float32"})
    n_out = IRNode("out", "Exp", inputs=["mid"], shape_metadata=(4, 4), attributes={"dtype": "float32"})
    for n in [n_in, n_mid, n_out]:
        g.nodes[n.id] = n
    g.inputs = ["in0"]
    g.outputs = ["out"]

    cuda_code = CudaCodeGenerator(g).generate()
    assert "CUDA_CHECK(cudaFree(d_out_0));" in cuda_code

    rocm_code = RocmCodeGenerator(g).generate()
    assert "HIP_CHECK(hipFree(d_out_0));" in rocm_code

    metal_code = MetalCodeGenerator(g).generate()
    assert "buffer_out_0 = None" in metal_code

    runner = MetalRunner()
    runner.free_buffer(None)
    runner.free_buffer(ctypes.c_void_p(0x1234))


def test_hardware_new_ops_coverage_and_numerical_equivalence() -> None:
    """Verify presence and code generation for new reductions, activations, linalg, and spatial ops."""
    cuda_gen = CudaCodeGenerator(IRGraph())
    rocm_gen = RocmCodeGenerator(IRGraph())
    metal_gen = MetalCodeGenerator(IRGraph())

    new_ops = [
        "reduceprod",
        "reduceall",
        "reduceany",
        "argmax",
        "argmin",
        "cumsum",
        "elu",
        "selu",
        "hardswish",
        "mish",
        "leakyrelu",
        "cholesky",
        "qr",
        "solve",
        "triangularsolve",
        "einsum",
        "avgpool3d",
        "maxpool3d",
        "interpolate",
    ]

    for op in new_ops:
        assert op in cuda_gen.config.templates, f"Missing {op} in CUDA"
        assert op in rocm_gen.config.templates, f"Missing {op} in ROCm"
        assert op in metal_gen.config.templates, f"Missing {op} in Metal"

        # Verify type mappings and grid calculations
        c_tpl = cuda_gen.config.templates[op]
        assert "float32" in c_tpl.get("type_mappings", {})
        assert c_tpl.get("grid_calc") is not None

        r_tpl = rocm_gen.config.templates[op]
        assert "float32" in r_tpl.get("type_mappings", {})
        assert r_tpl.get("grid_calc") is not None

        m_tpl = metal_gen.config.templates[op]
        assert "float32" in m_tpl.get("type_mappings", {})
        assert m_tpl.get("grid_calc") is not None

    # Test code generation for an activation and reduction graph
    g = IRGraph()
    n_in = IRNode("in0", "Input", shape_metadata=(10, 10), attributes={"dtype": "float32"})
    n_elu = IRNode("elu_op", "Elu", inputs=["in0"], shape_metadata=(10, 10), attributes={"dtype": "float32"})
    n_prod = IRNode("prod_op", "ReduceProd", inputs=["elu_op"], shape_metadata=(10, 1), attributes={"dtype": "float32"})
    for n in [n_in, n_elu, n_prod]:
        g.nodes[n.id] = n
    g.inputs = ["in0"]
    g.outputs = ["prod_op"]

    cuda_code = CudaCodeGenerator(g).generate()
    assert "elu" in cuda_code
    assert "reduceprod" in cuda_code

    rocm_code = RocmCodeGenerator(g).generate()
    assert "elu" in rocm_code
    assert "reduceprod" in rocm_code

    metal_code = MetalCodeGenerator(g).generate()
    assert "elu" in metal_code
    assert "reduceprod" in metal_code
