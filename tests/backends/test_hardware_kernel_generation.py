"""Comprehensive hardware code generation and compiler validation test suite.

Tests code generation for registered IR operations across CUDA, ROCm, Metal, and LLVM/C++,
verifying compiler protocol compliance, syntax validation, and declarative execution limits.
"""

from __future__ import annotations

from ml_switcheroo_compiler.backends.cuda.cuda import CudaCodeGenerator
from ml_switcheroo_compiler.backends.hardware_compilers import (
    ClangCompiler,
    CUDACompiler,
    HardwareCompilerProtocol,
    HIPCompiler,
    MetalCompiler,
)
from ml_switcheroo_compiler.backends.hardware_config_models import (
    HardwareDeviceProfilesManifestModel,
    load_hardware_device_profiles,
)
from ml_switcheroo_compiler.backends.llvm_cpp.generator import CppGenerator
from ml_switcheroo_compiler.backends.metal.metal import MetalCodeGenerator
from ml_switcheroo_compiler.backends.rocm.rocm import RocmCodeGenerator
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY


def test_hardware_compiler_protocol_conformance() -> None:
    """Verify that all hardware compilers implement HardwareCompilerProtocol."""
    compilers = [
        CUDACompiler(),
        HIPCompiler(),
        MetalCompiler(),
        ClangCompiler(),
    ]
    for comp in compilers:
        assert isinstance(comp, HardwareCompilerProtocol)
        assert isinstance(comp.target_name, str)
        assert len(comp.target_name) > 0


def test_hardware_compilers_syntax_validation() -> None:
    """Verify deterministic syntax validation across all accelerator compilers."""
    cuda_comp = CUDACompiler()
    assert cuda_comp.validate_syntax("__global__ void test_k(float* A) { int i = 0; }")
    assert not cuda_comp.validate_syntax("invalid cuda code missing global")
    assert not cuda_comp.validate_syntax("__global__ void test_k( { unbalanced")

    hip_comp = HIPCompiler()
    assert hip_comp.validate_syntax("__global__ void hip_k(float* A) { int i = 0; }")
    assert not hip_comp.validate_syntax("unbalanced { ( }")

    metal_comp = MetalCompiler()
    assert metal_comp.validate_syntax("kernel void msl_k() { int i = 0; }")
    assert not metal_comp.validate_syntax("void non_kernel() {}")

    clang_comp = ClangCompiler()
    assert clang_comp.validate_syntax("void compute_cpp() { int x = 42; }")
    assert not clang_comp.validate_syntax("unbalanced [")


def test_hardware_compilers_compilation_results() -> None:
    """Verify HardwareCompilationResult structure on simulated/available compilation."""
    cuda_comp = CUDACompiler()
    res_cuda = cuda_comp.compile("__global__ void k(float* out) { out[0] = 1.0f; }")
    assert res_cuda.success is True
    assert res_cuda.compiler_cli in ("nvcc", "nvcc_emulator")

    hip_comp = HIPCompiler()
    res_hip = hip_comp.compile("__global__ void k(float* out) { out[0] = 1.0f; }")
    assert res_hip.success is True

    metal_comp = MetalCompiler()
    res_metal = metal_comp.compile("kernel void k() {}")
    assert res_metal.success is True

    clang_comp = ClangCompiler()
    res_clang = clang_comp.compile("void k() {}")
    assert res_clang.success is True


def test_hardware_device_profiles_limits() -> None:
    """Verify declarative hardware device profiles execution limits and tile sizes."""
    manifest = load_hardware_device_profiles()
    assert isinstance(manifest, HardwareDeviceProfilesManifestModel)

    profiles = manifest.profiles
    assert "cuda" in profiles
    assert "rocm" in profiles
    assert "metal" in profiles
    assert "llvm_cpp" in profiles

    # CUDA limits
    cuda_prof = profiles["cuda"]
    assert cuda_prof.warp_size == 32
    assert cuda_prof.execution_unit_size == 32
    assert cuda_prof.max_threads_per_block == 1024
    assert cuda_prof.shared_memory_limit_bytes == 49152
    assert cuda_prof.dynamic_tile_sizes.tile_1d == 256
    assert cuda_prof.dynamic_tile_sizes.tile_2d == [16, 16]

    # ROCm limits
    rocm_prof = profiles["rocm"]
    assert rocm_prof.wavefront_size == 64
    assert rocm_prof.execution_unit_size == 64
    assert rocm_prof.shared_memory_limit_bytes == 65536

    # Metal limits
    metal_prof = profiles["metal"]
    assert metal_prof.simdgroup_size == 32
    assert metal_prof.execution_unit_size == 32
    assert metal_prof.threadgroup_memory_limit_bytes == 32768


def test_advanced_hardware_kernel_patterns() -> None:
    """Verify code generation for advanced reduction, matrix multiplication, and convolution patterns."""
    for pattern in ["warp_shuffle_reduce", "shared_memory_tree_reduce", "tiled_matmul_2d", "conv2d_shared_halo", "maxpool2d_dynamic", "avgpool2d_dynamic"]:
        g = IRGraph(name=f"test_{pattern}")
        in0 = IRNode(id="in0", op_type="Input", shape_metadata=(16, 16))
        node = IRNode(id="p_op", op_type=pattern, inputs=["in0"], shape_metadata=(16, 16))
        g.nodes["in0"] = in0
        g.nodes["p_op"] = node
        g.inputs = ["in0"]
        g.outputs = ["p_op"]
        g.sorted_nodes = [in0, node]

        cuda_code = CudaCodeGenerator(g).generate()
        assert pattern in cuda_code or "kernel" in cuda_code
        assert CUDACompiler().validate_syntax(cuda_code)

        rocm_code = RocmCodeGenerator(g).generate()
        assert pattern in rocm_code or "kernel" in rocm_code
        assert HIPCompiler().validate_syntax(rocm_code)

        metal_code = MetalCodeGenerator(g).generate()
        assert pattern in metal_code or "kernel" in metal_code
        assert MetalCompiler().validate_syntax(metal_code)

        cpp_code = CppGenerator(g).generate()
        assert len(cpp_code) > 0
        assert ClangCompiler().validate_syntax(cpp_code)


def test_all_registered_operations_hardware_codegen_parity() -> None:
    """Test code generation for registered operations across CUDA, ROCm, Metal, and LLVM/C++."""
    supported_ops = [
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Relu",
        "MatMul",
        "BatchMatMul",
        "Conv2D",
        "Conv3D",
        "Sigmoid",
        "Tanh",
        "GELU",
        "SiLU",
        "ELU",
        "LeakyReLU",
        "HardSwish",
        "Exp",
        "Log",
        "Sqrt",
        "Pow",
        "Abs",
        "Neg",
        "Sin",
        "Cos",
        "Minimum",
        "Maximum",
        "Floor",
        "Ceil",
        "BatchNorm",
        "LayerNorm",
        "RMSNorm",
        "GroupNorm",
        "MaxPool3D",
        "AvgPool2D",
        "AvgPool3D",
        "GlobalAvgPool2D",
        "ReduceSum",
        "ReduceMean",
        "ReduceMax",
        "ReduceMin",
        "ReduceProd",
        "ReduceAny",
        "ReduceAll",
        "Transpose",
        "Einsum",
        "Softmax",
        "LogSoftmax",
    ]
    cuda_comp = CUDACompiler()
    hip_comp = HIPCompiler()
    metal_comp = MetalCompiler()
    clang_comp = ClangCompiler()

    binary_ops = {"Add", "Sub", "Mul", "Div", "Pow", "Minimum", "Maximum", "MatMul", "BatchMatMul", "Einsum", "Conv2D", "Conv3D"}
    for op_name in supported_ops:
        g = IRGraph(name=f"test_supp_{op_name}")
        in0 = IRNode(id="in0", op_type="Input", shape_metadata=(4, 4))
        g.nodes["in0"] = in0
        if op_name in binary_ops:
            in1 = IRNode(id="in1", op_type="Input", shape_metadata=(4, 4))
            g.nodes["in1"] = in1
            node = IRNode(id=f"{op_name}_node", op_type=op_name, inputs=["in0", "in1"], shape_metadata=(4, 4))
            g.inputs = ["in0", "in1"]
            g.sorted_nodes = [in0, in1, node]
        else:
            node = IRNode(id=f"{op_name}_node", op_type=op_name, inputs=["in0"], shape_metadata=(4, 4))
            g.inputs = ["in0"]
            g.sorted_nodes = [in0, node]
        g.nodes[node.id] = node
        g.outputs = [node.id]

        cuda_src = CudaCodeGenerator(g).generate()
        assert cuda_comp.validate_syntax(cuda_src)

        rocm_src = RocmCodeGenerator(g).generate()
        assert hip_comp.validate_syntax(rocm_src)

        metal_src = MetalCodeGenerator(g).generate()
        assert metal_comp.validate_syntax(metal_src)

        cpp_src = CppGenerator(g).generate()
        assert clang_comp.validate_syntax(cpp_src)

    all_ops = list(_YAML_REGISTRY.keys())
    assert len(all_ops) >= 4000

    sample_ops = all_ops[::80]  # ~50 representative operations covering the spectrum

    for op_name in sample_ops:
        g = IRGraph(name=f"test_{op_name}")
        in0 = IRNode(id="in0", op_type="Input", shape_metadata=(4, 4))
        node = IRNode(id=f"{op_name}_node", op_type=op_name, inputs=["in0"], shape_metadata=(4, 4))
        g.nodes["in0"] = in0
        g.nodes[node.id] = node
        g.inputs = ["in0"]
        g.outputs = [node.id]
        g.sorted_nodes = [in0, node]

        # CUDA code generation and syntax validation
        try:
            cuda_src = CudaCodeGenerator(g).generate()
            assert cuda_comp.validate_syntax(cuda_src)
        except BackendNotSupportedError as e:
            assert "not supported" in str(e).lower()

        # ROCm code generation and syntax validation
        try:
            rocm_src = RocmCodeGenerator(g).generate()
            assert hip_comp.validate_syntax(rocm_src)
        except BackendNotSupportedError as e:
            assert "not supported" in str(e).lower()

        # Metal code generation and syntax validation
        try:
            metal_src = MetalCodeGenerator(g).generate()
            assert metal_comp.validate_syntax(metal_src)
        except BackendNotSupportedError as e:
            assert "not supported" in str(e).lower()

        # LLVM/C++ code generation and syntax validation
        cpp_src = CppGenerator(g).generate()
        assert clang_comp.validate_syntax(cpp_src)
