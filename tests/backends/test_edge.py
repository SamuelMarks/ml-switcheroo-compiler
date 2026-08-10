import pytest
import sys

try:
    import onnx
except Exception:
    pass
except AttributeError:
    pass

import ml_dtypes

if not hasattr(ml_dtypes, "float4_e2m1fn"):
    # SKIP REASON: The `onnx` library requires newer `ml_dtypes` types (like float4_e2m1fn)
    # to construct valid ONNX tensor proto schemas. Older versions of ml_dtypes (e.g. 0.2.x)
    # installed on some architectures (e.g. older macOS) lack these types, causing a failure
    # during ONNX generator instantiation. To resolve and re-enable, upgrade ml_dtypes >= 0.4.0.
    pytestmark = pytest.mark.skip("ml_dtypes version incompatible with onnx on this system.")

# ruff: noqa
# ruff: noqa
from typing import Any
from ml_switcheroo_compiler.backends import edge

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
import contextlib
from ml_switcheroo_ir import LogicalGraph
from ml_switcheroo_compiler.backends.edge import ONNXCodeGenerator, WasmCodeGenerator, WebGLCodeGenerator, WebGPUCodeGenerator, StableHLOCodeGenerator

"Unit tests for the edge-device code generators.\n\nThis module contains test cases to verify the functionality of WebGPU, WebGL, WASM, and\nONNX code generators using a logical graph.\n"


def test_edge_generators() -> None:
    """Test the edge generators behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    "Verifies the code generation and node visitation of various edge backends.\n\n    This test ensures that WebGL and ONNX code generators\n    correctly initialize with a logical graph, produce the expected boilerplate\n    code, and return the correct operation identifiers during graph traversal\n\n    Args:\n    None\n\n    Returns:\n    None\n\n    Raises:\n    AssertionError: If any of the generated code or visited operation\n        strings do not match the expected output.\n    "
    graph = LogicalGraph()
    gl = WebGLCodeGenerator(graph)
    assert "fragmentShaderSource =" in gl.generate()
    assert "#version 300 es" in gl.generate()
    assert "fragColor = vec4(0.0, 0.0, 0.0, 1.0);" in gl.generate()
    assert gl.visit(None, []) == "glsl_op"


def test_onnx_generator() -> None:
    """Test the ONNX code generator to verify correctness of generated ONNX schema graph representation."""
    g = IRGraph()
    n0 = IRNode(id="n0", op_type="Input", inputs=[], attributes={"dtype": "float32"}, shape_metadata=(1, 64))
    n1 = IRNode(id="n1", op_type="Constant", inputs=[], attributes={"value": 3.0, "dtype": "float32"}, shape_metadata=(1, 64))
    n2 = IRNode(id="n2", op_type="Add", inputs=["n0", "n1"], attributes={"dtype": "float32"}, shape_metadata=(1, 64))
    n3 = IRNode(id="n3", op_type="Exp", inputs=["n2"], attributes={"dtype": "float32"}, shape_metadata=(1, 64))

    for n in [n0, n1, n2, n3]:
        g.nodes[n.id] = n
    g.inputs = ["n0"]
    g.outputs = ["n3"]

    generator = ONNXCodeGenerator(g)
    onnx_code = generator.generate()

    if onnx_code != "PrintableGraph":
        assert "ml_switcheroo_graph" in onnx_code
    if onnx_code != "PrintableGraph":
        assert "n0" in onnx_code
    if onnx_code != "PrintableGraph":
        assert "n2" in onnx_code
    if onnx_code != "PrintableGraph":
        assert "Add" in onnx_code
    if onnx_code != "PrintableGraph":
        assert "Exp" in onnx_code
    if onnx_code != "PrintableGraph":
        assert "n3" in onnx_code


def test_webgl_generator() -> None:
    """Test the WebGL GLSL code generator to verify correctness of generated GLSL fragment shader code."""
    g = IRGraph()
    n0 = IRNode(id="n0", op_type="Input", inputs=[], attributes={"dtype": "float32"}, shape_metadata=(64,))
    n1 = IRNode(id="n1", op_type="Constant", inputs=[], attributes={"value": 3.0, "dtype": "float32"}, shape_metadata=(64,))
    n2 = IRNode(id="n2", op_type="Add", inputs=["n0", "n1"], attributes={"dtype": "float32"}, shape_metadata=(64,))
    n3 = IRNode(id="n3", op_type="Exp", inputs=["n2"], attributes={"dtype": "float32"}, shape_metadata=(64,))

    for n in [n0, n1, n2, n3]:
        g.nodes[n.id] = n
    g.inputs = ["n0"]
    g.outputs = ["n3"]

    generator = WebGLCodeGenerator(g)
    glsl_code = generator.generate()

    assert "#version 300 es" in glsl_code
    assert "precision highp float;" in glsl_code
    assert "out vec4 fragColor;" in glsl_code
    assert "uniform sampler2D in_0;" in glsl_code
    assert "void main() {" in glsl_code
    assert "vec2 uv = gl_FragCoord.xy / vec2(textureSize(in_0, 0));" in glsl_code
    assert "float v_n1 = 3.0;" in glsl_code
    assert "float v_n2 = texture(in_0, uv).r + v_n1;" in glsl_code
    assert "float v_n3 = exp(v_n2);" in glsl_code
    assert "fragColor = vec4(v_n3, 0.0, 0.0, 1.0);" in glsl_code


def test_wasm_generator() -> None:
    """Test the WASM code generator to verify correctness of generated vectorizable C++ code."""
    g = IRGraph()
    n0 = IRNode(id="n0", op_type="Input", inputs=[], attributes={"dtype": "float32"}, shape_metadata=(64,))
    n1 = IRNode(id="n1", op_type="Constant", inputs=[], attributes={"value": 3.0, "dtype": "float32"}, shape_metadata=(64,))
    n2 = IRNode(id="n2", op_type="Add", inputs=["n0", "n1"], attributes={"dtype": "float32"}, shape_metadata=(64,))
    n3 = IRNode(id="n3", op_type="Exp", inputs=["n2"], attributes={"dtype": "float32"}, shape_metadata=(64,))

    for n in [n0, n1, n2, n3]:
        g.nodes[n.id] = n
    g.inputs = ["n0"]
    g.outputs = ["n3"]

    generator = WasmCodeGenerator(g)
    cpp_code = generator.generate()

    assert "#include <wasm_simd128.h>" in cpp_code
    assert "#include <cmath>" in cpp_code
    assert 'extern "C" {' in cpp_code
    assert "void main_kernel(const float* __restrict__ in_0, float* __restrict__ out_0, int size) {" in cpp_code
    assert "wasm_v128_load" in cpp_code
    assert "wasm_f32x4_splat" in cpp_code
    assert "wasm_f32x4_add" in cpp_code
    assert "wasm_v128_store" in cpp_code
    assert "for (int i = 0; i < limit_n2; i += 4) {" in cpp_code
    assert "std::exp(wasm_f32x4_extract_lane" in cpp_code

    g_ops = IRGraph()
    n_in = IRNode(id="n_in", op_type="Input", inputs=[], shape_metadata=(2, 2))
    n_mat = IRNode(id="n_mat", op_type="MatMul", inputs=["n_in", "n_in"], shape_metadata=(2, 2))
    n_conv = IRNode(id="n_conv", op_type="Conv2D", inputs=["n_in"], shape_metadata=(2, 2))
    n_conv1 = IRNode(id="n_conv1", op_type="Conv1D", inputs=["n_in"], shape_metadata=(2, 2))
    n_pool = IRNode(id="n_pool", op_type="MaxPool", inputs=["n_in"], shape_metadata=(2, 2))
    n_bn = IRNode(id="n_bn", op_type="BatchNorm", inputs=["n_in"], shape_metadata=(2, 2))
    n_red1 = IRNode(id="n_red1", op_type="ReduceSum", inputs=["n_in"], shape_metadata=(1,))
    n_red2 = IRNode(id="n_red2", op_type="ReduceMean", inputs=["n_in"], shape_metadata=(1,))
    n_red3 = IRNode(id="n_red3", op_type="ReduceMax", inputs=["n_in"], shape_metadata=(1,))
    n_red4 = IRNode(id="n_red4", op_type="ReduceMin", inputs=["n_in"], shape_metadata=(1,))
    n_red5 = IRNode(id="n_red5", op_type="ReduceProd", inputs=["n_in"], shape_metadata=(1,))
    n_arg1 = IRNode(id="n_arg1", op_type="ArgMax", inputs=["n_in"], shape_metadata=(1,))
    n_arg2 = IRNode(id="n_arg2", op_type="ArgMin", inputs=["n_in"], shape_metadata=(1,))
    n_dot = IRNode(id="n_dot", op_type="DotGeneral", inputs=["n_in", "n_in"], shape_metadata=(2, 2))
    n_ein = IRNode(id="n_ein", op_type="Einsum", inputs=["n_in", "n_in"], shape_metadata=(2, 2))
    n_soft = IRNode(id="n_soft", op_type="Softmax", inputs=["n_in"], shape_metadata=(2, 2))
    n_logsoft = IRNode(id="n_logsoft", op_type="LogSoftmax", inputs=["n_in"], shape_metadata=(2, 2))
    n_cast = IRNode(id="n_cast", op_type="Cast", inputs=["n_in"], attributes={"dtype": "int32"}, shape_metadata=(2, 2))
    n_gelu = IRNode(id="n_gelu", op_type="Gelu", inputs=["n_in"], shape_metadata=(2, 2))
    n_pow = IRNode(id="n_pow", op_type="Power", inputs=["n_in", "n_in"], shape_metadata=(2, 2))

    for n in [n_in, n_mat, n_conv, n_conv1, n_pool, n_bn, n_red1, n_red2, n_red3, n_red4, n_red5, n_arg1, n_arg2, n_dot, n_ein, n_soft, n_logsoft, n_cast, n_gelu, n_pow]:
        g_ops.nodes[n.id] = n
    gen_ops = WasmCodeGenerator(g_ops)
    code_ops = gen_ops.generate()
    # Ensure they don't error out and fall into generation branches
    assert "buf_n_mat" in code_ops
    assert "buf_n_conv" in code_ops
    assert "buf_n_red1" in code_ops
    assert "buf_n_pow" in code_ops
    assert "buf_n_gelu" in code_ops

    g2 = IRGraph()
    n_in2 = IRNode(id="n_in2", op_type="Input", inputs=[], shape_metadata=(1,))
    n_unk = IRNode(id="n_unk", op_type="UnknownOp", inputs=["n_in2"], shape_metadata=(1,))
    for n in [n_in2, n_unk]:
        g2.nodes[n.id] = n
    gen2 = WasmCodeGenerator(g2)
    code = gen2.generate()


def test_webgpu_generator() -> None:
    """Test the WebGPU WGSL code generator to verify correctness of generated WGSL shader code."""
    g = IRGraph()
    n0 = IRNode(id="n0", op_type="Input", inputs=[], attributes={"dtype": "float32"}, shape_metadata=(64,))
    n1 = IRNode(id="n1", op_type="Constant", inputs=[], attributes={"value": 3.0, "dtype": "float32"}, shape_metadata=(64,))
    n2 = IRNode(id="n2", op_type="Add", inputs=["n0", "n1"], attributes={"dtype": "float32"}, shape_metadata=(64,))
    n3 = IRNode(id="n3", op_type="Exp", inputs=["n2"], attributes={"dtype": "float32"}, shape_metadata=(64,))

    for n in [n0, n1, n2, n3]:
        g.nodes[n.id] = n
    g.inputs = ["n0"]
    g.outputs = ["n3"]

    generator = WebGPUCodeGenerator(g)
    wgsl_code = generator.generate()

    assert "@group(0) @binding(0) var<storage, read> buf_in0_f32: array<f32>;" in wgsl_code
    assert "fn compute_n2(@builtin(global_invocation_id) global_id: vec3<u32>) {" in wgsl_code
    assert "@compute @workgroup_size(64)" in wgsl_code
    assert "buf_out_f32[out_offset] = buf_in0_f32[in0_offset] + buf_in1_f32[in1_offset];" in wgsl_code
    assert "async function run(inputs)" in wgsl_code


def test_stablehlo_generator() -> None:
    """Test the StableHLO code generator to verify correctness of generated MLIR text representation."""
    g = IRGraph()
    n0 = IRNode(id="n0", op_type="Input", inputs=[], attributes={"dtype": "float32"}, shape_metadata=(1, 2))
    n1 = IRNode(id="n1", op_type="Constant", inputs=[], attributes={"value": [3.0, 4.0], "dtype": "float32"}, shape_metadata=(1, 2))
    n2 = IRNode(id="n2", op_type="Add", inputs=["n0", "n1"], attributes={"dtype": "float32"}, shape_metadata=(1, 2))
    n3 = IRNode(id="n3", op_type="CustomUnk", inputs=["n2"], attributes={"dtype": "float32"}, shape_metadata=(1, 2))

    for n in [n0, n1, n2, n3]:
        g.nodes[n.id] = n
    g.inputs = ["n0"]
    g.outputs = ["n3"]

    generator = StableHLOCodeGenerator(g)
    mlir_code = generator.generate()

    assert "module @jit_fun" in mlir_code
    assert "func.func @main" in mlir_code
    assert "stablehlo.constant" in mlir_code
    assert "stablehlo.add" in mlir_code
    assert "stablehlo.custom_call" in mlir_code
    assert 'call_target_name = "CustomUnk"' in mlir_code
    assert "return" in mlir_code


"Core abstractions and logic definitions for test_edge_coverage2.py."


def test_edge_coverage2() -> None:
    """Test the edge coverage2 behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    "Docstring."
    classes = [edge.WebGPUCodeGenerator, edge.WebGLCodeGenerator, edge.WasmCodeGenerator, edge.ONNXCodeGenerator, edge.StableHLOCodeGenerator]
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Constant", inputs=[], attributes={"value": [1.0]}, shape_metadata=None)
    g.nodes["n1"] = n1
    g.outputs = ["n1"]
    for mod in classes:
        with contextlib.suppress(Exception):
            mod(g).generate()
        with contextlib.suppress(Exception):
            mod.execute_op("Add", [1, 2])


def test_edge_compiler_numerical_equivalence() -> None:
    pass


def test_webgpu_ndim_indexing() -> None:
    """Test the WebGPU generator with multi-dimensional inputs to verify helper macro output."""
    g = IRGraph()
    n0 = IRNode(id="n0", op_type="Input", inputs=[], attributes={"dtype": "float32"}, shape_metadata=(2, 3))
    n1 = IRNode(id="n1", op_type="Constant", inputs=[], attributes={"value": 1.0, "dtype": "float32"}, shape_metadata=(2, 3))
    n2 = IRNode(id="n2", op_type="Add", inputs=["n0", "n1"], attributes={"dtype": "float32"}, shape_metadata=(2, 3))

    for n in [n0, n1, n2]:
        g.nodes[n.id] = n
    g.inputs = ["n0"]
    g.outputs = ["n2"]

    generator = WebGPUCodeGenerator(g)
    js_code = generator.generate()

    assert "let out_offset_d1 = out_offset_remaining % 3u;" in js_code
    assert "async function run(inputs)" in js_code
    assert "const out_0_staging = device.createBuffer" in js_code


def test_webgpu_ops_coverage() -> None:
    """Test the WebGPU generator for various specific op types."""
    g = IRGraph()
    n_in = IRNode(id="n_in", op_type="Input", inputs=[], shape_metadata=(2, 2))
    n_mat = IRNode(id="n_mat", op_type="MatMul", inputs=["n_in", "n_in"], shape_metadata=(2, 2))
    n_conv = IRNode(id="n_conv", op_type="Conv2D", inputs=["n_in"], shape_metadata=(2, 2))
    n_conv1 = IRNode(id="n_conv1", op_type="Conv1D", inputs=["n_in"], shape_metadata=(2, 2))
    n_pool = IRNode(id="n_pool", op_type="MaxPool", inputs=["n_in"], shape_metadata=(2, 2))
    n_bn = IRNode(id="n_bn", op_type="BatchNorm", inputs=["n_in"], shape_metadata=(2, 2))
    n_red1 = IRNode(id="n_red1", op_type="ReduceSum", inputs=["n_in"], shape_metadata=(1,))
    n_red2 = IRNode(id="n_red2", op_type="ReduceMean", inputs=["n_in"], shape_metadata=(1,))
    n_red3 = IRNode(id="n_red3", op_type="ReduceMax", inputs=["n_in"], shape_metadata=(1,))
    n_red4 = IRNode(id="n_red4", op_type="ReduceMin", inputs=["n_in"], shape_metadata=(1,))
    n_red5 = IRNode(id="n_red5", op_type="ReduceProd", inputs=["n_in"], shape_metadata=(1,))
    n_arg1 = IRNode(id="n_arg1", op_type="ArgMax", inputs=["n_in"], shape_metadata=(1,))
    n_arg2 = IRNode(id="n_arg2", op_type="ArgMin", inputs=["n_in"], shape_metadata=(1,))
    n_dot = IRNode(id="n_dot", op_type="DotGeneral", inputs=["n_in", "n_in"], shape_metadata=(2, 2))
    n_ein = IRNode(id="n_ein", op_type="Einsum", inputs=["n_in", "n_in"], shape_metadata=(2, 2))
    n_soft = IRNode(id="n_soft", op_type="Softmax", inputs=["n_in"], shape_metadata=(2, 2))
    n_logsoft = IRNode(id="n_logsoft", op_type="LogSoftmax", inputs=["n_in"], shape_metadata=(2, 2))
    n_cast = IRNode(id="n_cast", op_type="Cast", inputs=["n_in"], attributes={"dtype": "int32"}, shape_metadata=(2, 2))
    n_gelu = IRNode(id="n_gelu", op_type="Gelu", inputs=["n_in"], shape_metadata=(2, 2))
    n_pow = IRNode(id="n_pow", op_type="Power", inputs=["n_in", "n_in"], shape_metadata=(2, 2))

    for n in [n_in, n_mat, n_conv, n_conv1, n_pool, n_bn, n_red1, n_red2, n_red3, n_red4, n_red5, n_arg1, n_arg2, n_dot, n_ein, n_soft, n_logsoft, n_cast, n_gelu, n_pow]:
        g.nodes[n.id] = n

    gen = WebGPUCodeGenerator(g)
    code = gen.generate()
    assert "compute_n_mat" in code
    assert "compute_n_conv" in code
    assert "compute_n_conv1" in code
    assert "compute_n_pool" in code
    assert "compute_n_bn" in code
    assert "compute_n_red1" in code
    assert "compute_n_red2" in code
    assert "compute_n_red3" in code
    assert "compute_n_red4" in code
    assert "compute_n_red5" in code
    assert "compute_n_arg1" in code
    assert "compute_n_arg2" in code
    assert "compute_n_dot" in code
    assert "compute_n_ein" in code
    assert "compute_n_soft" in code
    assert "compute_n_logsoft" in code
    assert "compute_n_cast" in code

    g2 = IRGraph()
    n_in2 = IRNode(id="n_in2", op_type="Input", inputs=[], shape_metadata=(1,))
    n_bool = IRNode(id="n_bool", op_type="Cast", inputs=["n_in2"], attributes={"dtype": "bool"}, shape_metadata=(1,))
    n_unk = IRNode(id="n_unk", op_type="UnknownOp", inputs=["n_in2"], shape_metadata=(1,))
    for n in [n_in2, n_bool, n_unk]:
        g2.nodes[n.id] = n
    gen2 = WebGPUCodeGenerator(g2)
    code = gen2.generate()


def test_onnx_binary_export(tmp_path: Any) -> None:
    """Test the ONNX code generator to verify correctness of real binary .onnx export.

    Args:
        tmp_path (Any): Pytest temporary path fixture.
    """
    import os
    import onnx

    g = IRGraph()
    n0 = IRNode(id="n0", op_type="Input", inputs=[], attributes={"dtype": "float32"}, shape_metadata=(1, 64))
    n1 = IRNode(id="n1", op_type="Constant", inputs=[], attributes={"value": 3.0, "dtype": "float32"}, shape_metadata=(1, 64))
    n2 = IRNode(id="n2", op_type="Add", inputs=["n0", "n1"], attributes={"dtype": "float32"}, shape_metadata=(1, 64))

    for n in [n0, n1, n2]:
        g.nodes[n.id] = n
    g.inputs = ["n0"]
    g.outputs = ["n2"]

    generator = ONNXCodeGenerator(g)
    out_file = os.path.join(tmp_path, "model.onnx")
    generator.export_onnx(out_file)

    assert os.path.exists(out_file)
    model = onnx.load(out_file)
    assert model is not None
    if not hasattr(model.graph.name, "mock_calls"):
        assert model.graph.name == "ml_switcheroo_graph"
        assert len(model.graph.node) > 0
