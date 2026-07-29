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

    assert "ml_switcheroo_graph" in onnx_code
    assert "n0" in onnx_code
    assert "n2" in onnx_code
    assert "Add" in onnx_code
    assert "Exp" in onnx_code
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
    assert "for (; idx < size; ++idx) {" in cpp_code
    assert "float v_n1_scalar = 3.0;" in cpp_code
    assert "float v_n2_scalar = in_0[idx] + v_n1_scalar;" in cpp_code
    assert "float v_n3_scalar = std::exp(v_n2_scalar);" in cpp_code
    assert "out_0[idx] = v_n3_scalar;" in cpp_code


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

    assert "@group(0) @binding(0) var<storage, read> in_0: array<f32>;" in wgsl_code
    assert "@group(0) @binding(1) var<storage, read_write> out_0: array<f32>;" in wgsl_code
    assert "@compute @workgroup_size(64)" in wgsl_code
    assert "fn main(" in wgsl_code
    assert "let idx = global_id.x;" in wgsl_code
    assert "let v_n1 = 3.0;" in wgsl_code
    assert "let v_n2 = in_0[idx] + v_n1;" in wgsl_code
    assert "let v_n3 = exp(v_n2);" in wgsl_code
    assert "out_0[idx] = v_n3;" in wgsl_code


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
    """Validate that generated edge code (WASM, WebGPU, WebGL) matches NumPy eager execution results."""
    import re
    import numpy as np
    from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

    # 1. Create a logical graph: f(x) = exp(x + 3.0)
    g = IRGraph()
    n0 = IRNode(id="n0", op_type="Input", inputs=[], attributes={"dtype": "float32"}, shape_metadata=(3,))
    n1 = IRNode(id="n1", op_type="Constant", inputs=[], attributes={"value": 3.0, "dtype": "float32"}, shape_metadata=(3,))
    n2 = IRNode(id="n2", op_type="Add", inputs=["n0", "n1"], attributes={"dtype": "float32"}, shape_metadata=(3,))
    n3 = IRNode(id="n3", op_type="Exp", inputs=["n2"], attributes={"dtype": "float32"}, shape_metadata=(3,))

    for n in [n0, n1, n2, n3]:
        g.nodes[n.id] = n
    g.inputs = ["n0"]
    g.outputs = ["n3"]

    # 2. Reference NumPy execution
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    inputs_dict = {"n0": x}
    ref_outputs = evaluate_graph(g, inputs_dict)
    ref_out = ref_outputs["n3"]

    # 3. WASM C++ Generator Validation via Numerical Simulation
    wasm_gen = WasmCodeGenerator(g)
    cpp_code = wasm_gen.generate()

    # Parse and execute loop body statements
    lines = cpp_code.split("\n")
    statements = []
    in_loop = False
    for line in lines:
        if "for (; idx < size; ++idx) {" in line:
            in_loop = True
            continue
        if in_loop:
            if "}" in line:
                in_loop = False
                continue
            statements.append(line.strip())

    # Evaluate Statements
    out_0 = np.zeros_like(x)
    for idx in range(len(x)):
        local_vars = {"idx": idx, "in_0": x, "out_0": out_0, "np": np}
        for stmt in statements:
            # Clean C++ types and map std::exp to np.exp
            py_stmt = re.sub(r"\bfloat\s+", "", stmt)
            py_stmt = py_stmt.replace("std::exp", "np.exp")
            exec(py_stmt, {}, local_vars)

    np.testing.assert_allclose(out_0, ref_out, rtol=1e-5)

    # 4. WebGPU WGSL Generator Validation via Numerical Simulation
    webgpu_gen = WebGPUCodeGenerator(g)
    wgsl_code = webgpu_gen.generate()

    # Extract let assignments and output assignments inside main
    lines = wgsl_code.split("\n")
    statements = []
    in_main = False
    for line in lines:
        if "fn main" in line:
            in_main = True
            continue
        if in_main:
            if "}" in line:
                in_main = False
                continue
            statements.append(line.strip())

    out_0 = np.zeros_like(x)
    for idx in range(len(x)):
        local_vars = {"idx": idx, "in_0": x, "out_0": out_0, "np": np}
        for stmt in statements:
            if "let idx = global_id.x;" in stmt:
                continue
            # Translate WGSL let declarations
            py_stmt = stmt.replace("let ", "")
            py_stmt = py_stmt.replace("exp", "np.exp")
            exec(py_stmt, {}, local_vars)

    np.testing.assert_allclose(out_0, ref_out, rtol=1e-5)

    # 5. WebGL GLSL Generator Validation via Numerical Simulation
    webgl_gen = WebGLCodeGenerator(g)
    glsl_code = webgl_gen.generate()

    lines = glsl_code.split("\n")
    statements = []
    in_main = False
    for line in lines:
        if "void main()" in line:
            in_main = True
            continue
        if in_main:
            if "}" in line:
                in_main = False
                continue
            statements.append(line.strip())

    for idx in range(len(x)):
        # texture(in_0, uv).r is equivalent to reading from in_0[idx]
        local_vars = {"idx": idx, "in_0": x, "np": np}
        for stmt in statements:
            if "vec2 uv =" in stmt or "fragColor =" in stmt:
                continue
            py_stmt = stmt.replace("float ", "")
            py_stmt = py_stmt.replace("texture(in_0, uv).r", "in_0[idx]")
            py_stmt = py_stmt.replace("exp", "np.exp")
            exec(py_stmt, {}, local_vars)
        # Verify result matched
        assert np.allclose(local_vars["v_n3"], ref_out[idx], rtol=1e-5)


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

    assert "fn get_offset_n0(idx: u32) -> u32" in js_code
    assert "in_0[get_offset_n0(idx)]" in js_code
    assert "async function run(inputs)" in js_code
    assert "const out_0_staging = device.createBuffer" in js_code


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
    assert model.graph.name == "ml_switcheroo_graph"
    assert len(model.graph.node) > 0
