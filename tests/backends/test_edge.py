"""Unit tests for the edge-device code generators.

This module contains test cases to verify the functionality of WebGPU, WebGL, WASM, and
ONNX code generators using a logical graph.
"""

from ml_switcheroo_ir import LogicalGraph

from ml_switcheroo_compiler.backends.edge import (
    ONNXCodeGenerator,
    WasmCodeGenerator,
    WebGLCodeGenerator,
    WebGPUCodeGenerator,
)


def test_edge_generators() -> None:
    """Verifies the code generation and node visitation of various edge backends.

    This test ensures that WebGPU, WebGL, WASM, and ONNX code generators
    correctly initialize with a logical graph, produce the expected boilerplate
    code, and return the correct operation identifiers during graph traversal

    Args:
    None

    Returns:
    None

    Raises:
    AssertionError: If any of the generated code or visited operation
        strings do not match the expected output.
    """
    graph = LogicalGraph()
    wg = WebGPUCodeGenerator(graph)
    assert wg.generate() == "/* WGSL WebGPU Generated Code */"
    assert wg.visit(None, []) == "wgsl_op"

    gl = WebGLCodeGenerator(graph)
    assert gl.generate() == "/* GLSL WebGL Generated Code */"
    assert gl.visit(None, []) == "glsl_op"

    wa = WasmCodeGenerator(graph)
    assert wa.generate() == "/* WASM SIMD Generated Code */"
    assert wa.visit(None, []) == "wasm_op"

    ox = ONNXCodeGenerator(graph)
    assert ox.generate() == "/* ONNX Generated Code */"
    assert ox.visit(None, []) == "onnx_op"
