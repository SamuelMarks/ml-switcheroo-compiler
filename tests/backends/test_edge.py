"""Tests for edge.py."""

from ml_switcheroo.backends.edge import (
    WebGPUCodeGenerator,
    WebGLCodeGenerator,
    WasmCodeGenerator,
    ONNXCodeGenerator,
)
from ml_switcheroo_ir import LogicalGraph


def test_edge_generators() -> None:
    """Docstring."""
    graph = LogicalGraph()
    wg = WebGPUCodeGenerator(graph)
    assert wg.generate() == "/* WGSL WebGPU Generated Code */"
    assert wg._dispatch_op_template(None) == "wgsl_op"

    wl = WebGLCodeGenerator(graph)
    assert wl.generate() == "/* GLSL WebGL Generated Code */"
    assert wl._dispatch_op_template(None) == "glsl_op"

    wa = WasmCodeGenerator(graph)
    assert wa.generate() == "/* WASM SIMD Generated Code */"
    assert wa._dispatch_op_template(None) == "wasm_op"

    ox = ONNXCodeGenerator(graph)
    assert ox.generate() == "/* ONNX Generated Code */"
    assert ox._dispatch_op_template(None) == "onnx_op"
