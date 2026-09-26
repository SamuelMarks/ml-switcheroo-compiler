"""Unit tests asserting 100% import and symbol coverage across standalone edge backend packages."""

from __future__ import annotations


def test_edge_inits_symbol_exports() -> None:
    """Verify that all standalone edge and webgpu packages export canonical generators."""
    import ml_switcheroo_compiler.backends.edge_mlir as edge_mlir
    import ml_switcheroo_compiler.backends.edge_onnx as edge_onnx
    import ml_switcheroo_compiler.backends.edge_stablehlo as edge_stablehlo
    import ml_switcheroo_compiler.backends.edge_wasm as edge_wasm
    import ml_switcheroo_compiler.backends.edge_webgl as edge_webgl
    import ml_switcheroo_compiler.backends.webgpu as webgpu

    assert hasattr(edge_mlir, "MLIRBytecodeEncoder")
    assert edge_mlir.MLIRBytecodeEncoder is not None

    assert hasattr(edge_onnx, "ONNXCodeGenerator")
    assert edge_onnx.ONNXCodeGenerator is not None

    assert hasattr(edge_stablehlo, "StableHLOCodeGenerator")
    assert edge_stablehlo.StableHLOCodeGenerator is not None

    assert hasattr(edge_wasm, "WasmCodeGenerator")
    assert edge_wasm.WasmCodeGenerator is not None

    assert hasattr(edge_webgl, "WebGLCodeGenerator")
    assert edge_webgl.WebGLCodeGenerator is not None

    assert hasattr(webgpu, "WebGPUCodeGenerator")
    assert webgpu.WebGPUCodeGenerator is not None
