"""Unit tests asserting 100% import and symbol coverage across standalone edge backend packages."""

from __future__ import annotations

from ml_switcheroo_compiler.core.shape import _has_attr


def test_edge_inits_symbol_exports() -> None:
    """Verify that canonical edge package and BackendRegistry export all edge generators."""
    import ml_switcheroo_compiler.backends.edge as edge
    import ml_switcheroo_compiler.backends.edge.distributed_webrtc as edge_webrtc
    import ml_switcheroo_compiler.backends.llvm_cpp as llvm_cpp
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    assert _has_attr(edge, "MLIRBytecodeEncoder")
    assert edge.MLIRBytecodeEncoder is not None
    assert _has_attr(edge, "ONNXCodeGenerator")
    assert edge.ONNXCodeGenerator is not None
    assert _has_attr(edge, "StableHLOCodeGenerator")
    assert edge.StableHLOCodeGenerator is not None
    assert _has_attr(edge, "WasmCodeGenerator")
    assert edge.WasmCodeGenerator is not None
    assert _has_attr(edge, "WebGLCodeGenerator")
    assert edge.WebGLCodeGenerator is not None
    assert _has_attr(edge, "WebGPUCodeGenerator")
    assert edge.WebGPUCodeGenerator is not None

    assert _has_attr(edge_webrtc, "SignalingHandler")
    assert edge_webrtc.SignalingHandler is not None
    assert _has_attr(edge_webrtc, "SignalingServer")
    assert edge_webrtc.SignalingServer is not None
    assert _has_attr(edge_webrtc, "WebRTCDataChannelStream")
    assert edge_webrtc.WebRTCDataChannelStream is not None

    assert _has_attr(llvm_cpp, "CppGenerator")
    assert llvm_cpp.CppGenerator is not None
    assert _has_attr(llvm_cpp, "LLVMCPPRunner")
    assert llvm_cpp.LLVMCPPRunner is not None

    assert BackendRegistry.get("edge") is edge.ONNXCodeGenerator
    assert BackendRegistry.get("edge_onnx") is edge.ONNXCodeGenerator
    assert BackendRegistry.get("edge_stablehlo") is edge.StableHLOCodeGenerator
    assert BackendRegistry.get("edge_wasm") is edge.WasmCodeGenerator
    assert BackendRegistry.get("edge_wasm_simd") is edge.WasmCodeGenerator
    assert BackendRegistry.get("edge_webgl") is edge.WebGLCodeGenerator
    assert BackendRegistry.get("edge_wgsl") is edge.WebGPUCodeGenerator
    assert BackendRegistry.get("webgpu") is edge.WebGPUCodeGenerator
