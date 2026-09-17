"""Unit tests for edge WebRTC collective emitter routines."""

from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_init, emit_webrtc_op


def test_emit_webrtc_init() -> None:
    """Verify WebRTC initialization JavaScript emission."""
    init_code = emit_webrtc_init()
    assert "RTCPeerConnection" in init_code
    assert "ml_switcheroo_collective" in init_code
    assert "onmessage" in init_code


def test_emit_webrtc_op() -> None:
    """Verify WebRTC collective op emission for all supported operations."""
    code_reduce = emit_webrtc_op("AllReduce", "bufferX", "op_1")
    assert "AllReduce" in code_reduce
    assert "bufferX" in code_reduce
    assert "window.__ml_collective.allReduce" in code_reduce

    code_gather = emit_webrtc_op("AllGather", "bufferY", "op_2")
    assert "AllGather" in code_gather
    assert "window.__ml_collective.allGather" in code_gather

    code_to_all = emit_webrtc_op("AllToAll", "bufferZ", "op_3")
    assert "AllToAll" in code_to_all
    assert "window.__ml_collective.allToAll" in code_to_all

    code_scatter = emit_webrtc_op("ReduceScatter", "bufferW", "op_4")
    assert "ReduceScatter" in code_scatter
    assert "window.__ml_collective.reduceScatter" in code_scatter

    code_invalid = emit_webrtc_op("NonExistentOp", "buf", "op_5")
    assert code_invalid == ""


def test_webgpu_webrtc() -> None:
    """Verify WebGPU WebRTC emission integration."""
    init_code = emit_webrtc_init()
    op_code = emit_webrtc_op("AllReduce", "localTensor", "reduce_0")
    assert len(init_code) > 0
    assert len(op_code) > 0


def test_wasm_webrtc() -> None:
    """Verify WASM WebRTC emission integration."""
    op_code = emit_webrtc_op("AllGather", "wasmBuffer", "gather_0")
    assert "wasmBuffer" in op_code
