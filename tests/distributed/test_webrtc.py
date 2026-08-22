"""Tests for WebRTC edge distributed generation."""

from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_init, emit_webrtc_op
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_emit_webrtc_init():
    res = emit_webrtc_init()
    assert "RTCPeerConnection" in res
    assert "createDataChannel" in res


def test_emit_webrtc_op():
    res = emit_webrtc_op("AllReduce", "my_data", "op1")
    assert "AllReduce" in res
    assert "my_data" in res

    res2 = emit_webrtc_op("AllGather", "my_data", "op2")
    assert "AllGather" in res2

    res3 = emit_webrtc_op("AllToAll", "my_data", "op3")
    assert "AllToAll" in res3

    res4 = emit_webrtc_op("Unknown", "my_data", "op4")
    assert res4 == ""


def test_webgpu_webrtc():
    graph = IRGraph()
    n1 = IRNode(id="in", op_type="Input", shape_metadata=[10])
    n2 = IRNode(id="allreduce", op_type="AllReduce", inputs=["in"], shape_metadata=[10])
    graph.nodes = {"in": n1, "allreduce": n2}
    graph.outputs = ["allreduce"]

    gen = WebGPUCodeGenerator(graph)
    code = gen.generate()
    assert "RTCPeerConnection" in code
    assert "ALLREDUCE" in code


def test_wasm_webrtc():
    graph = IRGraph()
    n1 = IRNode(id="in", op_type="Input", shape_metadata=[10])
    n2 = IRNode(id="allgather", op_type="AllGather", inputs=["in"], shape_metadata=[10])
    n3 = IRNode(id="reducescatter", op_type="ReduceScatter", inputs=["allgather"], shape_metadata=[10])
    graph.nodes = {"in": n1, "allgather": n2, "reducescatter": n3}
    graph.outputs = ["reducescatter"]

    gen = WasmCodeGenerator(graph)
    code = gen.generate()
    assert "AllGather" in code
    assert "ReduceScatter" in code
    assert "JS Orcherstrator:" in code
