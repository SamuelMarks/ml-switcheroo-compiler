"""Test Edge control flow generators."""

from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_webgpu_while_loop() -> None:
    graph = IRGraph()
    graph.nodes["in0"] = IRNode(id="in0", op_type="Input", inputs=[])
    graph.nodes["in1"] = IRNode(id="in1", op_type="Input", inputs=[])
    graph.nodes["while"] = IRNode(id="while", op_type="WhileLoop", inputs=["in0", "in1"])
    graph.outputs = ["while"]

    gen = WebGPUCodeGenerator(graph)
    res = gen.generate()
    assert "while" in res
    assert "compute_while" in res
    assert "current_state < 10.0" in res


def test_wasm_cond() -> None:
    graph = IRGraph()
    graph.nodes["in0"] = IRNode(id="in0", op_type="Input", inputs=[])
    graph.nodes["cond"] = IRNode(id="cond", op_type="Cond", inputs=["in0"])
    graph.outputs = ["cond"]

    gen = WasmCodeGenerator(graph)
    res = gen.generate()
    assert "if (" in res
    assert "else" in res


def test_webgpu_scan() -> None:
    graph = IRGraph()
    graph.nodes["in0"] = IRNode(id="in0", op_type="Input", inputs=[])
    graph.nodes["scan"] = IRNode(id="scan", op_type="Scan", inputs=["in0"])
    graph.outputs = ["scan"]

    gen = WebGPUCodeGenerator(graph)
    res = gen.generate()
    assert "acc + buf_in0_f32[i]" in res


def test_wasm_scan() -> None:
    graph = IRGraph()
    graph.nodes["in0"] = IRNode(id="in0", op_type="Input", inputs=[])
    graph.nodes["scan"] = IRNode(id="scan", op_type="Scan", inputs=["in0"])
    graph.outputs = ["scan"]

    gen = WasmCodeGenerator(graph)
    res = gen.generate()
    assert "acc_scan + buf_in0[i]" in res
