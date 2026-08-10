from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode


def test_wasm_simd_broadcast_generation():
    graph = IRGraph()
    n1 = LogicalNode(id="in1", op_type="Input")
    n1.shape_metadata = (32,)
    n2 = LogicalNode(id="in2", op_type="Input")
    n2.shape_metadata = (32, 64)
    n3 = LogicalNode(id="out", op_type="Add", inputs=["in1", "in2"])
    n3.shape_metadata = (32, 64)
    graph.nodes = {"in1": n1, "in2": n2, "out": n3}

    gen = WasmCodeGenerator(graph)
    code = gen.generate()

    assert "wasm_f32x4_add" in code


def test_wasm_simd_tanh():
    graph = IRGraph()
    n1 = LogicalNode(id="in1", op_type="Input")
    n1.shape_metadata = (32,)
    n2 = LogicalNode(id="out", op_type="Tanh", inputs=["in1"])
    graph.nodes = {"in1": n1, "out": n2}

    gen = WasmCodeGenerator(graph)
    code = gen.generate()

    assert "std::tanh" in code or "wasm_f32x4" in code
