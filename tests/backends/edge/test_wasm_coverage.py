from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_wasm_coverage_unknown_op():
    g = IRGraph()
    n = IRNode("dummy", "UnknownOp")
    n.inputs = ["dummy_in"]
    n.shape_metadata = 1
    g.inputs = []
    g.outputs = [n]
    g._nodes = {"dummy": n}
    gen = WasmCodeGenerator(g)
    gen.var_names = {"dummy_in": "dummy_in"}
    gen.sorted_nodes = g.inputs + [n]
    code = gen.generate()
    assert "_scalar_unknownop" in code

    n2 = IRNode("dummy_sigmoid", "Sigmoid")
    n2.inputs = ["dummy_in"]
    n2.shape_metadata = 10
    g._nodes["dummy_sigmoid"] = n2
    gen.sorted_nodes.append(n2)
    code = gen.generate()
    assert "std::exp" in code

    n3 = IRNode("dummy_exp", "Exp")
    n3.inputs = ["dummy_in"]
    n3.shape_metadata = 10
    g._nodes["dummy_exp"] = n3
    gen.sorted_nodes.append(n3)
    code = gen.generate()
    assert "std::exp" in code
