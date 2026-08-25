def test_wasm_missing_coverage():
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    graph = IRGraph()
    n1 = IRNode("if_node", "If", inputs=["in1"], attributes={"branch_graphs": [IRGraph()]})
    graph.nodes["if_node"] = n1
    graph.inputs = ["in1"]
    graph.outputs = ["if_node"]

    gen = WasmCodeGenerator(graph)
    gen.visit_If(n1, "If", "if_node", ["in1"], [10], 10)

    n2 = IRNode("cond_node", "Cond", inputs=["in1"], attributes={"branch_graphs": [IRGraph(), IRGraph()]})
    gen.visit_Cond(n2, "Cond", "cond_node", ["in1"], [10], 10)


def test_webgpu_missing_shader():
    from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    graph = IRGraph()
    n0 = IRNode("in1", "Input", inputs=[], shape_metadata=[1])
    n1 = IRNode("bad", "BadOp", inputs=["in1"], shape_metadata=[1])
    graph.nodes["in1"] = n0
    graph.nodes["bad"] = n1
    graph.inputs = ["in1"]
    graph.outputs = ["bad"]

    # Use a custom generator to force exception in _get_wgsl_for_op
    class MyGen(WebGPUCodeGenerator):
        def _get_wgsl_for_op(self, node, shape, nelem, clean_id):
            raise ValueError("Forced error")

    gen = MyGen(graph)
    gen.generate()


def test_webgl_missing_shader():
    from ml_switcheroo_compiler.backends.edge.webgl import WebGLCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    graph = IRGraph()
    n1 = IRNode("bad", "BadOp", inputs=["in1"])
    graph.nodes["bad"] = n1
    graph.inputs = ["in1"]
    graph.outputs = ["bad"]
    gen = WebGLCodeGenerator(graph)
    gen.generate()


def test_wasm_missing_coverage_empty_branches():
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    graph = IRGraph()
    n1 = IRNode("if_node", "If", inputs=["in1"], attributes={"branch_graphs": []})
    gen = WasmCodeGenerator(graph)
    gen.visit_If(n1, "If", "if_node", ["in1"], [10], 10)
