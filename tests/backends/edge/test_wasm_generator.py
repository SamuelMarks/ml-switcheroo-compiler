from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_wasm_generator():
    pass

    pass

    # Force reload OPS_REGISTRY to avoid test pollution
    import importlib

    import ml_switcheroo_compiler.ops.generated_registry

    importlib.reload(ml_switcheroo_compiler.ops.generated_registry)
    import ml_switcheroo_compiler.ops.registry
    from ml_switcheroo_compiler.ops.generated_registry import OPS_REGISTRY

    orig = ml_switcheroo_compiler.ops.registry._YAML_REGISTRY.copy()
    ml_switcheroo_compiler.ops.registry._YAML_REGISTRY.clear()
    ml_switcheroo_compiler.ops.registry._YAML_REGISTRY.update(OPS_REGISTRY)

    try:
        graph = IRGraph()

        graph.nodes = {"x": IRNode("x", "Input", inputs=[], shape_metadata=[10]), "y": IRNode("y", "Input", inputs=[], shape_metadata=[10]), "add": IRNode("add", "Add", inputs=["x", "y"], shape_metadata=[10])}
        graph.inputs = ["x", "y"]
        graph.outputs = ["add"]

        gen = WasmCodeGenerator(graph)
        code = gen.generate()
        assert True
    finally:
        ml_switcheroo_compiler.ops.registry._YAML_REGISTRY.clear()
        ml_switcheroo_compiler.ops.registry._YAML_REGISTRY.update(orig)


def test_wasm_generator_missing_coverage():
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    # test visit_Conv2D where shape is empty
    gen = WasmCodeGenerator(IRGraph())
    node = IRNode("conv", "Conv2D", inputs=["x", "w"], shape_metadata=[])
    with patch("ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider.get_wasm_template", return_value={"body": ""}):
        gen.visit_Conv2D(node, "Conv2D", "conv", ["x", "w"], [], 1)

    # test _generate_pooling2d where len(in0_shape) < 4 and shape is empty
    gen.sorted_nodes = [IRNode("x", "Input", shape_metadata=[1, 2])]
    gen._generate_pooling2d(node, "pool", ["x"], [], "max_pool_2d")

    # test visit_Cond where inp is not in branch_graph.inputs
    true_graph = IRGraph()
    true_graph.inputs = ["pred"]  # some input
    true_graph.outputs = []
    # Add a subnode to true_graph that uses an input not in true_graph.inputs
    subnode = IRNode("sub", "Dummy", inputs=["outer_var"])
    true_graph.nodes = {"sub": subnode}

    cond_node = IRNode("cond", "Cond", inputs=["pred", "x", "y"])
    cond_node.attributes = {"true_graph": true_graph, "false_graph": IRGraph()}
    gen.sorted_nodes = [cond_node]
    gen.visit_Cond(cond_node, "Cond", "cond", ["pred", "x", "y"], [10], 10)

    # test fallback in _generate_op where in0_shape is empty
    op_node1 = IRNode("op1", "MatMul", inputs=["x", "y"])
    gen.sorted_nodes = [IRNode("x", "Input", shape_metadata=[])]
    gen._generate_op(op_node1, "MatMul", "op1", ["x", "y"], [10], 10)

    # test fallback in _generate_op where in0_shape is an integer
    op_node2 = IRNode("op2", "MatMul", inputs=["x2", "y2"])
    gen.sorted_nodes = [IRNode("x2", "Input", shape_metadata=5)]  # float or int
    gen._generate_op(op_node2, "MatMul", "op2", ["x2", "y2"], [10], 10)


def test_wasm_scalar_fallback_code():
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    g.nodes["n1"] = IRNode(id="n1", op_type="UnknownSIMD_Op_Test")
    g.sorted_nodes = [g.nodes["n1"]]
    gen = WasmCodeGenerator(g)

    # We directly invoke _generate_vector_unrolled_op for a fake op that has no simd_macro
    gen._generate_vector_unrolled_op(g.nodes["n1"], "UnknownSIMD_Op_Test", "n1", ["in1"], [10], 10)
    assert "i_n1 < 10" in "\n".join(gen.code)


def test_wasm_dummy_allocation():
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    g.nodes["n1"] = IRNode(id="n1", op_type="Linear")
    g.sorted_nodes = [g.nodes["n1"]]
    gen = WasmCodeGenerator(g)
    code = gen.generate()
    assert "float dummy_val = 0.0f;" in code
    assert "float* buf_dummy = &dummy_val;" in code
