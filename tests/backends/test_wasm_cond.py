from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_wasm_cond_generation():
    graph = IRGraph()
    node_in = IRNode(id="node_in", op_type="Input", inputs=[], attributes={"dtype": "float32"}, shape_metadata=[3])

    # Create true branch graph
    true_graph = IRGraph()
    true_in = IRNode(id="true_in", op_type="Input", inputs=[], attributes={"dtype": "float32"}, shape_metadata=[3])
    true_add = IRNode(id="true_add", op_type="Add", inputs=["true_in", "true_in"], attributes={"dtype": "float32"}, shape_metadata=[3])
    true_graph.nodes["true_in"] = true_in
    true_graph.nodes["true_add"] = true_add
    true_graph.outputs = ["true_add"]
    true_graph.inputs = ["true_in"]

    # Create false branch graph
    false_graph = IRGraph()
    false_in = IRNode(id="false_in", op_type="Input", inputs=[], attributes={"dtype": "float32"}, shape_metadata=[3])
    false_sub = IRNode(id="false_sub", op_type="Subtract", inputs=["false_in", "false_in"], attributes={"dtype": "float32"}, shape_metadata=[3])
    false_graph.nodes["false_in"] = false_in
    false_graph.nodes["false_sub"] = false_sub
    false_graph.outputs = ["false_sub"]
    false_graph.inputs = ["false_in"]

    node_cond = IRNode(id="node_cond", op_type="Cond", inputs=["node_in", "node_in", "node_in"], attributes={"dtype": "float32", "branch_graphs": [true_graph, false_graph]}, shape_metadata=[3])

    graph.nodes["node_in"] = node_in
    graph.nodes["node_cond"] = node_cond
    graph.outputs = ["node_cond"]
    graph.inputs = ["node_in"]

    gen = WasmCodeGenerator(graph)
    code = gen.generate()

    assert "buf_true_add" in code
    assert "wasm_f32x4_add" in code
    assert "buf_false_sub" in code
    assert "wasm_f32x4_sub" in code


def test_wasm_generate_wat_and_control_flow():
    """Verify WAT emission and parameterized control flow generation."""
    import os

    from ml_switcheroo_compiler.backends.edge.config_models import (
        load_edge_control_flow_templates,
    )

    # 1. Test loading edge control flow templates YAML
    yaml_path = os.path.join(
        os.path.dirname(__file__),
        "../../src/ml_switcheroo_compiler/backends/edge/wasm_simd/control_flow.yaml",
    )
    cf_config = load_edge_control_flow_templates(yaml_path)
    assert "while_loop" in cf_config.control_flow_templates
    assert "cond" in cf_config.control_flow_templates
    assert "scan" in cf_config.control_flow_templates

    # 2. Test generate_wat with unary (neg, sqrt) and binary (mul, add)
    g = IRGraph()
    x = IRNode(id="x", op_type="Input")
    y = IRNode(id="y", op_type="Mul", inputs=["x", "x"])
    z = IRNode(id="z", op_type="Neg", inputs=["y"])
    g.nodes = {"x": x, "y": y, "z": z}
    g.inputs = ["x"]
    g.outputs = ["z"]

    gen = WasmCodeGenerator(g)
    wat = gen.generate_wat()
    assert "(module" in wat
    assert "f32x4.mul" in wat
    assert "f32x4.neg" in wat
    assert '(export "compute")' in wat

    # 3. Test WhileLoop with custom comparator and threshold
    g_loop = IRGraph()
    in_loop = IRNode(id="in0", op_type="Input")
    loop_node = IRNode(
        id="loop",
        op_type="WhileLoop",
        inputs=["in0"],
        attributes={"comparator": "<", "threshold": "50.0", "max_iters": 5},
    )
    g_loop.nodes = {"in0": in_loop, "loop": loop_node}
    g_loop.inputs = ["in0"]
    g_loop.outputs = ["loop"]

    gen_loop = WasmCodeGenerator(g_loop)
    loop_code = gen_loop.generate()
    assert "buf_in0[0] < 50.0" in loop_code

    # 4. Test Scan with custom init_val and scan_op_expr
    g_scan = IRGraph()
    in_scan = IRNode(id="s0", op_type="Input")
    scan_node = IRNode(
        id="scan",
        op_type="Scan",
        inputs=["s0"],
        attributes={"init_val": "1.0", "scan_op_expr": "acc * 2.0f"},
    )
    g_scan.nodes = {"s0": in_scan, "scan": scan_node}
    g_scan.inputs = ["s0"]
    g_scan.outputs = ["scan"]

    gen_scan = WasmCodeGenerator(g_scan)
    scan_code = gen_scan.generate()
    assert "acc * 2.0f" in scan_code
