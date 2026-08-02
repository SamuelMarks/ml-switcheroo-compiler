from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.ir.core import IRBlock, IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.loop_unrolling import _get_initial_constants, clone_subgraph, detect_static_bound, loop_unrolling_pass


def test_loop_unrolling_pass():
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="WhileLoop")
    n2 = IRNode(id="n2", op_type="Add")
    g.nodes["n1"] = n1
    g.nodes["n2"] = n2
    modified = loop_unrolling_pass(g)
    assert modified is False


def test_clone_subgraph():
    subgraph = IRBlock(id="b", nodes=[LogicalNode(id="in", op_type="Input"), LogicalNode(id="add", op_type="Add", inputs=["in", "in"]), LogicalNode(id="out", op_type="Output", inputs=["add"])], inputs=["in"], outputs=["out"])
    cloned, out_ids = clone_subgraph(subgraph, "suf", {"in": "outer_in"})
    assert "add_suf" in cloned
    assert cloned["add_suf"].inputs == ["outer_in", "outer_in"]
    assert out_ids == ["add_suf"]


def test_detect_static_bound_eval_error():
    cond_graph = IRBlock(id="c", nodes=[LogicalNode(id="c", op_type="NonExistentOp"), LogicalNode(id="out", op_type="Output", inputs=["c"])], outputs=["out"])
    assert detect_static_bound(cond_graph, cond_graph, {}) is None


def test_detect_static_bound_body_error():
    cond_graph = IRBlock(id="c", nodes=[LogicalNode(id="c", op_type="Constant", attributes={"value": True}), LogicalNode(id="out", op_type="Output", inputs=["c"])], outputs=["out"])
    body_graph = IRBlock(id="b", nodes=[LogicalNode(id="c", op_type="NonExistentOp"), LogicalNode(id="out", op_type="Output", inputs=["c"])], outputs=["out"])
    assert detect_static_bound(cond_graph, body_graph, {}) is None


def test_detect_static_bound_success():
    cond_graph = IRBlock(id="c", nodes=[LogicalNode(id="c", op_type="Constant", attributes={"value": False}), LogicalNode(id="out", op_type="Output", inputs=["c"])], outputs=["out"])
    body_graph = IRBlock(id="b", nodes=[], outputs=[])

    iters = detect_static_bound(cond_graph, body_graph, {})
    assert iters == 0


def test_get_initial_constants():
    cond = IRBlock(id="c", inputs=["c_in"])
    node = IRNode(id="loop", op_type="Loop", inputs=["outer_const"], attributes={"cond": cond})
    graph = IRGraph(nodes={"outer_const": IRNode(id="outer_const", op_type="Constant", attributes={"value": 42}), "loop": node})
    assert _get_initial_constants(node, graph) == {"c_in": 42}


def test_get_initial_constants_missing():
    node = IRNode(id="loop", op_type="Loop", inputs=[], attributes={})
    graph = IRGraph()
    assert _get_initial_constants(node, graph) == {}


def test_loop_unrolling_pass_unroll_iters():
    body_graph = IRBlock(id="b", nodes=[LogicalNode(id="b_in", op_type="Input"), LogicalNode(id="add", op_type="Add", inputs=["b_in", "b_in"]), LogicalNode(id="out", op_type="Output", inputs=["add"])], inputs=["b_in"], outputs=["out"])
    cond_graph = IRBlock(id="c", inputs=["c_in"], outputs=[])

    node = IRNode(id="loop", op_type="Loop", inputs=["init"], attributes={"cond": cond_graph, "body": body_graph, "unroll_iters": 2})
    g = IRGraph(nodes={"init": IRNode(id="init", op_type="Input"), "loop": node}, outputs=["loop"])

    assert loop_unrolling_pass(g) is True
    assert "add_unroll_loop_0" in g.nodes
    assert "add_unroll_loop_1" in g.nodes
    assert g.nodes["loop"].op_type == "Identity"
    assert g.nodes["loop"].inputs == ["add_unroll_loop_1"]


def test_loop_unrolling_pass_unroll_iters_tuple():
    body_graph = IRBlock(
        id="b",
        nodes=[LogicalNode(id="b_in", op_type="Input"), LogicalNode(id="b_in2", op_type="Input"), LogicalNode(id="add", op_type="Add", inputs=["b_in", "b_in"]), LogicalNode(id="sub", op_type="Sub", inputs=["b_in2", "b_in2"]), LogicalNode(id="out", op_type="Output", inputs=["add", "sub"])],
        inputs=["b_in", "b_in2"],
        outputs=["out"],
    )
    cond_graph = IRBlock(id="c", inputs=["c_in", "c_in2"], outputs=[])

    node = IRNode(id="loop", op_type="Loop", inputs=["init1", "init2"], attributes={"cond": cond_graph, "body": body_graph, "unroll_iters": 1})
    g = IRGraph(nodes={"init1": IRNode(id="init1", op_type="Input"), "init2": IRNode(id="init2", op_type="Input"), "loop": node}, outputs=["loop"])

    assert loop_unrolling_pass(g) is True
    assert g.nodes["loop"].op_type == "Tuple"
    assert g.nodes["loop"].inputs == ["add_unroll_loop_0", "sub_unroll_loop_0"]


def test_loop_unrolling_pass_unroll_zero():
    body_graph = IRBlock(id="b", nodes=[LogicalNode(id="b_in", op_type="Input"), LogicalNode(id="add", op_type="Add", inputs=["b_in", "b_in"]), LogicalNode(id="out", op_type="Output", inputs=["add"])], inputs=["b_in"], outputs=["out"])
    cond_graph = IRBlock(id="c", inputs=["c_in"], outputs=[])

    node = IRNode(id="loop", op_type="Loop", inputs=["init"], attributes={"cond": cond_graph, "body": body_graph, "unroll_iters": 0})
    g = IRGraph(nodes={"init": IRNode(id="init", op_type="Input"), "loop": node}, outputs=["loop"])

    assert loop_unrolling_pass(g) is True
    assert g.nodes["loop"].op_type == "Identity"
    assert g.nodes["loop"].inputs == ["init"]


def test_loop_unrolling_pass_unroll_zero_tuple():
    body_graph = IRBlock(id="b", nodes=[], inputs=["b_in", "b_in2"], outputs=[])
    cond_graph = IRBlock(id="c", inputs=["c_in", "c_in2"], outputs=[])

    node = IRNode(id="loop", op_type="Loop", inputs=["init1", "init2"], attributes={"cond": cond_graph, "body": body_graph, "unroll_iters": 0})
    g = IRGraph(nodes={"init1": IRNode(id="init1", op_type="Input"), "init2": IRNode(id="init2", op_type="Input"), "loop": node}, outputs=["loop"])

    assert loop_unrolling_pass(g) is True
    assert g.nodes["loop"].op_type == "Tuple"
    assert g.nodes["loop"].inputs == ["init1", "init2"]


def test_loop_unrolling_pass_detect_fails():
    cond_graph = IRBlock(id="c", nodes=[LogicalNode(id="c", op_type="NonExistentOp"), LogicalNode(id="out", op_type="Output", inputs=["c"])], outputs=["out"])
    body_graph = IRBlock(id="b", nodes=[], outputs=[])

    node = IRNode(
        id="loop",
        op_type="Loop",
        inputs=[],
        attributes={
            "cond": cond_graph,
            "body": body_graph,
        },
    )
    g = IRGraph(nodes={"loop": node}, outputs=["loop"])

    assert loop_unrolling_pass(g) is True
    assert g.nodes["loop"].attributes["unrolled"] is True


def test_detect_static_bound_success_multiple_iters():
    # Initial state c_in = 0
    # cond: c_in < 2 (we'll just use a mock backend or simple node to do this)
    # body: c_in + 1
    # We can use eager ops, but it's simpler to just patch evaluate_graph for this test.
    # Actually, we can use built-in ml_switcheroo_compiler ops that evaluator understands!
    # Evaluator understands Less, Add, Constant, Input, Output

    cond_graph = IRBlock(
        id="c", nodes=[LogicalNode(id="c_in", op_type="Input"), LogicalNode(id="const_2", op_type="Constant", attributes={"value": 2}), LogicalNode(id="less", op_type="Less", inputs=["c_in", "const_2"]), LogicalNode(id="out", op_type="Output", inputs=["less"])], inputs=["c_in"], outputs=["out"]
    )
    body_graph = IRBlock(
        id="b", nodes=[LogicalNode(id="b_in", op_type="Input"), LogicalNode(id="const_1", op_type="Constant", attributes={"value": 1}), LogicalNode(id="add", op_type="Add", inputs=["b_in", "const_1"]), LogicalNode(id="out", op_type="Output", inputs=["add"])], inputs=["b_in"], outputs=["out"]
    )

    iters = detect_static_bound(cond_graph, body_graph, {"c_in": 0})
    assert iters == 2


def test_detect_static_bound_tuple_state():
    cond_graph = IRBlock(
        id="c",
        nodes=[
            LogicalNode(id="in1", op_type="Input"),
            LogicalNode(id="in2", op_type="Input"),
            LogicalNode(id="out", op_type="Output", inputs=["in1"]),  # Just return in1
        ],
        inputs=["in1", "in2"],
        outputs=["out"],
    )
    body_graph = IRBlock(
        id="b", nodes=[LogicalNode(id="in1", op_type="Input"), LogicalNode(id="in2", op_type="Input"), LogicalNode(id="const_0", op_type="Constant", attributes={"value": False}), LogicalNode(id="out", op_type="Output", inputs=["const_0", "in2"])], inputs=["in1", "in2"], outputs=["out"]
    )
    iters = detect_static_bound(cond_graph, body_graph, {"in1": True, "in2": 0})
    assert iters == 1


def test_detect_static_bound_max_iters():
    cond_graph = IRBlock(id="c", nodes=[LogicalNode(id="c_in", op_type="Input"), LogicalNode(id="out", op_type="Output", inputs=["c_in"])], inputs=["c_in"], outputs=["out"])
    body_graph = IRBlock(id="b", nodes=[LogicalNode(id="b_in", op_type="Input"), LogicalNode(id="out", op_type="Output", inputs=["b_in"])], inputs=["b_in"], outputs=["out"])
    iters = detect_static_bound(cond_graph, body_graph, {"c_in": True}, max_iters=2)
    assert iters is None


def test_get_initial_constants_non_constant():
    cond = IRBlock(id="c", inputs=["c_in"])
    node = IRNode(id="loop", op_type="Loop", inputs=["outer_non_const"], attributes={"cond": cond})
    # outer_non_const is an Input, not a Constant
    graph = IRGraph(nodes={"outer_non_const": IRNode(id="outer_non_const", op_type="Input"), "loop": node})
    assert _get_initial_constants(node, graph) == {}
