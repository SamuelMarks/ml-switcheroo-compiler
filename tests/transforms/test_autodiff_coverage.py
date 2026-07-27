import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.transforms.autodiff import _invoke_jvp_rule, jvp


def test_jvp_primals_tangents_length_mismatch():
    graph = LogicalGraph()
    with pytest.raises(ValueError, match="primals and tangents must have the same length"):
        jvp(graph, ["p1", "p2"], ["t1"], ["o1"])


def test_jvp_output_not_found():
    graph = LogicalGraph()
    with pytest.raises(ValueError, match="Output node 'o1' not found in graph."):
        jvp(graph, ["p1"], ["t1"], ["o1"])


def test_invoke_jvp_rule_no_graph_node():
    def mock_jvp(a, b):
        return a + b

    res = _invoke_jvp_rule(mock_jvp, None, None, ["t1"])
    assert res == "mock_tangent"


def test_process_jvp_node_unimplemented():
    from unittest.mock import patch

    from ml_switcheroo_compiler.transforms.autodiff import _process_jvp_node

    graph = LogicalGraph()
    node = LogicalNode(id="n1", op_type="FakeOp", inputs=["p1"], shape_metadata=())
    graph.nodes["n1"] = node
    tangents = {"p1": "t1"}

    def raise_err(*args, **kwargs):
        raise ValueError()

    with patch("ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry.get_jvp", side_effect=raise_err):
        with pytest.raises(ValueError, match="Missing JVP rule for operation: FakeOp"):
            _process_jvp_node(graph, node, tangents)


def test_process_jvp_node_invoke_unimplemented():
    from unittest.mock import patch

    from ml_switcheroo_compiler.transforms.autodiff import _process_jvp_node

    graph = LogicalGraph()
    node = LogicalNode(id="n1", op_type="Add", inputs=["p1"], shape_metadata=())
    graph.nodes["n1"] = node
    tangents = {"p1": "t1"}

    def raise_err(*args, **kwargs):
        raise ValueError()

    with patch("ml_switcheroo_compiler.transforms.autodiff._invoke_jvp_rule", side_effect=raise_err):
        with pytest.raises(ValueError, match="Missing JVP rule for operation: Add"):
            _process_jvp_node(graph, node, tangents)


def test_dummy_jvp_execution():
    from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import get_jvp

    with pytest.raises(ValueError, match="Missing JVP rule for operation: NonExistentOp"):
        get_jvp("NonExistentOp")


def test_autodiff_grad_output_node_and_cotangent():
    from ml_switcheroo_compiler.transforms.autodiff import grad

    graph = LogicalGraph()
    # Build graph: Input -> Output
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input", shape_metadata=())
    graph.nodes["out1"] = LogicalNode(id="out1", op_type="Output", inputs=["in1"], shape_metadata=())

    # Test cotangent_id as string
    g2 = grad(graph, wrt=["in1"], output_id="out1", cotangent_id="cot_node")
    assert "cot_node" in g2.outputs

    # Test cotangent_id as dict
    g3 = grad(graph, wrt=["in1"], output_id="out1", cotangent_id={"out1": "cot_node2"})
    assert "cot_node2" in g3.outputs


def test_jvp_compile_expr_and_style2():
    from ml_switcheroo_compiler.transforms.autodiff import _compile_jvp_expr, _invoke_jvp_rule

    graph = LogicalGraph()
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input", shape_metadata=())
    graph.nodes["const_zero"] = LogicalNode(id="const_zero", op_type="Constant", attributes={"value": 0.0}, shape_metadata=())

    # Test _compile_jvp_expr
    inv_map = {"safe_id_0": "t1", "safe_id_1": "t2"}

    # Unary
    res = _compile_jvp_expr("-safe_id_0", graph, (), inv_map)
    assert graph.nodes[res].op_type == "Negative"
    assert graph.nodes[res].inputs[0] == "t1"

    # Unary plus (noop basically)
    res = _compile_jvp_expr("+safe_id_0", graph, (), inv_map)
    assert res == "t1"

    # BinOp Add, Sub, Mult, Div
    res = _compile_jvp_expr("safe_id_0 + safe_id_1", graph, (), inv_map)
    assert graph.nodes[res].op_type == "Add"

    res = _compile_jvp_expr("safe_id_0 - safe_id_1", graph, (), inv_map)
    assert graph.nodes[res].op_type == "Subtract"

    res = _compile_jvp_expr("safe_id_0 * safe_id_1", graph, (), inv_map)
    assert graph.nodes[res].op_type == "Multiply"

    res = _compile_jvp_expr("safe_id_0 / safe_id_1", graph, (), inv_map)
    assert graph.nodes[res].op_type == "TrueDivide"

    # Constant
    res = _compile_jvp_expr("2.0 * safe_id_0", graph, (), inv_map)
    assert graph.nodes[res].op_type == "Multiply"

    # Unsupported
    with pytest.raises(ValueError):
        _compile_jvp_expr("safe_id_0 ** 2", graph, (), inv_map)

    with pytest.raises(ValueError):
        _compile_jvp_expr("len(safe_id_0)", graph, (), inv_map)

    # Style2 JVP rule
    def my_jvp(x, tangent_x):
        return "x + tangent_x"

    class DummyNode:
        inputs = ["in1"]
        shape_metadata = ()

    res = _invoke_jvp_rule(my_jvp, graph, DummyNode(), ["t1"])
    assert res is not None

    def my_jvp_exc(x, tangent_x):
        raise ValueError("err")

    assert _invoke_jvp_rule(my_jvp_exc, graph, DummyNode(), ["t1"]) == "mock_tangent"


def test_process_jvp_node_output():
    from ml_switcheroo_compiler.transforms.autodiff import _process_jvp_node

    graph = LogicalGraph()
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input", shape_metadata=())
    graph.nodes["in2"] = LogicalNode(id="in2", op_type="Input", shape_metadata=())
    out_node = LogicalNode(id="out1", op_type="Output", inputs=["in1", "in2"], shape_metadata=())
    graph.nodes["out1"] = out_node

    tangents = {"in1": "t1"}  # in2 is missing, should create zeros

    _process_jvp_node(graph, out_node, tangents)
    assert "out1" in tangents
    out_t = tangents["out1"]
    assert graph.nodes[out_t].op_type == "Output"
    assert graph.nodes[out_t].inputs[0] == "t1"
    # The second input should be a zero constant
    assert graph.nodes[graph.nodes[out_t].inputs[1]].op_type == "Constant"


def test_jvp_invoke_rule_exc():
    from ml_switcheroo_compiler.transforms.autodiff import _invoke_jvp_rule

    def bad_jvp(graph, node, tangent):
        raise ValueError("err")

    assert _invoke_jvp_rule(bad_jvp, None, None, ["t1"]) == "mock_tangent"


def test_jvp_style2_returns_non_string():
    from ml_switcheroo_compiler.transforms.autodiff import _invoke_style2_jvp_rule

    def my_jvp(x, tangent_x):
        return 42

    import inspect

    sig = inspect.signature(my_jvp)

    class DummyNode:
        inputs = ["in1"]
        shape_metadata = ()

    res = _invoke_style2_jvp_rule(my_jvp, sig, None, DummyNode(), ["t1"])
    assert res == 42
