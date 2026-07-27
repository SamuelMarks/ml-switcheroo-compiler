"""Test module."""

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.core.state_manager import _build_functional_outputs, _process_assign_node, _rewrite_node, lift_state


def test_state_manager():
    # Test _process_assign_node
    n_assign = LogicalNode(id="a", op_type="Assign", domain="ml_switcheroo", inputs=["state1", "val1"])
    env = {"state1": "state1", "state2": "state2"}
    _process_assign_node(n_assign, env)
    assert env["state1"] == "val1"

    n_assign2 = LogicalNode(id="a2", op_type="Assign", domain="ml_switcheroo", inputs=["not_in_env", "val2"])
    _process_assign_node(n_assign2, env)
    assert "not_in_env" not in env

    # Test _rewrite_node
    n_op = LogicalNode(id="op1", op_type="Add", domain="ml_switcheroo", inputs=["state1", "state2", "const"])
    n_rewritten = _rewrite_node(n_op, env)
    assert n_rewritten.inputs == ["val1", "state2", "const"]

    # Test _build_functional_outputs
    out = _build_functional_outputs(["out1"], ["state1", "state2"], env)
    assert out == ["out1", "val1", "state2"]

    # Test lift_state
    g = LogicalGraph(name="test")
    g.nodes["state1"] = LogicalNode(id="state1", op_type="Variable", domain="ml_switcheroo")
    g.nodes["val1"] = LogicalNode(id="val1", op_type="Constant", domain="ml_switcheroo")
    g.nodes["a"] = n_assign
    g.nodes["op1"] = n_op
    g.outputs = ["op1"]

    g_func = lift_state(g, ["state1"])
    assert g_func.name == "test_functional"
    assert g_func.outputs == ["op1", "val1"]
    assert "op1" in g_func.nodes
    pass
