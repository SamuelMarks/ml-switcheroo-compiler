from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.state_lowering import state_lowering_pass


def test_state_lowering_pass():
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Input", attributes={"is_state": True, "name": "var_a"})
    n2 = IRNode(id="n2", op_type="Output", attributes={"is_state": True})
    n3 = IRNode(id="n3", op_type="Input")
    g.nodes["n1"] = n1
    g.nodes["n2"] = n2
    g.nodes["n3"] = n3

    modified = state_lowering_pass(g)
    assert modified is True

    assert n1.op_type == "ReadVariable"
    assert n1.attributes.get("variable_name") == "var_a"
    assert "name" not in n1.attributes

    assert n2.op_type == "AssignVariable"
    assert n2.attributes.get("variable_name") == "n2"
    assert "name" not in n2.attributes

    assert n3.op_type == "Input"

    # Run again, should not modify
    modified_again = state_lowering_pass(g)
    assert modified_again is False


def test_state_lowering_empty_graph():
    g = IRGraph()
    assert state_lowering_pass(g) is False


def test_state_lowering_output_with_name():
    g = IRGraph()
    n2 = IRNode(id="n2", op_type="Output", attributes={"is_state": True, "name": "var_b"})
    g.nodes["n2"] = n2

    modified = state_lowering_pass(g)
    assert modified is True

    assert n2.op_type == "AssignVariable"
    assert n2.attributes.get("variable_name") == "var_b"
    assert "name" not in n2.attributes


def test_state_lowering_input_without_name():
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Input", attributes={"is_state": True})
    g.nodes["n1"] = n1

    modified = state_lowering_pass(g)
    assert modified is True

    assert n1.op_type == "ReadVariable"
    assert n1.attributes.get("variable_name") == "n1"
    assert "name" not in n1.attributes
