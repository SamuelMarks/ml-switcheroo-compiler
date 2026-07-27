# ruff: noqa: E501
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.state_lifting import _get_node_items, state_lifting_pass


class DummyBlock:
    def __init__(self):
        self.nodes = {}
        self.outputs = []


def test_state_lifting():
    graph = IRGraph()
    n1 = IRNode("n1", "ReadVariable", attributes={"variable_name": "v1"})
    n2 = IRNode("n2", "AssignVariable", attributes={"variable_name": "v2"})
    n3 = IRNode("n3", "Add", attributes={})
    graph.nodes = {"n1": n1, "n2": n2, "n3": n3}
    graph.outputs = []
    res = state_lifting_pass(graph)
    assert res is True
    assert n1.op_type == "Input"
    assert n2.op_type == "Output"
    b = DummyBlock()
    n4 = IRNode("n4", "ReadVariable", attributes={})
    b.nodes = {"n4": n4}
    n5 = IRNode("n5", "If", attributes={"body": b})
    graph.nodes["n5"] = n5
    res = state_lifting_pass(graph)
    assert res is True


def test_get_node_items():
    assert _get_node_items(None) == []

    class DummyBlockList:
        def __init__(self):
            n = IRNode("n1", "Add", attributes={})
            self.nodes = [n]

    assert len(_get_node_items(DummyBlockList())) == 1


"Extra tests for state lifting pass."


def test_state_lifting_already_in_outputs() -> None:
    """Test state lifting when node id is already in outputs."""
    nodes = {"assign1": IRNode(id="assign1", op_type="AssignVariable", inputs=["some_input"], attributes={"variable_name": "my_var"})}
    graph = IRGraph(name="test", nodes=nodes, outputs=["assign1"])
    state_lifting_pass(graph)
    assert graph.nodes["assign1"].op_type == "Output"
    assert graph.outputs == ["assign1"]


def test_state_lifting_no_outputs() -> None:
    """Test state lifting when block has no outputs attribute."""

    class DummyBlock:
        def __init__(self) -> None:
            self.nodes = {"assign1": IRNode(id="assign1", op_type="AssignVariable", inputs=["some_input"], attributes={"variable_name": "my_var"})}

    block = DummyBlock()
    from ml_switcheroo_compiler.transforms.passes.state_lifting import _lift_block

    _lift_block(block)
    assert block.nodes["assign1"].op_type == "Output"
