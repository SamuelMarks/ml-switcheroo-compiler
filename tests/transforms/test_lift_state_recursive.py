from ml_switcheroo_compiler.ir.core import IRGraph, IRBlock, IRNode
from ml_switcheroo_compiler.transforms.passes.lift_state import lift_state_pass
from ml_switcheroo_compiler.transforms.passes.state_lifting import state_lifting_pass


def test_recursive_state_lifting():
    graph = IRGraph("test")
    block1 = IRBlock("block1", inputs=[], outputs=[], nodes=[])
    node1 = IRNode(
        id="assign",
        op_type="AssignVariable",
        inputs=["dummy"],
        attributes={"variable_name": "x"},
        shape_metadata=(),
    )
    block1.nodes = {"assign": node1}

    parent_node = IRNode(
        id="parent",
        op_type="While",
        inputs=[],
        attributes={"body": block1},
        shape_metadata=(),
    )
    graph.nodes = {"parent": parent_node}

    state_lifting_pass(graph)
    assert node1.op_type == "Output"


def test_recursive_lift_state():
    graph = IRGraph("test")
    block1 = IRBlock("block1", inputs=[], outputs=[], nodes=[])
    node1 = IRNode(
        id="assign",
        op_type="AssignVariable",
        inputs=["dummy"],
        attributes={"variable_name": "x"},
        shape_metadata=(),
    )
    block1.nodes = {"assign": node1}

    parent_node = IRNode(
        id="parent",
        op_type="While",
        inputs=[],
        attributes={"body": block1},
        shape_metadata=(),
    )
    graph.nodes = {"parent": parent_node}

    lift_state_pass(graph)
    assert node1.op_type == "Output"
