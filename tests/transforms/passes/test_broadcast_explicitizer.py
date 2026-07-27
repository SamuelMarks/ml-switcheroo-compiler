"""Test Broadcast Explicitizer Pass."""

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.broadcast_explicitizer import _inject_broadcast_node, _needs_broadcast, _process_broadcast_node, broadcast_explicitizer_pass


def test_inject_broadcast_node() -> None:
    graph = IRGraph(name="test", nodes={}, outputs=[])
    input_id = "input1"
    target_shape = (2, 2)
    new_id = _inject_broadcast_node(graph, input_id, target_shape)
    assert new_id in graph.nodes
    node = graph.nodes[new_id]
    assert node.op_type == "BroadcastTo"
    assert node.inputs == [input_id]
    assert node.shape_metadata == target_shape
    assert node.attributes["shape"] == target_shape


def test_needs_broadcast() -> None:
    assert _needs_broadcast(None, (1, 2)) is None
    assert _needs_broadcast((1, 2), None) is None
    assert _needs_broadcast((1, 2), (1, 2)) is None
    assert _needs_broadcast((1, 2), (2, 2)) == (2, 2)
    assert _needs_broadcast((3, 2), (2, 2)) is None  # Should raise ValueError in broadcast_shapes, returning None


def test_process_broadcast_node() -> None:
    # Op doesn't exist
    node1 = IRNode(id="node1", op_type="UnknownOp", inputs=[])
    graph = IRGraph(name="test", nodes={"node1": node1}, outputs=[])
    assert _process_broadcast_node(graph, node1) is False

    # Not 2 inputs
    node2 = IRNode(id="node2", op_type="Add", inputs=["in1"])
    graph = IRGraph(name="test", nodes={"node2": node2}, outputs=[])
    assert _process_broadcast_node(graph, node2) is False

    # Target shape None
    node3 = IRNode(id="node3", op_type="Add", inputs=["in1", "in2"])
    graph = IRGraph(name="test", nodes={"in1": IRNode(id="in1", op_type="Input", inputs=[], shape_metadata=(1, 2)), "in2": IRNode(id="in2", op_type="Input", inputs=[], shape_metadata=(1, 2)), "node3": node3}, outputs=[])
    assert _process_broadcast_node(graph, node3) is False

    # Broadcast in1
    node4 = IRNode(id="node4", op_type="Add", inputs=["in1", "in2"])
    graph = IRGraph(name="test", nodes={"in1": IRNode(id="in1", op_type="Input", inputs=[], shape_metadata=(1, 2)), "in2": IRNode(id="in2", op_type="Input", inputs=[], shape_metadata=(2, 2)), "node4": node4}, outputs=[])
    assert _process_broadcast_node(graph, node4) is True
    assert node4.inputs[0].startswith("broadcast_")
    assert node4.inputs[1] == "in2"

    # Broadcast in2
    node5 = IRNode(id="node5", op_type="Add", inputs=["in1", "in2"])
    graph = IRGraph(name="test", nodes={"in1": IRNode(id="in1", op_type="Input", inputs=[], shape_metadata=(2, 2)), "in2": IRNode(id="in2", op_type="Input", inputs=[], shape_metadata=(1, 2)), "node5": node5}, outputs=[])
    assert _process_broadcast_node(graph, node5) is True
    assert node5.inputs[0] == "in1"
    assert node5.inputs[1].startswith("broadcast_")

    # Broadcast both
    node6 = IRNode(id="node6", op_type="Add", inputs=["in1", "in2"])
    graph = IRGraph(name="test", nodes={"in1": IRNode(id="in1", op_type="Input", inputs=[], shape_metadata=(1, 2)), "in2": IRNode(id="in2", op_type="Input", inputs=[], shape_metadata=(2, 1)), "node6": node6}, outputs=[])
    assert _process_broadcast_node(graph, node6) is True
    assert node6.inputs[0].startswith("broadcast_")
    assert node6.inputs[1].startswith("broadcast_")


def test_broadcast_explicitizer_pass() -> None:
    node1 = IRNode(id="node1", op_type="Add", inputs=["in1", "in2"])
    graph = IRGraph(name="test", nodes={"in1": IRNode(id="in1", op_type="Input", inputs=[], shape_metadata=(1, 2)), "in2": IRNode(id="in2", op_type="Input", inputs=[], shape_metadata=(2, 2)), "node1": node1}, outputs=["node1"])

    modified = broadcast_explicitizer_pass(graph)
    assert modified is True
