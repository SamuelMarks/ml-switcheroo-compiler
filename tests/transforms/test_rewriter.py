from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.transforms.rewriter import shape_aware_rewrite


def test_shape_aware_rewrite():
    graph = LogicalGraph()
    node = LogicalNode(id="reshape", op_type="Reshape", inputs=[], shape_metadata=(10,), attributes={})
    graph.nodes["reshape"] = node

    node2 = LogicalNode(id="add", op_type="Add", inputs=[], shape_metadata=None, attributes={})
    graph.nodes["add"] = node2

    graph.outputs = ["add"]

    new_graph = shape_aware_rewrite(graph)
    assert new_graph.nodes["reshape"].attributes["explicit_shape"] == [10]
    assert new_graph.nodes["add"].attributes["requires_strict_cast"] is True
