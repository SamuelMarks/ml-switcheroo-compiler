# ruff: noqa: E501
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.transforms.rewriter import shape_aware_rewrite


def test_shape_aware_rewrite():
    graph = LogicalGraph("TestGraph", outputs=["n2"])
    n1 = LogicalNode("n1", "Reshape", shape_metadata=(2, 3), attributes={"newshape": (-1, 3)})
    n2 = LogicalNode("n2", "Add", inputs=["n1"], shape_metadata=(2, 3))
    n3 = LogicalNode("n3", "Other", shape_metadata=(2, 3))
    graph.nodes["n1"] = n1
    graph.nodes["n2"] = n2
    graph.nodes["n3"] = n3
    new_graph = shape_aware_rewrite(graph)
    assert new_graph.name == "TestGraph_rewritten"
    assert new_graph.outputs == ["n2"]
    assert new_graph.nodes["n1"].attributes["explicit_shape"] == [2, 3]
    assert new_graph.nodes["n2"].attributes["requires_strict_cast"] is True
    assert "requires_strict_cast" not in new_graph.nodes["n3"].attributes
