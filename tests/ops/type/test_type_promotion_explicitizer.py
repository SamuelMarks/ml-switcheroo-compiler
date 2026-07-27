# ruff: noqa: E501
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.transforms.passes.type_promotion_explicitizer import _inject_cast_node, _needs_cast, type_promotion_explicitizer_pass


def test_inject_cast_node():
    graph = LogicalGraph()
    n1 = LogicalNode("n1", "Input", shape_metadata=(2,))
    graph.nodes["n1"] = n1
    new_id = _inject_cast_node(graph, "n1", "float32")
    assert new_id in graph.nodes
    assert graph.nodes[new_id].op_type == "Cast"
    assert graph.nodes[new_id].attributes["dtype"] == "float32"


def test_needs_cast():
    assert _needs_cast(None, "float32") is None
    assert _needs_cast("float32", "float32") is None
    assert _needs_cast("int32", "float32") == "float32"
    assert _needs_cast("unknown", "float32") is None


def test_type_promotion_explicitizer_pass(mocker):
    mocker.patch("ml_switcheroo_compiler.transforms.passes.type_promotion_explicitizer.dtype_inference_pass")
    graph = LogicalGraph()
    n1 = LogicalNode("n1", "Input", attributes={"dtype": "int32"})
    n2 = LogicalNode("n2", "Input", attributes={"dtype": "float32"})
    n3 = LogicalNode("n3", "Add", inputs=["n1", "n2"])
    graph.nodes["n1"] = n1
    graph.nodes["n2"] = n2
    graph.nodes["n3"] = n3
    assert type_promotion_explicitizer_pass(graph) is True
    assert graph.nodes["n3"].inputs[0].startswith("cast_")
    n9 = LogicalNode("n9", "Input", attributes={"dtype": "float32"})
    n10 = LogicalNode("n10", "Input", attributes={"dtype": "int32"})
    n11 = LogicalNode("n11", "Add", inputs=["n9", "n10"])
    graph.nodes["n9"] = n9
    graph.nodes["n10"] = n10
    graph.nodes["n11"] = n11
    type_promotion_explicitizer_pass(graph)
    assert graph.nodes["n11"].inputs[1].startswith("cast_")
    graph2 = LogicalGraph()
    n4 = LogicalNode("n4", "Input", attributes={"dtype": "float32"})
    n5 = LogicalNode("n5", "Add", inputs=["n4"])
    graph2.nodes["n4"] = n4
    graph2.nodes["n5"] = n5
    assert type_promotion_explicitizer_pass(graph2) is False
    graph3 = LogicalGraph()
    n6 = LogicalNode("n6", "Input", attributes={"dtype": "float32"})
    n7 = LogicalNode("n7", "Input", attributes={"dtype": "float32"})
    n8 = LogicalNode("n8", "Add", inputs=["n6", "n7"])
    graph3.nodes["n6"] = n6
    graph3.nodes["n7"] = n7
    graph3.nodes["n8"] = n8
    assert type_promotion_explicitizer_pass(graph3) is False
