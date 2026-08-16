from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.serialization.ir_format import graph_to_flatbuffers, graph_to_json, graph_to_protobuf, json_to_graph


def test_ir_format():
    graph = IRGraph()
    node = IRNode(id="n1", op_type="add", inputs=["n2", "n3"], shape_metadata=())
    graph.nodes["n1"] = node

    j = graph_to_json(graph)
    assert '"n1"' in j
    assert '"add"' in j

    g2 = json_to_graph(j)
    assert "n1" in g2.nodes
    assert g2.nodes["n1"].op_type == "add"
    assert g2.nodes["n1"].inputs == ["n2", "n3"]

    assert graph_to_protobuf(graph) == b""
    assert graph_to_flatbuffers(graph) == b""
