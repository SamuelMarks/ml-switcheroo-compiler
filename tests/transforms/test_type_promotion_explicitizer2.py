from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.transforms.passes.type_promotion_explicitizer import _inject_cast_node, type_promotion_explicitizer_pass


def test_type_promotion_explicitizer_full():
    graph = LogicalGraph()
    # Mixed precision inputs to trigger promotion logic
    node1 = LogicalNode(id="a", op_type="Input", inputs=[], shape_metadata=(2,), attributes={"dtype": DType.Float32})
    node2 = LogicalNode(id="b", op_type="Input", inputs=[], shape_metadata=(2,), attributes={"dtype": DType.Float64})
    node3 = LogicalNode(id="add", op_type="Add", inputs=["a", "b"], shape_metadata=(2,), attributes={"dtype": DType.Float64})

    graph.nodes["a"] = node1
    graph.nodes["b"] = node2
    graph.nodes["add"] = node3

    _inject_cast_node(graph, "a", "float64")
    type_promotion_explicitizer_pass(graph)


def test_type_promotion_explicitizer_needs_cast_error():
    from ml_switcheroo_compiler.transforms.passes.type_promotion_explicitizer import _needs_cast

    assert _needs_cast("invalid1", "invalid2") is None


def test_type_promotion_explicitizer_dt2_cast():
    graph = LogicalGraph()
    # Cast second arg
    node1 = LogicalNode(id="a", op_type="Input", inputs=[], shape_metadata=(2,), attributes={"dtype": DType.Float64})
    node2 = LogicalNode(id="b", op_type="Input", inputs=[], shape_metadata=(2,), attributes={"dtype": DType.Float32})
    node3 = LogicalNode(id="add", op_type="Add", inputs=["a", "b"], shape_metadata=(2,), attributes={"dtype": DType.Float64})

    graph.nodes["a"] = node1
    graph.nodes["b"] = node2
    graph.nodes["add"] = node3
    type_promotion_explicitizer_pass(graph)
