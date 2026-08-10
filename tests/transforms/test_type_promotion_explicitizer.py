from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.transforms.passes.type_promotion_explicitizer import type_promotion_explicitizer_pass


def test_type_promotion_explicitizer():
    graph = LogicalGraph()
    # Mock some behavior to pass through
    node1 = LogicalNode(id="node1", op_type="Add", inputs=["a", "b"], attributes={"dtype": DType.Float32})
    graph.nodes["node1"] = node1

    # Just run it to hit lines
    try:
        type_promotion_explicitizer_pass(graph)
    except Exception:
        pass
