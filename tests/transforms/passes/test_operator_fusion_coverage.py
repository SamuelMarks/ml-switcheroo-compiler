from ml_switcheroo_compiler.ir.core import LogicalNode
from ml_switcheroo_compiler.transforms.passes.operator_fusion import estimate_node_cost


def test_operator_fusion_cost_fused():
    node = LogicalNode(id="n1", op_type="FusedMultiHeadAttention")
    cost = estimate_node_cost(node)
    assert cost == 2
