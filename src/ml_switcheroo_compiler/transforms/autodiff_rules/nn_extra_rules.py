"""Extra autodiff rules for neural network operations."""

from ml_switcheroo_compiler.ops.base import emit_ir_node
from ml_switcheroo_compiler.transforms.autodiff_rules.common import make_zero_jvp, make_zero_vjp
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import register_jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp


@register_vjp("TimeDistributed")
def time_distributed_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for TimeDistributed.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    res = emit_ir_node(graph, "TimeDistributed", [cotangent], graph.nodes[node.inputs[0]].shape_metadata, node.attributes)
    return (res,)


@register_jvp("TimeDistributed")
def time_distributed_jvp(graph: object, node: object, tangents: tuple) -> str:
    """JVP for TimeDistributed.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        tangents (tuple): The tangents parameter.

    Returns:
        str: Result.
    """
    t_x = tangents[0]
    if t_x is None:
        return None
    return emit_ir_node(graph, "TimeDistributed", [t_x], node.shape_metadata, node.attributes)


for op_name in ["GroupMean", "GroupNorm", "GroupVariance", "Rope", "ScaledDotProductAttention"]:
    register_vjp(op_name)(make_zero_vjp(op_name))
    register_jvp(op_name)(make_zero_jvp(op_name))
