"""Binary rules for trig."""

import enum

from ml_switcheroo_compiler.ops.base import emit_ir_node
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import register_jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp


class UnconnectedGradients(enum.Enum):
    """Unconnected gradients enum."""

    NONE = "none"
    ZERO = "zero"


@register_vjp("Atan2")
def atan2_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for Atan2."""
    y = node.inputs[0]
    x = node.inputs[1]
    y_sq = emit_ir_node(graph, "Square", [y], graph.nodes[y].shape_metadata)
    x_sq = emit_ir_node(graph, "Square", [x], graph.nodes[x].shape_metadata)
    denom = emit_ir_node(graph, "Add", [x_sq, y_sq], graph.nodes[x].shape_metadata)

    # dy: x / (x^2 + y^2)
    dy = emit_ir_node(graph, "TrueDivide", [x, denom], graph.nodes[y].shape_metadata)
    cot_dy = emit_ir_node(graph, "Multiply", [cotangent, dy], graph.nodes[y].shape_metadata)

    # dx: -y / (x^2 + y^2)
    dx = emit_ir_node(graph, "TrueDivide", [y, denom], graph.nodes[x].shape_metadata)
    dx_neg = emit_ir_node(graph, "Negative", [dx], graph.nodes[x].shape_metadata)
    cot_dx = emit_ir_node(graph, "Multiply", [cotangent, dx_neg], graph.nodes[x].shape_metadata)

    return (cot_dy, cot_dx)


@register_jvp("Atan2")
def atan2_jvp(tangent_y: str, tangent_x: str, y: str, x: str, **kwargs: object) -> str:
    """JVP for Atan2."""
    graph = kwargs["graph"]
    y_sq = emit_ir_node(graph, "Square", [y], graph.nodes[y].shape_metadata)
    x_sq = emit_ir_node(graph, "Square", [x], graph.nodes[x].shape_metadata)
    denom = emit_ir_node(graph, "Add", [x_sq, y_sq], graph.nodes[x].shape_metadata)

    dy = emit_ir_node(graph, "TrueDivide", [x, denom], graph.nodes[y].shape_metadata)
    dy_term = emit_ir_node(graph, "Multiply", [tangent_y, dy], graph.nodes[y].shape_metadata)

    dx = emit_ir_node(graph, "TrueDivide", [y, denom], graph.nodes[x].shape_metadata)
    dx_neg = emit_ir_node(graph, "Negative", [dx], graph.nodes[x].shape_metadata)
    dx_term = emit_ir_node(graph, "Multiply", [tangent_x, dx_neg], graph.nodes[x].shape_metadata)

    return emit_ir_node(graph, "Add", [dy_term, dx_term], graph.nodes[y].shape_metadata)
