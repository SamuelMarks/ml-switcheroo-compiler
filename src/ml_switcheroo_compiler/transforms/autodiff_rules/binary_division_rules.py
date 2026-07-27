"""Binary rules for division."""

import enum

from ml_switcheroo_compiler.ops.base import emit_ir_node
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import register_jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp


class UnconnectedGradients(enum.Enum):
    """Unconnected gradients enum."""

    NONE = "none"
    ZERO = "zero"


@register_vjp("TruncateDiv")
def truncatediv_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the TruncateDiv operation."""
    x, y = node.inputs
    zero = emit_ir_node(graph, "Constant", [], None, attributes={"value": 0.0})
    zero_x = emit_ir_node(graph, "Multiply", [x, zero], graph.nodes[x].shape_metadata)
    zero_y = emit_ir_node(graph, "Multiply", [y, zero], graph.nodes[y].shape_metadata)
    return (zero_x, zero_y)


@register_jvp("TruncateDiv")
def truncatediv_jvp(tangent_x: object, tangent_y: object, x: object, y: object, **kwargs: object) -> str:
    """Computes the Jacobian-Vector Product (JVP) for the TruncateDiv operation."""
    graph = kwargs.get("graph")
    zero = emit_ir_node(graph, "Constant", [], None, attributes={"value": 0.0})
    dz = emit_ir_node(graph, "Multiply", [x, zero], graph.nodes[x].shape_metadata)
    return dz


@register_vjp("TruncateMod")
def truncatemod_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the TruncateMod operation."""
    x, y = node.inputs
    # dx = cotangent
    # dy = cotangent * -TruncateDiv(x, y)
    trunc_div = emit_ir_node(graph, "TruncateDiv", [x, y], graph.nodes[x].shape_metadata)
    neg_trunc_div = emit_ir_node(graph, "Negative", [trunc_div], graph.nodes[x].shape_metadata)
    dy = emit_ir_node(graph, "Multiply", [cotangent, neg_trunc_div], graph.nodes[y].shape_metadata)
    return (cotangent, dy)


@register_jvp("TruncateMod")
def truncatemod_jvp(tangent_x: object, tangent_y: object, x: object, y: object, **kwargs: object) -> str:
    """Computes the Jacobian-Vector Product (JVP) for the TruncateMod operation."""
    graph = kwargs.get("graph")
    # dz = dx - dy * TruncateDiv(x, y)
    trunc_div = emit_ir_node(graph, "TruncateDiv", [x, y], graph.nodes[x].shape_metadata)
    term = emit_ir_node(graph, "Multiply", [tangent_y, trunc_div], graph.nodes[y].shape_metadata)
    dz = emit_ir_node(graph, "Subtract", [tangent_x, term], graph.nodes[x].shape_metadata)
    return dz


@register_vjp("FloorDivide")
def floordivide_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the FloorDivide operation."""
    return truncatediv_vjp(graph, node, cotangent)


@register_jvp("FloorDivide")
def floordivide_jvp(tangent_x: object, tangent_y: object, x: object, y: object, **kwargs: object) -> str:
    """Computes the Jacobian-Vector Product (JVP) for the FloorDivide operation."""
    return truncatediv_jvp(tangent_x, tangent_y, x, y, **kwargs)


@register_vjp("Remainder")
def remainder_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the Remainder operation.

    z = x % y = x - y * FloorDivide(x, y)
    dx = cotangent
    dy = cotangent * -FloorDivide(x, y)
    """
    x, y = node.inputs
    floor_div = emit_ir_node(graph, "FloorDivide", [x, y], graph.nodes[x].shape_metadata)
    neg_floor_div = emit_ir_node(graph, "Negative", [floor_div], graph.nodes[x].shape_metadata)
    dy = emit_ir_node(graph, "Multiply", [cotangent, neg_floor_div], graph.nodes[y].shape_metadata)
    return (cotangent, dy)


@register_jvp("Remainder")
def remainder_jvp(tangent_x: object, tangent_y: object, x: object, y: object, **kwargs: object) -> str:
    """Computes the Jacobian-Vector Product (JVP) for the Remainder operation."""
    graph = kwargs.get("graph")
    floor_div = emit_ir_node(graph, "FloorDivide", [x, y], graph.nodes[x].shape_metadata)
    term = emit_ir_node(graph, "Multiply", [tangent_y, floor_div], graph.nodes[y].shape_metadata)
    dz = emit_ir_node(graph, "Subtract", [tangent_x, term], graph.nodes[x].shape_metadata)
    return dz


@register_vjp("Fmod")
def fmod_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the Fmod operation."""
    return remainder_vjp(graph, node, cotangent)


@register_jvp("Fmod")
def fmod_jvp(tangent_x: object, tangent_y: object, x: object, y: object, **kwargs: object) -> str:
    """Computes the Jacobian-Vector Product (JVP) for the Fmod operation."""
    return remainder_jvp(tangent_x, tangent_y, x, y, **kwargs)
