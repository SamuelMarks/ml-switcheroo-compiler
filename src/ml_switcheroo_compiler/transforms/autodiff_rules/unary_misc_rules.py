"""Unary rules for misc."""

from ml_switcheroo_compiler.ops.base import emit_ir_node
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import register_jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp


def _zero_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Docstring."""
    """Docstring."""
    x = node.inputs[0]
    zero = emit_ir_node(graph, "Constant", inputs=[], attributes={"value": 0.0})
    zero_like_x = emit_ir_node(graph, "Multiply", inputs=[x, zero])
    return (zero_like_x,)


def _zero_jvp(graph: object, node: object, tangents: tuple) -> str:
    """Docstring."""
    """Docstring."""
    x = node.inputs[0]
    zero = emit_ir_node(graph, "Constant", inputs=[], attributes={"value": 0.0})
    zero_like_x = emit_ir_node(graph, "Multiply", inputs=[x, zero])
    return zero_like_x


@register_vjp("Conj")
def conj_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Docstring."""
    dx = emit_ir_node(graph, "Conj", [cotangent], graph.nodes[node.inputs[0]].shape_metadata)
    return (dx,)


@register_jvp("Conj")
def conj_jvp(graph: object, node: object, tangents: tuple) -> str:
    """Docstring."""
    dx = emit_ir_node(graph, "Conj", [tangents[0]], graph.nodes[node.inputs[0]].shape_metadata)
    return dx


@register_vjp("Cast")
def cast_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Docstring."""
    x = node.inputs[0]
    # We cast the cotangent to the original dtype of x.
    # Note: cotangent type doesn't need explicit cast if the IR node automatically handles it,
    # but Cast explicitly casts to a dtype.
    dx = emit_ir_node(graph, "Cast", [cotangent], graph.nodes[x].shape_metadata)
    return (dx,)


@register_jvp("Cast")
def cast_jvp(graph: object, node: object, tangents: tuple) -> str:
    """Docstring."""
    # We cast the tangent to the target dtype of Cast.
    dx = emit_ir_node(graph, "Cast", [tangents[0]], graph.nodes[node.id].shape_metadata, attributes=node.attributes)
    return dx
