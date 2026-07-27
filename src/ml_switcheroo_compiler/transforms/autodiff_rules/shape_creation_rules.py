"""Shape rules for creation."""

from ml_switcheroo_compiler.ops.base import emit_ir_node
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import register_jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp


@register_vjp("Zeros")
@register_vjp("Ones")
@register_vjp("Arange")
@register_vjp("Range")
@register_vjp("Logspace")
@register_vjp("ConstantOfShape")
@register_vjp("ManualSeed")
@register_vjp("Rand")
@register_vjp("Randn")
@register_vjp("Randint")
@register_vjp("HammingWindow")
@register_vjp("HannWindow")
@register_vjp("KaiserWindow")
def _creation_vjp(graph: object, node: object, cotangent: str) -> tuple:
    from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients

    return tuple([UnconnectedGradients.ZERO] * len(node.inputs))


@register_jvp("Zeros")
@register_jvp("ZerosLike")
@register_jvp("Ones")
@register_jvp("Arange")
@register_jvp("Range")
@register_jvp("Logspace")
@register_jvp("ConstantOfShape")
@register_jvp("ManualSeed")
@register_jvp("Rand")
@register_jvp("Randn")
@register_jvp("Randint")
@register_jvp("HammingWindow")
@register_jvp("HannWindow")
@register_jvp("KaiserWindow")
def _creation_jvp(graph: object, node: object, tangents: tuple) -> str:

    # output tangent is zero
    return emit_ir_node(graph, "ZerosLike", [node.id], node.shape_metadata)


@register_vjp("Full")
def full_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for Full."""
    from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients

    # inputs: shape, fill_value
    if len(node.inputs) < 2:
        return (UnconnectedGradients.ZERO,) * len(node.inputs)

    fill_value = node.inputs[1]
    # Gradient wrt shape is ZERO.
    # Gradient wrt fill_value is sum of cotangent over all axes.
    # We can just sum all elements.
    shape_metadata = graph.nodes[fill_value].shape_metadata
    grad_fill = emit_ir_node(graph, "Sum", [cotangent], shape_metadata, {})
    return (UnconnectedGradients.ZERO, grad_fill)


@register_jvp("Full")
def full_jvp(graph: object, node: object, tangents: tuple) -> str:
    """JVP for Full."""
    if len(tangents) < 2:
        return emit_ir_node(graph, "ZerosLike", [node.id], node.shape_metadata)

    t_fill = tangents[1]
    # tangent is a full tensor filled with t_fill. But t_fill is scalar or same shape as fill_value.
    # We can just broadcast t_fill to the output shape.
    return emit_ir_node(graph, "BroadcastToLike", [t_fill, node.id], node.shape_metadata, {})
