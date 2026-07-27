"""Shape rules for misc."""

from ml_switcheroo_compiler.ops.base import emit_ir_node


def _generic_shape_vjp(graph: object, node: object, cotangent: str) -> tuple:
    x = node.inputs[0]
    return (emit_ir_node(graph, "Reshape", [cotangent], graph.nodes[x].shape_metadata),)


def _generic_shape_jvp(graph: object, node: object, tangent: str) -> str:
    return emit_ir_node(graph, node.op_type, [tangent], graph.nodes[node.id].shape_metadata, attributes=node.attributes)
