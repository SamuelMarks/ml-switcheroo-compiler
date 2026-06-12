"""Standard Vector-Jacobian Product (VJP) rules."""

import uuid
from typing import Optional
from ml_switcheroo_ir import LogicalGraph, LogicalNode
from ml_switcheroo.grad import register_vjp


def _emit_node(
    graph: LogicalGraph, op_type: str, inputs: list[str], shape_metadata: object = None
) -> str:
    """Docstring."""
    nid = f"{op_type.lower()}_{uuid.uuid4().hex[:6]}"
    node = LogicalNode(
        id=nid, op_type=op_type, inputs=inputs, shape_metadata=shape_metadata
    )
    graph.nodes[nid] = node
    return nid


@register_vjp("Add")
def vjp_add(graph: LogicalGraph, node: LogicalNode, adj: str) -> list[Optional[str]]:
    """VJP for Add: dl/dx = dl/dy, dl/dz = dl/dy."""
    # Note: Does not handle broadcast reduction yet.
    return [adj, adj]


@register_vjp("Sub")
def vjp_sub(graph: LogicalGraph, node: LogicalNode, adj: str) -> list[Optional[str]]:
    """VJP for Sub: dl/dx = dl/dy, dl/dz = -dl/dy."""
    neg_adj = _emit_node(graph, "Neg", [adj], graph.nodes[adj].shape_metadata)
    return [adj, neg_adj]


@register_vjp("Mul")
def vjp_mul(graph: LogicalGraph, node: LogicalNode, adj: str) -> list[Optional[str]]:
    """VJP for Mul: dl/dx = dl/dy * z, dl/dz = dl/dy * x."""
    x, y = node.inputs
    adj_x = _emit_node(graph, "Mul", [adj, y], graph.nodes[x].shape_metadata)
    adj_y = _emit_node(graph, "Mul", [adj, x], graph.nodes[y].shape_metadata)
    return [adj_x, adj_y]


@register_vjp("Div")
def vjp_div(graph: LogicalGraph, node: LogicalNode, adj: str) -> list[Optional[str]]:
    """VJP for Div: dl/dx = dl/dy / y, dl/dy_in = -dl/dy * x / y^2."""
    x, y = node.inputs
    # dl/dx = adj / y
    adj_x = _emit_node(graph, "Div", [adj, y], graph.nodes[x].shape_metadata)

    # dl/dy_in = -adj * x / y^2
    neg_adj = _emit_node(graph, "Neg", [adj])
    neg_adj_x = _emit_node(graph, "Mul", [neg_adj, x])
    y_sq = _emit_node(graph, "Mul", [y, y])
    adj_y = _emit_node(graph, "Div", [neg_adj_x, y_sq], graph.nodes[y].shape_metadata)
    return [adj_x, adj_y]


@register_vjp("Exp")
def vjp_exp(graph: LogicalGraph, node: LogicalNode, adj: str) -> list[Optional[str]]:
    """VJP for Exp: dl/dx = dl/dy * exp(x)."""
    x = node.inputs[0]
    exp_x = _emit_node(graph, "Exp", [x])
    adj_x = _emit_node(graph, "Mul", [adj, exp_x], graph.nodes[x].shape_metadata)
    return [adj_x]


@register_vjp("Log")
def vjp_log(graph: LogicalGraph, node: LogicalNode, adj: str) -> list[Optional[str]]:
    """VJP for Log: dl/dx = dl/dy / x."""
    x = node.inputs[0]
    adj_x = _emit_node(graph, "Div", [adj, x], graph.nodes[x].shape_metadata)
    return [adj_x]


@register_vjp("Sum")
def vjp_sum(graph: LogicalGraph, node: LogicalNode, adj: str) -> list[Optional[str]]:
    """VJP for Sum."""
    x = node.inputs[0]
    # In a full implementation, Expand/Broadcast is needed based on the 'axes'
    # attribute.
    # For now we emit an Expand node placeholder.
    adj_x = _emit_node(graph, "Expand", [adj], graph.nodes[x].shape_metadata)
    return [adj_x]


@register_vjp("Mean")
def vjp_mean(graph: LogicalGraph, node: LogicalNode, adj: str) -> list[Optional[str]]:
    """VJP for Mean."""
    # Simplified placeholder
    x = node.inputs[0]
    adj_x = _emit_node(graph, "Expand", [adj], graph.nodes[x].shape_metadata)
    return [adj_x]


@register_vjp("Max")
def vjp_max(graph: LogicalGraph, node: LogicalNode, adj: str) -> list[Optional[str]]:
    """VJP for Max (Placeholder)."""
    return [adj]


@register_vjp("Min")
def vjp_min(graph: LogicalGraph, node: LogicalNode, adj: str) -> list[Optional[str]]:
    """VJP for Min (Placeholder)."""
    return [adj]


@register_vjp("MatMul")
def vjp_matmul(graph: LogicalGraph, node: LogicalNode, adj: str) -> list[Optional[str]]:
    """VJP for MatMul."""
    x, y = node.inputs
    x_t = _emit_node(graph, "Transpose", [x])
    y_t = _emit_node(graph, "Transpose", [y])
    adj_x = _emit_node(graph, "MatMul", [adj, y_t], graph.nodes[x].shape_metadata)
    adj_y = _emit_node(graph, "MatMul", [x_t, adj], graph.nodes[y].shape_metadata)
    return [adj_x, adj_y]


@register_vjp("Gemm")
def vjp_gemm(graph: LogicalGraph, node: LogicalNode, adj: str) -> list[Optional[str]]:
    """VJP for Gemm (Placeholder)."""
    return [adj, adj, adj][: len(node.inputs)]


@register_vjp("Transpose")
def vjp_transpose(
    graph: LogicalGraph, node: LogicalNode, adj: str
) -> list[Optional[str]]:
    """VJP for Transpose."""
    x = node.inputs[0]
    adj_x = _emit_node(graph, "Transpose", [adj], graph.nodes[x].shape_metadata)
    return [adj_x]


@register_vjp("Conv")
def vjp_conv(graph: LogicalGraph, node: LogicalNode, adj: str) -> list[Optional[str]]:
    """VJP for Conv (Placeholder)."""
    return [adj, adj, adj][: len(node.inputs)]


@register_vjp("MaxPool")
def vjp_maxpool(
    graph: LogicalGraph, node: LogicalNode, adj: str
) -> list[Optional[str]]:
    """VJP for MaxPool (Placeholder)."""
    return [adj]


@register_vjp("Relu")
def vjp_relu(graph: LogicalGraph, node: LogicalNode, adj: str) -> list[Optional[str]]:
    """VJP for Relu (Placeholder)."""
    x = node.inputs[0]
    zero_id = _emit_node(graph, "Constant", [], graph.nodes[x].shape_metadata)
    graph.nodes[zero_id].attributes["value"] = 0.0
    greater_id = _emit_node(graph, "Greater", [x, zero_id])
    adj_x = _emit_node(graph, "Mul", [adj, greater_id], graph.nodes[x].shape_metadata)
    return [adj_x]


@register_vjp("Softmax")
def vjp_softmax(
    graph: LogicalGraph, node: LogicalNode, adj: str
) -> list[Optional[str]]:
    """VJP for Softmax (Placeholder)."""
    return [adj]
