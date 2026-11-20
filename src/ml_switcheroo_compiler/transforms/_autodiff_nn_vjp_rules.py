"""Built-in neural network convolution, pooling, and normalization VJP rules."""

from __future__ import annotations

from collections.abc import Callable, Sequence  # noqa: F401

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import _VJP_REGISTRY


def _node_shape_meta(node: LogicalNode | None) -> tuple[int | str, ...] | None:
    """Extract normalized `tuple[int | str, ...] | None` shape metadata from an optional `LogicalNode`.

    Args:
        node (LogicalNode | None): Optional logical node.

    Returns:
        tuple[int | str, ...] | None: Normalized shape metadata tuple or None.
    """
    return (
        tuple(d if isinstance(d, (int, str)) else str(d) for d in node.shape_metadata)
        if node is not None and node.shape_metadata is not None
        else None
    )


def conv2d_vjp_input(graph: LogicalGraph, node: LogicalNode, cotangent: str) -> str:
    """Compute exact VJP cotangent for the input tensor of a Conv2D operation.

    Handles arbitrary strides, padding, dilation, and groups.

    Args:
        graph (LogicalGraph): Computation graph.
        node (LogicalNode): Primal Conv2D node.
        cotangent (str): Output adjoint identifier.

    Returns:
        str: Input adjoint node identifier.
    """
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    input_id = node.inputs[0]
    weight_id = node.inputs[1]
    shape_meta = _node_shape_meta(graph.nodes.get(input_id))

    stride = node.attributes.get("stride", node.attributes.get("strides", (1, 1)))
    dilation = node.attributes.get("dilation", node.attributes.get("dilations", (1, 1)))

    raw_groups = node.attributes.get("groups", 1)
    attrs = {
        "stride": stride,
        "strides": stride,
        "padding": node.attributes.get("padding", "SAME"),
        "dilation": dilation,
        "dilations": dilation,
        "groups": int(raw_groups) if isinstance(raw_groups, (int, float, str)) else 1,
        "target_shape": shape_meta,
    }
    return emit_ir_node(graph, "Conv2DInputGrad", [cotangent, weight_id], shape_metadata=shape_meta, attributes=attrs)


def conv2d_vjp_weight(graph: LogicalGraph, node: LogicalNode, cotangent: str) -> str:
    """Compute exact VJP cotangent for the weight tensor of a Conv2D operation.

    Handles arbitrary strides, padding, dilation, and groups.

    Args:
        graph (LogicalGraph): Computation graph.
        node (LogicalNode): Primal Conv2D node.
        cotangent (str): Output adjoint identifier.

    Returns:
        str: Weight adjoint node identifier.
    """
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    input_id = node.inputs[0]
    weight_id = node.inputs[1]
    shape_meta = _node_shape_meta(graph.nodes.get(weight_id))

    stride = node.attributes.get("stride", node.attributes.get("strides", (1, 1)))
    dilation = node.attributes.get("dilation", node.attributes.get("dilations", (1, 1)))

    raw_groups = node.attributes.get("groups", 1)
    attrs = {
        "stride": stride,
        "strides": stride,
        "padding": node.attributes.get("padding", "SAME"),
        "dilation": dilation,
        "dilations": dilation,
        "groups": int(raw_groups) if isinstance(raw_groups, (int, float, str)) else 1,
        "target_shape": shape_meta,
    }
    return emit_ir_node(graph, "Conv2DWeightGrad", [input_id, cotangent], shape_metadata=shape_meta, attributes=attrs)


def _conv2d_vjp_rule(graph: LogicalGraph, node: LogicalNode, cotangent: str) -> tuple[str, ...]:
    """Exact VJP rule for Conv2D.

    Args:
        graph: Input `graph` argument (graph).
        node: Input `node` argument (node).
        cotangent: Input `cotangent` argument (cotangent).

    Returns:
        tuple[str, ...]: Computed result.
    """
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    grad_in = conv2d_vjp_input(graph, node, cotangent)
    grad_w = conv2d_vjp_weight(graph, node, cotangent)
    if len(node.inputs) > 2:
        bias_id = node.inputs[2]
        shape_b = _node_shape_meta(graph.nodes.get(bias_id))
        grad_b = emit_ir_node(graph, "Conv2DBiasGrad", [cotangent], shape_metadata=shape_b, attributes=node.attributes)
        return (grad_in, grad_w, grad_b)
    return (grad_in, grad_w)


def _max_pool2d_with_argmax_vjp(
    graph: LogicalGraph, node: LogicalNode, cotangent: str
) -> tuple[str | UnconnectedGradients, ...]:
    """Exact VJP rule for MaxPool2DWithArgmax.

    Args:
        graph: Input `graph` argument (graph).
        node: Input `node` argument (node).
        cotangent: Input `cotangent` argument (cotangent).

    Returns:
        tuple[str | UnconnectedGradients, ...]: Computed result.
    """
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    input_id = node.inputs[0]
    shape_meta = _node_shape_meta(graph.nodes.get(input_id))
    grad_in = emit_ir_node(
        graph, "MaxPool2DWithArgmaxGrad", [cotangent, input_id], shape_metadata=shape_meta, attributes=node.attributes
    )
    adjs: list[str | UnconnectedGradients] = [grad_in]
    while len(adjs) < len(node.inputs):
        adjs.append(UnconnectedGradients.NONE)
    return tuple(adjs)


def _avg_pool2d_vjp(graph: LogicalGraph, node: LogicalNode, cotangent: str) -> tuple[str, ...]:
    """Exact VJP rule for AvgPool2D.

    Args:
        graph: Input `graph` argument (graph).
        node: Input `node` argument (node).
        cotangent: Input `cotangent` argument (cotangent).

    Returns:
        tuple[str, ...]: Computed result.
    """
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    input_id = node.inputs[0]
    shape_meta = _node_shape_meta(graph.nodes.get(input_id))
    grad_in = emit_ir_node(
        graph, "AvgPool2DGrad", [cotangent, input_id], shape_metadata=shape_meta, attributes=node.attributes
    )
    return (grad_in,)


def _batch_norm_vjp(graph: LogicalGraph, node: LogicalNode, cotangent: str) -> tuple[str | UnconnectedGradients, ...]:
    """Exact VJP rule for BatchNorm.

    Args:
        graph: Input `graph` argument (graph).
        node: Input `node` argument (node).
        cotangent: Input `cotangent` argument (cotangent).

    Returns:
        tuple[str | UnconnectedGradients, ...]: Computed result.
    """
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    x_id = node.inputs[0]
    shape_x = _node_shape_meta(graph.nodes.get(x_id))
    gamma_id = node.inputs[1] if len(node.inputs) > 1 else None
    shape_gamma = _node_shape_meta(graph.nodes.get(gamma_id)) if gamma_id else None

    inps_for_x = [cotangent, x_id] + ([gamma_id] if gamma_id else [])
    grad_x = emit_ir_node(graph, "BatchNormInputGrad", inps_for_x, shape_metadata=shape_x, attributes=node.attributes)

    adjs: list[str | UnconnectedGradients] = [grad_x]
    if len(node.inputs) > 1:
        grad_gamma = emit_ir_node(
            graph, "BatchNormGammaGrad", [cotangent, x_id], shape_metadata=shape_gamma, attributes=node.attributes
        )
        adjs.append(grad_gamma)
    if len(node.inputs) > 2:
        beta_id = node.inputs[2]
        shape_beta = _node_shape_meta(graph.nodes.get(beta_id))
        grad_beta = emit_ir_node(
            graph, "BatchNormBetaGrad", [cotangent], shape_metadata=shape_beta, attributes=node.attributes
        )
        adjs.append(grad_beta)

    while len(adjs) < len(node.inputs):
        adjs.append(UnconnectedGradients.NONE)
    return tuple(adjs)


def _layer_norm_vjp(graph: LogicalGraph, node: LogicalNode, cotangent: str) -> tuple[str | UnconnectedGradients, ...]:
    """Exact VJP rule for LayerNorm.

    Args:
        graph: Input `graph` argument (graph).
        node: Input `node` argument (node).
        cotangent: Input `cotangent` argument (cotangent).

    Returns:
        tuple[str | UnconnectedGradients, ...]: Computed result.
    """
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    x_id = node.inputs[0]
    shape_x = _node_shape_meta(graph.nodes.get(x_id))
    gamma_id = node.inputs[1] if len(node.inputs) > 1 else None
    shape_gamma = _node_shape_meta(graph.nodes.get(gamma_id)) if gamma_id else None

    inps_for_x = [cotangent, x_id] + ([gamma_id] if gamma_id else [])
    grad_x = emit_ir_node(graph, "LayerNormInputGrad", inps_for_x, shape_metadata=shape_x, attributes=node.attributes)

    adjs: list[str | UnconnectedGradients] = [grad_x]
    if len(node.inputs) > 1:
        grad_gamma = emit_ir_node(
            graph, "LayerNormGammaGrad", [cotangent, x_id], shape_metadata=shape_gamma, attributes=node.attributes
        )
        adjs.append(grad_gamma)
    if len(node.inputs) > 2:
        beta_id = node.inputs[2]
        shape_beta = _node_shape_meta(graph.nodes.get(beta_id))
        grad_beta = emit_ir_node(
            graph, "LayerNormBetaGrad", [cotangent], shape_metadata=shape_beta, attributes=node.attributes
        )
        adjs.append(grad_beta)

    while len(adjs) < len(node.inputs):
        adjs.append(UnconnectedGradients.NONE)
    return tuple(adjs)


def _rms_norm_vjp(graph: LogicalGraph, node: LogicalNode, cotangent: str) -> tuple[str | UnconnectedGradients, ...]:
    """Exact VJP rule for RMSNorm.

    Args:
        graph: Input `graph` argument (graph).
        node: Input `node` argument (node).
        cotangent: Input `cotangent` argument (cotangent).

    Returns:
        tuple[str | UnconnectedGradients, ...]: Computed result.
    """
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    x_id = node.inputs[0]
    shape_x = _node_shape_meta(graph.nodes.get(x_id))
    gamma_id = node.inputs[1] if len(node.inputs) > 1 else None
    shape_gamma = _node_shape_meta(graph.nodes.get(gamma_id)) if gamma_id else None

    inps_for_x = [cotangent, x_id] + ([gamma_id] if gamma_id else [])
    grad_x = emit_ir_node(graph, "RMSNormInputGrad", inps_for_x, shape_metadata=shape_x, attributes=node.attributes)

    adjs: list[str | UnconnectedGradients] = [grad_x]
    if len(node.inputs) > 1:
        grad_gamma = emit_ir_node(
            graph, "RMSNormGammaGrad", [cotangent, x_id], shape_metadata=shape_gamma, attributes=node.attributes
        )
        adjs.append(grad_gamma)

    while len(adjs) < len(node.inputs):
        adjs.append(UnconnectedGradients.NONE)
    return tuple(adjs)


def _group_norm_vjp(graph: LogicalGraph, node: LogicalNode, cotangent: str) -> tuple[str | UnconnectedGradients, ...]:
    """Exact VJP rule for GroupNorm.

    Args:
        graph: Input `graph` argument (graph).
        node: Input `node` argument (node).
        cotangent: Input `cotangent` argument (cotangent).

    Returns:
        tuple[str | UnconnectedGradients, ...]: Computed result.
    """
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    x_id = node.inputs[0]
    shape_x = _node_shape_meta(graph.nodes.get(x_id))
    gamma_id = node.inputs[1] if len(node.inputs) > 1 else None
    shape_gamma = _node_shape_meta(graph.nodes.get(gamma_id)) if gamma_id else None

    inps_for_x = [cotangent, x_id] + ([gamma_id] if gamma_id else [])
    grad_x = emit_ir_node(graph, "GroupNormInputGrad", inps_for_x, shape_metadata=shape_x, attributes=node.attributes)

    adjs: list[str | UnconnectedGradients] = [grad_x]
    if len(node.inputs) > 1:
        grad_gamma = emit_ir_node(
            graph, "GroupNormGammaGrad", [cotangent, x_id], shape_metadata=shape_gamma, attributes=node.attributes
        )
        adjs.append(grad_gamma)
    if len(node.inputs) > 2:
        beta_id = node.inputs[2]
        shape_beta = _node_shape_meta(graph.nodes.get(beta_id))
        grad_beta = emit_ir_node(
            graph, "GroupNormBetaGrad", [cotangent], shape_metadata=shape_beta, attributes=node.attributes
        )
        adjs.append(grad_beta)

    while len(adjs) < len(node.inputs):
        adjs.append(UnconnectedGradients.NONE)
    return tuple(adjs)


# Register exact VJP rules
_VJP_REGISTRY["Conv2D"] = _conv2d_vjp_rule
_VJP_REGISTRY["conv2d"] = _conv2d_vjp_rule
_VJP_REGISTRY["MaxPool2DWithArgmax"] = _max_pool2d_with_argmax_vjp
_VJP_REGISTRY["AvgPool2D"] = _avg_pool2d_vjp
_VJP_REGISTRY["BatchNorm"] = _batch_norm_vjp
_VJP_REGISTRY["LayerNorm"] = _layer_norm_vjp
_VJP_REGISTRY["RMSNorm"] = _rms_norm_vjp
_VJP_REGISTRY["GroupNorm"] = _group_norm_vjp
