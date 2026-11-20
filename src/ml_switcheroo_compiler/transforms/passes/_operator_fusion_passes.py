"""Vertical conv-bias-activation and horizontal sibling operator fusion passes."""

from __future__ import annotations

from ml_switcheroo_ir import IRGraph, IRNode

from ml_switcheroo_compiler.transforms.passes._operator_fusion_clusters import (
    ELEMENTWISE_OPS,
    _count_consumers,
    _has_graph_nodes,
)


def _find_conv_and_bias_inputs(
    graph: IRGraph,
    bias_node: IRNode,
) -> tuple[str | None, str | None]:
    """Identify the convolution input ID and bias tensor ID from a binary Add/BiasAdd node.

    Args:
        graph (IRGraph): The computation graph.
        bias_node (IRNode): The binary Add or BiasAdd node.

    Returns:
        tuple[str | None, str | None]: (conv_node_id, bias_value_id) or (None, None).
    """
    for b_in in bias_node.inputs:
        cand = graph.nodes.get(b_in)
        if cand is not None and cand.op_type in ("Conv2D", "Conv", "Conv3D"):
            remaining = [x for x in bias_node.inputs if x != b_in]
            return b_in, (remaining[0] if remaining else None)
    return None, None


def _try_fuse_conv_bias_act_node(
    graph: IRGraph,
    act_node: IRNode,
    consumer_counts: dict[str, int],
) -> bool:
    """Try fusing a single activation node with its preceding BiasAdd and Conv nodes.

    Args:
        graph (IRGraph): Target computation graph.
        act_node (IRNode): Candidate activation node.
        consumer_counts (dict[str, int]): Consumer counts map.

    Returns:
        bool: True if the Conv+Bias+Activation chain was fused, False otherwise.
    """
    if act_node.op_type not in ("Relu", "Gelu", "Silu", "Sigmoid", "Tanh") or len(act_node.inputs) != 1:
        return False

    bias_id = act_node.inputs[0]
    bias_node = graph.nodes.get(bias_id)
    if bias_node is None or bias_node.op_type not in ("Add", "BiasAdd") or len(bias_node.inputs) != 2:
        return False
    if consumer_counts.get(bias_id, 0) != 1:
        return False

    conv_id, bias_val_id = _find_conv_and_bias_inputs(graph, bias_node)
    if conv_id is None or bias_val_id is None or consumer_counts.get(conv_id, 0) != 1:
        return False

    conv_node = graph.nodes[conv_id]
    fused_attrs = dict(conv_node.attributes)
    fused_attrs["activation"] = act_node.op_type
    fused_attrs["has_bias"] = True

    fused_node = IRNode(
        id=act_node.id,
        op_type=f"Fused{conv_node.op_type}Bias{act_node.op_type}",
        inputs=list(conv_node.inputs) + [bias_val_id],
        shape_metadata=act_node.shape_metadata,
    )
    fused_node.attributes = fused_attrs

    del graph.nodes[conv_id]
    del graph.nodes[bias_id]
    graph.nodes[act_node.id] = fused_node
    return True


def fuse_conv_bias_relu(graph: IRGraph) -> bool:
    """Fuse Conv2D/Conv + Add/BiasAdd + Relu/Activation sequences into FusedConv2DBiasRelu nodes.

    Args:
        graph (IRGraph): Target computation graph to optimize.

    Returns:
        bool: True if any Conv-Bias-Activation sequences were fused, False otherwise.
    """
    if not _has_graph_nodes(graph):
        return False

    modified: bool = False
    changed: bool = True

    while changed:
        changed = False
        consumer_counts = _count_consumers(graph)
        for act_node in list(graph.nodes.values()):
            if _try_fuse_conv_bias_act_node(graph, act_node, consumer_counts):
                changed = True
                modified = True
                break

    return modified


def _try_fuse_elem_act_node(
    graph: IRGraph,
    act_node: IRNode,
    consumer_counts: dict[str, int],
) -> bool:
    """Try fusing a single activation node with its preceding single-consumer elementwise node.

    Args:
        graph (IRGraph): Target computation graph.
        act_node (IRNode): Candidate activation node.
        consumer_counts (dict[str, int]): Consumer counts map.

    Returns:
        bool: True if fused, False otherwise.
    """
    if act_node.op_type not in ("Relu", "Gelu", "Silu", "Sigmoid", "Tanh") or len(act_node.inputs) != 1:
        return False

    elem_id = act_node.inputs[0]
    elem_node = graph.nodes.get(elem_id)
    if elem_node is None or elem_node.op_type not in ELEMENTWISE_OPS or consumer_counts.get(elem_id, 0) != 1:
        return False

    fused_node = IRNode(
        id=act_node.id,
        op_type=f"{elem_node.op_type}{act_node.op_type}",
        inputs=list(elem_node.inputs),
        shape_metadata=act_node.shape_metadata,
    )
    fused_node.attributes = {
        "fused_ops": [elem_node.op_type, act_node.op_type],
        "activation": act_node.op_type,
    }
    del graph.nodes[elem_id]
    graph.nodes[act_node.id] = fused_node
    return True


def fuse_elementwise_activation(graph: IRGraph) -> bool:
    """Fuse elementwise operations followed by single-consumer activations into fused nodes.

    Args:
        graph (IRGraph): Target computation graph to optimize.

    Returns:
        bool: True if any elementwise-activation patterns were fused, False otherwise.
    """
    if not _has_graph_nodes(graph):
        return False

    modified: bool = False
    changed: bool = True

    while changed:
        changed = False
        consumer_counts = _count_consumers(graph)
        for act_node in list(graph.nodes.values()):
            if _try_fuse_elem_act_node(graph, act_node, consumer_counts):
                changed = True
                modified = True
                break

    return modified


class VerticalFusionPass:
    """Vertical fusion pass merging sequential producer-consumer operation chains."""

    def __init__(self) -> None:
        """Initialize VerticalFusionPass."""
        pass

    def run(self, graph: IRGraph) -> bool:
        """Run vertical fusion passes on the target graph.

        Args:
            graph (IRGraph): Target computation graph.

        Returns:
            bool: True if any vertical fusions were applied, False otherwise.
        """
        c_mod = fuse_conv_bias_relu(graph)
        e_mod = fuse_elementwise_activation(graph)
        return c_mod or e_mod


def fuse_horizontal_patterns(graph: IRGraph) -> bool:
    """Fuse sibling operations with identical inputs and compatible shapes into horizontal fused nodes.

    Args:
        graph (IRGraph): Target computation graph.

    Returns:
        bool: True if horizontal fusions were performed, False otherwise.
    """
    if not _has_graph_nodes(graph):
        return False

    modified: bool = False
    changed: bool = True

    while changed:
        changed = False
        groups: dict[tuple[str, ...], list[IRNode]] = {}
        for node in graph.nodes.values():
            if node.op_type in ("Input", "Constant", "HorizontalFusedOps", "TupleGetItem"):
                continue
            if node.inputs:
                key = tuple(node.inputs)
                groups.setdefault(key, []).append(node)

        for _key, siblings in groups.items():
            if len(siblings) >= 2:
                first_shape = siblings[0].shape_metadata
                compatible = [
                    s
                    for s in siblings
                    if s.shape_metadata == first_shape
                    and (
                        s.op_type == siblings[0].op_type
                        or (s.op_type in ELEMENTWISE_OPS and siblings[0].op_type in ELEMENTWISE_OPS)
                    )
                ]
                if len(compatible) >= 2:
                    fused_id = f"horizontal_{'_'.join(s.id for s in compatible)}"
                    fused_ops = [s.op_type for s in compatible]
                    sibling_ids = [s.id for s in compatible]

                    h_node = IRNode(
                        id=fused_id,
                        op_type="HorizontalFusedOps",
                        inputs=list(compatible[0].inputs),
                        shape_metadata=first_shape,
                    )
                    h_node.attributes = {
                        "fused_ops": fused_ops,
                        "sibling_ids": sibling_ids,
                    }
                    graph.nodes[fused_id] = h_node

                    for idx, s in enumerate(compatible):
                        extract_node = IRNode(
                            id=s.id,
                            op_type="TupleGetItem",
                            inputs=[fused_id],
                            shape_metadata=s.shape_metadata,
                        )
                        extract_node.attributes = {"index": idx}
                        graph.nodes[s.id] = extract_node

                    changed = True
                    modified = True
                    break

    return modified


class HorizontalFusionPass:
    """Horizontal fusion pass merging sibling operations sharing identical inputs."""

    def __init__(self) -> None:
        """Initialize HorizontalFusionPass."""
        pass

    def run(self, graph: IRGraph) -> bool:
        """Run horizontal fusion passes on the target graph.

        Args:
            graph (IRGraph): Target computation graph.

        Returns:
            bool: True if horizontal fusions were applied, False otherwise.
        """
        return fuse_horizontal_patterns(graph)
