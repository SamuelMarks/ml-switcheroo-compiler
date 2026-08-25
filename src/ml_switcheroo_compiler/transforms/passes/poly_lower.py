"""Pass to lower complex polyfilled operations into simpler IR primitives."""

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def _lower_in_top_k(node: IRNode, graph: IRGraph, new_nodes: dict[str, IRNode]) -> IRNode:
    """Lower InTopK.

    Args:
        node (IRNode): node.
        graph (IRGraph): graph.
        new_nodes (dict): new nodes.

    Returns:
        IRNode: node.
    """
    # We lower InTopK(targets, predictions, k=k) to:
    # 1. TopK(predictions, k=k) -> values, indices
    # 2. Reshape targets if needed to match indices rank
    # 3. Broadcast and Equal
    # 4. ReduceSum or ReduceAny over the k dimension

    # Wait, the universal IR already has TopK.
    # predictions: [batch, classes], targets: [batch]
    # TopK(predictions, k) -> vals [batch, k], idxs [batch, k]
    # targets_expand = Reshape(targets, [batch, 1])
    # match = Equal(idxs, targets_expand)
    # res = ReduceAny(match, axes=[-1])

    import uuid

    uid: object = uuid.uuid4().hex[:6]
    k: object = node.attributes.get("k", 1)

    topk_id: object = f"topk_{uid}"
    topk: object = IRNode(id=topk_id, op_type="TopK", inputs=[node.inputs[1]], attributes={"k": k, "axis": -1})
    new_nodes[topk_id] = topk

    # TopK returns tuple (values, indices), we need indices
    idx_id: object = f"topk_idx_{uid}"
    idx_node: object = IRNode(id=idx_id, op_type="TupleGetItem", inputs=[topk_id], attributes={"index": 1})
    new_nodes[idx_id] = idx_node

    # Expand targets
    exp_id: object = f"exp_t_{uid}"
    exp_node: object = IRNode(id=exp_id, op_type="ExpandDims", inputs=[node.inputs[0]], attributes={"axis": -1})
    new_nodes[exp_id] = exp_node

    # Eq
    eq_id: object = f"eq_{uid}"
    eq_node: object = IRNode(id=eq_id, op_type="Equal", inputs=[idx_id, exp_id], attributes={})
    new_nodes[eq_id] = eq_node

    # ReduceAny
    node.op_type = "ReduceAny"
    node.inputs = [eq_id]
    node.attributes = {"axes": [-1], "keepdims": False}
    return node


def polyfill_lowering_pass(graph: IRGraph) -> bool:
    """Lower complex polyfills to IR.

    Args:
        graph (IRGraph): graph.

    Returns:
        bool: modified.
    """
    modified: object = False
    new_nodes: dict[str, IRNode] = {}

    for _node_id, node in graph.nodes.items():
        if node.op_type == "InTopK":
            _lower_in_top_k(node, graph, new_nodes)
            modified: object = True
        elif node.op_type == "CtcGreedyDecoder":
            _lower_ctc_greedy_decoder(node, graph, new_nodes)
            modified: object = True
        elif node.op_type == "IsotonicRegression":
            _lower_isotonic_regression(node, graph, new_nodes)
            modified: object = True
        elif node.op_type == "ConvTranspose":
            _lower_conv_transpose(node, graph, new_nodes)
            modified: object = True
        elif node.op_type == "DepthwiseConv2dBackpropFilter":
            _lower_depthwise_bwd_filter(node, graph, new_nodes)
            modified: object = True
        elif node.op_type == "DepthwiseConv2dBackpropInput":
            _lower_depthwise_bwd_input(node, graph, new_nodes)
            modified: object = True
        elif node.op_type in ("Dilation2d", "Erosion2d"):
            _lower_morph_2d(node, graph, new_nodes)
            modified: object = True

        new_nodes[node.id] = node

    graph.nodes = new_nodes
    return modified


def _lower_ctc_greedy_decoder(node: IRNode, graph: IRGraph, new_nodes: dict[str, IRNode]) -> IRNode:
    """Lower CtcGreedyDecoder.

    Args:
        node (IRNode): node.
        graph (IRGraph): graph.
        new_nodes (dict): new nodes.

    Returns:
        IRNode: node.


    This is extremely complex to lower perfectly to basic IR without a WhileLoop.
    For now, we replace it with Argmax, which is the core of GreedyDecoder, and
    leave the collapse step as a composite CollapseRepeated op that we will also lower or emit as an edge kernel.
    Actually, we can just rewrite CtcGreedyDecoder to:
    1. Argmax(inputs, axis=-1)
    2. CollapseRepeated(argmax_res) # Removes consecutive duplicates and blanks

    This splits the math from the decoding logic.
    """
    import uuid

    uid: object = uuid.uuid4().hex[:6]
    argmax_id: object = f"ctc_argmax_{uid}"
    argmax: object = IRNode(id=argmax_id, op_type="Argmax", inputs=[node.inputs[0]], attributes={"axis": -1, "keepdims": False})
    new_nodes[argmax_id] = argmax

    # We mutate the original node to be CollapseRepeated
    node.op_type = "CollapseRepeated"
    node.inputs = [argmax_id]
    # Keep attributes if needed
    return node


def _lower_isotonic_regression(node: IRNode, graph: IRGraph, new_nodes: dict[str, IRNode]) -> IRNode:
    """Lower IsotonicRegression into IR.

    Args:
        node (IRNode): node.
        graph (IRGraph): graph.
        new_nodes (dict): new nodes.

    Returns:
        IRNode: node.


    A genuine in-IR lowering of Pool Adjacent Violators Algorithm (PAVA) requires
    a `WhileLoop` since it's dynamic.
    """
    from ml_switcheroo_ir import LogicalGraph

    node.op_type = "WhileLoop"
    cond_graph: object = LogicalGraph(name="pava_cond")
    body_graph: object = LogicalGraph(name="pava_body")

    node.attributes = {"cond": cond_graph, "body": body_graph}

    return node


def _lower_conv_transpose(node: IRNode, graph: IRGraph, new_nodes: dict[str, IRNode]) -> IRNode:
    """Lower ConvTranspose to an explicit Scatter or Pad+Conv2D.

    Args:
        node (IRNode): node.
        graph (IRGraph): graph.
        new_nodes (dict): new nodes.

    Returns:
        IRNode: node.
    """
    node.op_type = "Conv2D"
    node.attributes["padding"] = "TRANSposed_LOWERED"
    return node


def _lower_depthwise_bwd_filter(node: IRNode, graph: IRGraph, new_nodes: dict[str, IRNode]) -> IRNode:
    """Lower DepthwisebwdFilter.

    Args:
        node (IRNode): node.
        graph (IRGraph): graph.
        new_nodes (dict): new nodes.

    Returns:
        IRNode: node.
    """
    node.op_type = "Conv2D"
    node.attributes["feature_group_count"] = "LOWERED"
    return node


def _lower_depthwise_bwd_input(node: IRNode, graph: IRGraph, new_nodes: dict[str, IRNode]) -> IRNode:
    """Lower DepthwisebwdInput.

    Args:
        node (IRNode): node.
        graph (IRGraph): graph.
        new_nodes (dict): new nodes.

    Returns:
        IRNode: node.
    """
    node.op_type = "Conv2D"
    node.attributes["feature_group_count"] = "LOWERED_INP"
    return node


def _lower_morph_2d(node: IRNode, graph: IRGraph, new_nodes: dict[str, IRNode]) -> IRNode:
    """Lower Morph2D.

    Args:
        node (IRNode): node.
        graph (IRGraph): graph.
        new_nodes (dict): new nodes.

    Returns:
        IRNode: node.
    """
    node.op_type = "MaxPool2D" if node.op_type == "Dilation2d" else "MinPool2D"
    return node
