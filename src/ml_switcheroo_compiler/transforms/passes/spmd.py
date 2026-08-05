"""SPMD compiler pass."""

from __future__ import annotations

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def _get_sharding_axes(sharding: object) -> list[str]:
    """Evaluate _get_sharding_axes operation.

    Args:
        sharding (object): The sharding parameter.

    Returns:
        object: Result.
    """
    if not sharding or not hasattr(sharding, "mesh_mapping"):
        return []
    return [m for m in sharding.mesh_mapping if m is not None]


def _is_boundary_transition(inp_sharding: object, node_sharding: object) -> tuple[bool, bool]:
    """Evaluate _is_boundary_transition operation.

    Args:
        inp_sharding (object): The inp_sharding parameter.
        node_sharding (object): The node_sharding parameter.

    Returns:
        object: Result.
    """
    inp_sharded = bool(_get_sharding_axes(inp_sharding))
    node_sharded = bool(_get_sharding_axes(node_sharding))
    return inp_sharded, node_sharded


def _create_all_gather_node(inp_id: str, node_sharding: object) -> IRNode:
    """Evaluate _create_all_gather_node operation.

    Args:
        inp_id (str): The inp_id parameter.
        node_sharding (object): The node_sharding parameter.

    Returns:
        IRNode: Result.
    """
    return IRNode(id=f"{inp_id}_all_gather", op_type="all_gather", inputs=[inp_id], sharding=node_sharding, attributes={"dispatch_early": True})


def _create_reduce_scatter_node(inp_id: str, node_sharding: object) -> IRNode:
    """Evaluate _create_reduce_scatter_node operation.

    Args:
        inp_id (str): The inp_id parameter.
        node_sharding (object): The node_sharding parameter.

    Returns:
        IRNode: Result.
    """
    return IRNode(id=f"{inp_id}_reduce_scatter", op_type="reduce_scatter", inputs=[inp_id], sharding=node_sharding, attributes={"dispatch_early": True})


def _create_all_reduce_node(inp_id: str, node_sharding: object) -> IRNode:
    """Evaluate _create_all_reduce_node operation.

    Args:
        inp_id (str): The inp_id parameter.
        node_sharding (object): The node_sharding parameter.

    Returns:
        IRNode: Result.
    """
    return IRNode(id=f"{inp_id}_all_reduce", op_type="all_reduce", inputs=[inp_id], sharding=node_sharding, attributes={"dispatch_early": True})


def _create_all_to_all_node(inp_id: str, node_sharding: object) -> IRNode:
    """Evaluate _create_all_to_all_node operation.

    Args:
        inp_id (str): The inp_id parameter.
        node_sharding (object): The node_sharding parameter.

    Returns:
        IRNode: Result.
    """
    return IRNode(id=f"{inp_id}_all_to_all", op_type="all_to_all", inputs=[inp_id], sharding=node_sharding, attributes={"dispatch_early": True})


def _inject_all_gather(node: IRNode, idx: int, inp_id: str, node_sharding: object) -> IRNode:
    """Evaluate _inject_all_gather operation.

    Args:
        node (IRNode): The node parameter.
        idx (int): The idx parameter.
        inp_id (str): The inp_id parameter.
        node_sharding (object): The node_sharding parameter.

    Returns:
        IRNode: Result.
    """
    gather_node = _create_all_gather_node(inp_id, node_sharding)
    node.inputs[idx] = gather_node.id
    return gather_node


def _inject_reduce_scatter(node: IRNode, idx: int, inp_id: str, node_sharding: object) -> IRNode:
    """Evaluate _inject_reduce_scatter operation.

    Args:
        node (IRNode): The node parameter.
        idx (int): The idx parameter.
        inp_id (str): The inp_id parameter.
        node_sharding (object): The node_sharding parameter.

    Returns:
        IRNode: Result.
    """
    scatter_node = _create_reduce_scatter_node(inp_id, node_sharding)
    node.inputs[idx] = scatter_node.id
    return scatter_node


def _inject_all_reduce(node: IRNode, idx: int, inp_id: str, node_sharding: object) -> IRNode:
    """Evaluate _inject_all_reduce operation.

    Args:
        node (IRNode): The node parameter.
        idx (int): The idx parameter.
        inp_id (str): The inp_id parameter.
        node_sharding (object): The node_sharding parameter.

    Returns:
        IRNode: Result.
    """
    reduce_node = _create_all_reduce_node(inp_id, node_sharding)
    node.inputs[idx] = reduce_node.id
    return reduce_node


def _inject_all_to_all(node: IRNode, idx: int, inp_id: str, node_sharding: object) -> IRNode:
    """Evaluate _inject_all_to_all operation.

    Args:
        node (IRNode): The node parameter.
        idx (int): The idx parameter.
        inp_id (str): The inp_id parameter.
        node_sharding (object): The node_sharding parameter.

    Returns:
        IRNode: Result.
    """
    atoa_node = _create_all_to_all_node(inp_id, node_sharding)
    node.inputs[idx] = atoa_node.id
    return atoa_node


def _handle_inp_sharded_only(node: IRNode, idx: int, inp_id: str, node_sharding: object, is_reduction: bool, is_grad: bool) -> IRNode | None:
    """Handle communication when only the input is sharded.

    Args:
        node (IRNode): The target node.
        idx (int): The index of the input.
        inp_id (str): The input ID.
        node_sharding (object): The node sharding specification.
        is_reduction (bool): Whether the node is a reduction.
        is_grad (bool): Whether the node computes gradients.

    Returns:
        IRNode | None: The injected communication node if applicable.
    """
    if is_reduction or is_grad:
        return _inject_all_reduce(node, idx, inp_id, node_sharding)
    return _inject_all_gather(node, idx, inp_id, node_sharding)


def _handle_node_sharded_only(node: IRNode, idx: int, inp_id: str, node_sharding: object, is_grad: bool) -> IRNode | None:
    """Handle communication when only the node is sharded.

    Args:
        node (IRNode): The target node.
        idx (int): The index of the input.
        inp_id (str): The input ID.
        node_sharding (object): The node sharding specification.
        is_grad (bool): Whether the node computes gradients.

    Returns:
        IRNode | None: The injected communication node if applicable.
    """
    if is_grad:
        return _inject_reduce_scatter(node, idx, inp_id, node_sharding)
    return None


def _handle_both_sharded(node: IRNode, idx: int, inp_id: str, node_sharding: object, inp_axes: list, node_axes: list) -> IRNode | None:
    """Handle communication when both input and node are sharded.

    Args:
        node (IRNode): The target node.
        idx (int): The index of the input.
        inp_id (str): The input ID.
        node_sharding (object): The node sharding specification.
        inp_axes (list): The input sharded axes.
        node_axes (list): The node sharded axes.

    Returns:
        IRNode | None: The injected communication node if applicable.
    """
    if inp_axes != node_axes and len(inp_axes) == len(node_axes):
        return _inject_all_to_all(node, idx, inp_id, node_sharding)
    return None


def _determine_spmd_communication(
    node: IRNode,
    idx: int,
    inp_id: str,
    node_sharding: object,
    inp_axes: list,
    node_axes: list,
) -> IRNode | None:
    """Determine the type of SPMD communication needed.

    Args:
        node (IRNode): The target node.
        idx (int): The index of the input.
        inp_id (str): The input ID.
        node_sharding (object): The node sharding specification.
        inp_axes (list): The input sharded axes.
        node_axes (list): The node sharded axes.

    Returns:
        IRNode | None: The injected communication node if applicable.
    """
    inp_sharded = bool(inp_axes)
    node_sharded = bool(node_axes)

    is_grad = node.op_type == "Grad" or node.attributes.get("sync_gradients", False)
    is_reduction = node.op_type in ["Sum", "Mean", "Max", "Min"]

    if inp_sharded and not node_sharded:
        return _handle_inp_sharded_only(node, idx, inp_id, node_sharding, is_reduction, is_grad)
    if not inp_sharded and node_sharded:
        return _handle_node_sharded_only(node, idx, inp_id, node_sharding, is_grad)
    if inp_sharded and node_sharded:
        return _handle_both_sharded(node, idx, inp_id, node_sharding, inp_axes, node_axes)
    return None


def _process_spmd_input(node: IRNode, idx: int, inp_id: str, graph: IRGraph, node_sharding: object) -> IRNode | None:
    """Evaluate _process_spmd_input operation.

    Args:
        node (IRNode): The node parameter.
        idx (int): The idx parameter.
        inp_id (str): The inp_id parameter.
        graph (IRGraph): The graph parameter.
        node_sharding (object): The node_sharding parameter.

    Returns:
        object: Result.
    """
    if inp_id not in graph.nodes:
        return None

    inp_node = graph.nodes[inp_id]
    inp_sharding = getattr(inp_node, "sharding", None)

    if not inp_sharding:
        return None

    inp_axes = _get_sharding_axes(inp_sharding)
    node_axes = _get_sharding_axes(node_sharding)

    return _determine_spmd_communication(node, idx, inp_id, node_sharding, inp_axes, node_axes)


def _process_spmd_node(node: IRNode, graph: IRGraph) -> tuple[bool, list[IRNode]]:
    """Evaluate _process_spmd_node operation.

    Args:
        node (IRNode): The node parameter.
        graph (IRGraph): The graph parameter.

    Returns:
        tuple: Result.
    """
    modified = False
    injected_nodes = []

    node_sharding = getattr(node, "sharding", None)
    if not node_sharding:
        return False, []

    for idx, inp_id in enumerate(list(node.inputs)):
        inj_node = _process_spmd_input(node, idx, inp_id, graph, node_sharding)
        if inj_node:
            injected_nodes.append(inj_node)
            modified = True

    return modified, injected_nodes


def inject_spmd_communication_pass(graph: IRGraph) -> bool:
    """Injects all_gather, reduce_scatter, all_reduce for SPMD execution.

    Args:
        graph (IRGraph): The graph parameter.

    Returns:
        bool: Result.
    """
    modified = False
    new_nodes = {}

    for node_id, node in list(graph.nodes.items()):
        new_nodes[node_id] = node

        node_modified, injected = _process_spmd_node(node, graph)
        if node_modified:
            modified = True

        for inj_node in injected:
            new_nodes[inj_node.id] = inj_node

    graph.nodes = new_nodes
    return modified
