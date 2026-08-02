"""SPMD compiler pass."""

import typing

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def _get_sharding_axes(sharding: object) -> list[str]:
    """Extract non-None mesh mapping axes.

    Args:
        sharding (object): Required parameter for sharding.

    Returns:
        list[str]: The evaluated or processed output.
    """
    if not sharding or not hasattr(sharding, "mesh_mapping"):
        return []
    return [m for m in sharding.mesh_mapping if m is not None]


def _is_boundary_transition(inp_sharding: object, node_sharding: object) -> tuple[bool, bool]:
    """Evaluate and process the is boundary transition operation.

    Args:
        inp_sharding (object): Required parameter for inp_sharding.
        node_sharding (object): Required parameter for node_sharding.

    Returns:
        tuple: The evaluated or processed output.
    """
    inp_sharded = bool(_get_sharding_axes(inp_sharding))
    node_sharded = bool(_get_sharding_axes(node_sharding))
    return inp_sharded, node_sharded


def _create_all_gather_node(inp_id: str, node_sharding: object) -> IRNode:
    """Evaluate and process the create all gather node operation.

    Args:
        inp_id (str): Required parameter for inp_id.
        node_sharding (object): Required parameter for node_sharding.

    Returns:
        IRNode: The evaluated or processed output.
    """
    return IRNode(id=f"{inp_id}_all_gather", op_type="all_gather", inputs=[inp_id], sharding=node_sharding, attributes={"dispatch_early": True})


def _create_reduce_scatter_node(inp_id: str, node_sharding: object) -> IRNode:
    """Evaluate and process the create reduce scatter node operation.

    Args:
        inp_id (str): Required parameter for inp_id.
        node_sharding (object): Required parameter for node_sharding.

    Returns:
        IRNode: The evaluated or processed output.
    """
    return IRNode(id=f"{inp_id}_reduce_scatter", op_type="reduce_scatter", inputs=[inp_id], sharding=node_sharding, attributes={"dispatch_early": True})


def _create_all_reduce_node(inp_id: str, node_sharding: object) -> IRNode:
    """Evaluate and process the create all reduce node operation.

    Args:
        inp_id (str): Required parameter for inp_id.
        node_sharding (object): Required parameter for node_sharding.

    Returns:
        IRNode: The evaluated or processed output.
    """
    return IRNode(id=f"{inp_id}_all_reduce", op_type="all_reduce", inputs=[inp_id], sharding=node_sharding, attributes={"dispatch_early": True})


def _create_all_to_all_node(inp_id: str, node_sharding: object) -> IRNode:
    """Evaluate and process the create all to all node operation.

    Args:
        inp_id (str): Required parameter for inp_id.
        node_sharding (object): Required parameter for node_sharding.

    Returns:
        IRNode: The evaluated or processed output.
    """
    return IRNode(id=f"{inp_id}_all_to_all", op_type="all_to_all", inputs=[inp_id], sharding=node_sharding, attributes={"dispatch_early": True})


def _inject_all_gather(node: IRNode, idx: int, inp_id: str, node_sharding: object) -> IRNode:
    """Evaluate and process the inject all gather operation.

    Args:
        node (IRNode): Required parameter for node.
        idx (int): Required parameter for idx.
        inp_id (str): Required parameter for inp_id.
        node_sharding (object): Required parameter for node_sharding.

    Returns:
        IRNode: The evaluated or processed output.
    """
    gather_node = _create_all_gather_node(inp_id, node_sharding)
    node.inputs[idx] = gather_node.id
    return gather_node


def _inject_reduce_scatter(node: IRNode, idx: int, inp_id: str, node_sharding: object) -> IRNode:
    """Evaluate and process the inject reduce scatter operation.

    Args:
        node (IRNode): Required parameter for node.
        idx (int): Required parameter for idx.
        inp_id (str): Required parameter for inp_id.
        node_sharding (object): Required parameter for node_sharding.

    Returns:
        IRNode: The evaluated or processed output.
    """
    scatter_node = _create_reduce_scatter_node(inp_id, node_sharding)
    node.inputs[idx] = scatter_node.id
    return scatter_node


def _inject_all_reduce(node: IRNode, idx: int, inp_id: str, node_sharding: object) -> IRNode:
    """Evaluate and process the inject all reduce operation.

    Args:
        node (IRNode): Required parameter for node.
        idx (int): Required parameter for idx.
        inp_id (str): Required parameter for inp_id.
        node_sharding (object): Required parameter for node_sharding.

    Returns:
        IRNode: The evaluated or processed output.
    """
    reduce_node = _create_all_reduce_node(inp_id, node_sharding)
    node.inputs[idx] = reduce_node.id
    return reduce_node


def _inject_all_to_all(node: IRNode, idx: int, inp_id: str, node_sharding: object) -> IRNode:
    """Evaluate and process the inject all to all operation.

    Args:
        node (IRNode): Required parameter for node.
        idx (int): Required parameter for idx.
        inp_id (str): Required parameter for inp_id.
        node_sharding (object): Required parameter for node_sharding.

    Returns:
        IRNode: The evaluated or processed output.
    """
    atoa_node = _create_all_to_all_node(inp_id, node_sharding)
    node.inputs[idx] = atoa_node.id
    return atoa_node


def _process_spmd_input(node: IRNode, idx: int, inp_id: str, graph: IRGraph, node_sharding: object) -> typing.Optional[IRNode]:
    """Evaluate and process the process spmd input operation.

    Args:
        node (IRNode): Required parameter for node.
        idx (int): Required parameter for idx.
        inp_id (str): Required parameter for inp_id.
        graph (IRGraph): Required parameter for graph.
        node_sharding (object): Required parameter for node_sharding.

    Returns:
        Any: The evaluated or processed output.
    """
    if inp_id not in graph.nodes:
        return None

    inp_node = graph.nodes[inp_id]
    inp_sharding = getattr(inp_node, "sharding", None)

    if not inp_sharding:
        return None

    inp_axes = _get_sharding_axes(inp_sharding)
    node_axes = _get_sharding_axes(node_sharding)

    inp_sharded = bool(inp_axes)
    node_sharded = bool(node_axes)

    is_grad = node.op_type == "Grad" or node.attributes.get("sync_gradients", False)
    is_reduction = node.op_type in ["Sum", "Mean", "Max", "Min"]

    result = None

    if inp_sharded and not node_sharded:
        if is_reduction or is_grad:
            result = _inject_all_reduce(node, idx, inp_id, node_sharding)
        else:
            result = _inject_all_gather(node, idx, inp_id, node_sharding)
    elif not inp_sharded and node_sharded:
        if is_grad:
            result = _inject_reduce_scatter(node, idx, inp_id, node_sharding)
    elif inp_sharded and node_sharded and inp_axes != node_axes:
        if len(inp_axes) == len(node_axes):
            result = _inject_all_to_all(node, idx, inp_id, node_sharding)

    return result


def _process_spmd_node(node: IRNode, graph: IRGraph) -> tuple[bool, list[IRNode]]:
    """Evaluate and process the process spmd node operation.

    Args:
        node (IRNode): Required parameter for node.
        graph (IRGraph): Required parameter for graph.

    Returns:
        tuple: The evaluated or processed output.
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
    """Injects all_gather, reduce_scatter, all_reduce for SPMD execution."""
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
