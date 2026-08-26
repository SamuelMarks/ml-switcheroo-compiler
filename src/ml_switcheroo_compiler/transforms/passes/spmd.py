"""Module spmd.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""SPMD compiler pass."""


from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def _get_sharding_axes(sharding) -> list[str]:
    """Evaluate _get_sharding_axes operation.

    Args:
        sharding (object): The sharding parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    if not sharding or not hasattr(sharding, "mesh_mapping"):
        return []
    return [m for m in sharding.mesh_mapping if m is not None]


def _is_boundary_transition(inp_sharding, node_sharding) -> tuple[bool, bool]:
    """Evaluate _is_boundary_transition operation.

    Args:
        inp_sharding (object): The inp_sharding parameter.
        node_sharding (object): The node_sharding parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    inp_sharded = bool(_get_sharding_axes(inp_sharding))
    node_sharded = bool(_get_sharding_axes(node_sharding))
    return inp_sharded, node_sharded


def _create_all_gather_node(inp_id: str, node_sharding) -> IRNode:
    """Evaluate _create_all_gather_node operation.

    Args:
        inp_id (str): The inp_id parameter.
        node_sharding (object): The node_sharding parameter.

    Returns:
        IRNode: Result.
    """
    return IRNode(id=f"{inp_id}_all_gather", op_type="AllGather", inputs=[inp_id], sharding=node_sharding, attributes={"dispatch_early": True})


def _create_reduce_scatter_node(inp_id: str, node_sharding) -> IRNode:
    """Evaluate _create_reduce_scatter_node operation.

    Args:
        inp_id (str): The inp_id parameter.
        node_sharding (object): The node_sharding parameter.

    Returns:
        IRNode: Result.
    """
    return IRNode(id=f"{inp_id}_reduce_scatter", op_type="ReduceScatter", inputs=[inp_id], sharding=node_sharding, attributes={"dispatch_early": True})


def _create_all_reduce_node(inp_id: str, node_sharding) -> IRNode:
    """Evaluate _create_all_reduce_node operation.

    Args:
        inp_id (str): The inp_id parameter.
        node_sharding (object): The node_sharding parameter.

    Returns:
        IRNode: Result.
    """
    return IRNode(id=f"{inp_id}_all_reduce", op_type="AllReduce", inputs=[inp_id], sharding=node_sharding, attributes={"dispatch_early": True})


def _create_all_to_all_node(inp_id: str, node_sharding) -> IRNode:
    """Evaluate _create_all_to_all_node operation.

    Args:
        inp_id (str): The inp_id parameter.
        node_sharding (object): The node_sharding parameter.

    Returns:
        IRNode: Result.
    """
    return IRNode(id=f"{inp_id}_all_to_all", op_type="AllToAll", inputs=[inp_id], sharding=node_sharding, attributes={"dispatch_early": True})


def _inject_all_gather(node: IRNode, idx: int, inp_id: str, node_sharding) -> IRNode:
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


def _inject_reduce_scatter(node: IRNode, idx: int, inp_id: str, node_sharding) -> IRNode:
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


def _inject_all_reduce(node: IRNode, idx: int, inp_id: str, node_sharding) -> IRNode:
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


def _inject_all_to_all(node: IRNode, idx: int, inp_id: str, node_sharding) -> IRNode:
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


from pathlib import Path

import yaml

_SPMD_RULES = None


def _get_spmd_rules():
    """_get_spmd_rules function.

    Returns:
        object: Result.
    """
    global _SPMD_RULES
    if _SPMD_RULES is None:
        yaml_path = Path(__file__).parent / "spmd_mappings.yaml"
        with open(yaml_path) as f:
            _SPMD_RULES = yaml.safe_load(f)
    return _SPMD_RULES


def _determine_spmd_communication(
    node: IRNode,
    idx: int,
    inp_id: str,
    node_sharding,
    inp_axes,
    node_axes,
) -> IRNode | None:
    """Determine the type of SPMD communication needed using data-driven rules."""
    rules = _get_spmd_rules()

    inp_sharded = bool(inp_axes)
    node_sharded = bool(node_axes)
    state = [inp_sharded, node_sharded]

    is_reduction = getattr(node, "op_type", "") in rules.get("reductions", [])
    is_grad = "grad" in getattr(node, "id", "") or "adjoint" in getattr(node, "id", "")

    injected_op = "none"

    for rule in rules.get("communication_matrix", []):
        if rule["state"] == state:
            for cond in rule.get("conditions", []):
                if cond.get("default", False):
                    injected_op = cond.get("inject", "none")
                    break
                if cond.get("is_reduction", False) and is_reduction:
                    injected_op = cond.get("inject", "none")
                    break
                if cond.get("is_grad", False) and is_grad:
                    injected_op = cond.get("inject", "none")
                    break
                if cond.get("axes_match") is False and cond.get("axes_length_match") is True:
                    if inp_axes != node_axes and len(inp_axes) == len(node_axes):
                        injected_op = cond.get("inject", "none")
                        break
            break

    if injected_op == "AllReduce":
        return _inject_all_reduce(node, idx, inp_id, node_sharding)
    if injected_op == "AllGather":
        return _inject_all_gather(node, idx, inp_id, node_sharding)
    if injected_op == "ReduceScatter":
        return _inject_reduce_scatter(node, idx, inp_id, node_sharding)
    if injected_op == "AllToAll":
        return _inject_all_to_all(node, idx, inp_id, node_sharding)
    return None


def _process_spmd_input(node: IRNode, idx: int, inp_id: str, graph: IRGraph, node_sharding) -> IRNode | None:
    """Evaluate _process_spmd_input operation.

    Args:
        node (IRNode): The node parameter.
        idx (int): The idx parameter.
        inp_id (str): The inp_id parameter.
        graph (IRGraph): The graph parameter.
        node_sharding (object): The node_sharding parameter.

    Returns:
            tuple[int, ...]: Result.
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
