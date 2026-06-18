"""SPMD compiler pass."""

import typing


from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def _is_boundary_transition(inp_sharding: object, node_sharding: object) -> tuple[bool, bool]:
    inp_sharded = any(m is not None for m in inp_sharding.mesh_mapping)
    node_sharded = any(m is not None for m in node_sharding.mesh_mapping)
    return inp_sharded, node_sharded


def _create_all_gather_node(inp_id: str, node_sharding: object) -> IRNode:
    return IRNode(
        id=f"{inp_id}_all_gather",
        op_type="all_gather",
        inputs=[inp_id],
        sharding=node_sharding,
    )


def _create_reduce_scatter_node(inp_id: str, node_sharding: object) -> IRNode:
    return IRNode(
        id=f"{inp_id}_reduce_scatter",
        op_type="reduce_scatter",
        inputs=[inp_id],
        sharding=node_sharding,
    )


def _inject_all_gather(node: IRNode, idx: int, inp_id: str, node_sharding: object) -> IRNode:
    gather_node = _create_all_gather_node(inp_id, node_sharding)
    node.inputs[idx] = gather_node.id
    return gather_node


def _inject_reduce_scatter(node: IRNode, idx: int, inp_id: str, node_sharding: object) -> IRNode:
    scatter_node = _create_reduce_scatter_node(inp_id, node_sharding)
    node.inputs[idx] = scatter_node.id
    return scatter_node


def _process_spmd_input(
    node: IRNode, idx: int, inp_id: str, graph: IRGraph, node_sharding: object
) -> typing.Optional[IRNode]:
    if inp_id not in graph.nodes:
        return None

    inp_node = graph.nodes[inp_id]
    inp_sharding = getattr(inp_node, "sharding", None)

    if not inp_sharding:
        return None

    inp_sharded, node_sharded = _is_boundary_transition(inp_sharding, node_sharding)

    dispatch = {
        (True, False, False): _inject_all_gather,
        (False, True, True): _inject_reduce_scatter,
    }

    is_grad = node.op_type == "Grad"
    handler = dispatch.get((inp_sharded, node_sharded, is_grad))

    if handler:
        return handler(node, idx, inp_id, node_sharding)

    return None


def _process_spmd_node(node: IRNode, graph: IRGraph) -> tuple[bool, list[IRNode]]:
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
