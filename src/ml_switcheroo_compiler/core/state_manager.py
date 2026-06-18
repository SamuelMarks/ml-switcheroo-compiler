"""State functionalization pass."""

from ml_switcheroo_ir import LogicalGraph, LogicalNode, topological_sort


def _process_assign_node(node: LogicalNode, state_env: dict[str, str]) -> None:
    target = node.inputs[0]
    new_val = node.inputs[1]
    if target in state_env:
        state_env[target] = new_val


def _rewrite_node(node: LogicalNode, state_env: dict[str, str]) -> LogicalNode:
    new_inputs = [state_env.get(inp, inp) for inp in node.inputs]
    return LogicalNode(
        id=node.id,
        op_type=node.op_type,
        domain=node.domain,
        version=node.version,
        attributes=dict(node.attributes),
        inputs=new_inputs,
        shape_metadata=node.shape_metadata,
        source_ast_ref=node.source_ast_ref,
        sharding=node.sharding,
    )


def _build_functional_outputs(
    graph_outputs: list[str], state_vars: list[str], state_env: dict[str, str]
) -> list[str]:
    functional_outputs = list(graph_outputs)
    for v in state_vars:
        functional_outputs.append(state_env[v])
    return functional_outputs


def lift_state(graph: LogicalGraph, state_vars: list[str]) -> LogicalGraph:
    """Lifts mutable state into pure functional boundaries.

    In frameworks like Flax or PyTorch, models have mutable state
    (parameters, batch stats)
    This pass assumes that mutable updates are represented as specific nodes
    (e.g., 'Assign' or 'UpdateState' nodes). It rewrites the graph so that
    state variables are treated as explicit inputs and their updated values
    are explicit outputs

    graph (LogicalGraph): The input graph containing state nodes
    state_vars (List[str]): The IDs of nodes representing initial state variables

    Returns:
    LogicalGraph: The functionalized graph

    Args:
        graph (LogicalGraph): Argument graph
        state_vars (list[str]): Argument state_vars
    """
    new_graph = LogicalGraph(name=f"{graph.name}_functional", mesh=graph.mesh)
    sorted_nodes = topological_sort(graph)
    state_env: dict[str, str] = {v: v for v in state_vars}

    for node in sorted_nodes:
        if node.op_type == "Assign":
            _process_assign_node(node, state_env)
        else:
            new_graph.nodes[node.id] = _rewrite_node(node, state_env)

    new_graph.outputs = _build_functional_outputs(graph.outputs, state_vars, state_env)
    return new_graph
