"""Reverse-mode Automatic Differentiation (AD) Engine."""

import uuid

from ml_switcheroo_ir import LogicalGraph, LogicalNode


def _add_nodes(graph: LogicalGraph, n1_id: str, n2_id: str) -> str:
    """Emit an Add node for gradient accumulation.

    Args:
        graph (LogicalGraph): The graph parameter for the operation.
        n1_id (str): The n1_id parameter for the operation.
        n2_id (str): The n2_id parameter for the operation.

    Returns:
        str: The evaluated output resulting from this operation.
    """
    out_id = f"{n1_id}_add_{n2_id}_{uuid.uuid4().hex[:6]}"
    n1 = graph.nodes[n1_id]

    # Simple shape heuristic for accumulation
    node = LogicalNode(
        id=out_id,
        op_type="Add",
        inputs=[n1_id, n2_id],
        shape_metadata=n1.shape_metadata,
    )
    graph.nodes[out_id] = node
    return out_id


def _copy_graph(graph: LogicalGraph) -> LogicalGraph:
    """Copy the forward graph into a new graph.

    Args:
        graph (LogicalGraph): The original graph.

    Returns:
        LogicalGraph: The new graph.
    """
    from ml_switcheroo_compiler.ir.core import clone_logical_node

    new_graph = LogicalGraph(name=f"{graph.name}_grad")
    for nid, node in graph.nodes.items():
        new_graph.nodes[nid] = clone_logical_node(node)
    return new_graph


def _get_reachable_from_output(sorted_nodes: list[LogicalNode], output_id: str) -> set[str]:
    """Find all nodes reachable from the output.

    Args:
        sorted_nodes (list[LogicalNode]): The sorted nodes.
        output_id (str): The output id.

    Returns:
        set[str]: The reachable nodes.
    """
    reachable_from_output: set[str] = {output_id}
    for node in reversed(sorted_nodes):
        if node.id in reachable_from_output:
            for inp in node.inputs:
                reachable_from_output.add(inp)
    return reachable_from_output


def _accumulate_gradients(
    new_graph: LogicalGraph,
    node: LogicalNode,
    adj_id: str,
    adjoints: dict[str, str],
) -> None:
    """Execute _accumulate_gradients.

    Args:
        new_graph (Any): Argument new_graph.
        node (Any): Argument node.
        adj_id (Any): Argument adj_id.
        adjoints (Any): Argument adjoints.
    """
    from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import get_vjp

    try:
        vjp_func = get_vjp(node.op_type)
        input_adjs = vjp_func(new_graph, node, adj_id)
    except NotImplementedError:
        msg = f"Missing VJP rule for operation: {node.op_type}"
        raise ValueError(msg) from None

    if len(input_adjs) != len(node.inputs):
        msg = (
            f"VJP for {node.op_type} returned {len(input_adjs)} adjoints, "
            f"expected {len(node.inputs)}."
        )
        raise ValueError(msg)

    for inp_id, inp_adj_id in zip(node.inputs, input_adjs):
        if inp_adj_id is None:
            continue

        if inp_id in adjoints:
            adjoints[inp_id] = _add_nodes(new_graph, adjoints[inp_id], inp_adj_id)
        else:
            adjoints[inp_id] = inp_adj_id


def _backward_pass(
    new_graph: LogicalGraph,
    sorted_nodes: list[LogicalNode],
    reachable_from_output: set[str],
    adjoints: dict[str, str],
) -> None:
    """Perform the backward pass to compute gradients.

    Args:
        new_graph (LogicalGraph): The new graph.
        sorted_nodes (list[LogicalNode]): The sorted nodes.
        reachable_from_output (set[str]): The reachable nodes.
        adjoints (dict[str, str]): The adjoints map.

    Raises:
        ValueError: If VJP rule is missing or returns incorrect number of adjoints.
    """
    for node in reversed(sorted_nodes):
        nid = node.id
        if nid not in reachable_from_output or nid not in adjoints:
            continue

        if node.op_type in ("StopGradient", "Input", "Constant"):
            continue

        _accumulate_gradients(new_graph, node, adjoints[nid], adjoints)


def _extract_gradients(
    new_graph: LogicalGraph,
    wrt: list[str],
    adjoints: dict[str, str],
) -> list[str]:
    """Extract the required gradients.

    Args:
        new_graph (LogicalGraph): The new_graph parameter for the operation.
        wrt (list[str]): Target nodes.
        adjoints (dict[str, str]): Adjoints map.

    Returns:
        list[str]: Output gradient node IDs.

    Raises:
        ValueError: If a target node is not found.
    """
    grad_outputs = []
    for w in wrt:
        if w not in new_graph.nodes:
            msg = f"Target node '{w}' not found in graph."
            raise ValueError(msg)

        if w in adjoints:
            grad_outputs.append(adjoints[w])
        else:
            zero_id = f"grad_zeros_{uuid.uuid4().hex[:6]}"
            zeros_node = LogicalNode(
                id=zero_id,
                op_type="Constant",
                attributes={"value": 0.0},
                shape_metadata=new_graph.nodes[w].shape_metadata,
            )
            new_graph.nodes[zero_id] = zeros_node
            grad_outputs.append(zero_id)
    return grad_outputs


def grad(graph: LogicalGraph, wrt: list[str], output_id: str) -> LogicalGraph:
    """Compute the gradient of a scalar output with respect to specified inputs.

    graph (LogicalGraph): The forward pass graph
    wrt (List[str]): List of node IDs to compute gradients for
    output_id (str): The node ID of the scalar output

    Returns:
    LogicalGraph: A new graph containing both forward pass and
    gradient computations

    Raises:
    ValueError: If output node does not exist, or required VJPs are missing

    Args:
        graph (LogicalGraph): Argument graph
        wrt (list[str]): Argument wrt
        output_id (str): Argument output_id
    """
    if output_id not in graph.nodes:
        msg = f"Output node '{output_id}' not found in graph."
        raise ValueError(msg)

    new_graph = _copy_graph(graph)

    from ml_switcheroo_ir import topological_sort

    sorted_nodes = topological_sort(new_graph)

    reachable_from_output = _get_reachable_from_output(sorted_nodes, output_id)

    adjoints: dict[str, str] = {}
    one_id = f"grad_ones_{uuid.uuid4().hex[:6]}"
    ones_node = LogicalNode(
        id=one_id,
        op_type="Constant",
        attributes={"value": 1.0},
        shape_metadata=(),
    )
    new_graph.nodes[one_id] = ones_node
    adjoints[output_id] = one_id

    _backward_pass(new_graph, sorted_nodes, reachable_from_output, adjoints)

    new_graph.outputs = _extract_gradients(new_graph, wrt, adjoints)
    return new_graph
