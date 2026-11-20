"""Core reverse-mode graph cloning, rematerialization, and adjoint accumulation."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence  # noqa: F401

from ml_switcheroo_ir import AttributeValue, LogicalGraph, LogicalNode

import ml_switcheroo_compiler.transforms.autodiff as _ad_mod
from ml_switcheroo_compiler.core.errors import MissingJVPRuleError
from ml_switcheroo_compiler.ir.core import clone_logical_node


def _is_zero_node(graph: LogicalGraph, node_id: str) -> bool:
    """Check if node represents a zero or identity element for addition.

    Args:
        graph (LogicalGraph): The computation graph.
        node_id (str): Identifier of the node to check.

    Returns:
        bool: True if node represents zero.
    """
    if node_id not in graph.nodes:
        return False
    node = graph.nodes[node_id]
    if node.op_type in ("Zeros", "ZerosLike"):
        return True
    if node.op_type == "Constant":
        val = node.attributes.get("value")
        if isinstance(val, (int, float)) and val == 0:
            return True
    return False


def _add_nodes(graph: LogicalGraph, n1_id: str, n2_id: str) -> str:
    """Emit an Add node for gradient accumulation, avoiding redundant Add chains.

    Args:
        graph (LogicalGraph): The graph parameter for the operation.
        n1_id (str): The n1_id parameter for the operation.
        n2_id (str): The n2_id parameter for the operation.

    Returns:
        str: The computed result.
    """
    if _is_zero_node(graph, n1_id):
        return n2_id
    if _is_zero_node(graph, n2_id):
        return n1_id

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


def _load_rematerialization_rules() -> dict[str, AttributeValue]:
    """Load rematerialization rules from YAML config.

    Returns:
        dict[str, AttributeValue]: Loaded rematerialization rules configuration.
    """
    import os

    import yaml

    yaml_path = os.path.join(os.path.dirname(__file__), "passes", "rematerialization_rules.yaml")
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, yaml.YAMLError):
            return {}
    return {}


def _should_rematerialize(node: LogicalNode, rules: Mapping[str, AttributeValue] | None = None) -> bool:
    """Determine if a node should be rematerialized across backward boundary.

    Args:
        node (LogicalNode): The node to evaluate.
        rules (Mapping[str, AttributeValue] | None): Rematerialization rules config.

    Returns:
        bool: True if node should be rematerialized.
    """
    if node.attributes.get("rematerialize", False):
        return True
    if rules is None:
        rules = _load_rematerialization_rules()
    if not rules:
        return False
    high_cost = rules.get("high_cost_ops", [])
    if isinstance(high_cost, (list, tuple)) and node.op_type in high_cost:
        return False
    target_ops = rules.get("target_ops", [])
    return (
        isinstance(target_ops, (list, tuple))
        and node.op_type in target_ops
        and bool(node.attributes.get("checkpoint", False))
    )


def _recompute_subgraph(new_graph: LogicalGraph, node: LogicalNode) -> LogicalNode:
    """Recursively recompute a node and its rematerialized inputs.

    Args:
        new_graph (LogicalGraph): The new graph.
        node (LogicalNode): The node to recompute.

    Returns:
        LogicalNode: The cloned recomputed node.
    """
    import uuid

    from ml_switcheroo_compiler.ir.core import clone_logical_node

    recompute_node = clone_logical_node(node)
    recompute_node.id = f"{node.id}_recompute_{uuid.uuid4().hex[:6]}"

    # Recursively recompute inputs if they are also tagged for rematerialization
    new_inputs = []
    for inp_id in node.inputs:
        if inp_id in new_graph.nodes:
            inp_node = new_graph.nodes[inp_id]
            if _should_rematerialize(inp_node):
                # Need to recompute the input as well
                recomputed_inp = _recompute_subgraph(new_graph, inp_node)
                new_inputs.append(recomputed_inp.id)
            else:
                new_inputs.append(inp_id)
        else:
            new_inputs.append(inp_id)

    recompute_node.inputs = new_inputs
    new_graph.nodes[recompute_node.id] = recompute_node
    return recompute_node


def _accumulate_gradients(
    new_graph: LogicalGraph,
    node: LogicalNode,
    adj_id: str,
    adjoints: dict[str, str],
) -> None:
    """Accumulate gradients for a given node.

    Args:
        new_graph (LogicalGraph): The new graph.
        node (LogicalNode): The node to process.
        adj_id (str): The adjoint ID for the node.
        adjoints (dict[str, str]): The mapping of adjoints.

    Raises:
        MissingJVPRuleError: If VJP rule is missing or returns incorrect number of adjoints.
    """
    if _should_rematerialize(node):
        eval_node = _recompute_subgraph(new_graph, node)
    else:
        eval_node = node

    try:
        vjp_func = _ad_mod.get_vjp(node.op_type)
    except ValueError:
        msg = f"Missing VJP rule for operation: {node.op_type}"
        raise MissingJVPRuleError(msg) from None

    input_adjs = vjp_func(new_graph, eval_node, adj_id)

    if len(input_adjs) != len(node.inputs):
        msg = f"VJP for {node.op_type} returned {len(input_adjs)} adjoints, expected {len(node.inputs)}."
        raise ValueError(msg)

    for inp_id, inp_adj_id in zip(node.inputs, input_adjs):
        if (
            inp_adj_id is None
            or inp_adj_id == _ad_mod.UnconnectedGradients.NONE
            or inp_adj_id == _ad_mod.UnconnectedGradients.ZERO
        ):
            continue

        if inp_id in adjoints:
            adjoints[inp_id] = _ad_mod._add_nodes(new_graph, adjoints[inp_id], inp_adj_id)
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

    """
    for node in reversed(sorted_nodes):
        nid = node.id
        if nid not in reachable_from_output or nid not in adjoints:
            continue

        if node.op_type == "Output":
            for inp in node.inputs:
                adjoints[inp] = adjoints[node.id]
            continue

        if node.op_type in ("StopGradient", "Input", "Constant"):
            continue

        _ad_mod._accumulate_gradients(new_graph, node, adjoints[nid], adjoints)
