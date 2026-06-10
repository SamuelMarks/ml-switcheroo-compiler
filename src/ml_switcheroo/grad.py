"""Reverse-mode Automatic Differentiation (AD) Engine."""

from typing import Dict, List, Set, Callable
import uuid

from ml_switcheroo_ir import LogicalGraph, LogicalNode

# Registry for VJP (Vector-Jacobian Product) rules
VJPRegistryType = Dict[str, Callable]
_VJP_REGISTRY: VJPRegistryType = {}


def register_vjp(op_type: str) -> Callable:
    """Decorator to register a VJP rule for a specific operation.

    Args:
        op_type (str): The ONNX operation type (e.g., 'Add').

    Returns:
        Callable: The decorator function.
    """

    def decorator(fn: Callable) -> Callable:
        """Docstring."""
        _VJP_REGISTRY[op_type] = fn
        return fn

    return decorator


def _add_nodes(graph: LogicalGraph, n1_id: str, n2_id: str) -> str:
    """Helper to emit an Add node for gradient accumulation."""
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


def grad(graph: LogicalGraph, wrt: List[str], output_id: str) -> LogicalGraph:
    """Computes the gradient of a scalar output with respect to specified inputs.

    Args:
        graph (LogicalGraph): The forward pass graph.
        wrt (List[str]): List of node IDs to compute gradients for.
        output_id (str): The node ID of the scalar output.

    Returns:
        LogicalGraph: A new graph containing both forward pass and gradient computations.

    Raises:
        ValueError: If output node does not exist, or required VJPs are missing.
    """
    if output_id not in graph.nodes:
        raise ValueError(f"Output node '{output_id}' not found in graph.")

    # Copy forward graph into new graph
    new_graph = LogicalGraph(name=f"{graph.name}_grad")
    for nid, node in graph.nodes.items():
        # Copy nodes manually to avoid deepcopy overhead
        new_graph.nodes[nid] = LogicalNode(
            id=node.id,
            op_type=node.op_type,
            domain=node.domain,
            version=node.version,
            attributes=dict(node.attributes),
            inputs=list(node.inputs),
            shape_metadata=node.shape_metadata,
            source_ast_ref=node.source_ast_ref,
            sharding=node.sharding,
        )

    # 1. Topological sort to determine evaluation order
    from ml_switcheroo_ir import topological_sort

    sorted_nodes = topological_sort(new_graph)

    # Prune nodes that don't lead to the output
    # (Simple backward reachability check)
    reachable_from_output: Set[str] = {output_id}
    # Reverse iteration
    for node in reversed(sorted_nodes):
        if node.id in reachable_from_output:
            for inp in node.inputs:
                reachable_from_output.add(inp)

    # 2. Initialize adjoints (gradients)
    # Map from forward_node_id -> adjoint_node_id
    adjoints: Dict[str, str] = {}

    # The adjoint of the output is 1.0
    one_id = f"grad_ones_{uuid.uuid4().hex[:6]}"
    ones_node = LogicalNode(
        id=one_id, op_type="Constant", attributes={"value": 1.0}, shape_metadata=()
    )
    new_graph.nodes[one_id] = ones_node
    adjoints[output_id] = one_id

    # 3. Backward pass
    for node in reversed(sorted_nodes):
        nid = node.id
        if nid not in reachable_from_output or nid not in adjoints:
            continue

        adj_id = adjoints[nid]

        # StopGradient node passes 0.0 backwards (or breaks the chain)
        if node.op_type == "StopGradient":
            continue

        if node.op_type in ["Input", "Constant"]:
            continue

        if node.op_type not in _VJP_REGISTRY:
            raise ValueError(f"Missing VJP rule for operation: {node.op_type}")

        vjp_fn = _VJP_REGISTRY[node.op_type]
        # VJP returns a list of adjoints corresponding to the inputs
        input_adjs = vjp_fn(new_graph, node, adj_id)

        if len(input_adjs) != len(node.inputs):
            raise ValueError(
                f"VJP for {node.op_type} returned {len(input_adjs)} adjoints, expected {len(node.inputs)}."
            )

        for inp_id, inp_adj_id in zip(node.inputs, input_adjs):
            if inp_adj_id is None:
                continue

            # Gradient Accumulation
            if inp_id in adjoints:
                adjoints[inp_id] = _add_nodes(new_graph, adjoints[inp_id], inp_adj_id)
            else:
                adjoints[inp_id] = inp_adj_id

    # 4. Extract required gradients
    grad_outputs = []
    for w in wrt:
        if w not in new_graph.nodes:
            raise ValueError(f"Target node '{w}' not found in graph.")

        if w in adjoints:
            grad_outputs.append(adjoints[w])
        else:
            # Zero gradient
            zero_id = f"grad_zeros_{uuid.uuid4().hex[:6]}"
            zeros_node = LogicalNode(
                id=zero_id,
                op_type="Constant",
                attributes={"value": 0.0},
                shape_metadata=new_graph.nodes[w].shape_metadata,
            )
            new_graph.nodes[zero_id] = zeros_node
            grad_outputs.append(zero_id)

    new_graph.outputs = grad_outputs
    return new_graph
