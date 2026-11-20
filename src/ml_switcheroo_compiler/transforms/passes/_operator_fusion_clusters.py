"""Elementwise cluster scalar-expression fusion transformations."""

from __future__ import annotations

from ml_switcheroo_ir import IRGraph, IRNode

from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.transforms.passes._operator_fusion_patterns import (
    _GraphNodesProtocol,
)

ELEMENTWISE_OPS: set[str] = {
    "Add",
    "Sub",
    "Mul",
    "Div",
    "TrueDivide",
    "Relu",
    "Sigmoid",
    "Neg",
    "Abs",
    "Exp",
    "Log",
    "Sqrt",
    "Gelu",
    "Silu",
    "Tanh",
    "Pow",
    "Maximum",
    "Minimum",
}


_SCALAR_EXPR_TEMPLATES: dict[str, str] = {
    "add": "({a} + {b})",
    "sub": "({a} - {b})",
    "mul": "({a} * {b})",
    "div": "({a} / {b})",
    "truedivide": "({a} / {b})",
    "relu": "fmaxf(0.0f, {a})",
    "sigmoid": "(1.0f / (1.0f + expf(-{a})))",
    "neg": "(-{a})",
    "abs": "fabsf({a})",
    "exp": "expf({a})",
    "log": "logf({a})",
    "sqrt": "sqrtf({a})",
    "tanh": "tanhf({a})",
    "gelu": "(0.5f * {a} * (1.0f + tanhf(0.79788456f * ({a} + 0.044715f * {a} * {a} * {a}))))",
    "silu": "({a} / (1.0f + expf(-{a})))",
    "maximum": "fmaxf({a}, {b})",
    "minimum": "fminf({a}, {b})",
}


def _get_scalar_expression_snippet(op_type: str, args: list[str]) -> str:
    """Generate scalar expression snippet for an elementwise op.

    Args:
        op_type (str): Operation name.
        args (list[str]): Input argument identifiers.

    Returns:
        str: C/C++ style scalar expression.
    """
    a = args[0] if args else "0.0f"
    b = args[1] if len(args) > 1 else "0.0f"
    template = _SCALAR_EXPR_TEMPLATES.get(op_type.lower())
    return template.format(a=a, b=b) if template is not None else a


def _extract_node_ops_and_expr(node: IRNode) -> tuple[list[str], str]:
    """Extract the list of fused ops and scalar expression for an elementwise or fused node.

    Args:
        node (IRNode): Elementwise or FusedElementwise IR node.

    Returns:
        tuple[list[str], str]: List of constituent op names and scalar C expression string.
    """
    if node.op_type == "FusedElementwise":
        raw_ops = node.attributes.get("fused_ops", [])
        ops_list = [str(op) for op in raw_ops] if isinstance(raw_ops, (list, tuple)) else []
        return ops_list, str(node.attributes.get("scalar_expr", "0.0f"))
    args = [f"in_{k}" for k in range(len(node.inputs))]
    return [node.op_type], _get_scalar_expression_snippet(node.op_type, args)


def _fuse_single_edge(
    graph: IRGraph,
    node: IRNode,
    inp_id: str,
    inp_index: int,
    producer: IRNode,
) -> None:
    """Fuse a single producer node into a consumer node.

    Args:
        graph (IRGraph): Target computation graph.
        node (IRNode): Consumer node to fuse into.
        inp_id (str): Producer node id to remove.
        inp_index (int): Input slot index being replaced.
        producer (IRNode): Producer node being inlined.
    """
    p_ops, p_expr = _extract_node_ops_and_expr(producer)
    n_ops, n_expr = _extract_node_ops_and_expr(node)
    fused_ops = p_ops + n_ops

    new_inputs = list(node.inputs[:inp_index]) + list(producer.inputs) + list(node.inputs[inp_index + 1 :])
    unique_inputs: list[str] = list(dict.fromkeys(new_inputs))

    for k, p_in in enumerate(producer.inputs):
        p_expr = p_expr.replace(f"in_{k}", f"in{unique_inputs.index(p_in)}")

    for idx_in, inp in enumerate(node.inputs):
        replacement = p_expr if idx_in == inp_index else f"in{unique_inputs.index(inp)}"
        n_expr = n_expr.replace(f"in_{idx_in}", replacement)

    fused_node = IRNode(
        id=node.id,
        op_type="FusedElementwise",
        inputs=unique_inputs,
        shape_metadata=node.shape_metadata,
    )
    fused_node.attributes = {
        "fused_ops": fused_ops,
        "scalar_expr": n_expr,
    }

    del graph.nodes[inp_id]
    graph.nodes[node.id] = fused_node


def _count_consumers(graph: IRGraph) -> dict[str, int]:
    """Compute the number of downstream consumers for each node in the graph.

    Args:
        graph (IRGraph): The IR computation graph.

    Returns:
        dict[str, int]: Mapping from node ID to consumer reference count.
    """
    consumer_counts: dict[str, int] = {node_id: 0 for node_id in graph.nodes}
    for node in graph.nodes.values():
        for inp in node.inputs:
            if inp in consumer_counts:
                consumer_counts[inp] += 1
    graph_outputs = list(graph.outputs or [])
    for out_id in graph_outputs:
        if out_id in consumer_counts:
            consumer_counts[out_id] += 1
    return consumer_counts


def _is_fusable_elementwise_node(node: IRNode | None) -> bool:
    """Check whether a node is an elementwise or FusedElementwise operation.

    Args:
        node (IRNode | None): Node to check.

    Returns:
        bool: True if the node can participate in elementwise cluster fusion.
    """
    return node is not None and (node.op_type in ELEMENTWISE_OPS or node.op_type == "FusedElementwise")


def _try_fuse_cluster_step(graph: IRGraph, consumer_counts: dict[str, int]) -> bool:
    """Perform a single elementwise cluster fusion step if a valid edge exists.

    Args:
        graph (IRGraph): The computation graph to mutate.
        consumer_counts (dict[str, int]): Reference counts per node ID.

    Returns:
        bool: True if a producer-consumer pair was fused, False otherwise.
    """
    for node in list(graph.nodes.values()):
        if not _is_fusable_elementwise_node(node):
            continue
        for i, inp_id in enumerate(list(node.inputs)):
            producer = graph.nodes.get(inp_id)
            if not _is_fusable_elementwise_node(producer) or consumer_counts.get(inp_id, 0) != 1:
                continue
            assert producer is not None
            p_shape = producer.shape_metadata
            n_shape = node.shape_metadata
            if p_shape and n_shape and p_shape != n_shape:
                continue
            _fuse_single_edge(graph, node, inp_id, i, producer)
            return True
    return False


def _has_graph_nodes(graph: IRGraph) -> bool:
    """Check if `graph` has a non-empty `nodes` mapping.

    Args:
        graph (IRGraph): Candidate IRGraph.

    Returns:
        bool: True if `graph.nodes` exists and is non-empty.
    """
    raw_graph: OpValue | _GraphNodesProtocol = graph
    return isinstance(raw_graph, _GraphNodesProtocol) and bool(raw_graph.nodes)


def fuse_elementwise_clusters(graph: IRGraph) -> bool:
    """Fuse elementwise DAG clusters into composite FusedElementwise IR nodes.

    Args:
        graph (IRGraph): The IR computation graph to optimize.

    Returns:
        bool: True if any elementwise DAG clusters were fused, False otherwise.
    """
    if not _has_graph_nodes(graph):
        return False

    modified: bool = False
    while _try_fuse_cluster_step(graph, _count_consumers(graph)):
        modified = True

    return modified
