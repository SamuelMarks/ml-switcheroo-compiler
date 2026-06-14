"""IR evaluator using the OpRegistry."""

from typing import Any

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.interpreter.environment import Environment
from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort


def evaluate_graph(graph: LogicalGraph, inputs: dict[str, Any]) -> dict[str, Any]:
    """Evaluate an IR graph eagerly.

    graph (LogicalGraph): The graph to evaluate
    inputs (Dict[str, Any]): The input dictionary mapping node IDs to values

    Returns:
    Dict[str, Any]: The outputs mapping node IDs to values

    Args:
        graph (LogicalGraph): Argument graph
        inputs (dict[str, Any]): Argument inputs
    """
    env = Environment(inputs)
    sorted_nodes = topological_sort(graph)
    backend = get_active_backend()

    for node in sorted_nodes:
        _evaluate_node(node, env, backend)

    outputs = {}
    for out_id in graph.outputs:
        if out_id not in env:
            msg = f"Output node '{out_id}' was never evaluated."
            raise RuntimeError(msg)
        outputs[out_id] = env.get(out_id)
    return outputs


def _evaluate_node(node: LogicalNode, env: Environment, backend: object) -> None:
    """Execute _evaluate_node.

    Args:
        node (Any): Argument node.
        env (Any): Argument env.
        backend (Any): Argument backend.
    """
    if node.op_type == "Input":
        env.get(node.id)
        return

    if node.op_type == "Constant":
        env.set(node.id, backend.array(node.attributes["value"]))
        return

    in_vals = [env.get(inp) for inp in node.inputs]
    target_op = _get_op_alias(node.op_type)
    kwargs = _prepare_node_kwargs(node, target_op)

    result = backend.execute_op(target_op, *in_vals, **kwargs)
    env.set(node.id, result)


def _get_op_alias(op_type: str) -> str:
    """Execute _get_op_alias.

    Args:
        op_type (Any): Argument op_type.

    Returns:
    Any: The result.
    """
    op_alias = {
        "Sub": "Subtract",
        "Mul": "Multiply",
        "Div": "TrueDivide",
        "Neg": "Negative",
        "Pow": "Power",
        "MatMul": "Matmul",
        "Expand": "BroadcastTo",
        "Permute": "Transpose",
    }
    return op_alias.get(op_type, op_type)


def _prepare_node_kwargs(node: LogicalNode, target_op: str) -> dict:
    """Execute _prepare_node_kwargs.

    Args:
        node (Any): Argument node.
        target_op (Any): Argument target_op.

    Returns:
    Any: The result.
    """
    kwargs = {**node.attributes}
    if getattr(node, "shape_metadata", None):
        if target_op in ("Expand", "BroadcastTo") and "shape" not in kwargs:
            kwargs["shape"] = node.shape_metadata
        if target_op == "Reshape" and "newshape" not in kwargs:
            kwargs["newshape"] = node.shape_metadata
    return kwargs
