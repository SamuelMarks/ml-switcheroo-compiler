"""IR evaluator using the OpRegistry."""

import builtins

import numpy as np
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort
from ml_switcheroo_compiler.interpreter.environment import Environment


def evaluate_graph(graph: LogicalGraph, inputs: dict[str, object]) -> dict[str, object]:
    """Evaluate an IR graph eagerly.

    graph (LogicalGraph): The graph to evaluate
    inputs (Dict[str, Any]): The input dictionary mapping node IDs to values

    Returns:
    Dict[str, Any]: The outputs mapping node IDs to values

    Args:
        graph (LogicalGraph): Argument graph
        inputs (dict[str, object]): Argument inputs
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


def _handle_slice(
    node: LogicalNode,
    env: Environment,
    backend: object,
    in_vals: list[object],
    kwargs: dict[str, object],
) -> None:  # pragma: no cover
    """Function docstring."""
    if "slices" in kwargs:
        parsed_key = eval(
            kwargs["slices"],
            {
                "slice": builtins.slice,
                "Ellipsis": Ellipsis,
                "None": None,
                "np": np,
                "array": np.array,
            },
        )
        env.set(node.id, backend.array(np.array(in_vals[0])[parsed_key]))
        return

    dim = kwargs.get("dim", 0)
    start = kwargs.get("start", None)
    end = kwargs.get("end", None)
    step = kwargs.get("step", 1)

    sl = [builtins.slice(None)] * len(in_vals[0].shape)
    sl[dim] = builtins.slice(start, end, step)
    env.set(node.id, backend.array(np.array(in_vals[0])[tuple(sl)]))


def _handle_getitem(
    node: LogicalNode,
    env: Environment,
    backend: object,
    in_vals: list[object],
    kwargs: dict[str, object],
) -> None:  # pragma: no cover
    """Function docstring."""
    key = kwargs.get("key")

    parsed_key = eval(
        key,
        {
            "slice": builtins.slice,
            "Ellipsis": Ellipsis,
            "None": None,
            "np": np,
            "array": np.array,
        },
    )
    env.set(node.id, backend.array(np.array(in_vals[0])[parsed_key]))


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

    if target_op == "Slice":  # pragma: no cover
        _handle_slice(node, env, backend, in_vals, kwargs)
        return

    if target_op == "GetItem":  # pragma: no cover
        _handle_getitem(node, env, backend, in_vals, kwargs)
        return

    if target_op == "Meshgrid":  # pragma: no cover
        idx = kwargs.pop("output_index", 0)
        result = backend.execute_op(target_op, *in_vals, **kwargs)
        env.set(node.id, result[idx])
        return

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


def _prepare_node_kwargs(node: LogicalNode, target_op: str) -> dict[str, object]:
    """Execute _prepare_node_kwargs.

    Args:
        node (Any): Argument node.
        target_op (Any): Argument target_op.

    Returns:
    Any: The result.
    """
    kwargs = {**node.attributes}
    if getattr(node, "shape_metadata", None):
        if target_op in ("Expand", "BroadcastTo", "ConstantOfShape", "Zeros", "Ones", "Full") and "shape" not in kwargs:
            kwargs["shape"] = node.shape_metadata
        if target_op == "Reshape" and "newshape" not in kwargs:
            kwargs["newshape"] = node.shape_metadata
    return kwargs
