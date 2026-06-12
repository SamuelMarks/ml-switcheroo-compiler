"""IR evaluator using the OpRegistry."""

from typing import Any
import numpy as np
from ml_switcheroo_ir import LogicalGraph
from ml_switcheroo.transforms.pass_manager import DAGTopologicalSorter
from ml_switcheroo.interpreter.environment import Environment
from ml_switcheroo.ops import get_op


def evaluate_graph(graph: LogicalGraph, inputs: dict[str, Any]) -> dict[str, Any]:
    """Evaluate an IR graph eagerly.

    Args:
        graph (LogicalGraph): The graph to evaluate.
        inputs (Dict[str, Any]): The input dictionary mapping node IDs to values.

    Returns:
        Dict[str, Any]: The outputs mapping node IDs to values.
    """
    env = Environment(inputs)
    sorted_nodes = DAGTopologicalSorter.sort(graph)

    for node in sorted_nodes:
        if node.op_type == "Input":
            # Just ensure it exists
            env.get(node.id)
            continue

        if node.op_type == "Constant":
            env.set(node.id, np.array(node.attributes["value"]))
            continue

        # Get evaluated inputs
        in_vals = [env.get(inp) for inp in node.inputs]

        # Use OpRegistry for dispatch
        # Special alias mapping for older tests (like 'Sub' -> 'Subtract', 'Mul' ->
        # 'Multiply')
        op_alias = {
            "Sub": "Subtract",
            "Mul": "Multiply",
            "Div": "TrueDivide",
            "Neg": "Negative",
            "Pow": "Power",
            "MatMul": "Matmul",
            "Expand": "BroadcastTo",
        }

        target_op = op_alias.get(node.op_type, node.op_type)

        try:
            op_cls = get_op(target_op)
        except KeyError:
            # Special logic for tests that expect Relu, Greater
            if target_op == "Relu":
                env.set(node.id, np.maximum(in_vals[0], 0.0))
                continue
            elif target_op == "Greater":
                env.set(node.id, np.greater(in_vals[0], in_vals[1]))
                continue
            raise NotImplementedError(
                f"Operation '{node.op_type}' is not implemented in interpreter."
            ) from None

        op_instance = op_cls()

        kwargs = {**node.attributes}
        if getattr(node, "shape_metadata", None):
            if target_op in ("Expand", "BroadcastTo"):
                kwargs["shape"] = node.shape_metadata
            if target_op == "Reshape":
                kwargs["newshape"] = node.shape_metadata

        result = op_instance.numpy_eval(*in_vals, **kwargs)
        env.set(node.id, result)

    outputs = {}
    for out_id in graph.outputs:
        if out_id not in env:
            raise RuntimeError(f"Output node '{out_id}' was never evaluated.")
        outputs[out_id] = env.get(out_id)

    return outputs
