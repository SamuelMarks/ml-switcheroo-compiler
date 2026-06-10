"""Reference Interpreter for executing LogicalGraphs using pure numpy."""

import numpy as np
from typing import Any
from ml_switcheroo_ir import LogicalGraph, topological_sort


def evaluate_graph(
    graph: LogicalGraph, inputs: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Evaluates a LogicalGraph eagerly using numpy.

    Args:
        graph (LogicalGraph): The computational graph to execute.
        inputs (Dict[str, np.ndarray]): A mapping of Input node IDs to numpy arrays.

    Returns:
        Dict[str, np.ndarray]: A mapping of output IDs to their computed values.
    """
    env: dict[str, Any] = {}

    # Initialize inputs
    for k, v in inputs.items():
        env[k] = v

    sorted_nodes = topological_sort(graph)

    for node in sorted_nodes:
        if node.op_type == "Input":
            if node.id not in env:
                raise ValueError(f"Missing input value for node '{node.id}'")
            continue  # pragma: no cover

        elif node.op_type == "Constant":
            env[node.id] = np.array(node.attributes["value"])
            continue

        # Get evaluated inputs
        in_vals = [env[inp] for inp in node.inputs]

        op_lower = node.op_type.lower()
        op_map = {
            "sub": "subtract",
            "mul": "multiply",
            "div": "divide",
            "neg": "negative",
            "pow": "power",
        }

        np_func_name = op_map.get(op_lower, op_lower)

        # Custom / special mapping implementations
        if node.op_type == "Relu":
            env[node.id] = np.maximum(in_vals[0], 0.0)
        elif node.op_type == "Expand":
            env[node.id] = (
                np.broadcast_to(in_vals[0], node.shape_metadata)
                if node.shape_metadata
                else in_vals[0]
            )
        elif hasattr(np, np_func_name):
            np_func = getattr(np, np_func_name)
            env[node.id] = np_func(*in_vals)
        else:
            raise NotImplementedError(
                f"Interpreter missing implementation for: {node.op_type}"
            )

    # Extract outputs
    outputs = {}
    for out_id in graph.outputs:
        if out_id not in env:
            raise RuntimeError(f"Output node '{out_id}' was never evaluated.")
        outputs[out_id] = env[out_id]

    return outputs
