"""Reference Interpreter for executing LogicalGraphs using pure numpy."""

import numpy as np
from typing import Dict, Any
from ml_switcheroo_ir import LogicalGraph, topological_sort


def evaluate_graph(
    graph: LogicalGraph, inputs: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Evaluates a LogicalGraph eagerly using numpy.

    Args:
        graph (LogicalGraph): The computational graph to execute.
        inputs (Dict[str, np.ndarray]): A mapping of Input node IDs to numpy arrays.

    Returns:
        Dict[str, np.ndarray]: A mapping of output IDs to their computed values.
    """
    env: Dict[str, Any] = {}

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

        # Execute ops
        if node.op_type == "Add":
            env[node.id] = in_vals[0] + in_vals[1]
        elif node.op_type == "Sub":
            env[node.id] = in_vals[0] - in_vals[1]
        elif node.op_type == "Mul":
            env[node.id] = in_vals[0] * in_vals[1]
        elif node.op_type == "Div":
            env[node.id] = in_vals[0] / in_vals[1]
        elif node.op_type == "Neg":
            env[node.id] = -in_vals[0]
        elif node.op_type == "Exp":
            env[node.id] = np.exp(in_vals[0])
        elif node.op_type == "Log":
            env[node.id] = np.log(in_vals[0])
        elif node.op_type == "Pow":
            env[node.id] = np.power(in_vals[0], in_vals[1])
        elif node.op_type == "Sum":
            env[node.id] = np.sum(in_vals[0])
        elif node.op_type == "Mean":
            env[node.id] = np.mean(in_vals[0])
        elif node.op_type == "Max":
            env[node.id] = np.max(in_vals[0])
        elif node.op_type == "Min":
            env[node.id] = np.min(in_vals[0])
        elif node.op_type == "MatMul":
            env[node.id] = np.matmul(in_vals[0], in_vals[1])
        elif node.op_type == "Dot":
            env[node.id] = np.dot(in_vals[0], in_vals[1])
        elif node.op_type == "Transpose":
            env[node.id] = np.transpose(in_vals[0])
        elif node.op_type == "Relu":
            env[node.id] = np.maximum(in_vals[0], 0.0)
        elif node.op_type == "Greater":
            env[node.id] = in_vals[0] > in_vals[1]
        elif node.op_type == "Where":
            env[node.id] = np.where(in_vals[0], in_vals[1], in_vals[2])
        elif node.op_type == "Expand":
            env[node.id] = (
                np.broadcast_to(in_vals[0], node.shape_metadata)
                if node.shape_metadata
                else in_vals[0]
            )
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
