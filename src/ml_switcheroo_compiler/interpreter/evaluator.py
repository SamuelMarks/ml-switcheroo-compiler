"""IR evaluator using the OpRegistry."""

import ast
import builtins

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


def _parse_slice_call(node: ast.Call) -> object:
    if not isinstance(node.func, ast.Name):
        return None
    if node.func.id == "slice":
        return builtins.slice(*[_parse_slice_node(arg) for arg in node.args])
    if node.func.id == "array":
        return _parse_slice_node(node.args[0])
    return None


def _parse_tuple(node: ast.Tuple) -> object:
    return tuple(_parse_slice_node(elt) for elt in node.elts)


def _parse_list(node: ast.List) -> object:
    return list(_parse_slice_node(elt) for elt in node.elts)


def _parse_constant(node: ast.Constant) -> object:
    return node.value


def _parse_name(node: ast.Name) -> object:
    return {"None": None, "Ellipsis": Ellipsis, "False": False, "True": True}.get(node.id)


def _parse_unary(node: ast.UnaryOp) -> object:
    if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        return -node.operand.value
    return None


def _parse_slice_node(node: ast.AST) -> object:
    dispatch = {
        ast.Tuple: _parse_tuple,
        ast.List: _parse_list,
        ast.Constant: _parse_constant,
        ast.Name: _parse_name,
        ast.UnaryOp: _parse_unary,
        ast.Call: _parse_slice_call,
    }
    handler = dispatch.get(type(node))
    if handler:
        return handler(node)
    msg = f"Unsupported slice expression node: {type(node)}"
    raise ValueError(msg)


def _parse_slice_string(s: str) -> object:
    return _parse_slice_node(ast.parse(str(s), mode="eval").body)


def _handle_slice(
    node: LogicalNode,
    env: Environment,
    backend: object,
    in_vals: list[object],
    kwargs: dict[str, object],
) -> None:
    """Handle slice operations during evaluation."""
    if "slices" in kwargs:
        parsed_key = _parse_slice_string(str(kwargs["slices"]))
        env.set(node.id, backend.asarray(in_vals[0])[parsed_key])
        return

    dim = kwargs.get("dim", 0)
    start = kwargs.get("start", None)
    end = kwargs.get("end", None)
    step = kwargs.get("step", 1)

    sl = [builtins.slice(None)] * len(in_vals[0].shape)
    sl[dim] = builtins.slice(start, end, step)
    env.set(node.id, backend.asarray(in_vals[0])[tuple(sl)])


def _handle_getitem(
    node: LogicalNode,
    env: Environment,
    backend: object,
    in_vals: list[object],
    kwargs: dict[str, object],
) -> None:
    """Handle getitem operations during evaluation."""
    key = kwargs.get("key")

    parsed_key = _parse_slice_string(str(key))
    env.set(node.id, backend.asarray(in_vals[0])[parsed_key])


def _evaluate_node(node: LogicalNode, env: Environment, backend: object) -> None:
    """Evaluate a single logical node and update the environment.

    Args:
        node (LogicalNode): The node to evaluate.
        env (Environment): The local variable environment.
        backend (object): The active compute backend.
    """
    if node.op_type == "Input":
        env.get(node.id)
        return

    if node.op_type == "Output":
        in_vals = [env.get(inp) for inp in node.inputs]
        val = in_vals[0] if len(in_vals) == 1 else tuple(in_vals)
        env.set(node.id, val)
        return

    if node.op_type == "Constant":
        env.set(node.id, backend.array(node.attributes["value"]))
        return

    in_vals = [env.get(inp) for inp in node.inputs]
    target_op = _get_op_alias(node.op_type)
    kwargs = _prepare_node_kwargs(node, target_op)

    if target_op == "Slice":
        _handle_slice(node, env, backend, in_vals, kwargs)
        return

    if target_op == "GetItem":
        _handle_getitem(node, env, backend, in_vals, kwargs)
        return

    if target_op == "Meshgrid":
        idx = kwargs.pop("output_index", 0)
        result = backend.execute_op(target_op, *in_vals, **kwargs)
        env.set(node.id, result[idx])
        return

    result = backend.execute_op(target_op, *in_vals, **kwargs)
    env.set(node.id, result)


def _get_op_alias(op_type: str) -> str:
    """Resolve an operation alias to its standardized name.

    Args:
        op_type (str): The raw operation name.

    Returns:
        str: The standardized operation name.
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
    """Prepare the keyword arguments required for executing a node.

    Args:
        node (LogicalNode): The target node.
        target_op (str): The resolved operation name.

    Returns:
        dict: A mapping of argument names to values.
    """
    kwargs = {**node.attributes}
    if getattr(node, "shape_metadata", None):
        if target_op in ("Expand", "BroadcastTo", "ConstantOfShape", "Zeros", "Ones", "Full") and "shape" not in kwargs:
            kwargs["shape"] = node.shape_metadata
        if target_op == "Reshape" and "newshape" not in kwargs:
            kwargs["newshape"] = node.shape_metadata
    return kwargs
