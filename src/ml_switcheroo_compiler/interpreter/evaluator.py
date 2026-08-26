# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""IR evaluator using the OpRegistry."""

import ast
import builtins

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort
from ml_switcheroo_compiler.interpreter.environment import Environment


def evaluate_graph(graph: LogicalGraph, inputs):
    """Execute a logical computation graph using eager mode evaluation.

    Args:
        graph (LogicalGraph): The directed acyclic graph defining the operations to evaluate.
        inputs (dict[str, object]): A mapping of input node IDs to their corresponding concrete tensor values.

    Returns:
        dict[str, object]: A dictionary mapping output node IDs to their computed tensor results.

    Raises:
        RuntimeError: If an output node was never evaluated during the graph execution.
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


def _parse_slice_call(node: ast.Call):
    """Parse a slice call AST node into a python slice or array object.

    Args:
        node (ast.Call): The Call AST node.

    Returns: Tensor: The parsed slice object.
    """
    if not isinstance(node.func, ast.Name):
        return None
    if node.func.id == "slice":
        return builtins.slice(*[_parse_slice_node(arg) for arg in node.args])
    if node.func.id == "array":
        return _parse_slice_node(node.args[0])
    return None


def _parse_tuple(node: ast.Tuple):
    """Parse a tuple AST node.

    Args:
        node (ast.Tuple): The Tuple AST node.

    Returns: Tensor: The parsed tuple object.
    """
    return tuple(_parse_slice_node(elt) for elt in node.elts)


def _parse_list(node: ast.List):
    """Parse a list AST node.

    Args:
        node (ast.List): The List AST node.

    Returns: Tensor: The parsed list object.
    """
    return list(_parse_slice_node(elt) for elt in node.elts)


def _parse_constant(node: ast.Constant):
    """Parse a constant AST node.

    Args:
        node (ast.Constant): The Constant AST node.

    Returns: Tensor: The parsed constant value.
    """
    return node.value


def _parse_name(node: ast.Name):
    """Parse a name AST node into standard singletons.

    Args:
        node (ast.Name): The Name AST node.

    Returns: Tensor: The parsed name object (e.g., None, Ellipsis).
    """
    return {"None": None, "Ellipsis": Ellipsis, "False": False, "True": True}.get(node.id)


def _parse_unary(node: ast.UnaryOp):
    """Parse a unary operation AST node.

    Args:
        node (ast.UnaryOp): The UnaryOp AST node.

    Returns: Tensor: The parsed unary operation value.
    """
    if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        return -node.operand.value
    return None


def _parse_slice_node(node: ast.AST):
    """Dispatch and parse an AST node for a slice representation into native Python types.

    Args:
        node (ast.AST): The parsed AST node representing a portion of a slice index.

    Returns: Tensor: The interpreted Python slice object, tuple, list, or constant value.

    Raises:
        ValueError: If the AST node type is not supported in slice expressions.
    """
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


def _parse_slice_string(s: str):
    """Parse a string representation of a slice index.

    Args:
        s (str): The slice string to evaluate.

    Returns: Tensor: The parsed python slice object.
    """
    return _parse_slice_node(ast.parse(str(s), mode="eval").body)


def _handle_slice(
    node: LogicalNode,
    env: Environment,
    backend,
    in_vals,
    kwargs,
) -> None:
    """Execute a slicing operation on a tensor and store the result.

    Args:
        node (LogicalNode): The IR node representing the slice operation.
        env (Environment): The current local variable environment for storing results.
        backend (object): The active compute backend used for array manipulation.
        in_vals (list[object]): List containing the input tensor to be sliced.
        kwargs (dict[str, object]): Keyword arguments defining the slice parameters (e.g. 'slices', 'dim', 'start', 'end', 'step').
    """
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
    backend,
    in_vals,
    kwargs,
) -> None:
    """Execute a getitem operation (advanced indexing) and store the result.

    Args:
        node (LogicalNode): The IR node representing the getitem operation.
        env (Environment): The local variable environment.
        backend (object): The active compute backend for array manipulation.
        in_vals (list[object]): List containing the input tensor.
        kwargs (dict[str, object]): Keyword arguments containing the 'key' for indexing.
    """
    key = kwargs.get("key")

    parsed_key = _parse_slice_string(str(key))
    env.set(node.id, backend.asarray(in_vals[0])[parsed_key])


def _handle_checkpoint(
    node: LogicalNode,
    env: Environment,
    in_vals,
) -> None:
    """Execute a gradient checkpointing subgraph dynamically.

    Args:
        node (LogicalNode): The IR node representing the checkpoint wrapper.
        env (Environment): The local variable environment.
        in_vals (list[object]): List of input tensors to feed into the subgraph.
    """
    subgraph = node.attributes["subgraph"]
    sub_inputs = {}
    for inp_val, in_id in zip(in_vals, subgraph.inputs):
        sub_inputs[in_id] = inp_val

    from ml_switcheroo_ir import LogicalGraph

    sub_graph = LogicalGraph(name="checkpoint_subgraph")
    if isinstance(subgraph.nodes, list):
        for n in subgraph.nodes:
            sub_graph.nodes[n.id] = n
    else:
        for k, n in subgraph.nodes.items():
            sub_graph.nodes[k] = n
    sub_graph.inputs = subgraph.inputs
    sub_graph.outputs = subgraph.outputs

    sub_outputs = evaluate_graph(sub_graph, sub_inputs)

    if len(sub_graph.outputs) == 1:
        result = sub_outputs[sub_graph.outputs[0]]
    else:
        result = tuple(sub_outputs[out_id] for out_id in sub_graph.outputs)
    env.set(node.id, result)


def _handle_meshgrid(
    node: LogicalNode,
    env: Environment,
    backend,
    in_vals,
    kwargs,
) -> None:
    """Execute a meshgrid operation and extract the correct output index.

    Args:
        node (LogicalNode): The IR node representing the meshgrid operation.
        env (Environment): The local variable environment.
        backend (object): The active compute backend for execution.
        in_vals (list[object]): List of input 1D coordinate arrays.
        kwargs (dict[str, object]): Keyword arguments including 'output_index' to select the desired returned grid.
    """
    idx = kwargs.pop("output_index", 0)
    result = backend.execute_op("Meshgrid", *in_vals, **kwargs)
    env.set(node.id, result[idx])


def _dispatch_op(node: LogicalNode, env: Environment, backend, target_op: str, in_vals, kwargs) -> None:
    """Dispatch the execution of an operation to the appropriate handler.

    Args:
        node (LogicalNode): The IR node.
        env (Environment): The current local variable environment.
        backend (object): The active compute backend.
        target_op (str): The name of the target operation.
        in_vals (list): List of input values.
        kwargs (dict): Keyword arguments for the operation.
    """
    if target_op == "Slice":
        _handle_slice(node, env, backend, in_vals, kwargs)
    elif target_op == "GetItem":
        _handle_getitem(node, env, backend, in_vals, kwargs)
    elif target_op == "Checkpoint":
        _handle_checkpoint(node, env, in_vals)
    elif target_op == "Meshgrid":
        _handle_meshgrid(node, env, backend, in_vals, kwargs)
    else:
        result = backend.execute_op(target_op, *in_vals, **kwargs)
        env.set(node.id, result)


def _evaluate_if_node(node: LogicalNode, env: Environment, backend) -> None:
    """_evaluate_if_node function.

    Args:
        node (object): The node parameter.
        env (object): The env parameter.
        backend (object): The backend parameter.

    Returns:
        object: Result.
    """
    cond_val = env.get(node.inputs[0])
    is_true = bool(cond_val) if isinstance(cond_val, (bool, int, float)) else bool(getattr(cond_val, "item", lambda cr=cond_val: cr)())
    branch = node.attributes.get("then_branch" if is_true else "else_branch")
    if not branch:
        branch = node.attributes.get("true_branch" if is_true else "false_branch")
    if branch:
        nodes = branch.nodes if isinstance(branch.nodes, list) else list(branch.nodes.values())
        for sub_node in nodes:
            _evaluate_node(sub_node, env, backend)
        if hasattr(branch, "outputs") and branch.outputs:
            if isinstance(branch.outputs, list):
                env.set(node.id, env.get(branch.outputs[0]))
            else:
                env.set(node.id, env.get(branch.outputs))
        else:
            last_node = nodes[-1]
            env.set(node.id, env.get(last_node.id))


def _evaluate_loop_node(node: LogicalNode, env: Environment, backend) -> None:
    """_evaluate_loop_node function.

    Args:
        node (object): The node parameter.
        env (object): The env parameter.
        backend (object): The backend parameter.

    Returns:
        object: Result.
    """
    cond_graph = node.attributes.get("cond")
    body_graph = node.attributes.get("body")
    curr_val = env.get(node.inputs[0])
    while True:
        cond_input_id = cond_graph.nodes[0].id if isinstance(cond_graph.nodes, list) else list(cond_graph.nodes.keys())[0]
        env.set(cond_input_id, curr_val)
        nodes = cond_graph.nodes if isinstance(cond_graph.nodes, list) else cond_graph.nodes.values()
        for sub_node in nodes:
            if sub_node.id != cond_input_id:
                _evaluate_node(sub_node, env, backend)
        cond_res = env.get(cond_graph.outputs[0])
        is_true = bool(cond_res) if isinstance(cond_res, (bool, int, float)) else bool(getattr(cond_res, "item", lambda cr=cond_res: cr)())
        if not is_true:
            break
        body_input_id = body_graph.nodes[0].id if isinstance(body_graph.nodes, list) else list(body_graph.nodes.keys())[0]
        env.set(body_input_id, curr_val)
        nodes = body_graph.nodes if isinstance(body_graph.nodes, list) else body_graph.nodes.values()
        for sub_node in nodes:
            if sub_node.id != body_input_id:
                _evaluate_node(sub_node, env, backend)
        curr_val = env.get(body_graph.outputs[0])
    env.set(node.id, curr_val)


def _evaluate_node(node: LogicalNode, env: Environment, backend) -> None:
    """Execute a single logical node and update the local environment with its result.

    Args:
        node (LogicalNode): The graph node to evaluate.
        env (Environment): The local variable environment acting as memory for the execution.
        backend (object): The underlying backend engine to dispatch computation to.
    """
    if node.op_type == "Input":
        env.get(node.id)
    elif node.op_type == "Output":
        in_vals = [env.get(inp) for inp in node.inputs if inp]
        val = in_vals[0] if len(in_vals) == 1 else tuple(in_vals)
        env.set(node.id, val)
    elif node.op_type in ("If", "Cond"):
        _evaluate_if_node(node, env, backend)
    elif node.op_type in ("WhileLoop", "Loop"):
        _evaluate_loop_node(node, env, backend)
    elif node.op_type == "Constant":
        env.set(node.id, backend.array(node.attributes["value"]))
    elif node.op_type == "Recompute":
        in_vals = [env.get(inp) for inp in node.inputs]
        orig_op = node.attributes.get("original_op", node.op_type)
        target_op = _get_op_alias(orig_op)
        kwargs = node.attributes.get("original_attrs", {}).copy()
        # For simplicity, dispatch as original op
        dummy_node = LogicalNode(id=node.id, op_type=target_op, inputs=node.inputs, attributes=kwargs)
        # We need to extract kwargs correctly for the target op
        kwargs = _prepare_node_kwargs(dummy_node, target_op)
        _dispatch_op(dummy_node, env, backend, target_op, in_vals, kwargs)
    else:
        in_vals = [env.get(inp) for inp in node.inputs]
        target_op = _get_op_alias(node.op_type)
        kwargs = _prepare_node_kwargs(node, target_op)
        _dispatch_op(node, env, backend, target_op, in_vals, kwargs)


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


def _prepare_node_kwargs(node: LogicalNode, target_op: str):
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
