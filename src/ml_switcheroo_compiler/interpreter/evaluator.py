"""IR evaluator using the OpRegistry."""

from __future__ import annotations

import ast
import builtins

from ml_switcheroo_ir import AttributeValue, LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, as_tensor
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort
from ml_switcheroo_compiler.interpreter.environment import Environment

TensorLike = EagerTensorProtocol
IndexableArray = TensorLike
EvalValue = TensorOrScalar
as_tensor_like = as_tensor


def evaluate_graph(
    graph: LogicalGraph,
    inputs: dict[str, EvalValue],
    backend: type[BaseGenerator] | None = None,
) -> dict[str, EvalValue]:
    """Execute a logical computation graph using eager mode evaluation.

    Args:
        graph (LogicalGraph): The directed acyclic graph defining the operations to evaluate.
        inputs (dict[str, EvalValue]): A mapping of input node IDs to their corresponding concrete tensor values.
        backend (type[BaseGenerator] | None): The compute backend generator class to dispatch ops to. Defaults to active backend.

    Returns:
        dict[str, EvalValue]: A dictionary mapping output node IDs to their computed tensor results.

    Raises:
        RuntimeError: If an output node was never evaluated during the graph execution.
    """
    initial_env: dict[str, EvalValue] = dict(getattr(graph, "initializers", {}) or {})
    initial_env.update(inputs)
    env = Environment(initial_env)
    sorted_nodes = topological_sort(graph)
    active_backend = backend if backend is not None else get_active_backend()

    for node in sorted_nodes:
        _evaluate_node(node, env, active_backend)

    outputs: dict[str, EvalValue] = {}
    for out_id in graph.outputs:
        if out_id not in env:
            msg = f"Output node '{out_id}' was never evaluated."
            raise RuntimeError(msg)
        outputs[out_id] = env.get(out_id)
    return outputs


def _parse_slice_call(node: ast.Call) -> EvalValue:
    """Parse a slice call AST node into a python slice or array object.

    Args:
        node (ast.Call): The Call AST node.

    Returns:
        EvalValue: The parsed slice object.
    """
    if not isinstance(node.func, ast.Name):
        return None
    if node.func.id == "slice":
        slice_args: list[int | float | str | None] = [v if isinstance((v := _parse_slice_node(arg)), (int, float, str)) else None for arg in node.args]
        return builtins.slice(*slice_args)
    if node.func.id == "array":
        return _parse_slice_node(node.args[0])
    return None


def _parse_tuple(node: ast.Tuple) -> EvalValue:
    """Parse a tuple AST node.

    Args:
        node (ast.Tuple): The Tuple AST node.

    Returns:
        EvalValue: The parsed tuple object.
    """
    return tuple(_parse_slice_node(elt) for elt in node.elts)


def _parse_list(node: ast.List) -> EvalValue:
    """Parse a list AST node.

    Args:
        node (ast.List): The List AST node.

    Returns:
        EvalValue: The parsed list object.
    """
    return list(_parse_slice_node(elt) for elt in node.elts)


def _parse_constant(node: ast.Constant) -> EvalValue:
    """Parse a constant AST node.

    Args:
        node (ast.Constant): The Constant AST node.

    Returns:
        EvalValue: The parsed constant value.
    """
    val = node.value
    return val.decode("utf-8", errors="replace") if isinstance(val, bytes) else val


def _parse_name(node: ast.Name) -> EvalValue:
    """Parse a name AST node into standard singletons.

    Args:
        node (ast.Name): The Name AST node.

    Returns:
        EvalValue: The parsed name object (e.g., None, Ellipsis).
    """
    return {"None": None, "Ellipsis": Ellipsis, "False": False, "True": True}.get(node.id)


def _parse_unary(node: ast.UnaryOp) -> EvalValue:
    """Parse a unary operation AST node.

    Args:
        node (ast.UnaryOp): The UnaryOp AST node.

    Returns:
        EvalValue: The parsed unary operation value.
    """
    if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float)):
        return -node.operand.value
    return None


def _parse_slice_node(node: ast.AST) -> EvalValue:
    """Dispatch and parse an AST node for a slice representation into native Python types.

    Args:
        node (ast.AST): The parsed AST node representing a portion of a slice index.

    Returns:
        EvalValue: The interpreted Python slice, tuple, list, or constant value.

    Raises:
        ValueError: If the AST node type is not supported in slice expressions.
    """
    if isinstance(node, (ast.Tuple, ast.List)):
        return _parse_tuple(node) if isinstance(node, ast.Tuple) else _parse_list(node)
    if isinstance(node, ast.Constant):
        return _parse_constant(node)
    if isinstance(node, ast.Name):
        return _parse_name(node)
    if isinstance(node, ast.UnaryOp):
        return _parse_unary(node)
    if isinstance(node, ast.Call):
        return _parse_slice_call(node)
    msg = f"Unsupported slice expression node: {type(node)}"
    raise ValueError(msg)


def _parse_slice_string(s: str) -> EvalValue:
    """Parse a string representation of a slice index.

    Args:
        s (str): The slice string to evaluate.

    Returns:
        EvalValue: The parsed python slice object.
    """
    return _parse_slice_node(ast.parse(str(s), mode="eval").body)


def _handle_slice(
    node: LogicalNode,
    env: Environment,
    backend: type[BaseGenerator],
    in_vals: list[EvalValue],
    kwargs: dict[str, EvalValue],
) -> None:
    """Execute a slicing operation on a tensor and store the result.

    Args:
        node (LogicalNode): The IR node representing the slice operation.
        env (Environment): The current local variable environment for storing results.
        backend (type[BaseGenerator]): The active compute backend used for array manipulation.
        in_vals (list[EvalValue]): List containing the input tensor to be sliced.
        kwargs (dict[str, EvalValue]): Keyword arguments defining the slice parameters (e.g. 'slices', 'dim', 'start', 'end', 'step').
    """
    arr = backend.asarray(in_vals[0])
    if "slices" in kwargs:
        parsed_key = _parse_slice_string(str(kwargs["slices"]))
        env.set(node.id, getattr(arr, "__getitem__")(parsed_key))
        return

    dim = int(getattr(kwargs.get("dim", 0), "__int__", lambda: 0)())
    raw_start = kwargs.get("start", None)
    raw_end = kwargs.get("end", None)
    raw_step = kwargs.get("step", 1)
    start: int | float | str | None = raw_start if isinstance(raw_start, (int, float, str)) else None
    end: int | float | str | None = raw_end if isinstance(raw_end, (int, float, str)) else None
    step: int | float | str | None = raw_step if isinstance(raw_step, (int, float, str)) else None

    sl: list[EvalValue] = [builtins.slice(None)] * len(getattr(in_vals[0], "shape"))
    sl[dim] = builtins.slice(start, end, step)
    env.set(node.id, getattr(arr, "__getitem__")(tuple(sl)))


def _handle_getitem(
    node: LogicalNode,
    env: Environment,
    backend: type[BaseGenerator],
    in_vals: list[EvalValue],
    kwargs: dict[str, EvalValue],
) -> None:
    """Execute a getitem operation (advanced indexing) and store the result.

    Args:
        node (LogicalNode): The IR node representing the getitem operation.
        env (Environment): The local variable environment.
        backend (type[BaseGenerator]): The active compute backend for array manipulation.
        in_vals (list[EvalValue]): List containing the input tensor.
        kwargs (dict[str, EvalValue]): Keyword arguments containing the 'key' for indexing.
    """
    key = kwargs.get("key")
    parsed_key = _parse_slice_string(str(key))
    arr = backend.asarray(in_vals[0])
    env.set(node.id, getattr(arr, "__getitem__")(parsed_key))


def _handle_index_put(
    node: LogicalNode,
    env: Environment,
    backend: type[BaseGenerator],
    in_vals: list[EvalValue],
    kwargs: dict[str, EvalValue],
) -> None:
    """Execute an index_put/scatter operation placing update elements into tensor.

    Args:
        node (LogicalNode): The IR node representing the index_put operation.
        env (Environment): The local variable environment.
        backend (type[BaseGenerator]): The active compute backend for array manipulation.
        in_vals (list[EvalValue]): List containing the base tensor and update values.
        kwargs (dict[str, EvalValue]): Keyword arguments containing the target index or key.
    """
    key = kwargs.get("key", kwargs.get("index", 0))
    parsed_key = _parse_slice_string(str(key))
    arr = backend.execute_op("Clone", in_vals[0])
    val = in_vals[1]
    getattr(arr, "__setitem__")(parsed_key, val)
    env.set(node.id, arr)


def _handle_checkpoint(
    node: LogicalNode,
    env: Environment,
    in_vals: list[EvalValue],
) -> None:
    """Execute a gradient checkpointing subgraph dynamically.

    Args:
        node (LogicalNode): The IR node representing the checkpoint wrapper.
        env (Environment): The local variable environment.
        in_vals (list[EvalValue]): List of input tensors to feed into the subgraph.
    """
    subgraph = getattr(node, "subgraphs", {}).get("body") or getattr(node, "subgraphs", {}).get("subgraph") or node.attributes.get("subgraph")
    sub_inputs: dict[str, EvalValue] = {}
    for inp_val, in_id in zip(in_vals, getattr(subgraph, "inputs")):
        sub_inputs[in_id] = inp_val

    sub_graph = LogicalGraph(name="checkpoint_subgraph")
    if isinstance(getattr(subgraph, "nodes"), list):
        for n in getattr(subgraph, "nodes"):
            sub_graph.nodes[n.id] = n
    else:
        for k, n in getattr(subgraph, "nodes").items():
            sub_graph.nodes[k] = n
    setattr(sub_graph, "inputs", getattr(subgraph, "inputs"))
    setattr(sub_graph, "outputs", getattr(subgraph, "outputs"))

    sub_outputs = evaluate_graph(sub_graph, sub_inputs)

    if len(sub_graph.outputs) == 1:
        result: EvalValue = sub_outputs[sub_graph.outputs[0]]
    else:
        result = tuple(sub_outputs[out_id] for out_id in sub_graph.outputs)
    env.set(node.id, result)


def _handle_meshgrid(
    node: LogicalNode,
    env: Environment,
    backend: type[BaseGenerator],
    in_vals: list[EvalValue],
    kwargs: dict[str, EvalValue],
) -> None:
    """Execute a meshgrid operation and extract the correct output index.

    Args:
        node (LogicalNode): The IR node representing the meshgrid operation.
        env (Environment): The local variable environment.
        backend (type[BaseGenerator]): The active compute backend for execution.
        in_vals (list[EvalValue]): List of input 1D coordinate arrays.
        kwargs (dict[str, EvalValue]): Keyword arguments including 'output_index' to select the desired returned grid.
    """
    raw_idx = kwargs.pop("output_index", 0)
    idx = int(raw_idx) if isinstance(raw_idx, (int, float, str)) else 0
    result = backend.execute_op("Meshgrid", *in_vals, **kwargs)
    env.set(node.id, getattr(result, "__getitem__")(idx))


def _dispatch_op(node: LogicalNode, env: Environment, backend: type[BaseGenerator], target_op: str, in_vals: list[EvalValue], kwargs: dict[str, EvalValue]) -> None:
    """Dispatch the execution of an operation to the appropriate handler.

    Args:
        node (LogicalNode): The IR node.
        env (Environment): The current local variable environment.
        backend (type[BaseGenerator]): The active compute backend.
        target_op (str): The name of the target operation.
        in_vals (list[EvalValue]): List of input values.
        kwargs (dict[str, EvalValue]): Keyword arguments for the operation.
    """
    if target_op == "Slice":
        _handle_slice(node, env, backend, in_vals, kwargs)
    elif target_op == "GetItem":
        _handle_getitem(node, env, backend, in_vals, kwargs)
    elif target_op in ("IndexPut", "SetItem"):
        _handle_index_put(node, env, backend, in_vals, kwargs)
    elif target_op == "Checkpoint":
        _handle_checkpoint(node, env, in_vals)
    elif target_op == "Meshgrid":
        _handle_meshgrid(node, env, backend, in_vals, kwargs)
    else:
        result = backend.execute_op(target_op, *in_vals, **kwargs)
        env.set(node.id, result)


def _evaluate_if_node(node: LogicalNode, env: Environment, backend: type[BaseGenerator]) -> None:
    """Execute evaluate if node for compiler processing.

    Args:
        node (LogicalNode): The node parameter.
        env (Environment): The env parameter.
        backend (type[BaseGenerator]): The backend parameter.
    """
    cond_val = env.get(node.inputs[0])
    is_true = bool(cond_val) if isinstance(cond_val, (bool, int, float)) else bool(getattr(cond_val, "item", lambda cr=cond_val: cr)())
    branch = (
        getattr(node, "subgraphs", {}).get("then_branch" if is_true else "else_branch")
        or node.attributes.get("then_branch" if is_true else "else_branch")
        or getattr(node, "subgraphs", {}).get("true_branch" if is_true else "false_branch")
        or node.attributes.get("true_branch" if is_true else "false_branch")
    )
    if branch:
        nodes = getattr(branch, "nodes") if isinstance(getattr(branch, "nodes"), list) else list(getattr(branch, "nodes").values())
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


def _evaluate_loop_node(node: LogicalNode, env: Environment, backend: type[BaseGenerator]) -> None:
    """Execute evaluate loop node for compiler processing.

    Args:
        node (LogicalNode): The node parameter.
        env (Environment): The env parameter.
        backend (type[BaseGenerator]): The backend parameter.
    """
    cond_graph = getattr(node, "subgraphs", {}).get("cond") or node.attributes.get("cond")
    body_graph = getattr(node, "subgraphs", {}).get("body") or node.attributes.get("body")
    curr_val = env.get(node.inputs[0])
    while True:
        cond_input_id = getattr(cond_graph, "nodes")[0].id if isinstance(getattr(cond_graph, "nodes"), list) else list(getattr(cond_graph, "nodes").keys())[0]
        env.set(cond_input_id, curr_val)
        nodes = getattr(cond_graph, "nodes") if isinstance(getattr(cond_graph, "nodes"), list) else getattr(cond_graph, "nodes").values()
        for sub_node in nodes:
            if sub_node.id != cond_input_id:
                _evaluate_node(sub_node, env, backend)
        cond_res = env.get(getattr(cond_graph, "outputs")[0])
        is_true = bool(cond_res) if isinstance(cond_res, (bool, int, float)) else bool(getattr(cond_res, "item", lambda cr=cond_res: cr)())
        if not is_true:
            break
        body_input_id = getattr(body_graph, "nodes")[0].id if isinstance(getattr(body_graph, "nodes"), list) else list(getattr(body_graph, "nodes").keys())[0]
        env.set(body_input_id, curr_val)
        nodes = getattr(body_graph, "nodes") if isinstance(getattr(body_graph, "nodes"), list) else getattr(body_graph, "nodes").values()
        for sub_node in nodes:
            if sub_node.id != body_input_id:
                _evaluate_node(sub_node, env, backend)
        curr_val = env.get(getattr(body_graph, "outputs")[0])
    env.set(node.id, curr_val)


def _evaluate_identity_node(node: LogicalNode, env: Environment, backend: type[BaseGenerator]) -> None:
    """Evaluate an Identity logical node and update the environment.

    Args:
        node (LogicalNode): The Identity node to evaluate.
        env (Environment): The local variable environment.
        backend (type[BaseGenerator]): The active compute backend.
    """
    in_vals = [env.get(inp) for inp in node.inputs if inp]
    val: EvalValue = in_vals[0] if in_vals else None
    if hasattr(backend, "execute_op"):
        try:
            val = backend.execute_op("Identity", *in_vals)
        except Exception:
            val = in_vals[0] if in_vals else None
    env.set(node.id, val)


def _evaluate_vmap_node(node: LogicalNode, env: Environment, backend: type[BaseGenerator]) -> None:
    """Evaluate a Vmap logical node by vectorizing and executing its body graph.

    Args:
        node (LogicalNode): The Vmap node to evaluate.
        env (Environment): The local variable environment.
        backend (type[BaseGenerator]): The active compute backend.
    """
    raw_bg: EvalValue = node.attributes.get("body")
    in_axes = node.attributes.get("in_axes", 0)
    out_axes = node.attributes.get("out_axes", 0)

    in_vals = [env.get(inp) for inp in node.inputs if inp]
    first_val = in_vals[0] if in_vals else None
    axis = in_axes if isinstance(in_axes, int) else (int(in_axes[0]) if isinstance(in_axes, (list, tuple)) and in_axes and isinstance(in_axes[0], int) else 0)
    batch_size = getattr(first_val, "shape")[axis] if hasattr(first_val, "shape") and axis < len(getattr(first_val, "shape")) else 1

    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    body_graph: IRGraph
    if isinstance(raw_bg, IRGraph):
        body_graph = raw_bg
    else:
        b_graph = IRGraph(name=getattr(raw_bg, "id", "body"))
        b_graph.inputs = list(getattr(raw_bg, "inputs", []))
        b_graph.outputs = list(getattr(raw_bg, "outputs", []))
        b_nodes = getattr(raw_bg, "nodes") if isinstance(getattr(raw_bg, "nodes"), list) else list(getattr(raw_bg, "nodes", {}).values())
        for n in b_nodes:
            b_graph.nodes[n.id] = IRNode(**n.__dict__) if not isinstance(n, IRNode) else n
        body_graph = b_graph

    from ml_switcheroo_compiler.transforms.passes.vectorization import vectorization_pass, vectorize_graph

    norm_in_axes: int | tuple[int | None, ...] = in_axes if isinstance(in_axes, int) else tuple(int(x) if isinstance(x, int) else None for x in in_axes) if isinstance(in_axes, (list, tuple)) else 0
    norm_out_axes: int | tuple[int, ...] = out_axes if isinstance(out_axes, int) else tuple(int(x) for x in out_axes if isinstance(x, int)) if isinstance(out_axes, (list, tuple)) else 0
    v_body = vectorize_graph(body_graph, in_axes=norm_in_axes, batch_size=batch_size, out_axes=norm_out_axes)
    v_body = vectorization_pass(v_body)

    body_input_map: dict[str, EvalValue] = {b_in: in_val for b_in, in_val in zip(v_body.inputs, in_vals)}
    body_env_map = evaluate_graph(v_body, body_input_map, backend=backend)
    env.set(node.id, body_env_map[v_body.outputs[0]])


def _evaluate_recompute_node(node: LogicalNode, env: Environment, backend: type[BaseGenerator]) -> None:
    """Evaluate a Recompute logical node by dispatching its original operation.

    Args:
        node (LogicalNode): The Recompute node to evaluate.
        env (Environment): The local variable environment.
        backend (type[BaseGenerator]): The active compute backend.
    """
    in_vals = [env.get(inp) for inp in node.inputs]
    orig_op = str(node.attributes.get("original_op", node.op_type))
    target_op = _get_op_alias(orig_op)
    raw_orig_attrs = node.attributes.get("original_attrs", {})
    orig_attrs: dict[str, AttributeValue] = dict(raw_orig_attrs) if isinstance(raw_orig_attrs, dict) else {}
    resolved_node = LogicalNode(id=node.id, op_type=target_op, inputs=node.inputs, attributes=orig_attrs)
    kwargs = _prepare_node_kwargs(resolved_node, target_op)
    _dispatch_op(resolved_node, env, backend, target_op, in_vals, kwargs)


def _evaluate_node(node: LogicalNode, env: Environment, backend: type[BaseGenerator]) -> None:
    """Execute a single logical node and update the local environment with its result.

    Args:
        node (LogicalNode): The graph node to evaluate.
        env (Environment): The local variable environment acting as memory for the execution.
        backend (type[BaseGenerator]): The underlying backend engine to dispatch computation to.
    """
    op_type = node.op_type
    if op_type == "Input":
        env.get(node.id)
    elif op_type == "Output":
        in_vals = [env.get(inp) for inp in node.inputs if inp]
        env.set(node.id, in_vals[0] if len(in_vals) == 1 else tuple(in_vals))
    elif op_type == "Identity":
        _evaluate_identity_node(node, env, backend)
    elif op_type == "Vmap":
        _evaluate_vmap_node(node, env, backend)
    elif op_type in ("If", "Cond"):
        _evaluate_if_node(node, env, backend)
    elif op_type in ("WhileLoop", "Loop"):
        _evaluate_loop_node(node, env, backend)
    elif op_type == "Constant":
        env.set(node.id, backend.array(node.attributes["value"]))
    elif op_type == "Recompute":
        _evaluate_recompute_node(node, env, backend)
    else:
        in_vals = [env.get(inp) for inp in node.inputs]
        target_op = _get_op_alias(op_type)
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


def _prepare_node_kwargs(node: LogicalNode, target_op: str) -> dict[str, EvalValue]:
    """Prepare the keyword arguments required for executing a node.

    Args:
        node (LogicalNode): The target node.
        target_op (str): The resolved operation name.

    Returns:
        dict[str, EvalValue]: A mapping of argument names to values.
    """
    kwargs: dict[str, EvalValue] = {k: v for k, v in node.attributes.items() if k not in ("var_name", "is_parameter", "param_name")}
    if "strides" in kwargs and target_op not in {
        "Conv",
        "Conv1D",
        "Conv2D",
        "Conv3D",
        "ConvTranspose",
        "ConvTranspose2D",
        "MaxPool",
        "MaxPool1D",
        "MaxPool2D",
        "MaxPool3D",
        "AvgPool",
        "AvgPool1D",
        "AvgPool2D",
        "AvgPool3D",
        "Pool1D",
        "Pool2D",
        "Pool3D",
        "StridedSlice",
    }:
        kwargs.pop("strides", None)
    if getattr(node, "shape_metadata", None):
        meta_seq = node.shape_metadata or ()
        meta_tuple: EvalValue = tuple(int(d) if isinstance(d, int) else str(d) for d in meta_seq)
        if target_op in ("Expand", "BroadcastTo", "ConstantOfShape", "Zeros", "Ones", "Full") and "shape" not in kwargs:
            kwargs["shape"] = meta_tuple
        if target_op == "Reshape" and "newshape" not in kwargs:
            kwargs["newshape"] = meta_tuple
    return kwargs


__all__ = ["Environment", "EvalValue", "IndexableArray", "TensorLike", "evaluate_graph"]
