"""IR evaluator using the OpRegistry."""

from __future__ import annotations

import ast
import builtins
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional, Protocol, TypeVar, Union, cast, runtime_checkable  # noqa: F401

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort
from ml_switcheroo_compiler.interpreter.environment import Environment


@runtime_checkable
class TensorLike(Protocol):
    """Protocol defining the structural type contract for tensor/array instances in the interpreter."""

    if TYPE_CHECKING:
        dtype: Union[str, type, None]
        nbytes: int
        size: int

        def copy(self) -> TensorLike:
            """Create a copy of the tensor.

            Returns:
                TensorLike: Copied tensor instance.
            """
            ...

        def __setitem__(self, key: Union[int, slice, tuple[EvalValue, ...], list[EvalValue], str, builtins.ellipsis, None], value: EvalValue) -> None:
            """Set item or slice in tensor.

            Args:
                key (Union[int, slice, tuple, list, str, builtins.ellipsis, None]): Indexing key or slice.
                value (EvalValue): Value to assign.
            """
            ...

    def __getitem__(self, key: Union[int, slice, tuple[EvalValue, ...], list[EvalValue], str, builtins.ellipsis, None]) -> TensorLike:
        """Get item or slice from tensor.

        Args:
            key (Union[int, slice, tuple, list, str, builtins.ellipsis, None]): Indexing key or slice.

        Returns:
            TensorLike: Indexed tensor slice or element.
        """
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """Get multidimensional shape dimensions.

        Returns:
            tuple[int, ...]: Tensor shape tuple.
        """
        ...

    def item(self) -> Union[int, float, bool]:
        """Get item as scalar value.

        Returns:
            Union[int, float, bool]: Extracted Python scalar.
        """
        ...

    def __len__(self) -> int:
        """Get length of leading dimension.

        Returns:
            int: Size of the leading dimension.
        """
        ...


IndexableArray = TensorLike
EvalValue = Union[
    int,
    float,
    complex,
    str,
    bool,
    list["EvalValue"],
    tuple["EvalValue", ...],
    dict[Union[str, int], "EvalValue"],
    None,
    slice,
    "builtins.ellipsis",
    TensorLike,
]


def evaluate_graph(
    graph: LogicalGraph,
    inputs: dict[str, EvalValue],
    backend: Optional[type[BaseGenerator]] = None,
) -> dict[str, EvalValue]:
    """Execute a logical computation graph using eager mode evaluation.

    Args:
        graph (LogicalGraph): The directed acyclic graph defining the operations to evaluate.
        inputs (dict[str, EvalValue]): A mapping of input node IDs to their corresponding concrete tensor values.
        backend (Optional[type[BaseGenerator]]): The compute backend generator class to dispatch ops to. Defaults to active backend.

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
        outputs[out_id] = cast(EvalValue, env.get(out_id))
    return outputs


def _parse_slice_call(node: ast.Call) -> EvalValue:
    """Parse a slice call AST node into a python slice or array object.

    Args:
        node (ast.Call): The Call AST node.

    Returns: EvalValue: The parsed slice object.
    """
    if not isinstance(node.func, ast.Name):
        return None
    if node.func.id == "slice":
        return builtins.slice(*[_parse_slice_node(arg) for arg in node.args])
    if node.func.id == "array":
        return _parse_slice_node(node.args[0])
    return None


def _parse_tuple(node: ast.Tuple) -> EvalValue:
    """Parse a tuple AST node.

    Args:
        node (ast.Tuple): The Tuple AST node.

    Returns: EvalValue: The parsed tuple object.
    """
    return tuple(_parse_slice_node(elt) for elt in node.elts)


def _parse_list(node: ast.List) -> EvalValue:
    """Parse a list AST node.

    Args:
        node (ast.List): The List AST node.

    Returns: EvalValue: The parsed list object.
    """
    return list(_parse_slice_node(elt) for elt in node.elts)


def _parse_constant(node: ast.Constant) -> EvalValue:
    """Parse a constant AST node.

    Args:
        node (ast.Constant): The Constant AST node.

    Returns: EvalValue: The parsed constant value.
    """
    val = node.value
    return val.decode("utf-8", errors="replace") if isinstance(val, bytes) else val


def _parse_name(node: ast.Name) -> EvalValue:
    """Parse a name AST node into standard singletons.

    Args:
        node (ast.Name): The Name AST node.

    Returns: EvalValue: The parsed name object (e.g., None, Ellipsis).
    """
    return {"None": None, "Ellipsis": Ellipsis, "False": False, "True": True}.get(node.id)


def _parse_unary(node: ast.UnaryOp) -> EvalValue:
    """Parse a unary operation AST node.

    Args:
        node (ast.UnaryOp): The UnaryOp AST node.

    Returns: EvalValue: The parsed unary operation value.
    """
    if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        return -cast(float, node.operand.value)
    return None


def _parse_slice_node(node: ast.AST) -> EvalValue:
    """Dispatch and parse an AST node for a slice representation into native Python types.

    Args:
        node (ast.AST): The parsed AST node representing a portion of a slice index.

    Returns: EvalValue: The interpreted Python slice, tuple, list, or constant value.

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
        return cast("Callable[..., EvalValue]", handler)(node)
    msg = f"Unsupported slice expression node: {type(node)}"
    raise ValueError(msg)


def _parse_slice_string(s: str) -> EvalValue:
    """Parse a string representation of a slice index.

    Args:
        s (str): The slice string to evaluate.

    Returns: EvalValue: The parsed python slice object.
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
        backend: The active compute backend used for array manipulation.
        in_vals (list): List containing the input tensor to be sliced.
        kwargs (dict[str, EvalValue]): Keyword arguments defining the slice parameters (e.g. 'slices', 'dim', 'start', 'end', 'step').
    """
    if "slices" in kwargs:
        parsed_key = _parse_slice_string(str(kwargs["slices"]))
        getattr(env, "set")(
            node.id,
            cast(IndexableArray, backend.asarray(in_vals[0]))[cast(Union[int, slice, tuple[EvalValue, ...], list[EvalValue], str, "builtins.ellipsis", None], parsed_key)],
        )
        return

    dim = int(getattr(kwargs.get("dim", 0), "__int__", lambda: 0)())
    start = kwargs.get("start", None)
    end = kwargs.get("end", None)
    step = kwargs.get("step", 1)

    sl = [builtins.slice(None)] * len(getattr(in_vals[0], "shape"))
    sl[dim] = builtins.slice(start, end, step)
    getattr(env, "set")(
        node.id,
        cast(IndexableArray, backend.asarray(in_vals[0]))[cast(Union[int, slice, tuple[EvalValue, ...], list[EvalValue], str, "builtins.ellipsis", None], tuple(sl))],
    )


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
        backend: The active compute backend for array manipulation.
        in_vals (list): List containing the input tensor.
        kwargs (dict[str, EvalValue]): Keyword arguments containing the 'key' for indexing.
    """
    key = kwargs.get("key")

    parsed_key = _parse_slice_string(str(key))
    getattr(env, "set")(
        node.id,
        cast(IndexableArray, backend.asarray(in_vals[0]))[cast(Union[int, slice, tuple[EvalValue, ...], list[EvalValue], str, "builtins.ellipsis", None], parsed_key)],
    )


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
    arr = cast(IndexableArray, backend.execute_op("Clone", in_vals[0]))
    val = in_vals[1]
    arr[cast(Union[int, slice, tuple[EvalValue, ...], list[EvalValue], str, "builtins.ellipsis", None], parsed_key)] = val
    getattr(env, "set")(node.id, arr)


def _handle_checkpoint(
    node: LogicalNode,
    env: Environment,
    in_vals: list[EvalValue],
) -> None:
    """Execute a gradient checkpointing subgraph dynamically.

    Args:
        node (LogicalNode): The IR node representing the checkpoint wrapper.
        env (Environment): The local variable environment.
        in_vals (list): List of input tensors to feed into the subgraph.
    """
    subgraph = getattr(node, "subgraphs", {}).get("body") or getattr(node, "subgraphs", {}).get("subgraph") or node.attributes.get("subgraph")
    sub_inputs = {}
    for inp_val, in_id in zip(in_vals, getattr(subgraph, "inputs")):
        sub_inputs[in_id] = inp_val

    from ml_switcheroo_ir import LogicalGraph

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
    getattr(env, "set")(node.id, result)


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
        backend: The active compute backend for execution.
        in_vals (list): List of input 1D coordinate arrays.
        kwargs (dict[str, EvalValue]): Keyword arguments including 'output_index' to select the desired returned grid.
    """
    idx = int(cast(int, kwargs.pop("output_index", 0)))
    result = backend.execute_op("Meshgrid", *in_vals, **kwargs)
    getattr(env, "set")(node.id, cast(IndexableArray, result)[idx])


def _dispatch_op(node: LogicalNode, env: Environment, backend: type[BaseGenerator], target_op: str, in_vals: list[EvalValue], kwargs: dict[str, EvalValue]) -> None:
    """Dispatch the execution of an operation to the appropriate handler.

    Args:
        node (LogicalNode): The IR node.
        env (Environment): The current local variable environment.
        backend: The active compute backend.
        target_op (str): The name of the target operation.
        in_vals (list): List of input values.
        kwargs (dict): Keyword arguments for the operation.
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
        getattr(env, "set")(node.id, result)


def _evaluate_if_node(node: LogicalNode, env: Environment, backend: type[BaseGenerator]) -> None:
    """Execute evaluate if node for compiler processing.

    Args:
        node: The node parameter.
        env: The env parameter.
        backend: The backend parameter.

    Returns:
        EvalValue: Result.
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
        node: The node parameter.
        env: The env parameter.
        backend: The backend parameter.

    Returns:
        EvalValue: Result.
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
    in_vals = [cast(EvalValue, env.get(inp)) for inp in node.inputs if inp]
    val: EvalValue = in_vals[0] if in_vals else None
    if hasattr(backend, "execute_op"):
        try:
            val = cast(EvalValue, backend.execute_op("Identity", *in_vals))
        except Exception:
            val = in_vals[0] if in_vals else None
    getattr(env, "set")(node.id, val)


def _evaluate_vmap_node(node: LogicalNode, env: Environment, backend: type[BaseGenerator]) -> None:
    """Evaluate a Vmap logical node by vectorizing and executing its body graph.

    Args:
        node (LogicalNode): The Vmap node to evaluate.
        env (Environment): The local variable environment.
        backend (type[BaseGenerator]): The active compute backend.
    """
    raw_bg = node.attributes.get("body")
    body_graph: LogicalGraph | None = cast("LogicalGraph | None", raw_bg)
    in_axes = node.attributes.get("in_axes", 0)
    out_axes = node.attributes.get("out_axes", 0)

    in_vals = [env.get(inp) for inp in node.inputs if inp]
    first_val = in_vals[0] if in_vals else None
    in_axes_seq = cast(tuple[int, ...], in_axes)
    axis = in_axes if isinstance(in_axes, int) else (int(in_axes_seq[0]) if in_axes_seq else 0)
    batch_size = getattr(first_val, "shape")[axis] if hasattr(first_val, "shape") and axis < len(getattr(first_val, "shape")) else 1

    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    if body_graph is not None and not isinstance(body_graph, IRGraph) and hasattr(body_graph, "nodes"):
        b_graph = IRGraph(name=getattr(body_graph, "id", "body"))
        b_graph.inputs = list(getattr(body_graph, "inputs", []))
        b_graph.outputs = list(getattr(body_graph, "outputs", []))
        b_nodes = getattr(body_graph, "nodes") if isinstance(getattr(body_graph, "nodes"), list) else list(getattr(body_graph, "nodes").values())
        for n in b_nodes:
            b_graph.nodes[n.id] = IRNode(**n.__dict__) if not isinstance(n, IRNode) else n
        body_graph = b_graph

    from ml_switcheroo_compiler.transforms.passes.vectorization import vectorization_pass, vectorize_graph

    v_body = cast("Callable[..., IRGraph]", vectorize_graph)(cast("LogicalGraph", body_graph), in_axes=in_axes, batch_size=batch_size, out_axes=out_axes)
    v_body = vectorization_pass(v_body)

    body_input_map = {b_in: in_val for b_in, in_val in zip(v_body.inputs, in_vals)}
    body_env_map = evaluate_graph(v_body, cast("dict[str, EvalValue]", body_input_map), backend=backend)
    getattr(env, "set")(node.id, body_env_map[v_body.outputs[0]])


def _evaluate_recompute_node(node: LogicalNode, env: Environment, backend: type[BaseGenerator]) -> None:
    """Evaluate a Recompute logical node by dispatching its original operation.

    Args:
        node (LogicalNode): The Recompute node to evaluate.
        env (Environment): The local variable environment.
        backend (type[BaseGenerator]): The active compute backend.
    """
    from collections.abc import Mapping

    from ml_switcheroo_ir import AttributeValue

    in_vals = [env.get(inp) for inp in node.inputs]
    orig_op = str(node.attributes.get("original_op", node.op_type))
    target_op = _get_op_alias(orig_op)
    orig_attrs: dict[str, EvalValue] = cast(dict[str, EvalValue], node.attributes.get("original_attrs", {})).copy()
    resolved_node = LogicalNode(id=node.id, op_type=target_op, inputs=node.inputs, attributes=cast("Mapping[str, AttributeValue]", orig_attrs))
    kwargs = _prepare_node_kwargs(resolved_node, target_op)
    _dispatch_op(resolved_node, env, backend, target_op, cast("list[EvalValue]", in_vals), kwargs)


def _evaluate_node(node: LogicalNode, env: Environment, backend: type[BaseGenerator]) -> None:
    """Execute a single logical node and update the local environment with its result.

    Args:
        node (LogicalNode): The graph node to evaluate.
        env (Environment): The local variable environment acting as memory for the execution.
        backend: The underlying backend engine to dispatch computation to.
    """
    op_type = node.op_type
    if op_type == "Input":
        env.get(node.id)
    elif op_type == "Output":
        in_vals = [env.get(inp) for inp in node.inputs if inp]
        env.set(node.id, in_vals[0] if len(in_vals) == 1 else tuple(getattr(in_vals, "__iter__")()))
    elif op_type == "Identity":
        _evaluate_identity_node(node, env, backend)
    elif op_type == "Vmap":
        _evaluate_vmap_node(node, env, backend)
    elif op_type in ("If", "Cond"):
        _evaluate_if_node(node, env, backend)
    elif op_type in ("WhileLoop", "Loop"):
        _evaluate_loop_node(node, env, backend)
    elif op_type == "Constant":
        getattr(env, "set")(node.id, backend.array(node.attributes["value"]))
    elif op_type == "Recompute":
        _evaluate_recompute_node(node, env, backend)
    else:
        in_vals = [env.get(inp) for inp in node.inputs]
        target_op = _get_op_alias(op_type)
        kwargs = _prepare_node_kwargs(node, target_op)
        _dispatch_op(node, env, backend, target_op, cast("list[EvalValue]", in_vals), kwargs)


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
        dict: A mapping of argument names to values.
    """
    kwargs: dict[str, EvalValue] = cast(
        dict[str, EvalValue],
        {k: v for k, v in node.attributes.items() if k not in ("var_name", "is_parameter", "param_name")},
    )
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
        from collections.abc import Sequence

        meta_tuple = cast("EvalValue", tuple(cast("Sequence[int | str]", node.shape_metadata)))
        if target_op in ("Expand", "BroadcastTo", "ConstantOfShape", "Zeros", "Ones", "Full") and "shape" not in kwargs:
            kwargs["shape"] = meta_tuple
        if target_op == "Reshape" and "newshape" not in kwargs:
            kwargs["newshape"] = meta_tuple
    return kwargs


__all__ = ["Environment", "EvalValue", "IndexableArray", "TensorLike", "evaluate_graph"]
