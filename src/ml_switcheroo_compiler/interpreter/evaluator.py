"""IR graph evaluator dispatching operations against the active backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ml_switcheroo_ir import AttributeValue, LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort
from ml_switcheroo_compiler.interpreter.environment import Environment, InterpreterValue, as_eval_value
from ml_switcheroo_compiler.interpreter.op_handlers import (
    InterpreterBackend,
    SubgraphCandidate,
    _eval_truthy,
    _extract_subgraph_nodes,
    _get_op_alias,
    _handle_checkpoint,
    _handle_getitem,
    _handle_index_put,
    _handle_meshgrid,
    _handle_slice,
    _has_eval_shape,
    _has_execute_op,
    _has_named_subgraph_id,
    _has_subgraph_inputs,
    _has_subgraph_outputs,
    _prepare_node_kwargs,
)
from ml_switcheroo_compiler.interpreter.slice_parser import (
    _eval_to_int,
    _parse_constant,
    _parse_list,
    _parse_name,
    _parse_slice_call,
    _parse_slice_node,
    _parse_slice_string,
    _parse_tuple,
    _parse_unary,
)

TensorLike = EagerTensorProtocol
IndexableArray = TensorLike


def evaluate_graph(
    graph: LogicalGraph,
    inputs: Mapping[str, InterpreterValue | OpValue],
    backend: InterpreterBackend | None = None,
) -> dict[str, InterpreterValue]:
    """Execute a logical computation graph using eager mode evaluation.

    Args:
        graph (LogicalGraph): The directed acyclic graph defining the operations to evaluate.
        inputs (Mapping[str, InterpreterValue | OpValue]): Mapping of input node IDs to concrete tensor or scalar
            values.
        backend (InterpreterBackend | None): The compute backend generator or protocol to dispatch ops to. Defaults to
            the active backend.

    Returns:
        dict[str, InterpreterValue]: Dictionary mapping output node IDs to their computed tensor or scalar results.

    Raises:
        RuntimeError: If an output node was never evaluated during the graph execution.
    """
    initial_env: dict[str, InterpreterValue] = {str(k): as_eval_value(v) for k, v in (graph.initializers or {}).items()}
    for key, raw_val in inputs.items():
        initial_env[key] = as_eval_value(raw_val)
    env = Environment(initial_env)
    sorted_nodes = topological_sort(graph)
    active_backend: InterpreterBackend = backend if backend is not None else get_active_backend()

    for node in sorted_nodes:
        _evaluate_node(node, env, active_backend)

    outputs: dict[str, InterpreterValue] = {}
    for out_id in graph.outputs:
        if out_id not in env:
            msg = f"Output node '{out_id}' was never evaluated."
            raise RuntimeError(msg)
        outputs[out_id] = env.get(out_id)
    return outputs


def _dispatch_op(
    node: LogicalNode,
    env: Environment,
    backend: InterpreterBackend,
    target_op: str,
    in_vals: Sequence[InterpreterValue],
    kwargs: dict[str, AttributeValue],
) -> None:
    """Dispatch the execution of an operation to the appropriate handler.

    Args:
        node (LogicalNode): The IR node.
        env (Environment): The current local variable environment.
        backend (InterpreterBackend): The active compute backend.
        target_op (str): The name of the target operation.
        in_vals (Sequence[InterpreterValue]): Sequence of evaluated input values.
        kwargs (dict[str, AttributeValue]): Keyword arguments for the operation.
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
        from ml_switcheroo_compiler.backends.protocols import adapt_backend_module

        result = adapt_backend_module(backend).require_op("execute_op")(target_op, *in_vals, **kwargs)
        env.set(node.id, as_eval_value(result))


def _evaluate_if_node(node: LogicalNode, env: Environment, backend: InterpreterBackend) -> None:
    """Evaluate a conditional (`If` / `Cond`) node by dispatching the selected branch subgraph.

    Args:
        node (LogicalNode): Conditional IR node.
        env (Environment): Active variable environment.
        backend (InterpreterBackend): Active compute backend.
    """
    cond_val = env.get(node.inputs[0])
    is_true = _eval_truthy(cond_val)
    branch: SubgraphCandidate = (
        node.subgraphs.get("then_branch" if is_true else "else_branch")
        or node.attributes.get("then_branch" if is_true else "else_branch")
        or node.subgraphs.get("true_branch" if is_true else "false_branch")
        or node.attributes.get("true_branch" if is_true else "false_branch")
    )
    if branch is not None:
        nodes = _extract_subgraph_nodes(branch)
        for sub_node in nodes:
            _evaluate_node(sub_node, env, backend)
        if _has_subgraph_outputs(branch) and branch.outputs:
            if isinstance(branch.outputs, list):
                env.set(node.id, env.get(branch.outputs[0]))
            else:
                env.set(node.id, env.get(branch.outputs))
        else:
            last_node = nodes[-1]
            env.set(node.id, env.get(last_node.id))


def _evaluate_loop_node(node: LogicalNode, env: Environment, backend: InterpreterBackend) -> None:
    """Evaluate a `WhileLoop` / `Loop` node until its condition subgraph evaluates to False.

    Args:
        node (LogicalNode): Loop IR node.
        env (Environment): Active variable environment.
        backend (InterpreterBackend): Active compute backend.
    """
    cond_graph: SubgraphCandidate = node.subgraphs.get("cond") or node.attributes.get("cond")
    body_graph: SubgraphCandidate = node.subgraphs.get("body") or node.attributes.get("body")
    curr_val = env.get(node.inputs[0])
    while True:
        cond_nodes = _extract_subgraph_nodes(cond_graph)
        cond_input_id = cond_nodes[0].id
        env.set(cond_input_id, curr_val)
        for sub_node in cond_nodes:
            if sub_node.id != cond_input_id:
                _evaluate_node(sub_node, env, backend)
        cond_outs = cond_graph.outputs if _has_subgraph_outputs(cond_graph) else []
        cond_out_id = cond_outs[0] if isinstance(cond_outs, list) else str(cond_outs)
        cond_res = env.get(cond_out_id)
        if not _eval_truthy(cond_res):
            break
        body_nodes = _extract_subgraph_nodes(body_graph)
        body_input_id = body_nodes[0].id
        env.set(body_input_id, curr_val)
        for sub_node in body_nodes:
            if sub_node.id != body_input_id:
                _evaluate_node(sub_node, env, backend)
        body_outs = body_graph.outputs if _has_subgraph_outputs(body_graph) else []
        body_out_id = body_outs[0] if isinstance(body_outs, list) else str(body_outs)
        curr_val = env.get(body_out_id)
    env.set(node.id, curr_val)


def _evaluate_identity_node(node: LogicalNode, env: Environment, backend: InterpreterBackend) -> None:
    """Evaluate an `Identity` logical node and update the environment.

    Args:
        node (LogicalNode): The Identity node to evaluate.
        env (Environment): The local variable environment.
        backend (InterpreterBackend): The active compute backend.
    """
    in_vals = [env.get(inp) for inp in node.inputs if inp]
    val: InterpreterValue = in_vals[0] if in_vals else None
    raw_backend: OpValue | InterpreterBackend = backend
    if _has_execute_op(raw_backend):
        try:
            val = as_eval_value(raw_backend.execute_op("Identity", *in_vals))
        except (AttributeError, KeyError, NotImplementedError, RuntimeError, TypeError, ValueError):
            val = in_vals[0] if in_vals else None
    env.set(node.id, val)


def _evaluate_vmap_node(node: LogicalNode, env: Environment, backend: InterpreterBackend) -> None:
    """Evaluate a `Vmap` logical node by vectorizing and executing its body graph.

    Args:
        node (LogicalNode): The Vmap node to evaluate.
        env (Environment): The local variable environment.
        backend (InterpreterBackend): The active compute backend.
    """
    raw_bg: SubgraphCandidate = node.attributes.get("body")
    in_axes = node.attributes.get("in_axes", 0)
    out_axes = node.attributes.get("out_axes", 0)

    in_vals = [env.get(inp) for inp in node.inputs if inp]
    first_val = in_vals[0] if in_vals else None
    axis = (
        in_axes
        if isinstance(in_axes, int)
        else (int(in_axes[0]) if isinstance(in_axes, (list, tuple)) and in_axes and isinstance(in_axes[0], int) else 0)
    )
    batch_size = (
        _eval_to_int(first_val.shape[axis], 1) if _has_eval_shape(first_val) and axis < len(first_val.shape) else 1
    )

    from ml_switcheroo_ir import IRGraph

    body_graph: IRGraph
    if isinstance(raw_bg, IRGraph):
        body_graph = raw_bg
    else:
        bg_name = raw_bg.id if _has_named_subgraph_id(raw_bg) else "body"
        b_graph = IRGraph(name=bg_name)
        b_graph.inputs = list(raw_bg.inputs) if _has_subgraph_inputs(raw_bg) else []
        raw_outs: list[str] | str = raw_bg.outputs if _has_subgraph_outputs(raw_bg) else []
        b_graph.outputs = list(raw_outs) if isinstance(raw_outs, list) else [str(raw_outs)]
        b_nodes = _extract_subgraph_nodes(raw_bg)
        for n in b_nodes:
            b_graph.nodes[n.id] = n
        body_graph = b_graph

    from ml_switcheroo_compiler.transforms.passes.vectorization import vectorization_pass, vectorize_graph

    norm_in_axes: int | tuple[int | None, ...] = (
        in_axes
        if isinstance(in_axes, int)
        else tuple(int(x) if isinstance(x, int) else None for x in in_axes)
        if isinstance(in_axes, (list, tuple))
        else 0
    )
    norm_out_axes: int | tuple[int, ...] = (
        out_axes
        if isinstance(out_axes, int)
        else tuple(int(x) for x in out_axes if isinstance(x, int))
        if isinstance(out_axes, (list, tuple))
        else 0
    )
    v_body = vectorize_graph(body_graph, in_axes=norm_in_axes, batch_size=batch_size, out_axes=norm_out_axes)
    v_body = vectorization_pass(v_body)

    body_input_map: dict[str, InterpreterValue] = dict(zip(v_body.inputs, in_vals))
    body_env_map = evaluate_graph(v_body, body_input_map, backend=backend)
    env.set(node.id, body_env_map[v_body.outputs[0]])


def _evaluate_recompute_node(node: LogicalNode, env: Environment, backend: InterpreterBackend) -> None:
    """Evaluate a `Recompute` logical node by dispatching its original operation.

    Args:
        node (LogicalNode): The Recompute node to evaluate.
        env (Environment): The local variable environment.
        backend (InterpreterBackend): The active compute backend.
    """
    in_vals = [env.get(inp) for inp in node.inputs]
    orig_op = str(node.attributes.get("original_op", node.op_type))
    target_op = _get_op_alias(orig_op)
    raw_orig_attrs = node.attributes.get("original_attrs", {})
    orig_attrs: dict[str, AttributeValue] = dict(raw_orig_attrs) if isinstance(raw_orig_attrs, dict) else {}
    resolved_node = LogicalNode(id=node.id, op_type=target_op, inputs=node.inputs, attributes=orig_attrs)
    kwargs = _prepare_node_kwargs(resolved_node, target_op)
    _dispatch_op(resolved_node, env, backend, target_op, in_vals, kwargs)


def _evaluate_node(node: LogicalNode, env: Environment, backend: InterpreterBackend) -> None:
    """Execute a single logical node and update the local environment with its result.

    Args:
        node (LogicalNode): The graph node to evaluate.
        env (Environment): The local variable environment acting as memory for the execution.
        backend (InterpreterBackend): The underlying backend engine to dispatch computation to.
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
        from ml_switcheroo_compiler.backends.protocols import adapt_backend_module

        const_val = node.attributes["value"]
        arr_val = adapt_backend_module(backend).require_op("array")(const_val)
        env.set(node.id, as_eval_value(arr_val))
    elif op_type == "Recompute":
        _evaluate_recompute_node(node, env, backend)
    else:
        in_vals = [env.get(inp) for inp in node.inputs]
        target_op = _get_op_alias(op_type)
        kwargs = _prepare_node_kwargs(node, target_op)
        _dispatch_op(node, env, backend, target_op, in_vals, kwargs)


__all__ = [
    "Environment",
    "InterpreterValue",
    "IndexableArray",
    "TensorLike",
    "_dispatch_op",
    "_evaluate_if_node",
    "_evaluate_identity_node",
    "_evaluate_loop_node",
    "_evaluate_node",
    "_evaluate_recompute_node",
    "_evaluate_vmap_node",
    "_get_op_alias",
    "_handle_checkpoint",
    "_handle_getitem",
    "_handle_index_put",
    "_handle_meshgrid",
    "_handle_slice",
    "_parse_constant",
    "_parse_list",
    "_parse_name",
    "_parse_slice_call",
    "_parse_slice_node",
    "_parse_slice_string",
    "_parse_tuple",
    "_parse_unary",
    "_prepare_node_kwargs",
    "evaluate_graph",
]
