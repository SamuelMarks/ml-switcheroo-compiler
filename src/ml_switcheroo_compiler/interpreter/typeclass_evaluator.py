"""Pure Reader/dictionary-passing IR graph evaluator parameterized over CoreEagerTypeclass[TensorT]."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from ml_switcheroo_ir import (
    AttributeValue,
    Err,
    LogicalGraph,
    LogicalNode,
    Ok,
    Result,
    is_non_bool_int,
    prove_acyclic_sorted,
)

from ml_switcheroo_compiler.backends.typeclasses import CoreEagerTypeclass, TensorT


def _parse_int_tuple_attr(val: AttributeValue) -> tuple[int, ...] | None:
    """Extract a concrete integer tuple from an ``AttributeValue`` when valid.

    Args:
        val (AttributeValue): Candidate IR attribute value.

    Returns:
        tuple[int, ...] | None: Concrete integer tuple, or ``None`` if not an integer sequence.
    """
    if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
        if all(is_non_bool_int(item) for item in val):
            return tuple(int(item) for item in val if is_non_bool_int(item))
    return None


def _eval_creation_node(
    node: LogicalNode,
    inputs: Mapping[str, TensorT],
    eager_dict: CoreEagerTypeclass[TensorT],
) -> Result[str, TensorT]:
    """Evaluate a nullary creation or input node (``Input``, ``Constant``, ``Zeros``, ``Ones``).

    Args:
        node (LogicalNode): Target creation IR node.
        inputs (Mapping[str, TensorT]): Bound graph input tensor environment.
        eager_dict (CoreEagerTypeclass[TensorT]): Parameterized eager typeclass witness.

    Returns:
        Result[str, TensorT]: ``Ok(TensorT)`` or ``Err(str)`` diagnostic.
    """
    if node.op_type == "Input":
        return Ok(inputs[node.id]) if node.id in inputs else Err(f"Missing required input tensor for node {node.id!r}.")
    shape_tup = _parse_int_tuple_attr(node.attributes.get("shape")) or (1,)
    if node.op_type in ("Zeros", "Ones"):
        return Ok(eager_dict.zeros(shape_tup, None) if node.op_type == "Zeros" else eager_dict.ones(shape_tup, None))
    raw_val = node.attributes.get("value", 0.0)
    if isinstance(raw_val, (int, float)) and not isinstance(raw_val, bool):
        return Ok(eager_dict.array(float(raw_val), None))
    if isinstance(raw_val, Sequence) and not isinstance(raw_val, (str, bytes)):
        flat_vals = [float(v) for v in raw_val if isinstance(v, (int, float)) and not isinstance(v, bool)]
        return Ok(eager_dict.array(flat_vals, None))
    return Err(f"Unsupported Constant value attribute on node {node.id!r}: {raw_val!r}")


def _eval_unary_node(
    node: LogicalNode,
    arg: TensorT,
    eager_dict: CoreEagerTypeclass[TensorT],
) -> Result[str, TensorT]:
    """Evaluate a unary, shape, or reduction ``LogicalNode`` against ``eager_dict``.

    Args:
        node (LogicalNode): Unary or reduction IR node.
        arg (TensorT): Evaluated input tensor operand.
        eager_dict (CoreEagerTypeclass[TensorT]): Parameterized eager typeclass witness.

    Returns:
        Result[str, TensorT]: ``Ok(TensorT)`` or ``Err(str)`` diagnostic.
    """
    unary_table: dict[str, Callable[[TensorT], TensorT]] = {
        "Identity": lambda x: x,
        "Abs": eager_dict.abs,
        "Exp": eager_dict.exp,
        "Log": eager_dict.log,
        "Sqrt": eager_dict.sqrt,
        "Sin": eager_dict.sin,
        "Cos": eager_dict.cos,
        "Tanh": eager_dict.tanh,
        "Neg": lambda x: eager_dict.multiply(x, eager_dict.array(-1.0, None)),
    }
    direct_fn = unary_table.get(node.op_type)
    if direct_fn is not None:
        return Ok(direct_fn(arg))
    if node.op_type == "Reshape":
        target_shape = _parse_int_tuple_attr(node.attributes.get("shape"))
        if target_shape is None:
            return Err(f"Reshape node {node.id!r} requires concrete integer 'shape' attribute.")
        return Ok(eager_dict.reshape(arg, target_shape))
    if node.op_type == "Transpose":
        perm = _parse_int_tuple_attr(node.attributes.get("perm"))
        return Ok(eager_dict.transpose(arg, perm))
    if node.op_type in ("Sum", "ReduceSum", "Mean", "ReduceMean"):
        keepdims = bool(node.attributes.get("keepdims", False))
        raw_axis = node.attributes.get("axis")
        axis_arg: Sequence[int] | int | None = (
            int(raw_axis) if is_non_bool_int(raw_axis) else _parse_int_tuple_attr(raw_axis)
        )
        red_fn = eager_dict.sum if node.op_type in ("Sum", "ReduceSum") else eager_dict.mean
        return Ok(red_fn(arg, axis_arg, keepdims))
    return Err(f"Unsupported unary operation {node.op_type!r} on node {node.id!r}.")


def _eval_binary_node(
    node: LogicalNode,
    lhs: TensorT,
    rhs: TensorT,
    eager_dict: CoreEagerTypeclass[TensorT],
) -> Result[str, TensorT]:
    """Evaluate a binary arithmetic or linear-algebra ``LogicalNode`` against ``eager_dict``.

    Args:
        node (LogicalNode): Binary IR node.
        lhs (TensorT): Evaluated left operand.
        rhs (TensorT): Evaluated right operand.
        eager_dict (CoreEagerTypeclass[TensorT]): Parameterized eager typeclass witness.

    Returns:
        Result[str, TensorT]: ``Ok(TensorT)`` or ``Err(str)`` diagnostic.
    """
    binary_table: dict[str, Callable[[TensorT, TensorT], TensorT]] = {
        "Add": eager_dict.add,
        "Sub": eager_dict.subtract,
        "Subtract": eager_dict.subtract,
        "Mul": eager_dict.multiply,
        "Multiply": eager_dict.multiply,
        "Div": eager_dict.divide,
        "Divide": eager_dict.divide,
        "MatMul": eager_dict.matmul,
    }
    fn = binary_table.get(node.op_type)
    if fn is None:
        return Err(f"Unsupported binary operation {node.op_type!r} on node {node.id!r}.")
    return Ok(fn(lhs, rhs))


def _eval_single_node(
    node: LogicalNode,
    env: dict[str, TensorT],
    inputs: Mapping[str, TensorT],
    eager_dict: CoreEagerTypeclass[TensorT],
) -> Result[str, TensorT]:
    """Dispatch a single topologically ordered ``LogicalNode`` against ``env`` and ``eager_dict``.

    Args:
        node (LogicalNode): Current node in topological execution order.
        env (dict[str, TensorT]): Materialized value environment for upstream node IDs.
        inputs (Mapping[str, TensorT]): External input tensor bindings.
        eager_dict (CoreEagerTypeclass[TensorT]): Parameterized eager typeclass witness.

    Returns:
        Result[str, TensorT]: ``Ok(TensorT)`` result or ``Err(str)`` diagnostic.
    """
    if node.op_type in ("Input", "Constant", "Zeros", "Ones"):
        return _eval_creation_node(node, inputs, eager_dict)
    resolved_args: list[TensorT] = []
    for inp_name in node.inputs:
        if inp_name in env:
            resolved_args.append(env[inp_name])
        elif inp_name in inputs:
            resolved_args.append(inputs[inp_name])
        else:
            return Err(f"Node {node.id!r} references unbound input {inp_name!r}.")
    if len(resolved_args) == 1:
        return _eval_unary_node(node, resolved_args[0], eager_dict)
    if len(resolved_args) == 2:
        return _eval_binary_node(node, resolved_args[0], resolved_args[1], eager_dict)
    return Err(f"Unsupported operand arity {len(resolved_args)} for op {node.op_type!r} on node {node.id!r}.")


def evaluate_graph_with_typeclass(
    graph: LogicalGraph,
    inputs: Mapping[str, TensorT],
    eager_dict: CoreEagerTypeclass[TensorT],
) -> Result[str, dict[str, TensorT]]:
    """Evaluate ``graph`` in Reader/dictionary-passing style using ``eager_dict`` without global state.

    Args:
        graph (LogicalGraph): Computation graph to prove acyclic and evaluate.
        inputs (Mapping[str, TensorT]): Mapping from input names to concrete ``TensorT`` values.
        eager_dict (CoreEagerTypeclass[TensorT]): Explicit ``CoreEagerTypeclass`` witness dictionary.

    Returns:
        Result[str, dict[str, TensorT]]: ``Ok`` output tensor dictionary or ``Err(str)`` diagnostic.
    """
    proven = prove_acyclic_sorted(graph)
    if isinstance(proven, Err):
        return Err(str(proven.error))
    env: dict[str, TensorT] = {}
    for node in proven.value.ordered_nodes:
        step_res = _eval_single_node(node, env, inputs, eager_dict)
        if isinstance(step_res, Err):
            return step_res
        env[node.id] = step_res.value
    merged_scope: dict[str, TensorT] = {**inputs, **env}
    target_outputs = list(graph.outputs) if graph.outputs else [n.id for n in proven.value.ordered_nodes[-1:]]
    out_map: dict[str, TensorT] = {}
    for out_name in target_outputs:
        if out_name not in merged_scope:
            return Err(f"Declared graph output {out_name!r} was not produced during evaluation.")
        out_map[out_name] = merged_scope[out_name]
    return Ok(out_map)


__all__ = ["evaluate_graph_with_typeclass"]
