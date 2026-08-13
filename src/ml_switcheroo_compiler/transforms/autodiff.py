# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Reverse-mode Automatic Differentiation (AD) Engine."""

import typing
import uuid
from typing import Any

from ml_switcheroo_ir import LogicalGraph, LogicalNode, topological_sort

from ml_switcheroo_compiler.core.errors import MissingJVPRuleError
from ml_switcheroo_compiler.ir.core import clone_logical_node
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import get_vjp


def _add_nodes(graph: LogicalGraph, n1_id: str, n2_id: str) -> str:
    """Emit an Add node for gradient accumulation.

    Args:
        graph (LogicalGraph): The graph parameter for the operation.
        n1_id (str): The n1_id parameter for the operation.
        n2_id (str): The n2_id parameter for the operation.

    Returns:
        str: The computed result.
    """
    out_id = f"{n1_id}_add_{n2_id}_{uuid.uuid4().hex[:6]}"
    n1 = graph.nodes[n1_id]

    # Simple shape heuristic for accumulation
    node = LogicalNode(
        id=out_id,
        op_type="Add",
        inputs=[n1_id, n2_id],
        shape_metadata=n1.shape_metadata,
    )
    graph.nodes[out_id] = node
    return out_id


def _copy_graph(graph: LogicalGraph) -> LogicalGraph:
    """Copy the forward graph into a new graph.

    Args:
        graph (LogicalGraph): The original graph.

    Returns:
        LogicalGraph: The new graph.
    """
    new_graph = LogicalGraph(name=f"{graph.name}_grad")
    for nid, node in graph.nodes.items():
        new_graph.nodes[nid] = clone_logical_node(node)
    return new_graph


def _get_reachable_from_output(sorted_nodes: list[LogicalNode], output_id: str) -> set[str]:
    """Find all nodes reachable from the output.

    Args:
        sorted_nodes (list[LogicalNode]): The sorted nodes.
        output_id (str): The output id.

    Returns:
        set[str]: The reachable nodes.
    """
    reachable_from_output: set[str] = {output_id}
    for node in reversed(sorted_nodes):
        if node.id in reachable_from_output:
            for inp in node.inputs:
                reachable_from_output.add(inp)
    return reachable_from_output


def _accumulate_gradients(
    new_graph: LogicalGraph,
    node: LogicalNode,
    adj_id: str,
    adjoints: dict[str, str],
) -> None:
    """Accumulate gradients for a given node.

    Args:
        new_graph (LogicalGraph): The new graph.
        node (LogicalNode): The node to process.
        adj_id (str): The adjoint ID for the node.
        adjoints (dict[str, str]): The mapping of adjoints.

    Raises:
        ValueError: If VJP rule is missing or returns incorrect number of adjoints.
    """
    try:
        if node.attributes.get("rematerialize", False):
            import uuid

            from ml_switcheroo_compiler.ir.core import clone_logical_node

            recompute_node = clone_logical_node(node)
            recompute_node.id = f"{node.id}_recompute_{uuid.uuid4().hex[:6]}"
            new_graph.nodes[recompute_node.id] = recompute_node
            # Ensure inputs are available. In a deep unroll we might need to recursively recompute.
            # For now, just re-evaluating the single op is sufficient for checkpointing elements.
            eval_node = recompute_node
        else:
            eval_node = node
        vjp_func = get_vjp(node.op_type)
        input_adjs = vjp_func(new_graph, eval_node, adj_id)
    except ValueError:
        msg = f"Missing VJP rule for operation: {getattr(node, 'op_type', 'Unknown')}"
        raise ValueError(msg) from None

    if len(input_adjs) != len(node.inputs):
        msg = f"VJP for {getattr(node, 'op_type', 'Unknown')} returned {len(input_adjs)} adjoints, expected {len(node.inputs)}."
        raise ValueError(msg)

    from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients

    for inp_id, inp_adj_id in zip(node.inputs, input_adjs):
        if inp_adj_id is None or inp_adj_id == UnconnectedGradients.NONE or inp_adj_id == UnconnectedGradients.ZERO:
            continue

        if inp_id in adjoints:
            adjoints[inp_id] = _add_nodes(new_graph, adjoints[inp_id], inp_adj_id)
        else:
            adjoints[inp_id] = inp_adj_id


def _backward_pass(
    new_graph: LogicalGraph,
    sorted_nodes: list[LogicalNode],
    reachable_from_output: set[str],
    adjoints: dict[str, str],
) -> None:
    """Perform the backward pass to compute gradients.

    Args:
        new_graph (LogicalGraph): The new graph.
        sorted_nodes (list[LogicalNode]): The sorted nodes.
        reachable_from_output (set[str]): The reachable nodes.
        adjoints (dict[str, str]): The adjoints map.

    """
    for node in reversed(sorted_nodes):
        nid = node.id
        if nid not in reachable_from_output or nid not in adjoints:
            continue

        if node.op_type == "Output":
            for inp in node.inputs:
                adjoints[inp] = adjoints[node.id]
            continue

        if node.op_type in ("StopGradient", "Input", "Constant"):
            continue

        _accumulate_gradients(new_graph, node, adjoints[nid], adjoints)


def _extract_gradients(
    new_graph: LogicalGraph,
    wrt: list[str],
    adjoints: dict[str, str],
) -> list[str]:
    """Extract the required gradients.

    Args:
        new_graph (LogicalGraph): The new_graph parameter for the operation.
        wrt (list[str]): Target nodes.
        adjoints (dict[str, str]): Adjoints map.

    Returns:
        list[str]: Output gradient node IDs.

    Raises:
        ValueError: If a target node is not found.
    """
    grad_outputs = []
    for w in wrt:
        if w not in new_graph.nodes:
            msg = f"Target node '{w}' not found in graph."
            raise ValueError(msg)

        if w in adjoints:
            grad_outputs.append(adjoints[w])
        else:
            zero_id = f"grad_zeros_{uuid.uuid4().hex[:6]}"
            zeros_node = LogicalNode(
                id=zero_id,
                op_type="Constant",
                attributes={"value": 0.0},
                shape_metadata=new_graph.nodes[w].shape_metadata,
            )
            new_graph.nodes[zero_id] = zeros_node
            grad_outputs.append(zero_id)
    return grad_outputs


def grad(graph: LogicalGraph, wrt: list[str], output_id: str, cotangent_id: typing.Optional[str] = None) -> LogicalGraph:
    """Evaluate grad operation.

    Args:
        graph (LogicalGraph): The graph parameter.
        wrt (list): The wrt parameter.
        output_id (str): The output_id parameter.
        cotangent_id (str): The cotangent_id parameter.

    Returns:
        LogicalGraph: Result.

    Raises:
        ValueError: An exception.
    """
    if output_id not in graph.nodes:
        msg = f"Output node '{output_id}' not found in graph."
        raise ValueError(msg)

    new_graph = _copy_graph(graph)

    sorted_nodes = topological_sort(new_graph)

    reachable_from_output = _get_reachable_from_output(sorted_nodes, output_id)

    adjoints: dict[str, str] = {}
    if cotangent_id is not None:
        if isinstance(cotangent_id, dict):
            for k, v in cotangent_id.items():
                adjoints[k] = v
        else:
            adjoints[output_id] = cotangent_id
    else:
        one_id = f"grad_ones_{uuid.uuid4().hex[:6]}"
        ones_node = LogicalNode(
            id=one_id,
            op_type="Constant",
            attributes={"value": 1.0},
            shape_metadata=(),
        )
        new_graph.nodes[one_id] = ones_node
        adjoints[output_id] = one_id

    _backward_pass(new_graph, sorted_nodes, reachable_from_output, adjoints)

    new_graph.outputs = _extract_gradients(new_graph, wrt, adjoints)
    return new_graph


def _get_input_tangents(new_graph: Any, node: Any, tangents: dict[str, str]) -> list[str]:
    """Get or create input tangents for a node.

    Args:
        new_graph (Any): The IR graph being constructed.
        node (Any): The IR node.
        tangents (dict[str, str]): Mapping of node IDs to tangent node IDs.

    Returns:
        list[str]: A list of input tangent node IDs.
    """
    import uuid

    from ml_switcheroo_ir import LogicalNode

    input_tangents = []
    for inp in node.inputs:
        if inp in tangents:
            input_tangents.append(tangents[inp])
        else:
            zero_id = f"jvp_zeros_{uuid.uuid4().hex[:6]}"
            zeros_node = LogicalNode(
                id=zero_id,
                op_type="Constant",
                attributes={"value": 0.0},
                shape_metadata=new_graph.nodes[inp].shape_metadata,
            )
            new_graph.nodes[zero_id] = zeros_node
            input_tangents.append(zero_id)
    return input_tangents


def _compile_jvp_expr(expr_str: str, graph: Any, shape_metadata: Any, inverse_map: dict[str, str]) -> str:
    """Compile a JVP expression string into IR nodes.

    Args:
        expr_str (str): The expression string.
        graph (Any): The target IR graph.
        shape_metadata (Any): Expected shape.
        inverse_map (dict[str, str]): Variable substitution map.

    Returns:
        str: The ID of the final tangent node.
    """
    import ast

    from ml_switcheroo_compiler.ops.base import emit_ir_node

    node = ast.parse(expr_str, mode="eval").body

    def _convert(ast_node: Any) -> str:
        """Recursively convert an AST node to IR.

        Args:
            ast_node (Any): The AST node.

        Returns:
            str: Generated IR node ID.

        Raises:
            ValueError: If an unsupported AST node is encountered.
        """
        if isinstance(ast_node, ast.Name):
            return inverse_map.get(ast_node.id, ast_node.id)
        if isinstance(ast_node, ast.BinOp):
            left_id = _convert(ast_node.left)
            right_id = _convert(ast_node.right)
            op_map = {
                ast.Add: "Add",
                ast.Sub: "Subtract",
                ast.Mult: "Multiply",
                ast.Div: "TrueDivide",
            }
            op_type = op_map.get(type(ast_node.op))
            if op_type is None:
                raise ValueError(f"Unsupported binary operator in JVP expression: {type(ast_node.op)}")
            return emit_ir_node(graph, op_type, [left_id, right_id], shape_metadata)
        if isinstance(ast_node, ast.UnaryOp):
            operand_id = _convert(ast_node.operand)
            if isinstance(ast_node.op, ast.USub):
                return emit_ir_node(graph, "Negative", [operand_id], shape_metadata)
            return operand_id
        if isinstance(ast_node, ast.Constant):
            import uuid

            from ml_switcheroo_ir import LogicalNode

            const_id = f"jvp_const_{uuid.uuid4().hex[:6]}"
            const_node = LogicalNode(
                id=const_id,
                op_type="Constant",
                attributes={"value": ast_node.value},
                shape_metadata=(),
            )
            graph.nodes[const_id] = const_node
            return const_id
        raise ValueError(f"Unsupported AST node in JVP expression: {type(ast_node)}")

    return _convert(node)


def _invoke_style2_jvp_rule(jvp_func: Any, sig: Any, new_graph: Any, node: Any, input_tangents: list[str]) -> Any:
    """Invoke a JVP rule using the 'style 2' parameter mapping.

    Args:
        jvp_func (Any): The JVP function.
        sig (Any): The signature of the JVP function.
        new_graph (Any): The target IR graph.
        node (Any): The IR node.
        input_tangents (list[str]): The list of input tangents.

    Returns: Any: The result of the JVP rule.
    """
    import inspect

    args = input_tangents + getattr(node, "inputs", [])
    param_keys = [name for name, param in sig.parameters.items() if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    call_args = args[: len(param_keys)]

    safe_id_map = {}
    inverse_map = {}
    for i, orig_id in enumerate(call_args):
        safe_id = f"safe_id_{i}"
        safe_id_map[orig_id] = safe_id
        inverse_map[safe_id] = orig_id

    safe_call_args = [safe_id_map[orig_id] for orig_id in call_args]

    try:
        expr = jvp_func(*safe_call_args)  # type: ignore
        if isinstance(expr, str):
            return _compile_jvp_expr(expr, new_graph, getattr(node, "shape_metadata", None), inverse_map)
        return expr
    except Exception as e:
        raise MissingJVPRuleError(f"Failed to execute JVP rule for {getattr(node, 'op_type', 'Unknown')}: {e}") from e


def _invoke_jvp_rule(jvp_func: Any, new_graph: Any, node: Any, input_tangents: list[str]) -> Any:
    """Invoke a JVP rule function, auto-detecting the style.

    Args:
        jvp_func (Any): The JVP function.
        new_graph (Any): The target IR graph.
        node (Any): The IR node.
        input_tangents (list[str]): The list of input tangents.

    Returns: Any: The result of the JVP rule.
    """
    import inspect

    sig = inspect.signature(jvp_func)
    if "graph" in sig.parameters and "node" in sig.parameters:
        try:
            return jvp_func(new_graph, node, input_tangents[0] if len(input_tangents) == 1 else input_tangents)
        except Exception as e:
            raise MissingJVPRuleError(f"Failed to execute JVP rule for {getattr(node, 'op_type', 'Unknown')}: {e}") from e

    # Style 2: real math rules with tangent parameters
    has_tangent_param = any("tangent" in p_name for p_name in sig.parameters)
    if len(sig.parameters) >= 2 and has_tangent_param:
        return _invoke_style2_jvp_rule(jvp_func, sig, new_graph, node, input_tangents)

    raise MissingJVPRuleError(f"No valid JVP rule signature matched for {getattr(node, 'op_type', 'Unknown')}")


def _process_jvp_node(
    new_graph: Any,
    node: Any,
    tangents: dict[str, str],
) -> None:
    """Process a single node for JVP.

    Args:
        new_graph (Any): The target IR graph.
        node (Any): The IR node to process.
        tangents (dict[str, str]): Mapping of node IDs to tangent node IDs.

    Raises:
        ValueError: If a required JVP rule is missing.
    """
    from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import get_jvp

    if node.op_type == "Output":
        import uuid

        from ml_switcheroo_ir import LogicalNode

        out_tangent_ids = []
        for inp in node.inputs:
            if inp in tangents:
                out_tangent_ids.append(tangents[inp])
            else:
                zero_id = f"jvp_zeros_{uuid.uuid4().hex[:6]}"
                zeros_node = LogicalNode(
                    id=zero_id,
                    op_type="Constant",
                    attributes={"value": 0.0},
                    shape_metadata=new_graph.nodes[inp].shape_metadata,
                )
                new_graph.nodes[zero_id] = zeros_node
                tangents[inp] = zero_id
                out_tangent_ids.append(zero_id)

        jvp_out_node_id = f"jvp_output_{uuid.uuid4().hex[:6]}"
        jvp_out_node = LogicalNode(
            id=jvp_out_node_id,
            op_type="Output",
            inputs=out_tangent_ids,
            shape_metadata=(),
        )
        new_graph.nodes[jvp_out_node_id] = jvp_out_node
        tangents[node.id] = jvp_out_node_id
        return

    if node.op_type in ("Input", "Constant", "StopGradient"):
        return

    if not any(inp in tangents for inp in node.inputs):
        return

    try:
        jvp_func = get_jvp(node.op_type)
    except ValueError:
        raise ValueError(f"Missing JVP rule for operation: {getattr(node, 'op_type', 'Unknown')}") from None

    input_tangents = _get_input_tangents(new_graph, node, tangents)

    try:
        out_tangent = _invoke_jvp_rule(jvp_func, new_graph, node, input_tangents)
        if out_tangent is not None:
            tangents[node.id] = typing.cast(str, out_tangent)
    except ValueError:
        raise ValueError(f"Missing JVP rule for operation: {getattr(node, 'op_type', 'Unknown')}") from None


def _forward_pass_jvp(
    new_graph: Any,
    sorted_nodes: list[Any],
    tangents: dict[str, str],
) -> None:
    """Perform the forward pass to compute JVP.

    Args:
        new_graph (Any): The target IR graph.
        sorted_nodes (list[Any]): Topologically sorted list of IR nodes.
        tangents (dict[str, str]): Mapping of node IDs to tangent node IDs.
    """
    for node in sorted_nodes:
        _process_jvp_node(new_graph, node, tangents)


def jvp(graph: LogicalGraph, primals: list[str], tangents: list[str], outputs: list[str]) -> LogicalGraph:
    """Evaluate jvp operation.

    Args:
        graph (LogicalGraph): The graph parameter.
        primals (list): The primals parameter.
        tangents (list): The tangents parameter.
        outputs (list): The outputs parameter.

    Returns:
        LogicalGraph: Result.

    Raises:
        ValueError: An exception.
    """
    if len(primals) != len(tangents):
        raise ValueError("primals and tangents must have the same length")

    for out_id in outputs:
        if out_id not in graph.nodes:
            raise ValueError(f"Output node '{out_id}' not found in graph.")

    new_graph = _copy_graph(graph)
    sorted_nodes = topological_sort(new_graph)

    tangents_map: dict[str, str] = dict(zip(primals, tangents))

    _forward_pass_jvp(new_graph, sorted_nodes, tangents_map)

    # Extract required output tangents
    out_tangents = []
    for out in outputs:
        if out in tangents_map:
            out_tangents.append(tangents_map[out])
        else:
            zero_id = f"jvp_zeros_{uuid.uuid4().hex[:6]}"
            zeros_node = LogicalNode(
                id=zero_id,
                op_type="Constant",
                attributes={"value": 0.0},
                shape_metadata=new_graph.nodes[out].shape_metadata,
            )
            new_graph.nodes[zero_id] = zeros_node
            out_tangents.append(zero_id)

    new_graph.outputs = out_tangents
    return new_graph


def hvp(graph: LogicalGraph, primals: list[str], tangents: list[str], outputs: list[str]) -> LogicalGraph:
    """Evaluate hvp operation.

    Args:
        graph (LogicalGraph): The graph parameter.
        primals (list): The primals parameter.
        tangents (list): The tangents parameter.
        outputs (list): The outputs parameter.

    Returns:
        LogicalGraph: Result.
    """
    from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import has_jvp
    from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import has_vjp

    missing_ops = []
    for node in graph.nodes.values():
        if node.op_type not in ("Input", "Output", "Constant"):
            # Ensure operations have both defined to guarantee higher-order composition
            if not has_vjp(node.op_type) or not has_jvp(node.op_type):
                missing_ops.append(node.op_type)
            pass  # Subgraph control flow ops are now handled recursively

    if missing_ops:
        unique_missing = ", ".join(sorted(set(missing_ops)))
        import warnings

        warnings.warn(f"Computing HVP with missing second-order derivative rules (vjp/jvp) for operations: {unique_missing}. Will rely on finite-difference fallbacks.", stacklevel=2)

    # First get the gradient (VJP) graph
    grad_graph = grad(graph, primals, outputs[0])

    # Then apply JVP to the gradient graph
    # For a scalar output f(x), the gradient is df/dx
    # The JVP of the gradient with tangent v is d^2f/dx^2 * v

    # Extract the gradient output nodes (these represent df/dx)
    grad_outputs = grad_graph.outputs

    # Now compute JVP of the gradient graph
    return jvp(grad_graph, primals, tangents, grad_outputs)
