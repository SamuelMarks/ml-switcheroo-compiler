"""Forward-mode (JVP) expression compiler and per-node tangent propagation."""

from __future__ import annotations

import inspect
import typing
import uuid
from collections.abc import Callable, Sequence  # noqa: F401

from ml_switcheroo_ir import AttributeValue, IRGraph, LogicalNode

import ml_switcheroo_compiler.transforms.autodiff as _ad_mod
from ml_switcheroo_compiler.core.errors import MissingJVPRuleError
from ml_switcheroo_compiler.core.shape import ShapeLike
from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import JVPRuleCallable


@typing.runtime_checkable
class JvpRuleCallable(typing.Protocol):
    """Protocol for JVP rule functions (both graph-based and expression-based)."""

    def __call__(
        self, *args: IRGraph | LogicalNode | str | list[str]
    ) -> str | tuple[str, ...] | UnconnectedGradients | None:
        """Execute the JVP rule.

        Args:
            *args: Rule arguments.

        Returns:
            str | tuple[str, ...] | UnconnectedGradients | None: Output tangent node ID(s) or expression string.
        """
        ...


def _get_input_tangents(new_graph: IRGraph, node: LogicalNode, tangents: dict[str, str]) -> list[str]:
    """Get or create input tangents for a node.

    Args:
        new_graph (IRGraph): The IR graph being constructed.
        node (LogicalNode): The IR node.
        tangents (dict[str, str]): Mapping of node IDs to tangent node IDs.

    Returns:
        list[str]: A list of input tangent node IDs.
    """
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


def _compile_jvp_expr(expr_str: str, graph: IRGraph, shape_metadata: ShapeLike, inverse_map: dict[str, str]) -> str:
    """Compile a JVP expression string into IR nodes.

    Args:
        expr_str (str): The expression string.
        graph (IRGraph): The target IR graph.
        shape_metadata (ShapeLike): Expected shape.
        inverse_map (dict[str, str]): Variable substitution map.

    Returns:
        str: The ID of the final tangent node.
    """
    import ast

    from ml_switcheroo_compiler.ops.base import emit_ir_node

    node = ast.parse(expr_str, mode="eval").body
    norm_shape: tuple[int | str, ...] | list[int | str] | None = (
        tuple(int(d) if isinstance(d, int) else str(d) for d in shape_metadata)
        if isinstance(shape_metadata, (list, tuple))
        else None
    )

    def _convert(ast_node: ast.AST) -> str:
        """Recursively convert an AST node to IR.

        Args:
            ast_node (ast.AST): The AST node.

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
            return emit_ir_node(graph, op_type, [left_id, right_id], norm_shape)
        if isinstance(ast_node, ast.UnaryOp):
            operand_id = _convert(ast_node.operand)
            if isinstance(ast_node.op, ast.USub):
                return emit_ir_node(graph, "Negative", [operand_id], norm_shape)
            return operand_id
        if isinstance(ast_node, ast.Constant):
            import uuid

            from ml_switcheroo_ir import LogicalNode

            const_id = f"jvp_const_{uuid.uuid4().hex[:6]}"
            c_val: AttributeValue = ast_node.value if isinstance(ast_node.value, (int, float, str, bool)) else None
            const_node = LogicalNode(
                id=const_id,
                op_type="Constant",
                attributes={"value": c_val},
                shape_metadata=(),
            )
            graph.nodes[const_id] = const_node
            return const_id
        raise ValueError(f"Unsupported AST node in JVP expression: {type(ast_node)}")

    return _convert(node)


def _invoke_style2_jvp_rule(
    jvp_func: JvpRuleCallable | JVPRuleCallable,
    sig: inspect.Signature,
    new_graph: IRGraph,
    node: LogicalNode,
    input_tangents: list[str],
) -> str | tuple[str, ...] | UnconnectedGradients | None:
    """Invoke a JVP rule using the 'style 2' parameter mapping.

    Args:
        jvp_func (JvpRuleCallable | JVPRuleCallable): The JVP function.
        sig (inspect.Signature): The signature of the JVP function.
        new_graph (IRGraph): The target IR graph.
        node (LogicalNode): The IR node.
        input_tangents (list[str]): The list of input tangents.

    Returns:
        str | tuple[str, ...] | UnconnectedGradients | None: The result of the JVP rule.

    Raises:
        MissingJVPRuleError: If no forward-mode JVP rule is registered for the target operation in
            `_invoke_style2_jvp_rule`.
    """
    args = input_tangents + list(node.inputs)
    param_keys = [
        name
        for name, param in sig.parameters.items()
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    call_args = args[: len(param_keys)]

    safe_id_map = {}
    inverse_map = {}
    for i, orig_id in enumerate(call_args):
        safe_id = f"safe_id_{i}"
        safe_id_map[orig_id] = safe_id
        inverse_map[safe_id] = orig_id

    safe_call_args = [safe_id_map[orig_id] for orig_id in call_args]

    try:
        assert isinstance(jvp_func, JvpRuleCallable)
        expr = jvp_func(*safe_call_args)
        if isinstance(expr, str):
            return _ad_mod._compile_jvp_expr(expr, new_graph, node.shape_metadata, inverse_map)
        return expr
    except (
        AttributeError,
        ImportError,
        LookupError,
        OSError,
        ArithmeticError,
        RuntimeError,
        SyntaxError,
        TypeError,
        ValueError,
    ) as e:
        raise MissingJVPRuleError(f"Failed to execute JVP rule for {node.op_type}: {e}") from e


def _invoke_jvp_rule(
    jvp_func: JvpRuleCallable | JVPRuleCallable,
    new_graph: IRGraph,
    node: LogicalNode,
    input_tangents: list[str],
) -> str | tuple[str, ...] | UnconnectedGradients | None:
    """Invoke a JVP rule function, auto-detecting the style.

    Args:
        jvp_func (JvpRuleCallable | JVPRuleCallable): The JVP function.
        new_graph (IRGraph): The target IR graph.
        node (LogicalNode): The IR node.
        input_tangents (list[str]): The list of input tangents.

    Returns:
        str | tuple[str, ...] | UnconnectedGradients | None: The result of the JVP rule.

    Raises:
        MissingJVPRuleError: If no forward-mode JVP rule is registered for the target operation in `_invoke_jvp_rule`.
    """
    import inspect

    sig = inspect.signature(jvp_func)
    if "graph" in sig.parameters and "node" in sig.parameters:
        try:
            return jvp_func(new_graph, node, input_tangents[0] if len(input_tangents) == 1 else input_tangents)
        except (
            AttributeError,
            ImportError,
            LookupError,
            OSError,
            ArithmeticError,
            RuntimeError,
            SyntaxError,
            TypeError,
            ValueError,
        ) as e:
            raise MissingJVPRuleError(f"Failed to execute JVP rule for {node.op_type}: {e}") from e

    # Style 2: real math rules with tangent parameters
    has_tangent_param = any("tangent" in p_name for p_name in sig.parameters)
    if len(sig.parameters) >= 2 and has_tangent_param:
        return _ad_mod._invoke_style2_jvp_rule(jvp_func, sig, new_graph, node, input_tangents)

    raw_node: LogicalNode | None = node
    raise MissingJVPRuleError(
        f"No valid JVP rule signature matched for {raw_node.op_type if raw_node is not None else None}"
    )


def _process_jvp_node(
    new_graph: IRGraph,
    node: LogicalNode,
    tangents: dict[str, str],
) -> None:
    """Process a single node for JVP.

    Args:
        new_graph (IRGraph): The target IR graph.
        node (LogicalNode): The IR node to process.
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
        jvp_func: JvpRuleCallable | JVPRuleCallable = get_jvp(node.op_type)
    except ValueError:
        raise ValueError(f"Missing JVP rule for operation: {node.op_type}") from None

    input_tangents = _ad_mod._get_input_tangents(new_graph, node, tangents)

    try:
        out_tangent = _ad_mod._invoke_jvp_rule(jvp_func, new_graph, node, input_tangents)
        if out_tangent is not None:
            tangents[node.id] = str(out_tangent)
    except ValueError:
        raise ValueError(f"Missing JVP rule for operation: {node.op_type}") from None
