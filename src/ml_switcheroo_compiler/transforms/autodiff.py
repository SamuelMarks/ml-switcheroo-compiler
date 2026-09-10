# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Reverse-mode Automatic Differentiation (AD) Engine."""

import typing
import uuid
from collections.abc import Sequence
from typing import Any, Optional, Union

from ml_switcheroo_ir import LogicalGraph, LogicalNode, topological_sort

from ml_switcheroo_compiler.core.errors import MissingJVPRuleError
from ml_switcheroo_compiler.ir.core import IRGraph, clone_logical_node
from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import _VJP_REGISTRY, get_vjp


def _is_zero_node(graph: LogicalGraph, node_id: str) -> bool:
    """Check if node represents a zero or identity element for addition.

    Args:
        graph (LogicalGraph): The computation graph.
        node_id (str): Identifier of the node to check.

    Returns:
        bool: True if node represents zero.
    """
    if node_id not in graph.nodes:
        return False
    node = graph.nodes[node_id]
    if node.op_type in ("Zeros", "ZerosLike"):
        return True
    if node.op_type == "Constant":
        val = node.attributes.get("value")
        if isinstance(val, (int, float)) and val == 0:
            return True
    return False


def _add_nodes(graph: LogicalGraph, n1_id: str, n2_id: str) -> str:
    """Emit an Add node for gradient accumulation, avoiding redundant Add chains.

    Args:
        graph (LogicalGraph): The graph parameter for the operation.
        n1_id (str): The n1_id parameter for the operation.
        n2_id (str): The n2_id parameter for the operation.

    Returns:
        str: The computed result.
    """
    if _is_zero_node(graph, n1_id):
        return n2_id
    if _is_zero_node(graph, n2_id):
        return n1_id

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


def _load_rematerialization_rules() -> dict[str, Any]:
    """Load rematerialization rules from YAML config.

    Returns:
        dict[str, Any]: Loaded rematerialization rules configuration.
    """
    import os

    import yaml

    yaml_path = os.path.join(os.path.dirname(__file__), "passes", "rematerialization_rules.yaml")
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _should_rematerialize(node: LogicalNode, rules: Optional[dict[str, Any]] = None) -> bool:
    """Determine if a node should be rematerialized across backward boundary.

    Args:
        node (LogicalNode): The node to evaluate.
        rules (Optional[dict[str, Any]]): Rematerialization rules config.

    Returns:
        bool: True if node should be rematerialized.
    """
    if node.attributes.get("rematerialize", False):
        return True
    if rules is None:
        rules = _load_rematerialization_rules()
    if not rules:
        return False
    if node.op_type in rules.get("high_cost_ops", []):
        return False
    return node.op_type in rules.get("target_ops", []) and bool(node.attributes.get("checkpoint", False))


def _recompute_subgraph(new_graph: LogicalGraph, node: LogicalNode) -> LogicalNode:
    """Recursively recompute a node and its rematerialized inputs.

    Args:
        new_graph (LogicalGraph): The new graph.
        node (LogicalNode): The node to recompute.

    Returns:
        LogicalNode: The cloned recomputed node.
    """
    import uuid

    from ml_switcheroo_compiler.ir.core import clone_logical_node

    recompute_node = clone_logical_node(node)
    recompute_node.id = f"{node.id}_recompute_{uuid.uuid4().hex[:6]}"

    # Recursively recompute inputs if they are also tagged for rematerialization
    new_inputs = []
    for inp_id in getattr(node, "inputs", []):
        if inp_id in new_graph.nodes:
            inp_node = new_graph.nodes[inp_id]
            if _should_rematerialize(inp_node):
                # Need to recompute the input as well
                recomputed_inp = _recompute_subgraph(new_graph, inp_node)
                new_inputs.append(recomputed_inp.id)
            else:
                new_inputs.append(inp_id)
        else:
            new_inputs.append(inp_id)

    recompute_node.inputs = new_inputs
    new_graph.nodes[recompute_node.id] = recompute_node
    return recompute_node


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
        MissingJVPRuleError: If VJP rule is missing or returns incorrect number of adjoints.
    """
    if _should_rematerialize(node):
        eval_node = _recompute_subgraph(new_graph, node)
    else:
        eval_node = node

    try:
        vjp_func = get_vjp(node.op_type)
    except ValueError:
        msg = f"Missing VJP rule for operation: {getattr(node, 'op_type', 'Unknown')}"
        raise MissingJVPRuleError(msg) from None

    input_adjs = vjp_func(new_graph, eval_node, adj_id)

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
    if any(getattr(n, "op_type", "") == "Vmap" for n in graph.nodes.values()):
        from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
        from ml_switcheroo_compiler.transforms.passes.vectorization import vectorization_pass

        ir_g = IRGraph(name=graph.name)
        ir_g.inputs = list(graph.inputs)
        ir_g.outputs = list(graph.outputs)
        for nid, n in graph.nodes.items():
            ir_g.nodes[nid] = n if isinstance(n, IRNode) else IRNode(**n.__dict__)
        graph = vectorization_pass(ir_g)

    output_ids: list[str] = [output_id] if isinstance(output_id, str) else list(output_id)
    for oid in output_ids:
        if oid not in graph.nodes:
            msg = f"Output node '{oid}' not found in graph."
            raise ValueError(msg)

    new_graph = _copy_graph(graph)

    sorted_nodes = topological_sort(new_graph)

    reachable_from_output: set[str] = set()
    for oid in output_ids:
        reachable_from_output.update(_get_reachable_from_output(sorted_nodes, oid))

    adjoints: dict[str, str] = {}
    if cotangent_id is not None:
        if isinstance(cotangent_id, dict):
            for k, v in cotangent_id.items():
                adjoints[k] = v
        elif isinstance(cotangent_id, (list, tuple)):
            for oid, cid in zip(output_ids, cotangent_id):
                adjoints[oid] = cid
        else:
            if len(output_ids) == 1:
                adjoints[output_ids[0]] = cotangent_id
            else:
                for oid in output_ids:
                    adjoints[oid] = cotangent_id
    else:
        for oid in output_ids:
            one_id = f"grad_ones_{uuid.uuid4().hex[:6]}"
            target_node = new_graph.nodes[oid]
            target_shape = getattr(target_node, "shape_metadata", ())
            ones_node = LogicalNode(
                id=one_id,
                op_type="Constant",
                attributes={"value": 1.0},
                shape_metadata=target_shape,
            )
            new_graph.nodes[one_id] = ones_node
            adjoints[oid] = one_id

    _backward_pass(new_graph, sorted_nodes, reachable_from_output, adjoints)

    new_graph.outputs = _extract_gradients(new_graph, wrt, adjoints)
    return new_graph


def _get_input_tangents(new_graph: IRGraph, node, tangents: dict[str, str]) -> list[str]:
    """Get or create input tangents for a node.

    Args:
        new_graph (object): The IR graph being constructed.
        node (object): The IR node.
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


def _compile_jvp_expr(expr_str: str, graph: IRGraph, shape_metadata, inverse_map: dict[str, str]) -> str:
    """Compile a JVP expression string into IR nodes.

    Args:
        expr_str (str): The expression string.
        graph (object): The target IR graph.
        shape_metadata (object): Expected shape.
        inverse_map (dict[str, str]): Variable substitution map.

    Returns:
        str: The ID of the final tangent node.
    """
    import ast

    from ml_switcheroo_compiler.ops.base import emit_ir_node

    node = ast.parse(expr_str, mode="eval").body

    def _convert(ast_node) -> str:
        """Recursively convert an AST node to IR.

        Args:
            ast_node (object): The AST node.

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


def _invoke_style2_jvp_rule(jvp_func: Any, sig: Any, new_graph: IRGraph, node: Any, input_tangents: list[str]) -> Any:
    """Invoke a JVP rule using the 'style 2' parameter mapping.

    Args:
        jvp_func (object): The JVP function.
        sig (object): The signature of the JVP function.
        new_graph (object): The target IR graph.
        node (object): The IR node.
        input_tangents (list[str]): The list of input tangents.

    Returns: Tensor: The result of the JVP rule.
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
        expr = jvp_func(*safe_call_args)
        if isinstance(expr, str):
            return _compile_jvp_expr(expr, new_graph, getattr(node, "shape_metadata", None), inverse_map)
        return expr
    except Exception as e:
        raise MissingJVPRuleError(f"Failed to execute JVP rule for {getattr(node, 'op_type', 'Unknown')}: {e}") from e


def _invoke_jvp_rule(jvp_func: Any, new_graph: IRGraph, node: Any, input_tangents: list[str]) -> Any:
    """Invoke a JVP rule function, auto-detecting the style.

    Args:
        jvp_func (object): The JVP function.
        new_graph (object): The target IR graph.
        node (object): The IR node.
        input_tangents (list[str]): The list of input tangents.

    Returns: Tensor: The result of the JVP rule.
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
    new_graph: IRGraph,
    node,
    tangents: dict[str, str],
) -> None:
    """Process a single node for JVP.

    Args:
        new_graph (object): The target IR graph.
        node (object): The IR node to process.
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
            tangents[node.id] = str(out_tangent)
    except ValueError:
        raise ValueError(f"Missing JVP rule for operation: {getattr(node, 'op_type', 'Unknown')}") from None


def _forward_pass_jvp(
    new_graph: IRGraph,
    sorted_nodes,
    tangents: dict[str, str],
) -> None:
    """Perform the forward pass to compute JVP.

    Args:
        new_graph (object): The target IR graph.
        sorted_nodes (list[object]): Topologically sorted list of IR nodes.
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


def hvp(
    graph: LogicalGraph,
    primals: list[str],
    tangents: list[str],
    outputs: list[str],
    mode: str = "forward-over-reverse",
    projected_tangents: typing.Optional[list[str]] = None,
) -> LogicalGraph:
    """Evaluate Hessian-vector product (HVP) operation.

    Supports both forward-over-reverse and reverse-over-forward modes, and seamlessly
    handles multi-dimensional and non-scalar outputs via projected tangents.

    Args:
        graph (LogicalGraph): Base computation graph.
        primals (list[str]): Input primal variable node IDs.
        tangents (list[str]): Tangent vector node IDs corresponding to primals.
        outputs (list[str]): Output node IDs to differentiate.
        mode (str): HVP computation mode ('forward-over-reverse' or 'reverse-over-forward').
        projected_tangents (typing.Optional[list[str]]): Optional projected tangent node IDs
            for projecting non-scalar or multi-node outputs into a directional scalar/vector.

    Returns:
        LogicalGraph: Graph computing the Hessian-vector product.
    """
    if mode == "forward-over-reverse":
        if projected_tangents is not None:
            cot_dict = {out_id: pt for out_id, pt in zip(outputs, projected_tangents)}
            grad_graph = grad(graph, primals, outputs, cotangent_id=cot_dict)
        else:
            if len(outputs) != 1:
                raise ValueError("hvp requires a single output node when using forward-over-reverse mode without projected_tangents")
            grad_graph = grad(graph, primals, outputs[0])
        grad_outputs = grad_graph.outputs
        return jvp(grad_graph, primals, tangents, grad_outputs)
    elif mode == "reverse-over-forward":
        jvp_graph = jvp(graph, primals, tangents, outputs)
        jvp_outputs = jvp_graph.outputs
        if projected_tangents is not None:
            cot_dict = {out_id: pt for out_id, pt in zip(jvp_outputs, projected_tangents)}
            return grad(jvp_graph, primals, jvp_outputs, cotangent_id=cot_dict)
        else:
            if len(jvp_outputs) != 1:
                raise ValueError("hvp requires a single output node when using reverse-over-forward mode without projected_tangents")
            return grad(jvp_graph, primals, jvp_outputs[0])
    else:
        raise ValueError(f"Unknown HVP mode: {mode}")


def conv2d_vjp_input(graph: LogicalGraph, node: LogicalNode, cotangent: str) -> str:
    """Compute exact VJP cotangent for the input tensor of a Conv2D operation.

    Handles arbitrary strides, padding, dilation, and groups.

    Args:
        graph (LogicalGraph): Computation graph.
        node (LogicalNode): Primal Conv2D node.
        cotangent (str): Output adjoint identifier.

    Returns:
        str: Input adjoint node identifier.
    """
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    input_id = node.inputs[0]
    weight_id = node.inputs[1]
    input_node = graph.nodes.get(input_id)
    shape_meta = getattr(input_node, "shape_metadata", None)

    stride = node.attributes.get("stride", node.attributes.get("strides", (1, 1)))
    dilation = node.attributes.get("dilation", node.attributes.get("dilations", (1, 1)))

    attrs = {
        "stride": stride,
        "strides": stride,
        "padding": node.attributes.get("padding", "SAME"),
        "dilation": dilation,
        "dilations": dilation,
        "groups": int(node.attributes.get("groups", 1)),
        "target_shape": shape_meta,
    }
    return emit_ir_node(graph, "Conv2DInputGrad", [cotangent, weight_id], shape_metadata=shape_meta, attributes=attrs)


def conv2d_vjp_weight(graph: LogicalGraph, node: LogicalNode, cotangent: str) -> str:
    """Compute exact VJP cotangent for the weight tensor of a Conv2D operation.

    Handles arbitrary strides, padding, dilation, and groups.

    Args:
        graph (LogicalGraph): Computation graph.
        node (LogicalNode): Primal Conv2D node.
        cotangent (str): Output adjoint identifier.

    Returns:
        str: Weight adjoint node identifier.
    """
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    input_id = node.inputs[0]
    weight_id = node.inputs[1]
    weight_node = graph.nodes.get(weight_id)
    shape_meta = getattr(weight_node, "shape_metadata", None)

    stride = node.attributes.get("stride", node.attributes.get("strides", (1, 1)))
    dilation = node.attributes.get("dilation", node.attributes.get("dilations", (1, 1)))

    attrs = {
        "stride": stride,
        "strides": stride,
        "padding": node.attributes.get("padding", "SAME"),
        "dilation": dilation,
        "dilations": dilation,
        "groups": int(node.attributes.get("groups", 1)),
        "target_shape": shape_meta,
    }
    return emit_ir_node(graph, "Conv2DWeightGrad", [input_id, cotangent], shape_metadata=shape_meta, attributes=attrs)


def _conv2d_vjp_rule(graph: LogicalGraph, node: LogicalNode, cotangent: str) -> tuple[str, ...]:
    """Exact VJP rule for Conv2D."""
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    grad_in = conv2d_vjp_input(graph, node, cotangent)
    grad_w = conv2d_vjp_weight(graph, node, cotangent)
    if len(node.inputs) > 2:
        bias_id = node.inputs[2]
        shape_b = getattr(graph.nodes.get(bias_id), "shape_metadata", None)
        grad_b = emit_ir_node(graph, "Conv2DBiasGrad", [cotangent], shape_metadata=shape_b, attributes=node.attributes)
        return (grad_in, grad_w, grad_b)
    return (grad_in, grad_w)


def _max_pool2d_with_argmax_vjp(graph: LogicalGraph, node: LogicalNode, cotangent: str) -> tuple[str, ...]:
    """Exact VJP rule for MaxPool2DWithArgmax."""
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    input_id = node.inputs[0]
    shape_meta = getattr(graph.nodes.get(input_id), "shape_metadata", None)
    grad_in = emit_ir_node(graph, "MaxPool2DWithArgmaxGrad", [cotangent, input_id], shape_metadata=shape_meta, attributes=node.attributes)
    adjs = [grad_in]
    while len(adjs) < len(node.inputs):
        adjs.append(UnconnectedGradients.NONE)
    return tuple(adjs)


def _avg_pool2d_vjp(graph: LogicalGraph, node: LogicalNode, cotangent: str) -> tuple[str, ...]:
    """Exact VJP rule for AvgPool2D."""
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    input_id = node.inputs[0]
    shape_meta = getattr(graph.nodes.get(input_id), "shape_metadata", None)
    grad_in = emit_ir_node(graph, "AvgPool2DGrad", [cotangent, input_id], shape_metadata=shape_meta, attributes=node.attributes)
    return (grad_in,)


def _batch_norm_vjp(graph: LogicalGraph, node: LogicalNode, cotangent: str) -> tuple[str, ...]:
    """Exact VJP rule for BatchNorm."""
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    x_id = node.inputs[0]
    shape_x = getattr(graph.nodes.get(x_id), "shape_metadata", None)
    gamma_id = node.inputs[1] if len(node.inputs) > 1 else None
    shape_gamma = getattr(graph.nodes.get(gamma_id), "shape_metadata", None) if gamma_id else None

    inps_for_x = [cotangent, x_id] + ([gamma_id] if gamma_id else [])
    grad_x = emit_ir_node(graph, "BatchNormInputGrad", inps_for_x, shape_metadata=shape_x, attributes=node.attributes)

    adjs = [grad_x]
    if len(node.inputs) > 1:
        grad_gamma = emit_ir_node(graph, "BatchNormGammaGrad", [cotangent, x_id], shape_metadata=shape_gamma, attributes=node.attributes)
        adjs.append(grad_gamma)
    if len(node.inputs) > 2:
        beta_id = node.inputs[2]
        shape_beta = getattr(graph.nodes.get(beta_id), "shape_metadata", None)
        grad_beta = emit_ir_node(graph, "BatchNormBetaGrad", [cotangent], shape_metadata=shape_beta, attributes=node.attributes)
        adjs.append(grad_beta)

    while len(adjs) < len(node.inputs):
        adjs.append(UnconnectedGradients.NONE)
    return tuple(adjs)


def _layer_norm_vjp(graph: LogicalGraph, node: LogicalNode, cotangent: str) -> tuple[str, ...]:
    """Exact VJP rule for LayerNorm."""
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    x_id = node.inputs[0]
    shape_x = getattr(graph.nodes.get(x_id), "shape_metadata", None)
    gamma_id = node.inputs[1] if len(node.inputs) > 1 else None
    shape_gamma = getattr(graph.nodes.get(gamma_id), "shape_metadata", None) if gamma_id else None

    inps_for_x = [cotangent, x_id] + ([gamma_id] if gamma_id else [])
    grad_x = emit_ir_node(graph, "LayerNormInputGrad", inps_for_x, shape_metadata=shape_x, attributes=node.attributes)

    adjs = [grad_x]
    if len(node.inputs) > 1:
        grad_gamma = emit_ir_node(graph, "LayerNormGammaGrad", [cotangent, x_id], shape_metadata=shape_gamma, attributes=node.attributes)
        adjs.append(grad_gamma)
    if len(node.inputs) > 2:
        beta_id = node.inputs[2]
        shape_beta = getattr(graph.nodes.get(beta_id), "shape_metadata", None)
        grad_beta = emit_ir_node(graph, "LayerNormBetaGrad", [cotangent], shape_metadata=shape_beta, attributes=node.attributes)
        adjs.append(grad_beta)

    while len(adjs) < len(node.inputs):
        adjs.append(UnconnectedGradients.NONE)
    return tuple(adjs)


def _rms_norm_vjp(graph: LogicalGraph, node: LogicalNode, cotangent: str) -> tuple[str, ...]:
    """Exact VJP rule for RMSNorm."""
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    x_id = node.inputs[0]
    shape_x = getattr(graph.nodes.get(x_id), "shape_metadata", None)
    gamma_id = node.inputs[1] if len(node.inputs) > 1 else None
    shape_gamma = getattr(graph.nodes.get(gamma_id), "shape_metadata", None) if gamma_id else None

    inps_for_x = [cotangent, x_id] + ([gamma_id] if gamma_id else [])
    grad_x = emit_ir_node(graph, "RMSNormInputGrad", inps_for_x, shape_metadata=shape_x, attributes=node.attributes)

    adjs = [grad_x]
    if len(node.inputs) > 1:
        grad_gamma = emit_ir_node(graph, "RMSNormGammaGrad", [cotangent, x_id], shape_metadata=shape_gamma, attributes=node.attributes)
        adjs.append(grad_gamma)

    while len(adjs) < len(node.inputs):
        adjs.append(UnconnectedGradients.NONE)
    return tuple(adjs)


def _group_norm_vjp(graph: LogicalGraph, node: LogicalNode, cotangent: str) -> tuple[str, ...]:
    """Exact VJP rule for GroupNorm."""
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    x_id = node.inputs[0]
    shape_x = getattr(graph.nodes.get(x_id), "shape_metadata", None)
    gamma_id = node.inputs[1] if len(node.inputs) > 1 else None
    shape_gamma = getattr(graph.nodes.get(gamma_id), "shape_metadata", None) if gamma_id else None

    inps_for_x = [cotangent, x_id] + ([gamma_id] if gamma_id else [])
    grad_x = emit_ir_node(graph, "GroupNormInputGrad", inps_for_x, shape_metadata=shape_x, attributes=node.attributes)

    adjs = [grad_x]
    if len(node.inputs) > 1:
        grad_gamma = emit_ir_node(graph, "GroupNormGammaGrad", [cotangent, x_id], shape_metadata=shape_gamma, attributes=node.attributes)
        adjs.append(grad_gamma)
    if len(node.inputs) > 2:
        beta_id = node.inputs[2]
        shape_beta = getattr(graph.nodes.get(beta_id), "shape_metadata", None)
        grad_beta = emit_ir_node(graph, "GroupNormBetaGrad", [cotangent], shape_metadata=shape_beta, attributes=node.attributes)
        adjs.append(grad_beta)

    while len(adjs) < len(node.inputs):
        adjs.append(UnconnectedGradients.NONE)
    return tuple(adjs)


# Register exact VJP rules
_VJP_REGISTRY["Conv2D"] = _conv2d_vjp_rule
_VJP_REGISTRY["conv2d"] = _conv2d_vjp_rule
_VJP_REGISTRY["MaxPool2DWithArgmax"] = _max_pool2d_with_argmax_vjp
_VJP_REGISTRY["AvgPool2D"] = _avg_pool2d_vjp
_VJP_REGISTRY["BatchNorm"] = _batch_norm_vjp
_VJP_REGISTRY["LayerNorm"] = _layer_norm_vjp
_VJP_REGISTRY["RMSNorm"] = _rms_norm_vjp
_VJP_REGISTRY["GroupNorm"] = _group_norm_vjp
