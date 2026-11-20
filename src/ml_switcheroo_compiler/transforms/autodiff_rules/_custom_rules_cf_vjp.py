"""Reverse-mode (VJP) autodiff rules for checkpointing and structured control flow."""

from __future__ import annotations

from ml_switcheroo_ir import IRGraph, IRNode, LogicalGraph

from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_common import (
    _extract_graph_nodes,
    _inline_grad_subgraph,
    _inline_subgraph,
    _is_subgraph_branch,
    _is_subgraph_with_nodes,
    _make_node_attrs,
    _node_shape_meta,
)
from ml_switcheroo_compiler.transforms.autodiff_rules.common import (
    UnconnectedGradients,
)
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp


@register_vjp("Checkpoint")
def checkpoint_vjp(graph: IRGraph, node: IRNode, cotangent: str) -> tuple[str, ...] | str:
    """VJP for Checkpoint operation.

    Args:
        graph (IRGraph): Input `graph` argument (graph).
        node (IRNode): Input `node` argument (node).
        cotangent (str): Input `cotangent` argument (cotangent).

    Returns:
        tuple: Computed `tuple` output.
    """
    import uuid

    from ml_switcheroo_compiler.ir.core import clone_logical_node
    from ml_switcheroo_compiler.transforms.autodiff import grad as graph_grad

    fun = node.attributes.get("fun")
    if fun is not None:
        from ml_switcheroo_compiler.core.device import Device
        from ml_switcheroo_compiler.core.dtype import DType
        from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
        from ml_switcheroo_compiler.ir.callback_registry import CustomRuleHandle
        from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function
        from ml_switcheroo_compiler.tracing.tracer import ProxyTensor
        from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_common import (
            _StructuralSubgraphProtocol,
        )

        input_proxies = [
            Tensor(
                ProxyTensor(id=inp, shape=_node_shape_meta(graph, inp)),
                TensorConfig(_node_shape_meta(graph, inp), DType.Float32, Device("cpu")),
            )
            for inp in node.inputs
        ]
        resolved_fun = fun.resolve() if isinstance(fun, CustomRuleHandle) else fun
        subgraph: LogicalGraph | _StructuralSubgraphProtocol = _trace_function(
            resolved_fun, tuple(input_proxies), f"cp_recompute_{uuid.uuid4().hex[:6]}"
        )
    else:
        raw_sub = node.attributes["subgraph"]
        assert _is_subgraph_with_nodes(raw_sub)
        subgraph = raw_sub

    nodes_list: list[IRNode] = _extract_graph_nodes(subgraph.nodes)

    id_map = {n.id: f"cp_fwd_{n.id}_{uuid.uuid4().hex[:6]}" for n in nodes_list}
    for in_id, orig_in_id in zip(subgraph.inputs, node.inputs):
        id_map[in_id] = orig_in_id

    _inline_subgraph(graph, subgraph, node, id_map)

    sg = LogicalGraph(name="cp_sg")
    for n in nodes_list:
        sg.nodes[n.id] = clone_logical_node(n)
    sg.inputs = subgraph.inputs
    sg.outputs = subgraph.outputs

    cotangent_mapping = {sg.outputs[0]: cotangent}

    sg_grad = graph_grad(sg, wrt=sg.inputs, output_id=sg.outputs[0], cotangent_id=cotangent)

    adjoints = _inline_grad_subgraph(graph, sg_grad, sg, node, cotangent_mapping)
    return tuple(adjoints)


@register_vjp("If")
def _if_vjp(graph: IRGraph, node: IRNode, cotangent: str) -> tuple[str, ...] | str:
    """VJP for If operation.

    Args:
        graph (IRGraph): Input `graph` argument (graph).
        node (IRNode): Input `node` argument (node).
        cotangent (str): Input `cotangent` argument (cotangent).

    Returns:
        tuple: Computed `tuple` output.
    """
    from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients

    return (UnconnectedGradients.ZERO.value,)


@register_vjp("Loop")
def _loop_vjp(graph: IRGraph, node: IRNode, cotangent: str) -> tuple[str | UnconnectedGradients | None, ...]:
    """VJP for Loop operation.

    Args:
        graph (IRGraph): Input `graph` argument (graph).
        node (IRNode): Input `node` argument (node).
        cotangent (str): Input `cotangent` argument (cotangent).

    Returns:
        tuple: Computed `tuple` output.
    """
    from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients

    return tuple(UnconnectedGradients.ZERO for _ in node.inputs)


@register_vjp("Cond")
def _cond_vjp(graph: IRGraph, node: IRNode, cotangent: str) -> tuple[str | UnconnectedGradients | None, ...]:
    """VJP for Cond operation lowering into adjoint control flow structure.

    Args:
        graph (IRGraph): The computation graph.
        node (IRNode): The primal Cond node.
        cotangent (str): Cotangent identifier.

    Returns:
        tuple: Cotangents for condition and operands.
    """
    from ml_switcheroo_ir import IRNode

    from ml_switcheroo_compiler.transforms.autodiff import grad
    from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients

    tb = node.attributes.get("then_branch", node.attributes.get("true_branch"))
    eb = node.attributes.get("else_branch", node.attributes.get("false_branch"))

    if tb is not None and eb is not None and _is_subgraph_branch(tb) and _is_subgraph_branch(eb):
        tb_grad = grad(tb, tb.inputs, tb.outputs[0] if tb.outputs else "", cotangent_id=cotangent)
        eb_grad = grad(eb, eb.inputs, eb.outputs[0] if eb.outputs else "", cotangent_id=cotangent)

        adj_id = f"{node.id}_adj_cond"
        adj_node = IRNode(
            id=adj_id,
            op_type="Cond",
            inputs=list(node.inputs),
            attributes=_make_node_attrs(
                {
                    "then_branch": tb_grad,
                    "else_branch": eb_grad,
                    "true_branch": tb_grad,
                    "false_branch": eb_grad,
                }
            ),
            shape_metadata=node.shape_metadata,
        )
        graph.nodes[adj_id] = adj_node
        return tuple([UnconnectedGradients.ZERO] + [adj_id for _ in node.inputs[1:]])

    return tuple(UnconnectedGradients.ZERO for _ in node.inputs)


@register_vjp("Scan")
def _scan_vjp(graph: IRGraph, node: IRNode, cotangent: str) -> tuple[str | UnconnectedGradients | None, ...]:
    """VJP for Scan operation lowering into adjoint reverse scan.

    Args:
        graph (IRGraph): The computation graph.
        node (IRNode): The primal Scan node.
        cotangent (str): Cotangent identifier.

    Returns:
        tuple: Adjoint node outputs.
    """
    from ml_switcheroo_ir import IRNode

    from ml_switcheroo_compiler.transforms.autodiff import grad
    from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients

    body = node.attributes.get("body", node.attributes.get("body_subgraph"))
    if body is not None and _is_subgraph_branch(body) and body.inputs and body.outputs:
        body_grad = grad(body, body.inputs, body.outputs[0], cotangent_id=cotangent)
        adj_id = f"{node.id}_adj_scan"
        adj_node = IRNode(
            id=adj_id,
            op_type="Scan",
            inputs=list(node.inputs),
            attributes=_make_node_attrs(
                {
                    "body": body_grad,
                    "body_subgraph": body_grad,
                    "reverse": not node.attributes.get("reverse", False),
                }
            ),
            shape_metadata=node.shape_metadata,
        )
        graph.nodes[adj_id] = adj_node
        return tuple(adj_id for _ in node.inputs)

    return tuple(UnconnectedGradients.ZERO for _ in node.inputs)


@register_vjp("WhileLoop")
def _while_loop_vjp(graph: IRGraph, node: IRNode, cotangent: str) -> tuple[str | UnconnectedGradients | None, ...]:
    """VJP for WhileLoop operation lowering into adjoint control flow structure.

    Args:
        graph (IRGraph): The computation graph.
        node (IRNode): The primal WhileLoop node.
        cotangent (str): Cotangent identifier.

    Returns:
        tuple: Adjoint loop outputs.
    """
    from ml_switcheroo_ir import IRNode

    from ml_switcheroo_compiler.transforms.autodiff import grad
    from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients

    body = node.attributes.get("body", node.attributes.get("body_subgraph"))
    cond = node.attributes.get("cond", node.attributes.get("cond_graph"))
    if body is not None and _is_subgraph_branch(body) and body.inputs and body.outputs:
        body_grad = grad(body, body.inputs, body.outputs[0], cotangent_id=cotangent)
        adj_id = f"{node.id}_adj_while"
        adj_node = IRNode(
            id=adj_id,
            op_type="WhileLoop",
            inputs=list(node.inputs),
            attributes=_make_node_attrs(
                {
                    "body": body_grad,
                    "body_subgraph": body_grad,
                    "cond": cond,
                    "adjoint": True,
                }
            ),
            shape_metadata=node.shape_metadata,
        )
        graph.nodes[adj_id] = adj_node
        return tuple(adj_id for _ in node.inputs)

    return tuple(UnconnectedGradients.ZERO for _ in node.inputs)


@register_vjp("AssociativeScan")
def _assoc_scan_vjp(graph: IRGraph, node: IRNode, cotangent: str) -> tuple[str | UnconnectedGradients | None, ...]:
    """VJP for AssociativeScan operation.

    Args:
        graph (IRGraph): Input `graph` argument (graph).
        node (IRNode): Input `node` argument (node).
        cotangent (str): Input `cotangent` argument (cotangent).

    Returns:
        tuple[str | UnconnectedGradients | None, ...]: Zero cotangents for each input.
    """
    _ = (graph, cotangent)
    from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients

    return tuple(UnconnectedGradients.ZERO for _ in node.inputs)
