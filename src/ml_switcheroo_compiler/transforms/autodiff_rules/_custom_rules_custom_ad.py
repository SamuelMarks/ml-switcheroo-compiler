"""Autodiff rules for CustomVJP, CustomJVP, and Recompute higher-order primitives."""

from __future__ import annotations

from collections.abc import Sequence

from ml_switcheroo_ir import IRGraph, IRNode, LogicalGraph

from ml_switcheroo_compiler.transforms.autodiff_rules._custom_rules_common import (
    _coerce_node_attrs,
    _extract_graph_nodes,
    _inline_grad_subgraph,
    _inline_subgraph,
    _is_subgraph_with_nodes,
    _node_shape_meta,
)
from ml_switcheroo_compiler.transforms.autodiff_rules.common import (
    UnconnectedGradients,
)
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import register_jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp


@register_vjp("Recompute")
def recompute_vjp(graph: IRGraph, node: IRNode, cotangent: str) -> Sequence[str | UnconnectedGradients | None]:
    """VJP for Recompute operation.

    Delegates to the original operation's VJP rule.

    Args:
        graph (IRGraph): Input `graph` argument (graph).
        node (IRNode): Input `node` argument (node).
        cotangent (str): Input `cotangent` argument (cotangent).

    Returns:
        tuple: Computed `tuple` output.
    """
    from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import get_vjp

    orig_op = str(node.attributes.get("original_op", "Unknown"))

    # We must construct a dummy node that looks like the original node
    from ml_switcheroo_compiler.ir.core import clone_logical_node

    dummy = clone_logical_node(node)
    dummy.op_type = orig_op
    dummy.attributes = _coerce_node_attrs(node.attributes.get("original_attrs", {}))

    vjp_func = get_vjp(orig_op)
    res = vjp_func(graph, dummy, cotangent)
    return res


@register_vjp("CustomVJP")
def custom_vjp_vjp(graph: IRGraph, node: IRNode, cotangent: str) -> tuple[str, ...] | str:
    """VJP for CustomVJP operation.

    Args:
        graph (IRGraph): Input `graph` argument (graph).
        node (IRNode): Input `node` argument (node).
        cotangent (str): Input `cotangent` argument (cotangent).

    Returns:
        tuple: Computed `tuple` output.
    """
    import uuid

    from ml_switcheroo_ir import LogicalNode

    bwd_subgraph = (
        node.subgraphs.get("bwd") or node.attributes.get("bwd_graph") or node.attributes.get("custom_backward_subgraph")
    )
    if bwd_subgraph is not None and _is_subgraph_with_nodes(bwd_subgraph):
        id_map: dict[str, str] = {}
        if bwd_subgraph.inputs:
            id_map[bwd_subgraph.inputs[0]] = cotangent
            for i, inp_id in enumerate(bwd_subgraph.inputs[1:]):
                if i < len(node.inputs):
                    id_map[inp_id] = node.inputs[i]
        nodes_list = _extract_graph_nodes(bwd_subgraph.nodes)
        for n in nodes_list:
            if n.id in id_map:
                continue
            id_map[n.id] = f"cvjp_inlined_{uuid.uuid4().hex[:6]}"
        _inline_subgraph(graph, bwd_subgraph, node, id_map)
        return tuple(id_map.get(out_id, out_id) for out_id in bwd_subgraph.outputs)

    bwd_fn = node.attributes.get("bwd_fn")

    bwd_node_id = f"cvjp_bwd_{uuid.uuid4().hex[:6]}"
    bwd_node = LogicalNode(
        id=bwd_node_id,
        op_type="ProcessCustomVJPCall",
        inputs=[cotangent],
        attributes={},
        shape_metadata=node.shape_metadata,
    )
    bwd_node.attributes["bwd_fn"] = bwd_fn
    graph.nodes[bwd_node_id] = bwd_node

    adjoints = []
    for i in range(len(node.inputs)):
        ext_id = f"cvjp_ext_{i}_{uuid.uuid4().hex[:6]}"
        ext_node = LogicalNode(
            id=ext_id,
            op_type="TupleGetItem",
            inputs=[bwd_node_id],
            attributes={"index": i},
            shape_metadata=node.shape_metadata,
        )
        graph.nodes[ext_id] = ext_node
        adjoints.append(ext_id)

    return tuple(adjoints)


@register_jvp("CustomJVP")
def custom_jvp_jvp(graph: IRGraph, node: IRNode, tangents: Sequence[str] | str) -> tuple[str, ...] | str:
    """JVP for CustomJVP operation.

    Args:
        graph (IRGraph): The target IR graph.
        node (IRNode): The CustomJVP node.
        tangents (Sequence[str] | str): Tangents for inputs.

    Returns:
        str: Output tangent node ID.
    """
    import uuid

    from ml_switcheroo_ir import LogicalNode

    tangents_list: list[str] = [str(t) for t in tangents] if isinstance(tangents, (tuple, list)) else [str(tangents)]

    jvp_subgraph = (
        node.subgraphs.get("jvp") or node.attributes.get("jvp_graph") or node.attributes.get("custom_forward_subgraph")
    )
    if jvp_subgraph is not None and _is_subgraph_with_nodes(jvp_subgraph):
        id_map: dict[str, str] = {}
        combined_inputs = list(node.inputs) + tangents_list
        for i, sub_inp in enumerate(jvp_subgraph.inputs):
            if i < len(combined_inputs):
                id_map[sub_inp] = combined_inputs[i]
        nodes_list = _extract_graph_nodes(jvp_subgraph.nodes)
        for n in nodes_list:
            if n.id in id_map:
                continue
            id_map[n.id] = f"cjvp_inlined_{uuid.uuid4().hex[:6]}"
        _inline_subgraph(graph, jvp_subgraph, node, id_map)
        return (
            str(id_map.get(jvp_subgraph.outputs[0], jvp_subgraph.outputs[0]))
            if jvp_subgraph.outputs
            else str(tangents_list[0])
        )

    jvp_rule = node.attributes.get("jvp_rule")

    call_id = f"cjvp_call_{uuid.uuid4().hex[:6]}"
    call_node = LogicalNode(
        id=call_id,
        op_type="ProcessCustomJVPCall",
        inputs=list(node.inputs) + tangents_list,
        attributes={"num_primals": len(node.inputs)},
        shape_metadata=node.shape_metadata,
    )
    call_node.attributes["jvp_rule"] = jvp_rule
    call_node.attributes["fun"] = node.attributes.get("fun")
    graph.nodes[call_id] = call_node

    ext_id = f"cjvp_tan_{uuid.uuid4().hex[:6]}"
    ext_node = LogicalNode(
        id=ext_id,
        op_type="TupleGetItem",
        inputs=[call_id],
        attributes={"index": 1},
        shape_metadata=node.shape_metadata,
    )
    graph.nodes[ext_id] = ext_node
    return ext_id


@register_vjp("CustomJVP")
def custom_jvp_vjp(graph: IRGraph, node: IRNode, cotangent: str) -> tuple[str, ...] | str:
    """VJP for CustomJVP operation.

    Args:
        graph (IRGraph): The target IR graph.
        node (IRNode): The CustomJVP node.
        cotangent (str): Input `cotangent` argument (cotangent).

    Returns:
        tuple: Adjoints list for inputs.
    """
    import uuid

    from ml_switcheroo_compiler.core.device import Device
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.ir.callback_registry import CustomRuleHandle
    from ml_switcheroo_compiler.ir.core import clone_logical_node
    from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor
    from ml_switcheroo_compiler.transforms.autodiff import grad as graph_grad

    fun = node.attributes.get("fun")
    if fun is not None and node.inputs:
        input_proxies = [
            Tensor(
                ProxyTensor(id=inp, shape=_node_shape_meta(graph, inp)),
                TensorConfig(_node_shape_meta(graph, inp), DType.Float32, Device("cpu")),
            )
            for inp in node.inputs
        ]
        resolved_fun = fun.resolve() if isinstance(fun, CustomRuleHandle) else fun
        primal_block = _trace_function(resolved_fun, tuple(input_proxies), f"cjvp_primal_{uuid.uuid4().hex[:6]}")
        sg = LogicalGraph(name="cjvp_sg")
        nodes_list: list[IRNode] = _extract_graph_nodes(primal_block.nodes)
        for n in nodes_list:
            sg.nodes[n.id] = clone_logical_node(n)
        sg.inputs = primal_block.inputs
        sg.outputs = primal_block.outputs

        sg_grad = graph_grad(sg, wrt=sg.inputs, output_id=sg.outputs[0], cotangent_id=cotangent)
        cotangent_mapping = {sg.outputs[0]: cotangent}
        adjoints = _inline_grad_subgraph(graph, sg_grad, sg, node, cotangent_mapping)
        return tuple(adjoints)

    return tuple(cotangent for _ in node.inputs)
