"""Autodiff rules for custom and platform-specific kernel operations."""

from __future__ import annotations

import typing
from collections.abc import Mapping, Sequence

from ml_switcheroo_ir import AttributeValue, LogicalGraph

from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.autodiff_rules.common import (
    UnconnectedGradients,
    make_zero_jvp,
    make_zero_vjp,
)
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import register_jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp


def _is_attr_dict_holder(
    val: dict[str, AttributeValue],
) -> typing.TypeGuard[dict[str, LogicalGraph | AttributeValue]]:
    """Type guard narrowing an attributes dictionary to accept `LogicalGraph` entries.

    Args:
        val (dict[str, AttributeValue]): Attributes dictionary.

    Returns:
        bool: True if val is a dict.
    """
    return isinstance(val, dict)


def _make_node_attrs(
    items: Mapping[str, LogicalGraph | AttributeValue],
) -> dict[str, AttributeValue]:
    """Construct a node attributes dictionary containing `LogicalGraph` or `AttributeValue` entries.

    Args:
        items (Mapping[str, LogicalGraph | AttributeValue]): Key-value pairs for node attributes.

    Returns:
        dict[str, AttributeValue]: Populated attributes dictionary.
    """
    out: dict[str, AttributeValue] = {}
    target = out
    assert _is_attr_dict_holder(target)
    for k, v in items.items():
        target[str(k)] = v
    return out


def _coerce_node_attrs(raw: AttributeValue | LogicalGraph) -> dict[str, AttributeValue]:
    """Coerce a raw attribute value into a `dict[str, AttributeValue]`.

    Args:
        raw (AttributeValue | LogicalGraph): Raw attributes dictionary or value.

    Returns:
        dict[str, AttributeValue]: Coerced attributes dictionary.
    """
    return {str(k): v for k, v in raw.items()} if isinstance(raw, dict) else {}


@typing.runtime_checkable
class _SubgraphBranchProtocol(typing.Protocol):
    """Protocol for control-flow subgraph branches with inputs and outputs."""

    inputs: list[str]
    outputs: list[str]


@typing.runtime_checkable
class _SubgraphWithNodesProtocol(typing.Protocol):
    """Protocol for custom VJP/JVP subgraphs with nodes, inputs, and outputs."""

    nodes: dict[str, IRNode] | list[IRNode]
    inputs: list[str]
    outputs: list[str]


def _is_subgraph_branch(val: AttributeValue | LogicalGraph) -> typing.TypeGuard[LogicalGraph]:
    """Return True if `val` is a subgraph branch exposing `inputs` and `outputs`.

    Args:
        val (AttributeValue | LogicalGraph): Candidate branch attribute value.

    Returns:
        bool: True if `val` conforms to `_SubgraphBranchProtocol`.
    """
    return isinstance(val, _SubgraphBranchProtocol)


def _is_subgraph_with_nodes(val: AttributeValue | LogicalGraph) -> typing.TypeGuard[LogicalGraph]:
    """Return True if `val` is a subgraph exposing `nodes`, `inputs`, and `outputs`.

    Args:
        val (AttributeValue | LogicalGraph): Candidate subgraph value.

    Returns:
        bool: True if `val` conforms to `_SubgraphWithNodesProtocol`.
    """
    return isinstance(val, _SubgraphWithNodesProtocol)


def _extract_graph_nodes(nodes: Mapping[str, IRNode] | Sequence[IRNode]) -> list[IRNode]:
    """Extract a flat list of `IRNode` instances from a graph `nodes` mapping or sequence.

    Args:
        nodes (Mapping[str, IRNode] | Sequence[IRNode]): Graph nodes container.

    Returns:
        list[IRNode]: Flat list of IRNodes.
    """
    if isinstance(nodes, Mapping):
        return list(nodes.values())
    return list(nodes)


def _node_shape_meta(graph: IRGraph, node_id: str) -> tuple[int, ...]:
    """Extract integer shape metadata for `node_id` in `graph` if present.

    Args:
        graph (IRGraph): Computation graph.
        node_id (str): Target node identifier.

    Returns:
        tuple[int, ...]: Shape tuple or empty tuple.
    """
    target = graph.nodes.get(node_id)
    if target is None or target.shape_metadata is None:
        return ()
    return tuple(int(d) for d in target.shape_metadata if isinstance(d, int))


for op_name in [
    "CudaKernel",
    "MetalKernel",
    "PrecompiledCudaKernel",
    "TopK",
    "Cholesky",
    "CholeskyEx",
    "Eig",
    "Eigh",
    "Eigvals",
    "Eigvalsh",
    "FFT",
    "IFFT",
    "Sort",
    "SortComplex",
    "SortKeyVal",
    "Argsort",
    "Fftconvolve",
    "Fft",
    "Rfft",
    "Fft2",
    "Fftfreq",
    "Irfft",
    "Ihfft",
    "Ifft",
    "Fftn",
    "Ifftn",
    "Rfftn",
    "Irfftn",
    "Ifft2",
    "Rfft2",
    "Irfft2",
    "Fftnd",
    "Ifftnd",
    "Rfftnd",
    "Irfftnd",
    "Fftshift",
    "Ifftshift",
    "Hfft",
    "Rfftfreq",
]:
    register_vjp(op_name)(make_zero_vjp(op_name))
    register_jvp(op_name)(make_zero_jvp(op_name))


def _inline_subgraph(graph: IRGraph, subgraph: IRGraph, node: TensorOrScalar, id_map: dict[str, str]) -> None:
    """Inline a subgraph into the main graph.

    Args:
        graph (IRGraph): Input `graph` argument (graph).
        subgraph (IRGraph): Input `subgraph` argument (subgraph).
        node (TensorOrScalar): Input `node` argument (node).
        id_map (dict[str, str]): Input `id_map` argument (id map).
    """
    _ = node

    from ml_switcheroo_compiler.ir.core import clone_logical_node

    new_subgraph = LogicalGraph(name="cp_fwd_inline")
    nodes_to_process: list[IRNode] = _extract_graph_nodes(subgraph.nodes)

    for n in nodes_to_process:
        if n.op_type == "Input":
            continue
        new_n = clone_logical_node(n)
        new_n.id = id_map[n.id]
        new_n.inputs = [id_map.get(inp, inp) for inp in n.inputs]
        new_subgraph.nodes[new_n.id] = new_n
        graph.nodes[new_n.id] = new_n


def _inline_grad_subgraph(
    graph: IRGraph,
    sg_grad: IRGraph,
    sg: IRGraph,
    node: IRNode,
    cotangent_mapping: dict[str, str],
) -> list[str]:
    """Inline the gradient subgraph into the main graph.

    Args:
        graph (IRGraph): Input `graph` argument (graph).
        sg_grad (IRGraph): Input `sg_grad` argument (sg grad).
        sg (IRGraph): Input `sg` argument (sg).
        node (IRNode): Input `node` argument (node).
        cotangent_mapping (dict[str, str]): Input `cotangent_mapping` argument (cotangent mapping).

    Returns:
        list[str]: Inlined output gradient node IDs.
    """
    import uuid

    from ml_switcheroo_compiler.ir.core import clone_logical_node

    grad_id_map = {}
    for n in sg_grad.nodes.values():
        grad_id_map[n.id] = f"cp_bwd_{n.id}_{uuid.uuid4().hex[:6]}"

    for in_id, orig_in_id in zip(sg.inputs, node.inputs):
        grad_id_map[in_id] = orig_in_id

    for _, cot in cotangent_mapping.items():
        grad_id_map[cot] = cot

    for n in sg_grad.nodes.values():
        if n.op_type in ("Input", "Output") or n.id in cotangent_mapping.values():
            continue
        new_n = clone_logical_node(n)
        new_n.id = grad_id_map.get(n.id, n.id)
        new_n.inputs = [grad_id_map.get(inp, inp) for inp in n.inputs]
        graph.nodes[new_n.id] = new_n

    return [grad_id_map.get(out_id, out_id) for out_id in sg_grad.outputs]


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
    if fun is not None and node.inputs:
        from ml_switcheroo_compiler.core.device import Device
        from ml_switcheroo_compiler.core.dtype import DType
        from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
        from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function
        from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

        input_proxies = [
            Tensor(
                ProxyTensor(id=inp, shape=_node_shape_meta(graph, inp)),
                TensorConfig(_node_shape_meta(graph, inp), DType.Float32, Device("cpu")),
            )
            for inp in node.inputs
        ]
        subgraph = _trace_function(fun, tuple(input_proxies), f"cp_recompute_{uuid.uuid4().hex[:6]}")
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
        node (IRNode | Sequence[str] | str): The primal Cond node.
        cotangent (str): Cotangent identifier.

    Returns:
        tuple: Cotangents for condition and operands.
    """
    from ml_switcheroo_compiler.ir.core import IRNode
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
        node (IRNode | Sequence[str] | str): The primal Scan node.
        cotangent (str): Cotangent identifier.

    Returns:
        tuple: Adjoint node outputs.
    """
    from ml_switcheroo_compiler.ir.core import IRNode
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
        node (IRNode | Sequence[str] | str): The primal WhileLoop node.
        cotangent (str): Cotangent identifier.

    Returns:
        tuple: Adjoint loop outputs.
    """
    from ml_switcheroo_compiler.ir.core import IRNode
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


@register_jvp("If")
def _if_jvp(graph: IRGraph, node: IRNode, tangents: Sequence[str] | str) -> str:
    """JVP for If operation.

    Args:
        graph (IRGraph): Input `graph` argument (graph).
        node (IRNode): Input `node` argument (node).
        tangents (Sequence[str] | str): Input `tangents` argument (tangents).

    Returns:
        str: Tangent output node ID.
    """
    from ml_switcheroo_compiler.ir.core import IRNode
    from ml_switcheroo_compiler.transforms.autodiff import jvp

    attrs = node.attributes or {}
    if "then_branch" in attrs and "else_branch" in attrs:
        tb = attrs["then_branch"]
        eb = attrs["else_branch"]
        assert _is_subgraph_branch(tb) and _is_subgraph_branch(eb)
        tangents_seq: list[str] = [str(t) for t in tangents] if isinstance(tangents, (tuple, list)) else [str(tangents)]
        primals = []
        tangent_ids = []
        for i, in_id in enumerate(node.inputs):
            primals.append(in_id)
            if i < len(tangents_seq):
                tangent_ids.append(tangents_seq[i])

        then_jvp = jvp(tb, primals, tangent_ids, tb.outputs)
        else_jvp = jvp(eb, primals, tangent_ids, eb.outputs)

        new_id = f"{node.id}_jvp"
        new_node = IRNode(
            id=new_id,
            op_type="If",
            inputs=list(node.inputs),
            attributes=_make_node_attrs({"then_branch": then_jvp, "else_branch": else_jvp}),
        )
        graph.nodes[new_id] = new_node
        return new_id

    return "mock_tangent"


@register_jvp("Cond")
def _cond_jvp(graph: IRGraph, node: IRNode, tangents: Sequence[str] | str) -> str:
    """JVP for Cond operation.

    Args:
        graph (IRGraph): The computation graph.
        node (IRNode): The primal Cond node.
        tangents (Sequence[str] | str): Input tangent identifiers.

    Returns:
        str: Tangent output node identifier.
    """
    from ml_switcheroo_compiler.ir.core import IRNode
    from ml_switcheroo_compiler.transforms.autodiff import jvp

    attrs = node.attributes or {}
    tb = attrs.get("then_branch", attrs.get("true_branch"))
    eb = attrs.get("else_branch", attrs.get("false_branch"))

    if tb is not None and eb is not None and _is_subgraph_branch(tb) and _is_subgraph_branch(eb):
        primals = list(node.inputs)
        tangent_ids = [str(t) for t in tangents] if isinstance(tangents, (tuple, list)) else [str(tangents)]
        then_jvp = jvp(tb, primals, tangent_ids, tb.outputs)
        else_jvp = jvp(eb, primals, tangent_ids, eb.outputs)

        new_id = f"{node.id}_jvp"
        new_node = IRNode(
            id=new_id,
            op_type="Cond",
            inputs=list(node.inputs),
            attributes=_make_node_attrs(
                {
                    "then_branch": then_jvp,
                    "else_branch": else_jvp,
                    "true_branch": then_jvp,
                    "false_branch": else_jvp,
                }
            ),
            shape_metadata=node.shape_metadata,
        )
        graph.nodes[new_id] = new_node
        return new_id

    return "mock_tangent"


@register_jvp("WhileLoop")
def _while_loop_jvp(graph: IRGraph, node: IRNode, tangents: Sequence[str] | str) -> str:
    """JVP for WhileLoop operation.

    Args:
        graph (IRGraph): The computation graph.
        node (IRNode): The primal WhileLoop node.
        tangents (Sequence[str] | str): Input tangent identifiers.

    Returns:
        str: Tangent output node identifier.
    """
    from ml_switcheroo_compiler.ir.core import IRNode
    from ml_switcheroo_compiler.transforms.autodiff import jvp

    attrs = node.attributes or {}
    body = attrs.get("body", attrs.get("body_subgraph"))
    if body is not None and _is_subgraph_branch(body) and body.inputs and body.outputs:
        primals = list(node.inputs)
        tangent_ids = [str(t) for t in tangents] if isinstance(tangents, (tuple, list)) else [str(tangents)]
        body_jvp = jvp(body, primals, tangent_ids, body.outputs)

        new_id = f"{node.id}_jvp"
        new_node = IRNode(
            id=new_id,
            op_type="WhileLoop",
            inputs=list(node.inputs),
            attributes=_make_node_attrs(
                {
                    "body": body_jvp,
                    "body_subgraph": body_jvp,
                    "cond": attrs.get("cond"),
                }
            ),
            shape_metadata=node.shape_metadata,
        )
        graph.nodes[new_id] = new_node
        return new_id

    return "mock_tangent"


@register_jvp("Loop")
def _loop_jvp(graph: IRGraph, node: TensorOrScalar, tangents: TensorOrScalar) -> str:
    """JVP for Loop operation.

    Args:
        graph (IRGraph): Input `graph` argument (graph).
        node (TensorOrScalar): Input `node` argument (node).
        tangents (TensorOrScalar): Input `tangents` argument (tangents).

    Returns:
        str: Tangent node ID.
    """
    _ = (graph, node, tangents)
    return ""


@register_jvp("Scan")
def _scan_jvp(graph: IRGraph | None, node: IRNode | None, tangents: Sequence[str] | str) -> str:
    """JVP for Scan operation.

    Args:
        graph (IRGraph | None): The computation graph.
        node (IRNode | None): The primal Scan node.
        tangents (Sequence[str] | str): Tangents.

    Returns:
        str: Tangent node ID.
    """
    from ml_switcheroo_compiler.ir.core import IRNode
    from ml_switcheroo_compiler.transforms.autodiff import jvp

    if node is None or graph is None:
        return ""

    attrs = node.attributes or {}
    body = attrs.get("body", attrs.get("body_subgraph"))
    if body is not None and _is_subgraph_branch(body) and body.inputs and body.outputs:
        primals = list(node.inputs)
        tangent_ids = [str(t) for t in tangents] if isinstance(tangents, (tuple, list)) else [str(tangents)]
        body_jvp = jvp(body, primals, tangent_ids, body.outputs)

        new_id = f"{node.id}_jvp"
        new_node = IRNode(
            id=new_id,
            op_type="Scan",
            inputs=list(node.inputs),
            attributes=_make_node_attrs(
                {
                    "body": body_jvp,
                    "body_subgraph": body_jvp,
                    "reverse": attrs.get("reverse", False),
                }
            ),
            shape_metadata=node.shape_metadata,
        )
        graph.nodes[new_id] = new_node
        return new_id

    return ""


@register_jvp("AssociativeScan")
def _assoc_scan_jvp(graph: IRGraph, node: TensorOrScalar, tangents: TensorOrScalar) -> str:
    """JVP for AssociativeScan operation.

    Args:
        graph (IRGraph): Input `graph` argument (graph).
        node (TensorOrScalar): Input `node` argument (node).
        tangents (TensorOrScalar): Input `tangents` argument (tangents).

    Returns:
        str: Tangent node ID.
    """
    _ = (graph, node, tangents)
    return ""


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

    bwd_subgraph = node.subgraphs.get("bwd") or node.attributes.get("bwd_graph") or node.attributes.get("custom_backward_subgraph")
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

    jvp_subgraph = node.subgraphs.get("jvp") or node.attributes.get("jvp_graph") or node.attributes.get("custom_forward_subgraph")
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
        return str(id_map.get(jvp_subgraph.outputs[0], jvp_subgraph.outputs[0])) if jvp_subgraph.outputs else str(tangents_list[0])

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
        primal_block = _trace_function(fun, tuple(input_proxies), f"cjvp_primal_{uuid.uuid4().hex[:6]}")
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


@register_vjp("GetItem")
def getitem_vjp(graph: IRGraph, node: IRNode, cotangent: str) -> tuple[str, ...]:
    """VJP for GetItem indexing operation.

    Args:
        graph (IRGraph): Target computation graph.
        node (IRNode | Sequence[str] | str): GetItem node.
        cotangent (str): Upstream cotangent node ID.

    Returns:
        tuple[str, ...]: Adjoint for input array.
    """
    import uuid

    from ml_switcheroo_ir import LogicalNode

    inp_id = node.inputs[0] if node.inputs else ""
    key = node.attributes.get("key", 0)
    inp_shape = _node_shape_meta(graph, inp_id)

    zero_in_id = f"zero_in_{uuid.uuid4().hex[:6]}"
    graph.nodes[zero_in_id] = LogicalNode(
        id=zero_in_id,
        op_type="ZerosLike",
        inputs=[inp_id] if inp_id else [],
        shape_metadata=inp_shape,
    )

    put_id = f"index_put_{uuid.uuid4().hex[:6]}"
    graph.nodes[put_id] = LogicalNode(
        id=put_id,
        op_type="IndexPut",
        inputs=[zero_in_id, cotangent],
        attributes={"key": key},
        shape_metadata=inp_shape,
    )
    return (put_id,)


@register_jvp("IndexPut")
def index_put_jvp(graph: IRGraph, node: IRNode, tangents: Sequence[str] | str) -> str:
    """JVP for IndexPut operation.

    Args:
        graph (IRGraph): Target computation graph.
        node (IRNode | Sequence[str] | str): IndexPut node.
        tangents (IRNode | Sequence[str] | str): Tangent identifiers.

    Returns:
        str: Tangent node ID.
    """
    import uuid

    from ml_switcheroo_ir import LogicalNode

    tangents_list: list[str] = [str(t) for t in tangents] if isinstance(tangents, (tuple, list)) else [str(tangents)]
    t_target = tangents_list[0] if len(tangents_list) > 0 else "0"
    t_val = tangents_list[1] if len(tangents_list) > 1 else "0"
    out_id = f"index_put_tan_{uuid.uuid4().hex[:6]}"
    graph.nodes[out_id] = LogicalNode(
        id=out_id,
        op_type="IndexPut",
        inputs=[t_target, t_val],
        attributes=dict(node.attributes or {}),
        shape_metadata=node.shape_metadata,
    )
    return out_id


@register_jvp("ZerosLike")
def zeros_like_jvp(graph: IRGraph, node: IRNode, tangents: Sequence[str] | str) -> str:
    """JVP for ZerosLike operation.

    Args:
        graph (IRGraph): Target computation graph.
        node (IRNode | Sequence[str] | str): ZerosLike node.
        tangents (IRNode | Sequence[str] | str): Tangent identifiers.

    Returns:
        str: Tangent node ID.
    """
    import uuid

    from ml_switcheroo_ir import LogicalNode

    zero_id = f"zeros_like_tan_{uuid.uuid4().hex[:6]}"
    graph.nodes[zero_id] = LogicalNode(
        id=zero_id,
        op_type="ZerosLike",
        inputs=list(node.inputs),
        attributes=dict(node.attributes or {}),
        shape_metadata=node.shape_metadata,
    )
    return zero_id


@register_vjp("LogPoissonLoss")
def log_poisson_loss_vjp(graph: IRGraph, node: IRNode, cotangent: str) -> tuple[str, ...]:
    """VJP for LogPoissonLoss operation.

    Args:
        graph (IRGraph): Target computation graph.
        node (IRNode | Sequence[str] | str): LogPoissonLoss node.
        cotangent (str): Upstream cotangent node ID.

    Returns:
        tuple[str, ...]: Adjoint node IDs for targets and log_input arguments.
    """
    import uuid

    from ml_switcheroo_ir import LogicalNode

    targets_id: str = node.inputs[0] if len(node.inputs) > 0 else ""
    log_input_id: str = node.inputs[1] if len(node.inputs) > 1 else ""
    attrs: dict[str, AttributeValue] = dict(node.attributes or {})
    log_input_flag: bool = bool(attrs.get("log_input", True))

    exp_id = f"exp_{uuid.uuid4().hex[:6]}"
    sub_id = f"sub_{uuid.uuid4().hex[:6]}"
    mul_id = f"mul_{uuid.uuid4().hex[:6]}"

    if log_input_flag:
        graph.nodes[exp_id] = LogicalNode(id=exp_id, op_type="Exp", inputs=[log_input_id], shape_metadata=node.shape_metadata)
        graph.nodes[sub_id] = LogicalNode(id=sub_id, op_type="Sub", inputs=[exp_id, targets_id], shape_metadata=node.shape_metadata)
        graph.nodes[mul_id] = LogicalNode(id=mul_id, op_type="Mul", inputs=[sub_id, cotangent], shape_metadata=node.shape_metadata)
    else:
        one_id = f"one_{uuid.uuid4().hex[:6]}"
        div_id = f"div_{uuid.uuid4().hex[:6]}"
        graph.nodes[one_id] = LogicalNode(id=one_id, op_type="Constant", attributes={"value": 1.0}, shape_metadata=node.shape_metadata)
        graph.nodes[div_id] = LogicalNode(id=div_id, op_type="Div", inputs=[targets_id, log_input_id], shape_metadata=node.shape_metadata)
        graph.nodes[sub_id] = LogicalNode(id=sub_id, op_type="Sub", inputs=[one_id, div_id], shape_metadata=node.shape_metadata)
        graph.nodes[mul_id] = LogicalNode(id=mul_id, op_type="Mul", inputs=[sub_id, cotangent], shape_metadata=node.shape_metadata)

    target_zero_id = f"zero_{uuid.uuid4().hex[:6]}"
    graph.nodes[target_zero_id] = LogicalNode(id=target_zero_id, op_type="ZerosLike", inputs=[targets_id] if targets_id else [], shape_metadata=node.shape_metadata)
    return (target_zero_id, mul_id)


@register_jvp("LogPoissonLoss")
def log_poisson_loss_jvp(graph: IRGraph, node: IRNode, tangents: Sequence[str] | str) -> str:
    """JVP for LogPoissonLoss operation.

    Args:
        graph (IRGraph): Target computation graph.
        node (IRNode | Sequence[str] | str): LogPoissonLoss node.
        tangents (IRNode | Sequence[str] | str): Tangent identifiers.

    Returns:
        str: Tangent node ID.
    """
    import uuid

    from ml_switcheroo_ir import LogicalNode

    targets_id: str = node.inputs[0] if len(node.inputs) > 0 else ""
    log_input_id: str = node.inputs[1] if len(node.inputs) > 1 else ""
    t_tangent: str = tangents[1] if len(tangents) > 1 else tangents[0]

    attrs: dict[str, AttributeValue] = dict(node.attributes or {})
    log_input_flag: bool = bool(attrs.get("log_input", True))

    exp_id = f"exp_tan_{uuid.uuid4().hex[:6]}"
    sub_id = f"sub_tan_{uuid.uuid4().hex[:6]}"
    mul_id = f"mul_tan_{uuid.uuid4().hex[:6]}"

    if log_input_flag:
        graph.nodes[exp_id] = LogicalNode(id=exp_id, op_type="Exp", inputs=[log_input_id], shape_metadata=node.shape_metadata)
        graph.nodes[sub_id] = LogicalNode(id=sub_id, op_type="Sub", inputs=[exp_id, targets_id], shape_metadata=node.shape_metadata)
        graph.nodes[mul_id] = LogicalNode(id=mul_id, op_type="Mul", inputs=[sub_id, t_tangent], shape_metadata=node.shape_metadata)
    else:
        one_id = f"one_tan_{uuid.uuid4().hex[:6]}"
        div_id = f"div_tan_{uuid.uuid4().hex[:6]}"
        graph.nodes[one_id] = LogicalNode(id=one_id, op_type="Constant", attributes={"value": 1.0}, shape_metadata=node.shape_metadata)
        graph.nodes[div_id] = LogicalNode(id=div_id, op_type="Div", inputs=[targets_id, log_input_id], shape_metadata=node.shape_metadata)
        graph.nodes[sub_id] = LogicalNode(id=sub_id, op_type="Sub", inputs=[one_id, div_id], shape_metadata=node.shape_metadata)
        graph.nodes[mul_id] = LogicalNode(id=mul_id, op_type="Mul", inputs=[sub_id, t_tangent], shape_metadata=node.shape_metadata)

    return mul_id
