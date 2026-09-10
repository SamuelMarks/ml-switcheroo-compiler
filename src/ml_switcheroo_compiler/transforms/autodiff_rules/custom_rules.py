# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Autodiff rules for custom and platform-specific kernel operations."""

from collections.abc import Sequence
from typing import Any, Optional, Union

from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.transforms.autodiff_rules.common import make_zero_jvp, make_zero_vjp
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import register_jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp

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


def _inline_subgraph(graph: IRGraph, subgraph: IRGraph, node, id_map: dict[str, str]) -> None:
    """Inline a subgraph into the main graph.

    Args:
        graph (object): The graph parameter.
        subgraph (object): The subgraph parameter.
        node (object): The node parameter.
        id_map (dict): The id_map parameter.
    """
    from ml_switcheroo_ir import LogicalGraph

    from ml_switcheroo_compiler.ir.core import clone_logical_node

    new_subgraph = LogicalGraph(name="cp_fwd_inline")
    nodes_to_process = subgraph.nodes if isinstance(subgraph.nodes, list) else subgraph.nodes.values()

    for n in nodes_to_process:
        if n.op_type == "Input":
            continue
        new_n = clone_logical_node(n)
        new_n.id = id_map[n.id]
        new_n.inputs = [id_map.get(inp, inp) for inp in n.inputs]
        new_subgraph.nodes[new_n.id] = new_n
        graph.nodes[new_n.id] = new_n


def _inline_grad_subgraph(graph: IRGraph, sg_grad, sg, node, cotangent_mapping: dict[str, str]) -> list[str]:
    """Inline the gradient subgraph into the main graph.

    Args:
        graph (object): The graph parameter.
        sg_grad (object): The sg_grad parameter.
        sg (object): The sg parameter.
        node (object): The node parameter.
        cotangent_mapping (object): The cotangent_mapping parameter.

    Returns:
            tuple[int, ...]: Result.
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
def checkpoint_vjp(graph: IRGraph, node: Any, cotangent: str) -> Any:
    """VJP for Checkpoint operation.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    import uuid

    from ml_switcheroo_ir import LogicalGraph

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
                ProxyTensor(id=inp, shape=getattr(graph.nodes.get(inp), "shape_metadata", ())),
                TensorConfig(getattr(graph.nodes.get(inp), "shape_metadata", ()), DType.Float32, Device("cpu")),
            )
            for inp in node.inputs
        ]
        subgraph = _trace_function(fun, tuple(input_proxies), f"cp_recompute_{uuid.uuid4().hex[:6]}")
    else:
        subgraph = node.attributes["subgraph"]

    nodes_list = subgraph.nodes if isinstance(subgraph.nodes, list) else subgraph.nodes.values()

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
def _if_vjp(graph: IRGraph, node: Any, cotangent: str) -> Any:
    """VJP for If operation.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients

    return (UnconnectedGradients.ZERO,)


@register_vjp("Loop")
def _loop_vjp(graph: IRGraph, node: Any, cotangent: str) -> Any:
    """VJP for Loop operation.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients

    return tuple(UnconnectedGradients.ZERO for _ in node.inputs)


@register_vjp("Cond")
def _cond_vjp(graph: IRGraph, node: Any, cotangent: str) -> tuple:
    """VJP for Cond operation lowering into adjoint control flow structure.

    Args:
        graph (IRGraph): The computation graph.
        node (Any): The primal Cond node.
        cotangent (str): Cotangent identifier.

    Returns:
        tuple: Cotangents for condition and operands.
    """
    from ml_switcheroo_compiler.ir.core import IRNode
    from ml_switcheroo_compiler.transforms.autodiff import grad
    from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients

    tb = node.attributes.get("then_branch", node.attributes.get("true_branch"))
    eb = node.attributes.get("else_branch", node.attributes.get("false_branch"))

    if tb is not None and eb is not None and hasattr(tb, "inputs") and hasattr(eb, "inputs"):
        tb_grad = grad(tb, tb.inputs, tb.outputs[0] if tb.outputs else "", cotangent_id=cotangent)
        eb_grad = grad(eb, eb.inputs, eb.outputs[0] if eb.outputs else "", cotangent_id=cotangent)

        adj_id = f"{node.id}_adj_cond"
        adj_node = IRNode(
            id=adj_id,
            op_type="Cond",
            inputs=list(node.inputs),
            attributes={
                "then_branch": tb_grad,
                "else_branch": eb_grad,
                "true_branch": tb_grad,
                "false_branch": eb_grad,
            },
            shape_metadata=node.shape_metadata,
        )
        graph.nodes[adj_id] = adj_node
        return tuple([UnconnectedGradients.ZERO] + [adj_id for _ in node.inputs[1:]])

    return tuple(UnconnectedGradients.ZERO for _ in node.inputs)


@register_vjp("Scan")
def _scan_vjp(graph: IRGraph, node: Any, cotangent: str) -> Any:
    """VJP for Scan operation lowering into adjoint reverse scan.

    Args:
        graph (IRGraph): The computation graph.
        node (Any): The primal Scan node.
        cotangent (str): Cotangent identifier.

    Returns:
        tuple: Adjoint node outputs.
    """
    from ml_switcheroo_compiler.ir.core import IRNode
    from ml_switcheroo_compiler.transforms.autodiff import grad
    from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients

    body = node.attributes.get("body", node.attributes.get("body_subgraph"))
    if body is not None and hasattr(body, "inputs") and hasattr(body, "outputs"):
        body_grad = grad(body, body.inputs, body.outputs[0] if body.outputs else "", cotangent_id=cotangent)
        adj_id = f"{node.id}_adj_scan"
        adj_node = IRNode(
            id=adj_id,
            op_type="Scan",
            inputs=list(node.inputs),
            attributes={
                "body": body_grad,
                "body_subgraph": body_grad,
                "reverse": not node.attributes.get("reverse", False),
            },
            shape_metadata=node.shape_metadata,
        )
        graph.nodes[adj_id] = adj_node
        return tuple(adj_id for _ in node.inputs)

    return tuple(UnconnectedGradients.ZERO for _ in node.inputs)


@register_vjp("WhileLoop")
def _while_loop_vjp(graph: IRGraph, node: Any, cotangent: str) -> Any:
    """VJP for WhileLoop operation lowering into adjoint control flow structure.

    Args:
        graph (IRGraph): The computation graph.
        node (Any): The primal WhileLoop node.
        cotangent (str): Cotangent identifier.

    Returns:
        tuple: Adjoint loop outputs.
    """
    from ml_switcheroo_compiler.ir.core import IRNode
    from ml_switcheroo_compiler.transforms.autodiff import grad
    from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients

    body = node.attributes.get("body", node.attributes.get("body_subgraph"))
    cond = node.attributes.get("cond", node.attributes.get("cond_graph"))
    if body is not None and hasattr(body, "inputs") and hasattr(body, "outputs"):
        body_grad = grad(body, body.inputs, body.outputs[0] if body.outputs else "", cotangent_id=cotangent)
        adj_id = f"{node.id}_adj_while"
        adj_node = IRNode(
            id=adj_id,
            op_type="WhileLoop",
            inputs=list(node.inputs),
            attributes={
                "body": body_grad,
                "body_subgraph": body_grad,
                "cond": cond,
                "adjoint": True,
            },
            shape_metadata=node.shape_metadata,
        )
        graph.nodes[adj_id] = adj_node
        return tuple(adj_id for _ in node.inputs)

    return tuple(UnconnectedGradients.ZERO for _ in node.inputs)


@register_vjp("AssociativeScan")
def _assoc_scan_vjp(graph: IRGraph, node: Any, cotangent: str) -> Any:
    """VJP for AssociativeScan operation.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    from ml_switcheroo_compiler.transforms.autodiff_rules.common import UnconnectedGradients

    return tuple(UnconnectedGradients.ZERO for _ in node.inputs)


@register_jvp("If")
def _if_jvp(graph: IRGraph, node, tangents) -> str:
    """JVP for If operation."""
    from ml_switcheroo_compiler.ir.core import IRNode
    from ml_switcheroo_compiler.transforms.autodiff import jvp

    if "then_branch" in node.attributes and "else_branch" in node.attributes:
        tb = node.attributes["then_branch"]
        eb = node.attributes["else_branch"]

        primals = []
        tangent_ids = []
        for i, in_id in enumerate(node.inputs):
            primals.append(in_id)
            if i < len(tangents):
                tangent_ids.append(tangents[i])

        then_jvp = jvp(tb, primals, tangent_ids, tb.outputs)
        else_jvp = jvp(eb, primals, tangent_ids, eb.outputs)

        new_id = f"{node.id}_jvp"
        new_node = IRNode(id=new_id, op_type="If", inputs=node.inputs, attributes={"then_branch": then_jvp, "else_branch": else_jvp})
        graph.nodes[new_id] = new_node
        return new_id

    return "mock_tangent"


@register_jvp("Cond")
def _cond_jvp(graph: IRGraph, node, tangents) -> str:
    """JVP for Cond operation.

    Args:
        graph (IRGraph): The computation graph.
        node (Any): The primal Cond node.
        tangents (Sequence[str]): Input tangent identifiers.

    Returns:
        str: Tangent output node identifier.
    """
    from ml_switcheroo_compiler.ir.core import IRNode
    from ml_switcheroo_compiler.transforms.autodiff import jvp

    tb = node.attributes.get("then_branch", node.attributes.get("true_branch"))
    eb = node.attributes.get("else_branch", node.attributes.get("false_branch"))

    if tb is not None and eb is not None and hasattr(tb, "inputs") and hasattr(eb, "inputs"):
        primals = list(node.inputs)
        tangent_ids = list(tangents)
        then_jvp = jvp(tb, primals, tangent_ids, tb.outputs)
        else_jvp = jvp(eb, primals, tangent_ids, eb.outputs)

        new_id = f"{node.id}_jvp"
        new_node = IRNode(
            id=new_id,
            op_type="Cond",
            inputs=node.inputs,
            attributes={
                "then_branch": then_jvp,
                "else_branch": else_jvp,
                "true_branch": then_jvp,
                "false_branch": else_jvp,
            },
            shape_metadata=node.shape_metadata,
        )
        graph.nodes[new_id] = new_node
        return new_id

    return "mock_tangent"


@register_jvp("WhileLoop")
def _while_loop_jvp(graph: IRGraph, node, tangents) -> str:
    """JVP for WhileLoop operation.

    Args:
        graph (IRGraph): The computation graph.
        node (Any): The primal WhileLoop node.
        tangents (Sequence[str]): Input tangent identifiers.

    Returns:
        str: Tangent output node identifier.
    """
    from ml_switcheroo_compiler.ir.core import IRNode
    from ml_switcheroo_compiler.transforms.autodiff import jvp

    body = node.attributes.get("body", node.attributes.get("body_subgraph"))
    if body is not None and hasattr(body, "inputs") and hasattr(body, "outputs"):
        primals = list(node.inputs)
        tangent_ids = list(tangents)
        body_jvp = jvp(body, primals, tangent_ids, body.outputs)

        new_id = f"{node.id}_jvp"
        new_node = IRNode(
            id=new_id,
            op_type="WhileLoop",
            inputs=node.inputs,
            attributes={
                "body": body_jvp,
                "body_subgraph": body_jvp,
                "cond": node.attributes.get("cond"),
            },
            shape_metadata=node.shape_metadata,
        )
        graph.nodes[new_id] = new_node
        return new_id

    return "mock_tangent"


@register_jvp("Loop")
def _loop_jvp(graph: IRGraph, node, tangents) -> str:
    """JVP for Loop operation.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        tangents (list): The tangents parameter.

    Returns:
        str: Result.
    """
    return ""


@register_jvp("Scan")
def _scan_jvp(graph: IRGraph, node, tangents) -> str:
    """JVP for Scan operation.

    Args:
        graph (IRGraph): The computation graph.
        node (Any): The primal Scan node.
        tangents (Sequence[str]): Tangents.

    Returns:
        str: Tangent node ID.
    """
    from ml_switcheroo_compiler.ir.core import IRNode
    from ml_switcheroo_compiler.transforms.autodiff import jvp

    if node is None or graph is None:
        return ""

    body = node.attributes.get("body", node.attributes.get("body_subgraph"))
    if body is not None and hasattr(body, "inputs") and hasattr(body, "outputs"):
        primals = list(node.inputs)
        tangent_ids = list(tangents)
        body_jvp = jvp(body, primals, tangent_ids, body.outputs)

        new_id = f"{node.id}_jvp"
        new_node = IRNode(
            id=new_id,
            op_type="Scan",
            inputs=node.inputs,
            attributes={
                "body": body_jvp,
                "body_subgraph": body_jvp,
                "reverse": node.attributes.get("reverse", False),
            },
            shape_metadata=node.shape_metadata,
        )
        graph.nodes[new_id] = new_node
        return new_id

    return ""


@register_jvp("AssociativeScan")
def _assoc_scan_jvp(graph: IRGraph, node, tangents) -> str:
    """JVP for AssociativeScan operation.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        tangents (list): The tangents parameter.

    Returns:
        str: Result.
    """
    return ""


@register_vjp("Recompute")
def recompute_vjp(graph: IRGraph, node: Any, cotangent: str) -> Any:
    """VJP for Recompute operation.

    Delegates to the original operation's VJP rule.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import get_vjp

    orig_op = node.attributes.get("original_op", "Unknown")

    # We must construct a dummy node that looks like the original node
    from ml_switcheroo_compiler.ir.core import clone_logical_node

    dummy = clone_logical_node(node)
    dummy.op_type = orig_op
    dummy.attributes = node.attributes.get("original_attrs", {})

    vjp_func = get_vjp(orig_op)
    res = vjp_func(graph, dummy, cotangent)
    return res


@register_vjp("CustomVJP")
def custom_vjp_vjp(graph: IRGraph, node: Any, cotangent: str) -> Any:
    """VJP for CustomVJP operation.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    import uuid

    from ml_switcheroo_ir import LogicalNode

    bwd_fn = node.attributes["bwd_fn"]

    bwd_node_id = f"cvjp_bwd_{uuid.uuid4().hex[:6]}"
    bwd_node = LogicalNode(
        id=bwd_node_id,
        op_type="ProcessCustomVJPCall",
        inputs=[cotangent],
        attributes={"bwd_fn": bwd_fn},
        shape_metadata=node.shape_metadata,
    )
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
def custom_jvp_jvp(graph: IRGraph, node: Any, tangents: Any) -> Any:
    """JVP for CustomJVP operation.

    Args:
        graph (object): The target IR graph.
        node (object): The CustomJVP node.
        tangents (object): Tangents for inputs.

    Returns:
        str: Output tangent node ID.
    """
    import uuid

    from ml_switcheroo_ir import LogicalNode

    jvp_rule = node.attributes.get("jvp_rule")
    tangents_list = list(tangents) if isinstance(tangents, (tuple, list)) else [tangents]

    call_id = f"cjvp_call_{uuid.uuid4().hex[:6]}"
    call_node = LogicalNode(
        id=call_id,
        op_type="ProcessCustomJVPCall",
        inputs=list(node.inputs) + tangents_list,
        attributes={"jvp_rule": jvp_rule, "num_primals": len(node.inputs), "fun": node.attributes.get("fun")},
        shape_metadata=node.shape_metadata,
    )
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
def custom_jvp_vjp(graph: IRGraph, node: Any, cotangent: str) -> Any:
    """VJP for CustomJVP operation.

    Args:
        graph (object): The target IR graph.
        node (object): The CustomJVP node.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Adjoints list for inputs.
    """
    import uuid

    from ml_switcheroo_ir import LogicalGraph

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
                ProxyTensor(id=inp, shape=getattr(graph.nodes.get(inp), "shape_metadata", ())),
                TensorConfig(getattr(graph.nodes.get(inp), "shape_metadata", ()), DType.Float32, Device("cpu")),
            )
            for inp in node.inputs
        ]
        primal_block = _trace_function(fun, tuple(input_proxies), f"cjvp_primal_{uuid.uuid4().hex[:6]}")
        sg = LogicalGraph(name="cjvp_sg")
        nodes_list = primal_block.nodes if isinstance(primal_block.nodes, list) else primal_block.nodes.values()
        for n in nodes_list:
            sg.nodes[n.id] = clone_logical_node(n)
        sg.inputs = primal_block.inputs
        sg.outputs = primal_block.outputs

        sg_grad = graph_grad(sg, wrt=sg.inputs, output_id=sg.outputs[0], cotangent_id=cotangent)
        cotangent_mapping = {sg.outputs[0]: cotangent}
        adjoints = _inline_grad_subgraph(graph, sg_grad, sg, node, cotangent_mapping)
        return tuple(adjoints)

    return tuple(cotangent for _ in node.inputs)
