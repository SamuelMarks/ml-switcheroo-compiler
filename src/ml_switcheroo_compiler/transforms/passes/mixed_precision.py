"""Module mixed_precision.py."""

from typing import Any

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Mixed Precision and Loss Scaling passes."""

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode, clone_logical_node
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter

# Ops that are safe and beneficial to run in float16/bfloat16
SAFE_FP16_OPS: set[str] = {"MatMul", "Conv2D", "BatchMatMul"}

# Ops that might cause underflow/overflow or loss of precision and should run in fp32
FP32_OPS: set[str] = {"Exp", "Log", "Softmax", "ReduceSum"}


def _cast_inputs(node: IRNode, graph: IRGraph, target_dtype: str, new_nodes: dict[str, IRNode]) -> tuple[list[str], bool]:
    """Cast inputs of the given node to the target dtype.

    Args:
        node (IRNode): The node parameter.
        graph (IRGraph): The graph parameter.
        target_dtype (str): The target_dtype parameter.
        new_nodes (dict): The new_nodes parameter.

    Returns:
        tuple: Result.
    """
    new_inputs = []
    modified = False
    for inp_id in node.inputs:
        inp_node = new_nodes.get(inp_id) or graph.nodes.get(inp_id)
        current_dtype = inp_node.attributes.get("dtype", "float32") if inp_node else "float32"
        if current_dtype != target_dtype:
            # Insert a cast
            cast_id = f"{inp_id}_cast_{target_dtype}"
            if cast_id not in new_nodes:
                cast_node = IRNode(
                    id=cast_id,
                    op_type="Cast",
                    inputs=[inp_id],
                    attributes={"to": target_dtype},
                )
                new_nodes[cast_id] = cast_node
                modified = True
            new_inputs.append(cast_id)
        else:
            new_inputs.append(inp_id)
    return new_inputs, modified


def _process_node(node: IRNode, graph: IRGraph, target_dtype: str, new_nodes: dict[str, IRNode]) -> bool:
    """Process a single node and cast it to the target dtype if needed.

    Args:
        node (IRNode): The node parameter.
        graph (IRGraph): The graph parameter.
        target_dtype (str): The target_dtype parameter.
        new_nodes (dict): The new_nodes parameter.

    Returns:
        bool: Result.
    """
    modified = False
    new_inputs, inputs_modified = _cast_inputs(node, graph, target_dtype, new_nodes)
    if inputs_modified:
        modified = True

    if new_inputs != node.inputs:
        new_node = clone_logical_node(node, inputs=new_inputs)
        new_node.attributes["dtype"] = target_dtype
        new_nodes[node.id] = new_node
        modified = True
    elif node.attributes.get("dtype") != target_dtype:
        node.attributes["dtype"] = target_dtype
        modified = True
    return modified


def mixed_precision_pass(graph: IRGraph, target_dtype: str = "float16") -> bool:
    """In-place Mixed Precision Casting pass.

    Automatically inserts Cast operations to lower precision (e.g., float16)
    for compute-heavy safe ops, and upcasts back for precision-sensitive ops.

    Args:
        graph (IRGraph): The input graph to mutate.
        target_dtype (str): The target lower precision dtype.

    Returns:
        bool: True if the graph was modified, False otherwise.
    """
    sorted_nodes = DAGTopologicalSorter.sort(graph)
    if not sorted_nodes:
        return False

    new_nodes: dict[str, IRNode] = dict(graph.nodes)
    graph_modified = False

    for node in sorted_nodes:
        if node.op_type in SAFE_FP16_OPS:
            if _process_node(node, graph, target_dtype, new_nodes):
                graph_modified = True
        elif node.op_type in FP32_OPS:
            if _process_node(node, graph, "float32", new_nodes):
                graph_modified = True

    if graph_modified:
        graph.nodes.clear()
        graph.nodes.update(new_nodes)
        return True

    return False


def _get_scale_nodes(new_nodes: dict[str, IRNode], scale: float) -> tuple[str, str]:
    """Retrieve or create scaling constant nodes.

    Args:
        new_nodes (Any): The new_nodes parameter.
        scale (float): The scale parameter.

    Returns: Any: Result.
    """
    scale_node_id = "loss_scale_factor"
    inv_scale_node_id = "loss_scale_inv_factor"
    if scale_node_id not in new_nodes:
        new_nodes[scale_node_id] = IRNode(
            id=scale_node_id,
            op_type="Constant",
            attributes={"value": scale},
        )
    if inv_scale_node_id not in new_nodes:
        new_nodes[inv_scale_node_id] = IRNode(
            id=inv_scale_node_id,
            op_type="Constant",
            attributes={"value": 1.0 / scale},
        )
    return scale_node_id, inv_scale_node_id


def _scale_inputs(grad_inputs: list[str], new_nodes: dict[str, IRNode], scale_node_id: str) -> None:
    """Scale the given gradient inputs by the scale factor.

    Args:
        grad_inputs (list): The grad_inputs parameter.
        new_nodes (dict): The new_nodes parameter.
        scale_node_id (str): The scale_node_id parameter.
    """
    for g_id in grad_inputs:
        mul_id = f"{g_id}_scaled"
        if mul_id not in new_nodes:
            new_nodes[mul_id] = IRNode(id=mul_id, op_type="Mul", inputs=[g_id, scale_node_id])

        # Replace uses of g_id with mul_id
        for n_id, n in list(new_nodes.items()):
            if n_id == mul_id:
                continue
            if g_id in n.inputs:
                new_inputs = [mul_id if i == g_id else i for i in n.inputs]
                new_nodes[n_id] = clone_logical_node(n, inputs=new_inputs)


def _unscale_outputs(graph: IRGraph, new_nodes: dict[str, IRNode], inv_scale_node_id: str) -> list[str]:
    """Unscale the final outputs using the inverse scale factor.

    Args:
        graph (IRGraph): The graph parameter.
        new_nodes (dict): The new_nodes parameter.
        inv_scale_node_id (str): The inv_scale_node_id parameter.

    Returns:
        list: Result.
    """
    new_outputs = []
    for out_id in graph.outputs:
        unscaled_id = f"{out_id}_unscaled"
        if unscaled_id not in new_nodes:
            new_nodes[unscaled_id] = IRNode(id=unscaled_id, op_type="Mul", inputs=[out_id, inv_scale_node_id])
        new_outputs.append(unscaled_id)
    return new_outputs


def loss_scaling_pass(graph: IRGraph, scale: float = 1024.0) -> bool:
    """Implement loss scaling logic for gradient graphs to prevent underflow.

    This pass searches for gradients inputs (usually identified by metadata)
    and multiplies them by the scale factor before they are used, and
    divides the final gradients by the scale factor.

    Args:
        graph (IRGraph): The input graph to mutate.
        scale (float): The scaling factor.

    Returns:
        bool: True if modified, False otherwise.
    """
    grad_inputs = []
    for node in graph.nodes.values():
        if node.op_type == "Input" and node.attributes.get("is_grad") is True:
            grad_inputs.append(node.id)

    if not grad_inputs:
        return False

    new_nodes = dict(graph.nodes)
    scale_node_id, inv_scale_node_id = _get_scale_nodes(new_nodes, scale)

    _scale_inputs(grad_inputs, new_nodes, scale_node_id)

    new_outputs = _unscale_outputs(graph, new_nodes, inv_scale_node_id)

    graph.outputs = new_outputs
    graph.nodes.clear()
    graph.nodes.update(new_nodes)

    return True
