"""Shape rules for shape."""

from ml_switcheroo_compiler.ops.base import emit_ir_node
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import register_jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp


@register_vjp("Reshape")
def reshape_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Compute the Vector-Jacobian Product (VJP) for a Reshape operation.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    x = node.inputs[0]
    return (
        emit_ir_node(
            graph,
            "Reshape",
            [cotangent],
            graph.nodes[x].shape_metadata,
            attributes={"newshape": graph.nodes[x].shape_metadata},
        ),
    )


@register_jvp("Reshape")
def reshape_jvp(graph: object, node: object, tangent: str) -> str:
    """Compute the Jacobian-Vector Product (JVP) for a Reshape operation.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        tangent (str): The tangent parameter.

    Returns:
        str: Result.
    """
    return emit_ir_node(
        graph,
        "Reshape",
        [tangent],
        node.shape_metadata,
        attributes={"newshape": node.attributes.get("newshape")},
    )


@register_vjp("Transpose")
def transpose_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Compute the Vector-Jacobian Product (VJP) for a Transpose operation.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    x = node.inputs[0]
    axes = node.attributes.get("axes")
    if axes is not None:
        rev_axes = [0] * len(axes)
        for i, ax in enumerate(axes):
            rev_axes[ax] = i
    else:
        rev_axes = None
    return (
        emit_ir_node(
            graph,
            "Transpose",
            [cotangent],
            graph.nodes[x].shape_metadata,
            attributes={"axes": rev_axes},
        ),
    )


@register_jvp("Transpose")
def transpose_jvp(graph: object, node: object, tangent: str) -> str:
    """Compute the Jacobian-Vector Product (JVP) for a Transpose operation.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        tangent (str): The tangent parameter.

    Returns:
        str: Result.
    """
    return emit_ir_node(
        graph,
        "Transpose",
        [tangent],
        node.shape_metadata,
        attributes={"axes": node.attributes.get("axes")},
    )


@register_vjp("BroadcastTo")
def broadcast_to_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Compute the Vector-Jacobian Product (VJP) for the BroadcastTo operation.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    x = node.inputs[0]
    res = emit_ir_node(graph, "Sum", [cotangent], graph.nodes[x].shape_metadata)
    return (res,)


@register_jvp("BroadcastTo")
def broadcast_to_jvp(graph: object, node: object, tangent: str) -> str:
    """Compute the Jacobian-Vector Product (JVP) for a BroadcastTo operation.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        tangent (str): The tangent parameter.

    Returns:
        str: Result.
    """
    return emit_ir_node(
        graph,
        "BroadcastTo",
        [tangent],
        node.shape_metadata,
        attributes={"shape": node.attributes.get("shape")},
    )


@register_vjp("Split")
def split_vjp(graph: object, node: object, cotangents: tuple) -> tuple:
    """VJP for Split.

    Args:
        graph: Graph.
        node: Node.
        cotangents: Cotangents.

    Returns:
        tuple: Zeros or gradients.
    """
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    if not cotangents:
        return ()

    axis = node.attributes.get("axis", 0) if hasattr(node, "attributes") else 0

    concat_grad = emit_ir_node(graph, "Concat", inputs=list(cotangents), attributes={"axis": axis})

    return (concat_grad,)


@register_jvp("Split")
def split_jvp(graph: object, node: object, tangent: object) -> str:
    """JVP for Split.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        tangent (object): The tangent parameter.

    Returns:
        str: Result.
    """
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    # Split the tangent exactly the same way the primal is split
    t_x = tangent[0] if isinstance(tangent, (tuple, list)) else tangent
    return emit_ir_node(graph, "Split", [t_x], node.shape_metadata, attributes=node.attributes)


@register_jvp("GetItem")
def getitem_jvp(graph: object, node: object, tangent: object) -> str:
    """Compute the Jacobian-Vector Product (JVP) for a GetItem operation.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        tangent (object): The tangent parameter.

    Returns:
        str: Result.
    """
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    tangent_x = tangent[0] if isinstance(tangent, (tuple, list)) else tangent
    key = node.attributes.get("key", "0")
    return emit_ir_node(
        graph,
        "GetItem",
        [tangent_x],
        node.shape_metadata,
        attributes={"key": key},
    )


@register_vjp("GetItem")
def getitem_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Compute the Vector-Jacobian Product (VJP) for a GetItem operation.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    x = node.inputs[0]
    key = node.attributes.get("key", "0")
    zeros = emit_ir_node(graph, "ZerosLike", [x], graph.nodes[x].shape_metadata)
    return (
        emit_ir_node(
            graph,
            "SetItem",
            [zeros, cotangent],
            graph.nodes[x].shape_metadata,
            attributes={"key": key},
        ),
    )


@register_jvp("SetItem")
def setitem_jvp(graph: object, node: object, tangent: object) -> str:
    """Compute the Jacobian-Vector Product (JVP) for a SetItem operation.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        tangent (object): The tangent parameter.

    Returns:
        str: Result.
    """
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    # inputs are [x, value]
    tangent_x = tangent[0] if isinstance(tangent, (tuple, list)) else tangent
    # Check if there is a second tangent (for the value argument)
    if isinstance(tangent, (tuple, list)) and len(tangent) > 1:
        tangent_val = tangent[1]
    else:
        # Create zero tangent if value is constant/non-differentiable
        value = node.inputs[1]
        tangent_val = emit_ir_node(graph, "ZerosLike", [value], graph.nodes[value].shape_metadata)

    key = node.attributes.get("key", "0")
    return emit_ir_node(
        graph,
        "SetItem",
        [tangent_x, tangent_val],
        node.shape_metadata,
        attributes={"key": key},
    )


@register_vjp("SetItem")
def setitem_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Compute the Vector-Jacobian Product (VJP) for a SetItem operation.

    Args:
        graph (object): The graph parameter.
        node (object): The node parameter.
        cotangent (str): The cotangent parameter.

    Returns:
        tuple: Result.
    """
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    x = node.inputs[0]
    value = node.inputs[1]
    key = node.attributes.get("key", "0")

    # Grad wrt x: set key slice of cotangent to zero
    zeros_val = emit_ir_node(graph, "ZerosLike", [value], graph.nodes[value].shape_metadata)
    grad_x = emit_ir_node(graph, "SetItem", [cotangent, zeros_val], graph.nodes[x].shape_metadata, attributes={"key": key})

    # Grad wrt value: extract key slice from cotangent
    grad_value = emit_ir_node(graph, "GetItem", [cotangent], graph.nodes[value].shape_metadata, attributes={"key": key})

    return grad_x, grad_value
