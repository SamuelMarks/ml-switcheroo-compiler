"""Defines Vector-Jacobian Product (VJP) and Jacobian-Vector Product (JVP) rules for.

shape-

manipulation operations

This module registers differentiation rules for operations like Reshape, Transpose, and
BroadcastTo, allowing the autodiff system to propagate gradients through shape changes
"""

from ml_switcheroo_compiler.ops.base import emit_ir_node
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import register_jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp


@register_vjp("Reshape")
def reshape_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for a Reshape operation.

    This rule reshapes the incoming cotangent back to the original shape of the
    input tensor

    Args:
        graph (object): The computation graph
        node (object): The Reshape node
        cotangent (str): The cotangent variable name

    Returns:
    tuple: A tuple containing the name of the emitted node representing the
    reshaped cotangent
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
    """Computes the Jacobian-Vector Product (JVP) for a Reshape operation.

    Generates a JAX-compatible code string that reshapes the tangent vector to
    the target shape

    Args:
        graph (object): The computation graph
        node (object): The Reshape node
        tangent (str): The tangent variable name

    Returns:
    str: A code string representing the JAX reshape operation
    """
    return f"jnp.reshape({tangent}, {node.attributes['newshape']})"


@register_vjp("Transpose")
def transpose_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for a Transpose operation.

    This rule transposes the incoming cotangent using the inverse permutation
    of the original transpose axes to match the input's original shape

    Args:
        graph (object): The computation graph
        node (object): The Transpose node
        cotangent (str): The cotangent variable name

    Returns:
    tuple: A tuple containing the name of the emitted node representing the
    transposed cotangent
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
    """Computes the Jacobian-Vector Product (JVP) for a Transpose operation.

    Generates a JAX-compatible code string that transposes the tangent vector
    using the same axes as the original operation

    Args:
        graph (object): The computation graph
        node (object): The Transpose node
        tangent (str): The tangent variable name

    Returns:
    str: A code string representing the JAX transpose operation
    """
    axes = node.attributes.get("axes")
    if axes is None:
        return f"jnp.transpose({tangent})"
    return f"jnp.transpose({tangent}, axes={axes})"


@register_vjp("BroadcastTo")
def broadcast_to_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for the BroadcastTo operation.

    Args:
        graph (object): The computation graph
        node (object): The BroadcastTo operation node
        cotangent (str): The identifier of the cotangent tensor

    Returns:
    tuple: A tuple containing the identifier of the summed cotangent tensor
    """
    x = node.inputs[0]
    res = emit_ir_node(graph, "Sum", [cotangent], graph.nodes[x].shape_metadata)
    return (res,)


@register_jvp("BroadcastTo")
def broadcast_to_jvp(graph: object, node: object, tangent: str) -> str:
    """Computes the Jacobian-Vector Product (JVP) for a BroadcastTo operation.

    Generates a JAX-compatible code string that broadcasts the tangent vector
    to the target shape

    Args:
        graph (object): The computation graph
        node (object): The BroadcastTo node
        tangent (str): The tangent variable name

    Returns:
    str: A code string representing the JAX broadcast_to operation
    """
    shape = node.attributes.get("shape")
    return f"jnp.broadcast_to({tangent}, {shape})"
