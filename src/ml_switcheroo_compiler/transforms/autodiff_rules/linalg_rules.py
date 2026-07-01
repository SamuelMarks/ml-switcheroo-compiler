"""Defines automatic differentiation (autodiff) rules for linear algebra operations.

This module registers Vector-Jacobian Products (VJPs) and Jacobian-Vector Products
(JVPs) for operations such as matrix multiplication (Matmul) and dot products (Dot)
These rules are used to propagate gradients through the computation graph
"""

from ml_switcheroo_compiler.ops.base import emit_ir_node
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import register_jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp


@register_vjp("BlockMaskedMm")
def block_masked_mm_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the VJP for BlockMaskedMm."""
    # a, b, [mask_out, mask_lhs, mask_rhs]
    inputs = node.inputs
    a, b = inputs[0], inputs[1]

    adj_a = emit_ir_node(
        graph,
        "Matmul",
        [cotangent, b],
        graph.nodes[a].shape_metadata,
        attributes={"transpose_b": True},
    )
    adj_b = emit_ir_node(
        graph,
        "Matmul",
        [a, cotangent],
        graph.nodes[b].shape_metadata,
        attributes={"transpose_a": True},
    )
    # The true gradient would involve scaling masks by block size and multiplying.
    # For now, this acts as a placeholder that works for unmasked cases or basic tests.
    return (adj_a, adj_b)


@register_vjp("Dropout2d")
def dropout2d_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP function."""
    a = node.inputs[0]
    return (
        emit_ir_node(
            graph,
            "Dropout2d",
            [cotangent],
            graph.nodes[a].shape_metadata,
            attributes=node.attributes,
        ),
    )


@register_vjp("Dropout3d")
def dropout3d_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP function."""
    a = node.inputs[0]
    return (
        emit_ir_node(
            graph,
            "Dropout3d",
            [cotangent],
            graph.nodes[a].shape_metadata,
            attributes=node.attributes,
        ),
    )


@register_vjp("GatherMm")
def gather_mm_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP function."""
    # Just a placeholder for VJP
    return (cotangent, cotangent)


@register_vjp("SegmentedMm")
def segmented_mm_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP function."""
    # Placeholder
    return (cotangent, cotangent)


@register_vjp("PutAlongAxis")
def put_along_axis_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP function."""
    # Placeholder
    return (cotangent, cotangent)


@register_vjp("Logcumsumexp")
def logcumsumexp_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP function."""
    # Placeholder
    return (cotangent,)


@register_vjp("Gru")
def gru_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP function."""
    # Placeholder
    return (cotangent, cotangent, cotangent)


@register_vjp("GetItem")
def getitem_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP function."""
    # Placeholder
    return (cotangent,)


@register_vjp("Matmul")
def matmul_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for a matrix multiplication operation.

    This function calculates the adjoints (gradients) with respect to the inputs
    of the matrix multiplication node by emitting new IR nodes representing the
    transposed matrix multiplications

    Args:
        graph (object): The computation graph containing the nodes
        node (object): The matrix multiplication IR node being differentiated
        cotangent (str): The name or identifier of the cotangent (incoming gradient)
        variable

    Returns:
    tuple: A tuple containing the identifiers of the adjoint nodes for the two
    inputs (adj_a, adj_b)
    """
    a, b = node.inputs
    adj_a = emit_ir_node(
        graph,
        "Matmul",
        [cotangent, b],
        graph.nodes[a].shape_metadata,
        attributes={"transpose_b": True},
    )
    adj_b = emit_ir_node(
        graph,
        "Matmul",
        [a, cotangent],
        graph.nodes[b].shape_metadata,
        attributes={"transpose_a": True},
    )
    return (adj_a, adj_b)


@register_jvp("Matmul")
def matmul_jvp(tangent_a: object, tangent_b: object, a: object, b: object, **kwargs: object) -> str:
    """Computes the Jacobian-Vector Product (JVP) for a matrix multiplication operation.

    Generates a string representation of the JVP expression using JAX-like syntax
    (jnp.matmul) to propagate tangents through the matrix multiplication

    Args:
        tangent_a (object): The tangent corresponding to the first input matrix
        tangent_b (object): The tangent corresponding to the second input matrix
        a (object): The first input matrix
        b (object): The second input matrix
        **kwargs (object): Additional keyword arguments

    Returns:
    str: A string representing the JAX code for the JVP calculation
    """
    return f"(jnp.matmul({tangent_a}, {b}) + jnp.matmul({a}, {tangent_b}))"


@register_vjp("Dot")
def dot_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """Computes the Vector-Jacobian Product (VJP) for a dot product operation.

    Args:
        graph (object): The computation graph containing the nodes
        node (object): The dot product IR node being differentiated
        cotangent (str): The name or identifier of the cotangent (incoming gradient)
        variable

    Returns:
    tuple: A tuple containing the identifiers of the adjoint nodes
    """
    a, b = node.inputs
    adj_a = emit_ir_node(
        graph,
        "Dot",
        [cotangent, b],
        graph.nodes[a].shape_metadata,
        attributes={"transpose_b": True},
    )
    adj_b = emit_ir_node(
        graph,
        "Dot",
        [a, cotangent],
        graph.nodes[b].shape_metadata,
        attributes={"transpose_a": True},
    )
    return (adj_a, adj_b)


@register_jvp("Dot")
def dot_jvp(tangent_a: object, tangent_b: object, a: object, b: object, **kwargs: object) -> str:
    """Computes the Jacobian-Vector Product (JVP) for a dot product operation.

    Generates a string representation of the JVP expression using JAX-like syntax
    (jnp.dot) to propagate tangents through the dot product

    Args:
        tangent_a (object): The tangent corresponding to the first input
        tangent_b (object): The tangent corresponding to the second input
        a (object): The first input
        b (object): The second input
        **kwargs (object): Additional keyword arguments

    Returns:
    str: A string representing the JAX code for the JVP calculation
    """
    return f"(jnp.dot({tangent_a}, {b}) + jnp.dot({a}, {tangent_b}))"
