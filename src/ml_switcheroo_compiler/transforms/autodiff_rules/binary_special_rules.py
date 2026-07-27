"""Binary rules for special."""

import enum

from ml_switcheroo_compiler.ops.base import emit_ir_node
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import register_jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import register_vjp


class UnconnectedGradients(enum.Enum):
    """Unconnected gradients enum."""

    NONE = "none"
    ZERO = "zero"


@register_vjp("Igamma")
def igamma_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for Igamma."""
    a, x = node.inputs

    # da = igamma_grad_a(a, x) * cotangent
    grad_a_base = emit_ir_node(graph, "IgammaGradA", [a, x], graph.nodes[a].shape_metadata)
    da = emit_ir_node(graph, "Multiply", [cotangent, grad_a_base], graph.nodes[a].shape_metadata)

    # dx = exp(-x + (a-1)*log(x) - lgamma(a)) * cotangent
    grad_x_base = _compute_igamma_dx(graph, a, x)
    dx = emit_ir_node(graph, "Multiply", [cotangent, grad_x_base], graph.nodes[x].shape_metadata)

    return (da, dx)


@register_jvp("Igamma")
def igamma_jvp(graph: object, node: object, tangents: tuple) -> str:
    """JVP for Igamma."""
    a, x = node.inputs
    t_a, t_x = tangents
    grad_a_base = emit_ir_node(graph, "IgammaGradA", [a, x], graph.nodes[a].shape_metadata)
    term_a = emit_ir_node(graph, "Multiply", [t_a, grad_a_base], graph.nodes[a].shape_metadata)
    grad_x_base = _compute_igamma_dx(graph, a, x)
    term_x = emit_ir_node(graph, "Multiply", [t_x, grad_x_base], graph.nodes[x].shape_metadata)
    return emit_ir_node(graph, "Add", [term_a, term_x], node.shape_metadata)


@register_vjp("Igammac")
def igammac_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for Igammac."""
    # igammac(a, x) = 1 - igamma(a, x)
    # So gradients are just negative of igamma
    da, dx = igamma_vjp(graph, node, cotangent)
    neg_da = emit_ir_node(graph, "Negative", [da], graph.nodes[node.inputs[0]].shape_metadata)
    neg_dx = emit_ir_node(graph, "Negative", [dx], graph.nodes[node.inputs[1]].shape_metadata)
    return (neg_da, neg_dx)


@register_jvp("Igammac")
def igammac_jvp(graph: object, node: object, tangents: tuple) -> str:
    """JVP for Igammac."""
    dy = igamma_jvp(graph, node, tangents)
    return emit_ir_node(graph, "Negative", [dy], node.shape_metadata)


@register_vjp("Zeta")
def zeta_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for Zeta."""
    x, q = node.inputs

    dx = UnconnectedGradients.ZERO

    one = emit_ir_node(graph, "Constant", [], None, {"value": 1.0})
    x_plus_1 = emit_ir_node(graph, "Add", [x, one], graph.nodes[x].shape_metadata)
    zeta_x1_q = emit_ir_node(graph, "Zeta", [x_plus_1, q], graph.nodes[q].shape_metadata)
    neg_x = emit_ir_node(graph, "Negative", [x], graph.nodes[x].shape_metadata)
    dq_base = emit_ir_node(graph, "Multiply", [neg_x, zeta_x1_q], graph.nodes[q].shape_metadata)
    dq = emit_ir_node(graph, "Multiply", [cotangent, dq_base], graph.nodes[q].shape_metadata)
    return (dx, dq)


@register_jvp("Zeta")
def zeta_jvp(graph: object, node: object, tangents: tuple) -> str:
    """JVP for Zeta."""
    x, q = node.inputs
    t_x, t_q = tangents

    if t_q is None:
        return None

    # dz/dq = -x * zeta(x+1, q) * t_q
    one = emit_ir_node(graph, "Constant", [], None, {"value": 1.0})
    x_plus_1 = emit_ir_node(graph, "Add", [x, one], graph.nodes[x].shape_metadata)
    zeta_x1_q = emit_ir_node(graph, "Zeta", [x_plus_1, q], graph.nodes[q].shape_metadata)
    neg_x = emit_ir_node(graph, "Negative", [x], graph.nodes[x].shape_metadata)
    dq_base = emit_ir_node(graph, "Multiply", [neg_x, zeta_x1_q], graph.nodes[q].shape_metadata)
    return emit_ir_node(graph, "Multiply", [t_q, dq_base], node.shape_metadata)


@register_vjp("Polygamma")
def polygamma_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for Polygamma."""
    n, x = node.inputs

    dn = UnconnectedGradients.ZERO

    one = emit_ir_node(graph, "Constant", [], None, {"value": 1.0})
    n_plus_1 = emit_ir_node(graph, "Add", [n, one], graph.nodes[n].shape_metadata)
    dx_base = emit_ir_node(graph, "Polygamma", [n_plus_1, x], graph.nodes[x].shape_metadata)
    dx = emit_ir_node(graph, "Multiply", [cotangent, dx_base], graph.nodes[x].shape_metadata)
    return (dn, dx)


@register_jvp("Polygamma")
def polygamma_jvp(graph: object, node: object, tangents: tuple) -> str:
    """JVP for Polygamma."""
    n, x = node.inputs
    t_n, t_x = tangents
    # dz/dn = 0
    # dz/dx = polygamma(n+1, x) * t_x
    one = emit_ir_node(graph, "Constant", [], None, {"value": 1.0})
    n_plus_1 = emit_ir_node(graph, "Add", [n, one], graph.nodes[n].shape_metadata)
    dx_base = emit_ir_node(graph, "Polygamma", [n_plus_1, x], graph.nodes[x].shape_metadata)
    return emit_ir_node(graph, "Multiply", [t_x, dx_base], node.shape_metadata)


@register_vjp("Betainc")
def betainc_vjp(graph: object, node: object, cotangent: str) -> tuple:
    """VJP for Betainc."""
    a, b, x = node.inputs

    da = UnconnectedGradients.ZERO
    db = UnconnectedGradients.ZERO

    dx_base = _compute_betainc_dx(graph, a, b, x)
    dx = emit_ir_node(graph, "Multiply", [cotangent, dx_base], graph.nodes[x].shape_metadata)

    return (da, db, dx)


@register_jvp("Betainc")
def betainc_jvp(graph: object, node: object, tangents: tuple) -> str:
    """JVP for Betainc."""
    a, b, x = node.inputs
    t_a, t_b, t_x = tangents
    # dz/da = 0, dz/db = 0
    dx_base = _compute_betainc_dx(graph, a, b, x)
    return emit_ir_node(graph, "Multiply", [t_x, dx_base], node.shape_metadata)


def _compute_igamma_dx(graph: object, a: str, x: str) -> str:
    """Evaluate and process the compute igamma dx operation.

    Args:
        graph (object): Required parameter for graph.
        a (str): Required parameter for a.
        x (str): Required parameter for x.

    Returns:
        str: The evaluated or processed output.
    """
    one = emit_ir_node(graph, "Constant", [], None, {"value": 1.0})
    a_minus_1 = emit_ir_node(graph, "Subtract", [a, one], graph.nodes[a].shape_metadata)
    log_x = emit_ir_node(graph, "Log", [x], graph.nodes[x].shape_metadata)
    term1 = emit_ir_node(graph, "Multiply", [a_minus_1, log_x], graph.nodes[x].shape_metadata)
    neg_x = emit_ir_node(graph, "Negative", [x], graph.nodes[x].shape_metadata)
    lgamma_a = emit_ir_node(graph, "Lgamma", [a], graph.nodes[a].shape_metadata)
    term2 = emit_ir_node(graph, "Add", [neg_x, term1], graph.nodes[x].shape_metadata)
    term3 = emit_ir_node(graph, "Subtract", [term2, lgamma_a], graph.nodes[x].shape_metadata)
    return emit_ir_node(graph, "Exp", [term3], graph.nodes[x].shape_metadata)


def _compute_betainc_dx(graph: object, a: str, b: str, x: str) -> str:
    """Evaluate and process the compute betainc dx operation.

    Args:
        graph (object): Required parameter for graph.
        a (str): Required parameter for a.
        b (str): Required parameter for b.
        x (str): Required parameter for x.

    Returns:
        str: The evaluated or processed output.
    """
    one = emit_ir_node(graph, "Constant", [], None, {"value": 1.0})
    term1, term2 = _compute_betainc_dx_terms(graph, a, b, x, one)
    log_beta = _compute_betainc_log_beta(graph, a, b)

    log_dx = emit_ir_node(graph, "Add", [term1, term2], graph.nodes[x].shape_metadata)
    log_dx = emit_ir_node(graph, "Subtract", [log_dx, log_beta], graph.nodes[x].shape_metadata)
    return emit_ir_node(graph, "Exp", [log_dx], graph.nodes[x].shape_metadata)


def _compute_betainc_log_beta(graph: object, a: str, b: str) -> str:
    """Evaluate and process the compute betainc log beta operation.

    Args:
        graph (object): Required parameter for graph.
        a (str): Required parameter for a.
        b (str): Required parameter for b.

    Returns:
        str: The evaluated or processed output.
    """
    lgamma_a = emit_ir_node(graph, "Lgamma", [a], graph.nodes[a].shape_metadata)
    lgamma_b = emit_ir_node(graph, "Lgamma", [b], graph.nodes[b].shape_metadata)
    a_plus_b = emit_ir_node(graph, "Add", [a, b], graph.nodes[a].shape_metadata)
    lgamma_ab = emit_ir_node(graph, "Lgamma", [a_plus_b], graph.nodes[a].shape_metadata)

    log_beta = emit_ir_node(graph, "Add", [lgamma_a, lgamma_b], graph.nodes[a].shape_metadata)
    return emit_ir_node(graph, "Subtract", [log_beta, lgamma_ab], graph.nodes[a].shape_metadata)


def _compute_betainc_dx_terms(graph: object, a: str, b: str, x: str, one: str) -> tuple[str, str]:
    """Evaluate and process the compute betainc dx terms operation.

    Args:
        graph (object): Required parameter for graph.
        a (str): Required parameter for a.
        b (str): Required parameter for b.
        x (str): Required parameter for x.
        one (str): Required parameter for one.

    Returns:
        tuple: The evaluated or processed output.
    """
    a_minus_1 = emit_ir_node(graph, "Subtract", [a, one], graph.nodes[a].shape_metadata)
    b_minus_1 = emit_ir_node(graph, "Subtract", [b, one], graph.nodes[b].shape_metadata)
    one_minus_x = emit_ir_node(graph, "Subtract", [one, x], graph.nodes[x].shape_metadata)

    log_x = emit_ir_node(graph, "Log", [x], graph.nodes[x].shape_metadata)
    log_1_minus_x = emit_ir_node(graph, "Log", [one_minus_x], graph.nodes[x].shape_metadata)

    term1 = emit_ir_node(graph, "Multiply", [a_minus_1, log_x], graph.nodes[x].shape_metadata)
    term2 = emit_ir_node(graph, "Multiply", [b_minus_1, log_1_minus_x], graph.nodes[x].shape_metadata)
    return term1, term2
