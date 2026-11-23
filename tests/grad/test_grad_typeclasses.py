"""Unit and law tests for scalar/symbolic cotangent monoids and DifferentiableOpDictionary MCD."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode
from ml_switcheroo_ir.typeclasses import Invalid, Valid

from ml_switcheroo_compiler.grad import (
    SCALAR_TANGENT_EQ,
    SCALAR_TANGENT_MONOID,
    DifferentiableOpDictionary,
    GraphCotangentElem,
    ScalarTangentElem,
    build_graph_cotangent_monoid,
    is_symbolic_zero_node,
    verify_jvp_vjp_adjoint_law,
)
from ml_switcheroo_compiler.grad.typeclasses import _dot_product
from ml_switcheroo_compiler.transforms._autodiff_vjp_core import _add_nodes, _is_zero_node


def test_scalar_tangent_monoid_laws() -> None:
    """Verify ZeroTangent identity, NoTangent absorption, canonical zero, and associativity."""
    z = ScalarTangentElem.zero()
    nt = ScalarTangentElem.no_tangent()
    assert ScalarTangentElem.of(0.0) == z

    samples = [z, nt, ScalarTangentElem.of(1.5), ScalarTangentElem.of(-1.5), ScalarTangentElem.of(3.25)]
    assert not SCALAR_TANGENT_EQ.eq(z, nt)
    assert SCALAR_TANGENT_EQ.eq(nt, nt)

    for a in samples:
        assert SCALAR_TANGENT_EQ.eq(SCALAR_TANGENT_MONOID.semigroup.combine(z, a), a)
        assert SCALAR_TANGENT_EQ.eq(SCALAR_TANGENT_MONOID.semigroup.combine(a, z), a)
        assert SCALAR_TANGENT_EQ.eq(SCALAR_TANGENT_MONOID.semigroup.combine(nt, a), nt)
        assert SCALAR_TANGENT_EQ.eq(SCALAR_TANGENT_MONOID.semigroup.combine(a, nt), nt)
        for b in samples:
            for c in samples:
                lhs = SCALAR_TANGENT_MONOID.semigroup.combine(
                    SCALAR_TANGENT_MONOID.semigroup.combine(a, b),
                    c,
                )
                rhs = SCALAR_TANGENT_MONOID.semigroup.combine(
                    a,
                    SCALAR_TANGENT_MONOID.semigroup.combine(b, c),
                )
                assert SCALAR_TANGENT_EQ.eq(lhs, rhs)


def test_symbolic_graph_cotangent_monoid_and_vjp_core() -> None:
    """Verify symbolic zero canonicalization, Add node emission, and _autodiff_vjp_core integration."""
    graph = LogicalGraph(
        name="cotangent_test",
        nodes=[
            LogicalNode(id="z0", op_type="Zeros", shape_metadata=(2, 3)),
            LogicalNode(id="c0", op_type="Constant", attributes={"value": 0}, shape_metadata=(2, 3)),
            LogicalNode(id="c_bool", op_type="Constant", attributes={"value": False}, shape_metadata=(2, 3)),
            LogicalNode(id="c1", op_type="Constant", attributes={"value": 2.0}, shape_metadata=(2, 3)),
            LogicalNode(id="g1", op_type="Input", shape_metadata=(2, 3)),
            LogicalNode(id="g2", op_type="Input", shape_metadata=(2, 3)),
        ],
    )
    assert is_symbolic_zero_node(graph, "z0") is True
    assert is_symbolic_zero_node(graph, "c0") is True
    assert is_symbolic_zero_node(graph, "c_bool") is False
    assert is_symbolic_zero_node(graph, "c1") is False
    assert is_symbolic_zero_node(graph, "missing_id") is False
    assert _is_zero_node(graph, "z0") is True

    monoid = build_graph_cotangent_monoid(graph)
    e_zero = monoid.empty()
    e_nt = GraphCotangentElem.no_tangent()
    e_from_z0 = GraphCotangentElem.from_node(graph, "z0")
    assert e_from_z0 == e_zero

    e_g1 = GraphCotangentElem.from_node(graph, "g1")
    e_g2 = GraphCotangentElem.from_node(graph, "g2")

    assert monoid.semigroup.combine(e_zero, e_g1) == e_g1
    assert monoid.semigroup.combine(e_g1, e_zero) == e_g1
    assert monoid.semigroup.combine(e_nt, e_g1) == e_nt
    assert monoid.semigroup.combine(e_g1, e_nt) == e_nt

    combined = monoid.semigroup.combine(e_g1, e_g2)
    assert combined.node_id is not None and combined.node_id in graph.nodes
    assert graph.nodes[combined.node_id].op_type == "Add"

    assert _add_nodes(graph, "z0", "g1") == "g1"
    assert _add_nodes(graph, "g1", "c0") == "g1"
    added_id = _add_nodes(graph, "g1", "g2")
    assert graph.nodes[added_id].op_type == "Add"


def test_differentiable_op_mcd_and_adjoint_laws() -> None:
    """Verify Minimal Complete Definitions (from_jvp, from_vjp, from_jvp_and_vjp) and adjoint law checks."""

    # Nonlinear R^2 -> R^2 map: f(x0, x1) = (x0 * sin(x1), x0^2 + 3 * x1)
    def _primal(x: Sequence[float]) -> tuple[float, float]:
        """Evaluate nonlinear R^2 -> R^2 test map.

        Args:
            x (Sequence[float]): Length-2 input vector.

        Returns:
            tuple[float, float]: Length-2 output vector.
        """
        return (x[0] * math.sin(x[1]), x[0] * x[0] + 3.0 * x[1])

    def _jvp(x: Sequence[float], dx: Sequence[float]) -> tuple[float, float]:
        """Evaluate exact Jacobian-vector product of _primal.

        Args:
            x (Sequence[float]): Primal point.
            dx (Sequence[float]): Input tangent vector.

        Returns:
            tuple[float, float]: Output tangent vector.
        """
        return (
            math.sin(x[1]) * dx[0] + x[0] * math.cos(x[1]) * dx[1],
            2.0 * x[0] * dx[0] + 3.0 * dx[1],
        )

    def _vjp(x: Sequence[float], dy: Sequence[float]) -> tuple[float, float]:
        """Evaluate exact vector-Jacobian product of _primal.

        Args:
            x (Sequence[float]): Primal point.
            dy (Sequence[float]): Output cotangent vector.

        Returns:
            tuple[float, float]: Input cotangent vector.
        """
        return (
            math.sin(x[1]) * dy[0] + 2.0 * x[0] * dy[1],
            x[0] * math.cos(x[1]) * dy[0] + 3.0 * dy[1],
        )

    op_from_jvp = DifferentiableOpDictionary.from_jvp("NonlinearFromJVP", _primal, _jvp)
    op_from_vjp = DifferentiableOpDictionary.from_vjp("NonlinearFromVJP", _primal, _vjp)
    op_from_transpose = DifferentiableOpDictionary.from_jvp_and_transpose("NonlinearTranspose", _primal, _jvp, _vjp)

    pt = (1.7, 0.6)
    v = (0.4, -0.9)
    u = (-1.2, 0.5)

    assert op_from_jvp.primal_fn(pt) == _primal(pt)
    assert op_from_transpose.vjp_fn(pt, u) == _vjp(pt, u)
    for g_derived, g_exact in zip(op_from_jvp.vjp_fn(pt, u), _vjp(pt, u)):
        assert math.isclose(g_derived, g_exact, abs_tol=1e-12)
    for t_derived, t_exact in zip(op_from_vjp.jvp_fn(pt, v), _jvp(pt, v)):
        assert math.isclose(t_derived, t_exact, abs_tol=1e-12)

    valid_jvp = verify_jvp_vjp_adjoint_law(op_from_jvp, pt, v, u)
    assert isinstance(valid_jvp, Valid)
    valid_vjp = verify_jvp_vjp_adjoint_law(op_from_vjp, pt, v, u)
    assert isinstance(valid_vjp, Valid)

    # Error branches: broken VJP, mismatched cotangent dim, mismatched tangent dim, and _dot_product error
    broken = DifferentiableOpDictionary.from_jvp_and_vjp(
        "BrokenAdjoint",
        _primal,
        _jvp,
        lambda _x, dy: (dy[0] + 10.0, dy[1]),
    )
    inv_law = verify_jvp_vjp_adjoint_law(broken, pt, v, u)
    assert isinstance(inv_law, Invalid) and "Adjoint law violated" in inv_law.errors[0]

    inv_co_dim = verify_jvp_vjp_adjoint_law(op_from_jvp, pt, v, (1.0,))
    assert isinstance(inv_co_dim, Invalid) and "cotangent dim" in inv_co_dim.errors[0]

    bad_vjp_dim = DifferentiableOpDictionary.from_jvp_and_vjp(
        "BadVJPDim",
        _primal,
        _jvp,
        lambda _x, _dy: (1.0,),
    )
    inv_tan_dim = verify_jvp_vjp_adjoint_law(bad_vjp_dim, pt, v, u)
    assert isinstance(inv_tan_dim, Invalid) and "tangent dim" in inv_tan_dim.errors[0]

    with pytest.raises(ValueError, match="Inner product dimension mismatch"):
        _dot_product((1.0,), (1.0, 2.0))
