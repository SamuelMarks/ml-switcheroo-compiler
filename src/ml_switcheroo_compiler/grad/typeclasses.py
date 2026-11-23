"""Haskell-inspired tangent monoids and DifferentiableOp minimal complete definitions for AD."""

from __future__ import annotations

import dataclasses
import enum
import math
import uuid
from collections.abc import Callable, Sequence

from ml_switcheroo_ir import LogicalGraph, LogicalNode
from ml_switcheroo_ir.typeclasses import Eq, Invalid, Monoid, Semigroup, Valid, Validation


class TangentKind(str, enum.Enum):
    """Discriminator tag for tangent and cotangent monoid elements."""

    ZERO = "zero"
    NO_TANGENT = "no_tangent"
    VALUE = "value"


@dataclasses.dataclass(frozen=True)
class ScalarTangentElem:
    """Scalar tangent/cotangent monoid element with ``ZERO`` identity and ``NO_TANGENT`` absorber."""

    kind: TangentKind
    value: float = 0.0

    @staticmethod
    def zero() -> ScalarTangentElem:
        """Construct the additive identity ``ZeroTangent`` element.

        Returns:
            ScalarTangentElem: Additive identity tangent element.
        """
        return ScalarTangentElem(kind=TangentKind.ZERO, value=0.0)

    @staticmethod
    def no_tangent() -> ScalarTangentElem:
        """Construct the non-differentiable ``NoTangent`` absorber element.

        Returns:
            ScalarTangentElem: Non-differentiable tangent absorber.
        """
        return ScalarTangentElem(kind=TangentKind.NO_TANGENT, value=0.0)

    @staticmethod
    def of(val: float) -> ScalarTangentElem:
        """Wrap a numeric tangent value, canonicalizing exact ``0.0`` to ``zero()``.

        Args:
            val (float): Numeric scalar tangent value.

        Returns:
            ScalarTangentElem: Canonical tangent element.
        """
        if val == 0.0:
            return ScalarTangentElem.zero()
        return ScalarTangentElem(kind=TangentKind.VALUE, value=float(val))


def _combine_scalar_tangents(lhs: ScalarTangentElem, rhs: ScalarTangentElem) -> ScalarTangentElem:
    """Combine two ``ScalarTangentElem`` values with ``ZERO`` identity and ``NO_TANGENT`` absorption.

    Args:
        lhs (ScalarTangentElem): Left tangent operand.
        rhs (ScalarTangentElem): Right tangent operand.

    Returns:
        ScalarTangentElem: Accumulated tangent element.
    """
    if lhs.kind == TangentKind.NO_TANGENT or rhs.kind == TangentKind.NO_TANGENT:
        return ScalarTangentElem.no_tangent()
    if lhs.kind == TangentKind.ZERO:
        return rhs
    if rhs.kind == TangentKind.ZERO:
        return lhs
    return ScalarTangentElem.of(lhs.value + rhs.value)


def _approx_eq_scalar_tangent(lhs: ScalarTangentElem, rhs: ScalarTangentElem, tol: float = 1e-9) -> bool:
    """Compare two ``ScalarTangentElem`` values up to floating-point tolerance ``tol``.

    Args:
        lhs (ScalarTangentElem): Left tangent operand.
        rhs (ScalarTangentElem): Right tangent operand.
        tol (float): Absolute floating-point tolerance.

    Returns:
        bool: ``True`` when kind matches and numeric values agree within ``tol``.
    """
    if lhs.kind != rhs.kind:
        return False
    if lhs.kind != TangentKind.VALUE:
        return True
    return math.isclose(lhs.value, rhs.value, abs_tol=tol, rel_tol=tol)


SCALAR_TANGENT_EQ: Eq[ScalarTangentElem] = Eq(eq=_approx_eq_scalar_tangent)
SCALAR_TANGENT_MONOID: Monoid[ScalarTangentElem] = Monoid(
    empty=ScalarTangentElem.zero,
    semigroup=Semigroup(combine=_combine_scalar_tangents),
)


def is_symbolic_zero_node(graph: LogicalGraph, node_id: str) -> bool:
    """Return ``True`` if ``node_id`` in ``graph`` denotes an exact additive zero tensor/scalar.

    Args:
        graph (LogicalGraph): Computation graph containing candidate cotangent nodes.
        node_id (str): Target node identifier in ``graph.nodes``.

    Returns:
        bool: ``True`` if ``node_id`` is ``Zeros``, ``ZerosLike``, or ``Constant(0)``.
    """
    node = graph.nodes.get(node_id)
    if node is None:
        return False
    if node.op_type in ("Zeros", "ZerosLike"):
        return True
    if node.op_type == "Constant":
        val = node.attributes.get("value")
        if isinstance(val, (int, float)) and not isinstance(val, bool) and val == 0:
            return True
    return False


@dataclasses.dataclass(frozen=True)
class GraphCotangentElem:
    """Symbolic reverse-mode cotangent monoid carrier over a ``LogicalGraph``."""

    kind: TangentKind
    node_id: str | None = None

    @staticmethod
    def zero() -> GraphCotangentElem:
        """Construct the symbolic ``ZeroTangent`` identity element.

        Returns:
            GraphCotangentElem: Symbolic zero cotangent element.
        """
        return GraphCotangentElem(kind=TangentKind.ZERO, node_id=None)

    @staticmethod
    def no_tangent() -> GraphCotangentElem:
        """Construct the symbolic ``NoTangent`` non-differentiable absorber.

        Returns:
            GraphCotangentElem: Symbolic non-differentiable cotangent element.
        """
        return GraphCotangentElem(kind=TangentKind.NO_TANGENT, node_id=None)

    @staticmethod
    def from_node(graph: LogicalGraph, node_id: str) -> GraphCotangentElem:
        """Wrap ``node_id`` as a ``GraphCotangentElem``, canonicalizing symbolic zero nodes to ``zero()``.

        Args:
            graph (LogicalGraph): Computation graph owning ``node_id``.
            node_id (str): Cotangent IR node identifier.

        Returns:
            GraphCotangentElem: ``zero()`` when ``is_symbolic_zero_node`` holds, else ``VALUE`` node ref.
        """
        if is_symbolic_zero_node(graph, node_id):
            return GraphCotangentElem.zero()
        return GraphCotangentElem(kind=TangentKind.VALUE, node_id=node_id)


def build_graph_cotangent_monoid(graph: LogicalGraph) -> Monoid[GraphCotangentElem]:
    """Construct an explicit ``Monoid[GraphCotangentElem]`` emitting ``Add`` nodes into ``graph`` on combine.

    Args:
        graph (LogicalGraph): Target reverse-mode ``LogicalGraph`` for emitted ``Add`` accumulation nodes.

    Returns:
        Monoid[GraphCotangentElem]: Lawful cotangent accumulation monoid bound to ``graph``.
    """

    def _combine(lhs: GraphCotangentElem, rhs: GraphCotangentElem) -> GraphCotangentElem:
        """Accumulate two symbolic cotangents, eliding ``Add`` nodes on ``ZERO`` and absorbing ``NO_TANGENT``.

        Args:
            lhs (GraphCotangentElem): Existing accumulated cotangent.
            rhs (GraphCotangentElem): Incoming adjoint contribution.

        Returns:
            GraphCotangentElem: Combined symbolic cotangent.
        """
        if lhs.kind == TangentKind.NO_TANGENT or rhs.kind == TangentKind.NO_TANGENT:
            return GraphCotangentElem.no_tangent()
        if lhs.kind == TangentKind.ZERO or lhs.node_id is None:
            return rhs
        if rhs.kind == TangentKind.ZERO or rhs.node_id is None:
            return lhs
        out_id = f"{lhs.node_id}_add_{rhs.node_id}_{uuid.uuid4().hex[:6]}"
        lhs_node = graph.nodes.get(lhs.node_id)
        shape_meta = lhs_node.shape_metadata if lhs_node is not None else None
        graph.nodes[out_id] = LogicalNode(
            id=out_id,
            op_type="Add",
            inputs=[lhs.node_id, rhs.node_id],
            shape_metadata=shape_meta,
        )
        return GraphCotangentElem(kind=TangentKind.VALUE, node_id=out_id)

    return Monoid(empty=GraphCotangentElem.zero, semigroup=Semigroup(combine=_combine))


def _dot_product(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute standard Euclidean inner product ``<vec_a, vec_b>``.

    Args:
        vec_a (Sequence[float]): First vector operand.
        vec_b (Sequence[float]): Second vector operand of matching length.

    Returns:
        float: Inner product sum.

    Raises:
        ValueError: If ``len(vec_a) != len(vec_b)``.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Inner product dimension mismatch: {len(vec_a)} != {len(vec_b)}")
    return sum(a * b for a, b in zip(vec_a, vec_b))


@dataclasses.dataclass(frozen=True)
class DifferentiableOpDictionary:
    """Haskell-style ``Differentiable`` typeclass dictionary with Minimal Complete Definitions (``jvp | vjp``)."""

    op_name: str
    primal_fn: Callable[[Sequence[float]], tuple[float, ...]]
    jvp_fn: Callable[[Sequence[float], Sequence[float]], tuple[float, ...]]
    vjp_fn: Callable[[Sequence[float], Sequence[float]], tuple[float, ...]]

    @staticmethod
    def from_jvp_and_vjp(
        op_name: str,
        primal_fn: Callable[[Sequence[float]], Sequence[float]],
        jvp_fn: Callable[[Sequence[float], Sequence[float]], Sequence[float]],
        vjp_fn: Callable[[Sequence[float], Sequence[float]], Sequence[float]],
    ) -> DifferentiableOpDictionary:
        """Construct ``DifferentiableOpDictionary`` from explicit ``jvp_fn`` and ``vjp_fn`` implementations.

        Args:
            op_name (str): Canonical operator identifier.
            primal_fn (Callable[[Sequence[float]], Sequence[float]]): Forward primal evaluation map.
            jvp_fn (Callable[[Sequence[float], Sequence[float]], Sequence[float]]): Forward-mode JVP map.
            vjp_fn (Callable[[Sequence[float], Sequence[float]], Sequence[float]]): Reverse-mode VJP map.

        Returns:
            DifferentiableOpDictionary: Complete differentiable operator witness.
        """
        return DifferentiableOpDictionary(
            op_name=op_name,
            primal_fn=lambda x: tuple(float(v) for v in primal_fn(x)),
            jvp_fn=lambda x, dx: tuple(float(v) for v in jvp_fn(x, dx)),
            vjp_fn=lambda x, dy: tuple(float(v) for v in vjp_fn(x, dy)),
        )

    @staticmethod
    def from_jvp_and_transpose(
        op_name: str,
        primal_fn: Callable[[Sequence[float]], Sequence[float]],
        jvp_fn: Callable[[Sequence[float], Sequence[float]], Sequence[float]],
        transpose_fn: Callable[[Sequence[float], Sequence[float]], Sequence[float]],
    ) -> DifferentiableOpDictionary:
        """Construct ``DifferentiableOpDictionary`` from ``jvp_fn`` and its explicit linear transpose rule.

        Models Elliott/JAX-style ``(linearize, linear_transpose)`` Minimal Complete Definitions without
        requiring basis probing.

        Args:
            op_name (str): Canonical operator identifier.
            primal_fn (Callable[[Sequence[float]], Sequence[float]]): Forward primal evaluation map.
            jvp_fn (Callable[[Sequence[float], Sequence[float]], Sequence[float]]): Forward-mode JVP map.
            transpose_fn (Callable[[Sequence[float], Sequence[float]], Sequence[float]]): Linear transpose rule.

        Returns:
            DifferentiableOpDictionary: Complete differentiable operator witness with derived ``vjp_fn``.
        """
        return DifferentiableOpDictionary.from_jvp_and_vjp(
            op_name=op_name,
            primal_fn=primal_fn,
            jvp_fn=jvp_fn,
            vjp_fn=transpose_fn,
        )

    @staticmethod
    def from_jvp(
        op_name: str,
        primal_fn: Callable[[Sequence[float]], Sequence[float]],
        jvp_fn: Callable[[Sequence[float], Sequence[float]], Sequence[float]],
    ) -> DifferentiableOpDictionary:
        """Derive exact ``vjp_fn`` automatically from ``jvp_fn`` via linear basis transposition.

        Satisfies ``vjp(x, u)_i = <jvp(x, e_i), u>`` with zero finite-difference approximation error.

        Args:
            op_name (str): Canonical operator identifier.
            primal_fn (Callable[[Sequence[float]], Sequence[float]]): Forward primal evaluation map.
            jvp_fn (Callable[[Sequence[float], Sequence[float]], Sequence[float]]): Forward-mode JVP map.

        Returns:
            DifferentiableOpDictionary: Complete dictionary with transposed ``vjp_fn``.
        """

        def _derived_vjp(primals: Sequence[float], cotangent: Sequence[float]) -> tuple[float, ...]:
            """Compute exact transposed VJP via standard basis probing of ``jvp_fn``.

            Args:
                primals (Sequence[float]): Primal input coordinates ``x``.
                cotangent (Sequence[float]): Output cotangent vector ``u``.

            Returns:
                tuple[float, ...]: Input cotangent vector ``L_x^*(u)``.
            """
            n_in = len(primals)
            grad_in: list[float] = []
            for idx in range(n_in):
                basis = [1.0 if j == idx else 0.0 for j in range(n_in)]
                jvp_col = [float(v) for v in jvp_fn(primals, basis)]
                grad_in.append(_dot_product(jvp_col, cotangent))
            return tuple(grad_in)

        return DifferentiableOpDictionary.from_jvp_and_vjp(op_name, primal_fn, jvp_fn, _derived_vjp)

    @staticmethod
    def from_vjp(
        op_name: str,
        primal_fn: Callable[[Sequence[float]], Sequence[float]],
        vjp_fn: Callable[[Sequence[float], Sequence[float]], Sequence[float]],
    ) -> DifferentiableOpDictionary:
        """Derive exact ``jvp_fn`` automatically from ``vjp_fn`` via dual cotangent basis transposition.

        Satisfies ``jvp(x, v)_j = <v, vjp(x, e_j)>`` with zero finite-difference approximation error.

        Args:
            op_name (str): Canonical operator identifier.
            primal_fn (Callable[[Sequence[float]], Sequence[float]]): Forward primal evaluation map.
            vjp_fn (Callable[[Sequence[float], Sequence[float]], Sequence[float]]): Reverse-mode VJP map.

        Returns:
            DifferentiableOpDictionary: Complete dictionary with dual-transposed ``jvp_fn``.
        """

        def _derived_jvp(primals: Sequence[float], tangent: Sequence[float]) -> tuple[float, ...]:
            """Compute exact dual-transposed JVP via output cotangent basis probing of ``vjp_fn``.

            Args:
                primals (Sequence[float]): Primal input coordinates ``x``.
                tangent (Sequence[float]): Input tangent vector ``v``.

            Returns:
                tuple[float, ...]: Output tangent vector ``L_x(v)``.
            """
            primal_out = tuple(float(y) for y in primal_fn(primals))
            m_out = len(primal_out)
            tangent_out: list[float] = []
            for j_idx in range(m_out):
                co_basis = [1.0 if k == j_idx else 0.0 for k in range(m_out)]
                vjp_row = [float(g) for g in vjp_fn(primals, co_basis)]
                tangent_out.append(_dot_product(tangent, vjp_row))
            return tuple(tangent_out)

        return DifferentiableOpDictionary.from_jvp_and_vjp(op_name, primal_fn, _derived_jvp, vjp_fn)


def verify_jvp_vjp_adjoint_law(
    op_dict: DifferentiableOpDictionary,
    primals: Sequence[float],
    tangent: Sequence[float],
    cotangent: Sequence[float],
    tol: float = 1e-7,
) -> Validation[tuple[str, ...], float]:
    """Verify the Hilbert-space adjoint identity ``<JVP(x, v), u> == <v, VJP(x, u)>``.

    Args:
        op_dict (DifferentiableOpDictionary): Differentiable operator witness to verify.
        primals (Sequence[float]): Primal evaluation point ``x``.
        tangent (Sequence[float]): Input tangent vector ``v``.
        cotangent (Sequence[float]): Output cotangent vector ``u``.
        tol (float): Allowed absolute/relative numerical tolerance.

    Returns:
        Validation[tuple[str, ...], float]: ``Valid(inner_product)`` if law holds, else ``Invalid`` diagnostic tuple.
    """
    jvp_vec = op_dict.jvp_fn(primals, tangent)
    if len(jvp_vec) != len(cotangent):
        return Invalid((f"{op_dict.op_name}: JVP output dim {len(jvp_vec)} != cotangent dim {len(cotangent)}",))
    vjp_vec = op_dict.vjp_fn(primals, cotangent)
    if len(vjp_vec) != len(tangent):
        return Invalid((f"{op_dict.op_name}: VJP output dim {len(vjp_vec)} != tangent dim {len(tangent)}",))
    lhs_ip = _dot_product(jvp_vec, cotangent)
    rhs_ip = _dot_product(tangent, vjp_vec)
    if not math.isclose(lhs_ip, rhs_ip, abs_tol=tol, rel_tol=tol):
        return Invalid((f"{op_dict.op_name}: Adjoint law violated (<JVP, u>={lhs_ip} != <v, VJP>={rhs_ip})",))
    return Valid(lhs_ip)


__all__ = [
    "DifferentiableOpDictionary",
    "GraphCotangentElem",
    "SCALAR_TANGENT_EQ",
    "SCALAR_TANGENT_MONOID",
    "ScalarTangentElem",
    "TangentKind",
    "build_graph_cotangent_monoid",
    "is_symbolic_zero_node",
    "verify_jvp_vjp_adjoint_law",
]
