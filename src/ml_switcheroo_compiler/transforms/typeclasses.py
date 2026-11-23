"""Phase-indexed Kleisli compiler passes, fixpoint Endos, and subgraph-aware Monoid/EndoRewrite folds."""

from __future__ import annotations

import dataclasses
import os
import typing
from collections.abc import Callable, Mapping, Sequence

from ml_switcheroo_ir import (
    AcyclicSortedPhase,
    AttributeValue,
    Endo,
    EndoRewrite,
    Eq,
    Err,
    GroundedPhase,
    Invalid,
    LogicalGraph,
    LogicalNode,
    Monoid,
    Ok,
    Pretty,
    ProvenGraph,
    Result,
    SchemaValidPhase,
    ShardingValidPhase,
    prove_acyclic_sorted,
    prove_grounded,
    prove_schema_valid,
    prove_sharding_valid,
)

from ml_switcheroo_compiler.ir.core import clone_logical_node, fold_logical_graph_monoid

_M = typing.TypeVar("_M")
_P1 = typing.TypeVar("_P1")
_P2 = typing.TypeVar("_P2")
_P3 = typing.TypeVar("_P3")

GroundingManifestArg = typing.Union[Mapping[str, typing.Union[Sequence[str], frozenset[str]]], str, None]  # noqa: UP007


def _subgraphs_entry_equal(
    lhs_val: LogicalGraph | AttributeValue,
    rhs_val: LogicalGraph | AttributeValue,
) -> bool:
    """Compare two ``node.subgraphs`` values safely under union typing.

    Args:
        lhs_val (LogicalGraph | AttributeValue): Left subgraph or attribute value.
        rhs_val (LogicalGraph | AttributeValue): Right subgraph or attribute value.

    Returns:
        bool: ``True`` if both are structurally equal ``LogicalGraph``s or equal ``AttributeValue``s.
    """
    if isinstance(lhs_val, LogicalGraph) and isinstance(rhs_val, LogicalGraph):
        return _graphs_equal(lhs_val, rhs_val)
    if isinstance(lhs_val, LogicalGraph) or isinstance(rhs_val, LogicalGraph):
        return False
    return lhs_val == rhs_val


def _nodes_equal(lhs: LogicalNode, rhs: LogicalNode) -> bool:
    """Compare two ``LogicalNode`` instances structurally, including nested ``subgraphs``.

    Args:
        lhs (LogicalNode): Left logical node.
        rhs (LogicalNode): Right logical node.

    Returns:
        bool: ``True`` if all fields and nested subgraphs match.
    """
    if (
        lhs.id != rhs.id
        or lhs.op_type != rhs.op_type
        or list(lhs.inputs) != list(rhs.inputs)
        or dict(lhs.attributes) != dict(rhs.attributes)
        or lhs.shape_metadata != rhs.shape_metadata
        or lhs.sharding != rhs.sharding
    ):
        return False
    if set(lhs.subgraphs.keys()) != set(rhs.subgraphs.keys()):
        return False
    return all(_subgraphs_entry_equal(lhs.subgraphs[k], rhs.subgraphs[k]) for k in lhs.subgraphs)


def _graphs_equal(lhs: LogicalGraph, rhs: LogicalGraph) -> bool:
    """Compare two ``LogicalGraph`` instances structurally across nodes, inputs, outputs, and subgraphs.

    Args:
        lhs (LogicalGraph): Left logical graph.
        rhs (LogicalGraph): Right logical graph.

    Returns:
        bool: ``True`` if structurally identical.
    """
    if lhs.name != rhs.name or list(lhs.inputs) != list(rhs.inputs) or list(lhs.outputs) != list(rhs.outputs):
        return False
    if set(lhs.nodes.keys()) != set(rhs.nodes.keys()):
        return False
    return all(_nodes_equal(lhs.nodes[nid], rhs.nodes[nid]) for nid in lhs.nodes)


def _render_logical_graph(graph: LogicalGraph) -> str:
    """Render a deterministic human-readable IR summary string for ``graph``.

    Args:
        graph (LogicalGraph): Target logical computation graph.

    Returns:
        str: Formatted multi-line textual representation.
    """
    header = f"LogicalGraph(name={graph.name!r}, nodes={len(graph.nodes)})"
    lines = [header]
    for nid, node in graph.nodes.items():
        lines.append(f"  %{nid}: {node.op_type}({', '.join(node.inputs)})")
    return os.linesep.join(lines)


def _map_logical_node_children(
    node: LogicalNode,
    fn: Callable[[LogicalNode], LogicalNode],
) -> LogicalNode:
    """Apply ``fn`` across all child ``LogicalNode``s inside ``node.subgraphs``.

    Args:
        node (LogicalNode): Parent control-flow or standard ``LogicalNode``.
        fn (Callable[[LogicalNode], LogicalNode]): Endomorphic node transformation.

    Returns:
        LogicalNode: Cloned ``LogicalNode`` with transformed ``subgraphs``.
    """
    cloned = clone_logical_node(node)
    if not cloned.subgraphs:
        return cloned
    updated_subgraphs: dict[str, LogicalGraph | AttributeValue] = {}
    for branch_name, sub_g in cloned.subgraphs.items():
        if isinstance(sub_g, LogicalGraph):
            new_sub = LogicalGraph(
                name=sub_g.name,
                inputs=list(sub_g.inputs),
                outputs=list(sub_g.outputs),
            )
            for sub_nid, sub_node in sub_g.nodes.items():
                transformed = fn(sub_node)
                new_sub.nodes[sub_nid] = transformed
            updated_subgraphs[branch_name] = new_sub
        else:
            updated_subgraphs[branch_name] = sub_g
    cloned.subgraphs = updated_subgraphs
    return cloned


LOGICAL_NODE_EQ: Eq[LogicalNode] = Eq(eq=_nodes_equal)
LOGICAL_GRAPH_EQ: Eq[LogicalGraph] = Eq(eq=_graphs_equal)
LOGICAL_GRAPH_PRETTY: Pretty[LogicalGraph] = Pretty(render=_render_logical_graph)
LOGICAL_NODE_REWRITE: EndoRewrite[LogicalNode] = EndoRewrite(map_children=_map_logical_node_children)


def rewrite_graph_recursive(
    graph: LogicalGraph,
    node_fn: Callable[[LogicalNode], LogicalNode],
) -> LogicalGraph:
    """Bottom-up catamorphic rewrite applying ``node_fn`` to every node in ``graph`` and nested ``subgraphs``.

    Args:
        graph (LogicalGraph): Input computation graph.
        node_fn (Callable[[LogicalNode], LogicalNode]): Pure node endomorphism.

    Returns:
        LogicalGraph: Rewritten computation graph with original graph unmutated.
    """

    def _visit_node(n: LogicalNode) -> LogicalNode:
        """Recursively rewrite nested subgraph children of ``n`` before applying ``node_fn``.

        Args:
            n (LogicalNode): Node to visit.

        Returns:
            LogicalNode: Transformed node.
        """
        children_mapped = LOGICAL_NODE_REWRITE.map_children(n, _visit_node)
        return node_fn(children_mapped)

    out_graph = LogicalGraph(
        name=graph.name,
        inputs=list(graph.inputs),
        outputs=list(graph.outputs),
    )
    for nid, node in graph.nodes.items():
        rewritten = _visit_node(node)
        out_graph.nodes[nid] = rewritten
    return out_graph


def fold_graph_monoid(
    graph: LogicalGraph,
    monoid: Monoid[_M],
    node_fn: Callable[[LogicalNode, LogicalGraph], _M],
) -> _M:
    """Fold every node in ``graph`` (and recursively all ``node.subgraphs``) into ``monoid``.

    Args:
        graph (LogicalGraph): Root ``LogicalGraph`` to fold.
        monoid (Monoid[_M]): Target monoid witness providing ``empty`` and ``semigroup.combine``.
        node_fn (Callable[[LogicalNode, LogicalGraph], _M]): Per-node projection receiving ``(node, owner_graph)``.

    Returns:
        _M: Monoidal summary accumulated across ``graph`` and all nested subgraphs.
    """
    return fold_logical_graph_monoid(graph, monoid, node_fn)


def fixpoint_endo_pass(
    step: Endo[LogicalGraph],
    eq_witness: Eq[LogicalGraph] = LOGICAL_GRAPH_EQ,
    max_iterations: int = 16,
) -> Endo[LogicalGraph]:
    """Lift a single-step ``Endo[LogicalGraph]`` pass into a bounded fixed-point ``Endo[LogicalGraph]``.

    Args:
        step (Endo[LogicalGraph]): Single optimization endomorphism step.
        eq_witness (Eq[LogicalGraph]): Structural graph equality witness detecting convergence.
        max_iterations (int): Maximum iteration budget (must be ``>= 1``).

    Returns:
        Endo[LogicalGraph]: Fixed-point endomorphism halting upon ``eq_witness`` convergence or budget exhaustion.

    Raises:
        ValueError: If ``max_iterations < 1``.
    """
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")

    def _iterate(g: LogicalGraph) -> LogicalGraph:
        """Apply ``step.app`` repeatedly until convergence or ``max_iterations``.

        Args:
            g (LogicalGraph): Initial graph.

        Returns:
            LogicalGraph: Converged or budget-bounded graph.
        """
        curr = g
        for _ in range(max_iterations):
            nxt = step.app(curr)
            if eq_witness.eq(curr, nxt):
                return nxt
            curr = nxt
        return curr

    return Endo(app=_iterate)


@dataclasses.dataclass(frozen=True)
class TypedCompilerPass(typing.Generic[_P1, _P2]):
    """Phase-indexed Kleisli compiler pass ``ProvenGraph[_P1] -> Result[str, ProvenGraph[_P2]]``."""

    name: str
    run_pass: Callable[[ProvenGraph[_P1]], Result[str, ProvenGraph[_P2]]]


def compose_typed_passes(
    first: TypedCompilerPass[_P1, _P2],
    second: TypedCompilerPass[_P2, _P3],
) -> TypedCompilerPass[_P1, _P3]:
    """Compose two phase-indexed passes via Kleisli composition (``first >=> second``).

    Args:
        first (TypedCompilerPass[_P1, _P2]): Upstream pass producing ``ProvenGraph[_P2]``.
        second (TypedCompilerPass[_P2, _P3]): Downstream pass consuming ``ProvenGraph[_P2]``.

    Returns:
        TypedCompilerPass[_P1, _P3]: Composed phase-checked pipeline pass.
    """

    def _composed(inp: ProvenGraph[_P1]) -> Result[str, ProvenGraph[_P3]]:
        """Run ``first`` then short-circuit on ``Err`` or feed ``ProvenGraph[_P2]`` into ``second``.

        Args:
            inp (ProvenGraph[_P1]): Input proven graph in phase ``_P1``.

        Returns:
            Result[str, ProvenGraph[_P3]]: Output proven graph in phase ``_P3`` or error message.
        """
        mid = first.run_pass(inp)
        if isinstance(mid, Err):
            return mid
        return second.run_pass(mid.value)

    return TypedCompilerPass(name=f"{first.name} >>> {second.name}", run_pass=_composed)


def _run_schema_proof(proven: ProvenGraph[AcyclicSortedPhase]) -> Result[str, ProvenGraph[SchemaValidPhase]]:
    """Transition ``AcyclicSortedPhase`` to ``SchemaValidPhase`` via ``prove_schema_valid``.

    Args:
        proven (ProvenGraph[AcyclicSortedPhase]): Acyclic topologically sorted proven graph.

    Returns:
        Result[str, ProvenGraph[SchemaValidPhase]]: ``Ok`` schema-proven graph or ``Err`` summary.
    """
    res = prove_schema_valid(proven.graph)
    if isinstance(res, Invalid):
        return Err("; ".join(str(e) for e in res.errors))
    return Ok(res.value)


def _run_sharding_proof(proven: ProvenGraph[SchemaValidPhase]) -> Result[str, ProvenGraph[ShardingValidPhase]]:
    """Transition ``SchemaValidPhase`` to ``ShardingValidPhase`` via ``prove_sharding_valid``.

    Args:
        proven (ProvenGraph[SchemaValidPhase]): Schema-validated proven graph.

    Returns:
        Result[str, ProvenGraph[ShardingValidPhase]]: ``Ok`` sharding-proven graph or ``Err`` summary.
    """
    res = prove_sharding_valid(proven.graph)
    if isinstance(res, Invalid):
        return Err("; ".join(str(e) for e in res.errors))
    return Ok(res.value)


SCHEMA_PROOF_PASS: TypedCompilerPass[AcyclicSortedPhase, SchemaValidPhase] = TypedCompilerPass(
    name="prove_schema_valid",
    run_pass=_run_schema_proof,
)

SHARDING_PROOF_PASS: TypedCompilerPass[SchemaValidPhase, ShardingValidPhase] = TypedCompilerPass(
    name="prove_sharding_valid",
    run_pass=_run_sharding_proof,
)


def build_grounding_proof_pass(
    snapshots_path: GroundingManifestArg = None,
) -> TypedCompilerPass[ShardingValidPhase, GroundedPhase]:
    """Build a ``ShardingValidPhase -> GroundedPhase`` pass parameterized by ``snapshots_path``.

    Args:
        snapshots_path (GroundingManifestArg): Optional snapshot manifest path or mapping.

    Returns:
        TypedCompilerPass[ShardingValidPhase, GroundedPhase]: Grounding proof compiler pass.
    """

    def _run_grounding(proven: ProvenGraph[ShardingValidPhase]) -> Result[str, ProvenGraph[GroundedPhase]]:
        """Verify upstream snapshot grounding for ``proven.graph``.

        Args:
            proven (ProvenGraph[ShardingValidPhase]): Sharding-validated proven graph.

        Returns:
            Result[str, ProvenGraph[GroundedPhase]]: ``Ok`` grounded graph or ``Err`` diagnostics.
        """
        res = prove_grounded(proven.graph, snapshots_path=snapshots_path)
        if isinstance(res, Invalid):
            return Err("; ".join(str(e) for e in res.errors))
        return Ok(res.value)

    return TypedCompilerPass(name="prove_grounded", run_pass=_run_grounding)


def prove_full_graph_pipeline(
    graph: LogicalGraph,
    snapshots_path: GroundingManifestArg = None,
) -> Result[str, ProvenGraph[GroundedPhase]]:
    """Execute the full Kleisli proof pipeline ``AcyclicSorted -> SchemaValid -> ShardingValid -> Grounded``.

    Args:
        graph (LogicalGraph): Unproven input ``LogicalGraph``.
        snapshots_path (GroundingManifestArg): Optional snapshot manifest path or mapping.

    Returns:
        Result[str, ProvenGraph[GroundedPhase]]: ``Ok(ProvenGraph[GroundedPhase])`` or ``Err(str)``.
    """
    acyclic_res = prove_acyclic_sorted(graph)
    if isinstance(acyclic_res, Err):
        return Err(str(acyclic_res.error))
    pipeline = compose_typed_passes(
        compose_typed_passes(SCHEMA_PROOF_PASS, SHARDING_PROOF_PASS),
        build_grounding_proof_pass(snapshots_path=snapshots_path),
    )
    return pipeline.run_pass(acyclic_res.value)


__all__ = [
    "LOGICAL_GRAPH_EQ",
    "LOGICAL_GRAPH_PRETTY",
    "LOGICAL_NODE_EQ",
    "LOGICAL_NODE_REWRITE",
    "SCHEMA_PROOF_PASS",
    "SHARDING_PROOF_PASS",
    "TypedCompilerPass",
    "build_grounding_proof_pass",
    "compose_typed_passes",
    "fixpoint_endo_pass",
    "fold_graph_monoid",
    "prove_full_graph_pipeline",
    "rewrite_graph_recursive",
]
