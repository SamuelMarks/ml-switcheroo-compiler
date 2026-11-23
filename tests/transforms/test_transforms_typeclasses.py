"""Unit tests for phase-indexed TypedCompilerPass, Endo fixpoints, recursive rewrites, and subgraph FLOP folds."""

from __future__ import annotations

import pytest
from ml_switcheroo_ir import (
    Endo,
    Err,
    LogicalGraph,
    LogicalMesh,
    LogicalNode,
    Ok,
    PartitionSpec,
)
from ml_switcheroo_ir.typeclasses import INT_SUM_MONOID

from ml_switcheroo_compiler.diagnostics.flop_counter import estimate_flops
from ml_switcheroo_compiler.ir.core import clone_logical_node
from ml_switcheroo_compiler.transforms import (
    LOGICAL_GRAPH_EQ,
    LOGICAL_GRAPH_PRETTY,
    LOGICAL_NODE_EQ,
    SCHEMA_PROOF_PASS,
    SHARDING_PROOF_PASS,
    TypedCompilerPass,
    build_grounding_proof_pass,
    compose_typed_passes,
    fixpoint_endo_pass,
    fold_graph_monoid,
    prove_full_graph_pipeline,
    rewrite_graph_recursive,
)


def _make_subgraph_ir() -> LogicalGraph:
    """Construct a LogicalGraph containing a nested control-flow subgraph for testing.

    Returns:
        LogicalGraph: Outer graph with an If node owning true/false subgraphs.
    """
    true_branch = LogicalGraph(
        name="true_branch",
        nodes=[
            LogicalNode(id="t_in", op_type="Input", shape_metadata=(4, 4)),
            LogicalNode(id="t_relu", op_type="Relu", inputs=["t_in"], shape_metadata=(4, 4)),
        ],
    )
    false_branch = LogicalGraph(
        name="false_branch",
        nodes=[
            LogicalNode(id="f_in", op_type="Input", shape_metadata=(4, 4)),
            LogicalNode(id="f_neg", op_type="Neg", inputs=["f_in"], shape_metadata=(4, 4)),
        ],
    )
    outer = LogicalGraph(
        name="outer_cfg",
        inputs=["x"],
        outputs=["if_out"],
        nodes=[
            LogicalNode(id="x", op_type="Input", shape_metadata=(4, 4)),
            LogicalNode(
                id="if_out",
                op_type="If",
                inputs=["x"],
                shape_metadata=(4, 4),
                subgraphs={"then": true_branch, "else": false_branch},
            ),
        ],
    )
    return outer


def test_logical_graph_eq_pretty_rewrite_and_subgraph_fold() -> None:
    """Verify LOGICAL_GRAPH_EQ, LOGICAL_GRAPH_PRETTY, rewrite_graph_recursive, and subgraph estimate_flops."""
    g1 = _make_subgraph_ir()
    g2 = _make_subgraph_ir()
    assert LOGICAL_GRAPH_EQ.eq(g1, g2)
    assert LOGICAL_NODE_EQ.eq(g1.nodes["x"], g2.nodes["x"])
    assert not LOGICAL_NODE_EQ.eq(g1.nodes["x"], g2.nodes["if_out"])

    g_diff_name = _make_subgraph_ir()
    g_diff_name.name = "other_name"
    assert not LOGICAL_GRAPH_EQ.eq(g1, g_diff_name)

    g_diff_keys = _make_subgraph_ir()
    g_diff_keys.nodes["extra"] = LogicalNode(id="extra", op_type="Identity", inputs=["x"])
    assert not LOGICAL_GRAPH_EQ.eq(g1, g_diff_keys)

    g_diff_subkeys = _make_subgraph_ir()
    g_diff_subkeys.nodes["if_out"].subgraphs = {"only_then": LogicalGraph(name="only_then")}
    assert not LOGICAL_GRAPH_EQ.eq(g1, g_diff_subkeys)

    rendered = LOGICAL_GRAPH_PRETTY.render(g1)
    assert "LogicalGraph(name='outer_cfg', nodes=2)" in rendered
    assert "%if_out: If(x)" in rendered

    # Rewrite Relu -> Gelu recursively inside nested subgraphs
    def _relu_to_gelu(node: LogicalNode) -> LogicalNode:
        """Replace Relu op_type with Gelu on cloned node.

        Args:
            node (LogicalNode): Candidate node.

        Returns:
            LogicalNode: Updated or unchanged node.
        """
        cloned = clone_logical_node(node)
        if cloned.op_type == "Relu":
            cloned.op_type = "Gelu"
        return cloned

    rewritten = rewrite_graph_recursive(g1, _relu_to_gelu)
    orig_then = g1.nodes["if_out"].subgraphs["then"]
    rewr_then = rewritten.nodes["if_out"].subgraphs["then"]
    assert isinstance(orig_then, LogicalGraph) and orig_then.nodes["t_relu"].op_type == "Relu"
    assert isinstance(rewr_then, LogicalGraph) and rewr_then.nodes["t_relu"].op_type == "Gelu"

    # Also verify non-LogicalGraph attribute entry in subgraphs is preserved and compared cleanly
    g_attr_sub1 = LogicalGraph(name="attr_sub", nodes=[LogicalNode(id="a", op_type="If", subgraphs={"tag": "v1"})])
    g_attr_sub2 = LogicalGraph(name="attr_sub", nodes=[LogicalNode(id="a", op_type="If", subgraphs={"tag": "v1"})])
    g_attr_mix = LogicalGraph(
        name="attr_sub",
        nodes=[LogicalNode(id="a", op_type="If", subgraphs={"tag": LogicalGraph(name="inner")})],
    )
    assert LOGICAL_GRAPH_EQ.eq(g_attr_sub1, g_attr_sub2)
    assert not LOGICAL_GRAPH_EQ.eq(g_attr_sub1, g_attr_mix)
    rewr_attr = rewrite_graph_recursive(g_attr_sub1, _relu_to_gelu)
    assert rewr_attr.nodes["a"].subgraphs["tag"] == "v1"
    assert fold_graph_monoid(g_attr_sub1, INT_SUM_MONOID, lambda _n, _g: 1) == 1

    node_count = fold_graph_monoid(rewritten, INT_SUM_MONOID, lambda _n, _g: 1)
    assert node_count == 6

    # Outer If (16) + true_branch Relu (16) + false_branch Neg (16) = 48 FLOPs
    assert estimate_flops(g1) == 48


def test_fixpoint_endo_pass_convergence_and_budget() -> None:
    """Verify fixpoint_endo_pass halts on Eq convergence, exhausts max_iterations, and rejects < 1."""
    with pytest.raises(ValueError, match="max_iterations must be >= 1"):
        fixpoint_endo_pass(Endo(app=lambda g: g), max_iterations=0)

    def _peel_identity_step(g: LogicalGraph) -> LogicalGraph:
        """Rewrite one Identity_2 -> Identity_1 -> Constant step per iteration.

        Args:
            g (LogicalGraph): Input graph.

        Returns:
            LogicalGraph: Graph stepped once toward convergence.
        """

        def _step_node(n: LogicalNode) -> LogicalNode:
            """Advance synthetic op_type countdown by 1 step.

            Args:
                n (LogicalNode): Node to update.

            Returns:
                LogicalNode: Stepped node.
            """
            c = clone_logical_node(n)
            if c.op_type == "Step2":
                c.op_type = "Step1"
            elif c.op_type == "Step1":
                c.op_type = "Converged"
            return c

        return rewrite_graph_recursive(g, _step_node)

    g_start = LogicalGraph(name="fp", nodes=[LogicalNode(id="n0", op_type="Step2")])
    converged = fixpoint_endo_pass(Endo(app=_peel_identity_step), max_iterations=5).app(g_start)
    assert converged.nodes["n0"].op_type == "Converged"

    budget_stopped = fixpoint_endo_pass(Endo(app=_peel_identity_step), max_iterations=1).app(g_start)
    assert budget_stopped.nodes["n0"].op_type == "Step1"


def test_typed_compiler_pass_kleisli_and_full_proof_pipeline() -> None:
    """Verify compose_typed_passes short-circuiting and prove_full_graph_pipeline across all 4 phases."""
    valid_g = LogicalGraph(
        name="valid_pipeline",
        inputs=["x"],
        outputs=["y"],
        nodes=[LogicalNode(id="y", op_type="Relu", inputs=["x"], shape_metadata=(2,))],
    )
    ok_grounded = prove_full_graph_pipeline(valid_g, snapshots_path=None)
    assert isinstance(ok_grounded, Ok)
    assert [n.id for n in ok_grounded.value.ordered_nodes] == ["y"]

    # Cyclic graph fails at phase 1 (prove_acyclic_sorted)
    cyclic_g = LogicalGraph(
        name="cyclic",
        nodes=[
            LogicalNode(id="a", op_type="Relu", inputs=["b"]),
            LogicalNode(id="b", op_type="Relu", inputs=["a"]),
        ],
    )
    err_cyclic = prove_full_graph_pipeline(cyclic_g)
    assert isinstance(err_cyclic, Err)

    # Invalid schema (dangling input reference) fails at SCHEMA_PROOF_PASS and short-circuits downstream
    dangling_g = LogicalGraph(
        name="dangling",
        nodes=[LogicalNode(id="y", op_type="Relu", inputs=["missing_input"])],
    )
    err_schema = prove_full_graph_pipeline(dangling_g)
    assert isinstance(err_schema, Err) and "missing_input" in err_schema.error

    # Invalid GSPMD elementwise sharding transition ('dp' -> 'tp' without collective) fails at SHARDING_PROOF_PASS
    bad_shard_g = LogicalGraph(
        name="bad_shard",
        inputs=["x"],
        outputs=["y"],
        mesh=LogicalMesh(shape={"dp": 2, "tp": 2}),
        nodes=[
            LogicalNode(
                id="a",
                op_type="Relu",
                inputs=["x"],
                sharding=PartitionSpec(axes=("dp",)),
            ),
            LogicalNode(
                id="y",
                op_type="Add",
                inputs=["a", "a"],
                sharding=PartitionSpec(axes=("tp",)),
            ),
        ],
    )
    err_shard = prove_full_graph_pipeline(bad_shard_g)
    assert isinstance(err_shard, Err)

    # Foreign ungrounded dialect op fails at build_grounding_proof_pass when manifest rejects it
    foreign_g = LogicalGraph(
        name="foreign_bad",
        inputs=["x"],
        outputs=["f"],
        nodes=[LogicalNode(id="f", op_type="nonexistent_op_xyz", domain="torch", inputs=["x"])],
    )
    err_ground = prove_full_graph_pipeline(foreign_g, snapshots_path={"torch": frozenset({"torch.add"})})
    assert isinstance(err_ground, Err)

    # Also invoke build_grounding_proof_pass directly to hit its Invalid branch
    from ml_switcheroo_ir import ProvenGraph, ShardingValidPhase

    direct_ground_err = build_grounding_proof_pass(snapshots_path={"torch": frozenset({"torch.add"})}).run_pass(
        ProvenGraph[ShardingValidPhase](graph=foreign_g, ordered_nodes=(foreign_g.nodes["f"],))
    )
    assert isinstance(direct_ground_err, Err) and "nonexistent_op_xyz" in direct_ground_err.error

    # Verify Kleisli composition naming
    composed = compose_typed_passes(
        compose_typed_passes(SCHEMA_PROOF_PASS, SHARDING_PROOF_PASS),
        build_grounding_proof_pass(),
    )
    assert composed.name == "prove_schema_valid >>> prove_sharding_valid >>> prove_grounded"
    from ml_switcheroo_ir import AcyclicSortedPhase, prove_acyclic_sorted

    proven_ok = prove_acyclic_sorted(valid_g)
    assert isinstance(proven_ok, Ok)
    custom_err_pass: TypedCompilerPass[AcyclicSortedPhase, AcyclicSortedPhase] = TypedCompilerPass(
        name="fail_first",
        run_pass=lambda _x: Err("boom"),
    )
    chained = compose_typed_passes(custom_err_pass, custom_err_pass)
    chained_res = chained.run_pass(proven_ok.value)
    assert isinstance(chained_res, Err) and chained_res.error == "boom"

    # PassManager integration: to_fixpoint_endo and prove_grounded_pipeline
    from ml_switcheroo_compiler.distributed import ShardingLatticeElem
    from ml_switcheroo_compiler.transforms import PassManager
    from ml_switcheroo_compiler.transforms.passes.spmd import (
        join_input_shardings_lattice,
        sharding_to_lattice_elem,
    )

    pm = PassManager()
    fp_endo = pm.to_fixpoint_endo(max_iterations=3)
    assert LOGICAL_GRAPH_EQ.eq(fp_endo.app(valid_g), valid_g)
    pm_grounded = PassManager.prove_grounded_pipeline(valid_g)
    assert isinstance(pm_grounded, Ok)

    # SPMD sharding lattice helpers (None, str axis, multi-axis tuple, compatible join, conflicting join)
    assert sharding_to_lattice_elem(None) == ShardingLatticeElem.bottom()
    assert sharding_to_lattice_elem(PartitionSpec(axes=("dp", None))) == ShardingLatticeElem.from_axes(["dp"])
    assert sharding_to_lattice_elem(PartitionSpec(axes=(("dp", "tp"),))) == ShardingLatticeElem.from_axes(["dp+tp"])

    spmd_g = LogicalGraph(
        name="spmd_lat",
        nodes=[
            LogicalNode(id="in_dp", op_type="Input", sharding=PartitionSpec(axes=("dp", None))),
            LogicalNode(id="in_tp", op_type="Input", sharding=PartitionSpec(axes=(None, "tp"))),
            LogicalNode(id="in_conflict", op_type="Input", sharding=PartitionSpec(axes=("tp", None))),
            LogicalNode(id="ok_join", op_type="Add", inputs=["in_dp", "in_tp", "missing_ghost"]),
            LogicalNode(id="bad_join", op_type="Add", inputs=["in_dp", "in_conflict"]),
        ],
    )
    assert join_input_shardings_lattice(spmd_g.nodes["ok_join"], spmd_g) == ShardingLatticeElem.from_axes(["dp", "tp"])
    assert join_input_shardings_lattice(spmd_g.nodes["bad_join"], spmd_g) == ShardingLatticeElem.top()
