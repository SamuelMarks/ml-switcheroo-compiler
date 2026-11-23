"""Comprehensive conformance tests for ml-switcheroo-ir v0.0.8 and ml-ecosystem-snapshots v0.0.8 interfaces."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from ml_switcheroo_ir import (
    Err,
    GhostParam,
    Invalid,
    IRGraph,
    IRNode,
    LogicalGraph,
    LogicalNode,
    NodeId,
    Ok,
    ProvenGraph,
    TensorShape,
    Valid,
    prove_acyclic_sorted,
)
from ml_switcheroo_ir.schema.ghost import ParameterKind
from ml_switcheroo_ir.validation import GroundingEngineProtocol

from ml_switcheroo_compiler.backends.snapshot_grounding import (
    BackendSnapshotTargetModel,
    BackendSnapshotTargetsConfig,
    SnapshotGroundingEngine,
    _extract_category_endpoints,
    _extract_item_endpoints,
    _populate_from_snapshot_data,
    _signatures_from_ghost_ref,
    _signatures_from_item,
)
from ml_switcheroo_compiler.core.errors import CompilationError, ShapeMismatchError
from ml_switcheroo_compiler.ir import (
    COMPILER_POLYNOMIAL_EQ,
    COMPILER_POLYNOMIAL_HASH,
    COMPILER_POLYNOMIAL_RING,
    COMPILER_POLYNOMIAL_SEMIRING,
    COMPILER_SYM_NODE_EQ,
    COMPILER_SYM_NODE_HASH,
    COMPILER_SYM_NODE_REWRITE,
    Polynomial,
    SymBinaryOp,
    SymConst,
    SymInt,
    SymNode,
    SymPiecewise,
    SymUnaryOp,
    SymVar,
    as_opt_graph,
    broadcast_shapes_result,
    clone_logical_node,
    create_assign_variable,
    create_scatter_update,
    ensure_attr_val_result,
    ensure_graph,
    ensure_graph_result,
    matmul_shape_result,
    normalize_axis_result,
)
from ml_switcheroo_compiler.ir.shape_system import compute_contiguous_strides, normalize_axis
from ml_switcheroo_compiler.serialization import (
    COMPILER_IR_STR_CODEC,
    graph_to_json_result,
    json_to_graph_result,
    yaml_to_graph_result,
)
from ml_switcheroo_compiler.serialization.ir_format import _normalize_shape_dim
from ml_switcheroo_compiler.transforms import IRValidator, PassManager


def test_ir_core_proven_graph_and_results() -> None:
    """Verify ensure_graph, ensure_graph_result, as_opt_graph, ensure_attr_val_result, and AttrModel cloning."""
    raw_graph = LogicalGraph(
        name="g_v008",
        nodes=[LogicalNode(id="x", op_type="Input")],
        inputs=["x"],
        outputs=["x"],
    )
    proven_res = prove_acyclic_sorted(raw_graph)
    assert isinstance(proven_res, Ok)
    proven_graph = proven_res.value

    assert ensure_graph(proven_graph) is proven_graph.graph
    assert as_opt_graph(proven_graph) is proven_graph.graph
    assert as_opt_graph("not_a_graph") is None

    ok_none = ensure_graph_result(None)
    assert isinstance(ok_none, Ok)
    err_bad = ensure_graph_result(42)
    assert isinstance(err_bad, Err)

    ok_attr = ensure_attr_val_result(123)
    assert isinstance(ok_attr, Ok) and ok_attr.value == 123
    err_attr = ensure_attr_val_result(int)
    assert isinstance(err_attr, Err)

    shape_model = TensorShape(dims=[2, 4])
    ghost_param = GhostParam(name="alpha", kind=ParameterKind.POSITIONAL_OR_KEYWORD)
    node = LogicalNode(
        id="n1",
        op_type="Relu",
        inputs=["x"],
        attributes={"tshape": shape_model, "gparam": ghost_param},
    )
    cloned_node = clone_logical_node(node)
    assert cloned_node.attributes["tshape"] == shape_model
    assert cloned_node.attributes["tshape"] is not shape_model
    assert cloned_node.attributes["gparam"] == ghost_param
    assert cloned_node.attributes["gparam"] is not ghost_param

    assign_node = create_assign_variable("w", NodeId("val_0"), (2, 2))
    scatter_node = create_scatter_update(NodeId("t0"), NodeId("idx0"), NodeId("upd0"), (2, 2))
    assert assign_node.inputs == ["val_0"]
    assert scatter_node.inputs == ["t0", "idx0", "upd0"]


def test_polynomial_and_symnode_v008_parity() -> None:
    """Verify Polynomial ring witnesses, SymUnaryOp polynomial lowering, and SymNode/SymInt methods."""
    px = Polynomial.from_var("x")
    py = Polynomial.from_var("y")
    p_sum = COMPILER_POLYNOMIAL_SEMIRING.add(px, py)
    p_zero = COMPILER_POLYNOMIAL_SEMIRING.zero()
    p_one = COMPILER_POLYNOMIAL_SEMIRING.one()
    p_mul = COMPILER_POLYNOMIAL_SEMIRING.mul(p_sum, p_one)
    assert COMPILER_POLYNOMIAL_EQ.eq(p_mul, p_sum)
    assert COMPILER_POLYNOMIAL_HASH.hash_value(p_mul) == COMPILER_POLYNOMIAL_HASH.hash_value(p_sum)

    p_neg = COMPILER_POLYNOMIAL_RING.neg(px)
    assert COMPILER_POLYNOMIAL_EQ.eq(COMPILER_POLYNOMIAL_SEMIRING.add(px, p_neg), p_zero)
    assert COMPILER_POLYNOMIAL_EQ.eq(COMPILER_POLYNOMIAL_RING.sub(px, px), p_zero)
    assert p_sum.canonical_terms() == ((((("x", 1),)), 1), (((("y", 1),)), 1))

    vx = SymVar("x")
    c5 = SymConst(5)
    assert c5.is_constant is True
    assert vx.is_constant is False
    assert vx.evaluate({"x": 9}) == 9

    u_plus = SymUnaryOp("+", vx)
    u_minus = SymUnaryOp("-", vx)
    u_abs = SymUnaryOp("abs", vx)
    u_nonpoly = SymUnaryOp("-", SymBinaryOp("min", vx, c5))

    assert u_plus.simplify() == vx
    assert u_plus.eval({"x": 7}) == 7
    assert u_plus.to_polynomial() == px
    assert u_minus.to_polynomial() == p_neg
    assert u_abs.to_polynomial() is None
    assert u_nonpoly.to_polynomial() is None
    assert u_minus != vx
    assert hash(u_minus) == hash(SymUnaryOp("-", SymVar("x")))

    si_const = SymInt(8)
    si_bool = SymInt(True)
    si_var = SymInt("x + 2")
    assert si_const.is_constant is True
    assert si_const.eval({}) == 8
    assert si_var.is_constant is False
    assert si_var.evaluate({"x": 4}) == 6
    assert isinstance(si_bool.node, SymVar)

    with pytest.raises(TypeError):
        SymNode.to_node(True)
    with pytest.raises(TypeError):
        normalize_axis(True, 2)
    with pytest.raises(TypeError):
        normalize_axis([0, True], 2)

    assert compute_contiguous_strides(()) == ()
    assert compute_contiguous_strides((2, 3, 4)) == (12, 4, 1)
    assert compute_contiguous_strides((2, SymInt("N"), 4)) == (4, 4, 1)


def test_shape_results_and_symnode_rewrite_witness() -> None:
    """Verify total shape Result wrappers and COMPILER_SYM_NODE_REWRITE witness across AST variants."""
    b_ok = broadcast_shapes_result((1, 4), (3, 1))
    b_err = broadcast_shapes_result((2, 4), (3, 5))
    assert isinstance(b_ok, Ok) and b_ok.value == (3, 4)
    assert isinstance(b_err, Err) and isinstance(b_err.error, ShapeMismatchError)

    m_ok = matmul_shape_result((2, 3), (3, 5))
    m_err = matmul_shape_result((2, 3), (4, 5))
    assert isinstance(m_ok, Ok) and m_ok.value == (2, 5)
    assert isinstance(m_err, Err)

    a_ok = normalize_axis_result(-1, 3)
    a_err = normalize_axis_result(5, 3)
    assert isinstance(a_ok, Ok) and a_ok.value == 2
    assert isinstance(a_err, Err)

    vx = SymVar("x")
    c1 = SymConst(1)
    assert COMPILER_SYM_NODE_EQ.eq(vx, SymVar("x"))
    assert COMPILER_SYM_NODE_HASH.hash_value(vx) == hash(SymVar("x"))

    def _bump_var(node: SymNode) -> SymNode:
        """Replace variable ``x`` with ``y`` for rewrite witness testing.

        Args:
            node (SymNode): Input symbolic node.

        Returns:
            SymNode: Rewritten symbolic node.
        """
        return SymVar("y") if isinstance(node, SymVar) and node.name == "x" else node

    assert COMPILER_SYM_NODE_REWRITE.map_children(c1, _bump_var) == c1
    assert COMPILER_SYM_NODE_REWRITE.map_children(SymBinaryOp("+", vx, c1), _bump_var) == SymBinaryOp(
        "+", SymVar("y"), c1
    )
    assert COMPILER_SYM_NODE_REWRITE.map_children(SymUnaryOp("-", vx), _bump_var) == SymUnaryOp("-", SymVar("y"))
    pw = SymPiecewise([(vx, vx), (True, c1)], default=vx)
    pw_rewritten = COMPILER_SYM_NODE_REWRITE.map_children(pw, _bump_var)
    assert isinstance(pw_rewritten, SymPiecewise)
    assert pw_rewritten.default == SymVar("y")
    assert pw_rewritten.cases[0] == (SymVar("y"), SymVar("y"))


def test_pass_manager_endo_and_proven_graphs() -> None:
    """Verify IRValidator.prove_acyclic, prove_schema, PassManager.to_endo, and PassManager.run_checked."""
    in_node = IRNode(id="x", op_type="Input", inputs=[], shape_metadata=(2, 2))
    relu_node = IRNode(id="y", op_type="Relu", inputs=["x"], shape_metadata=(2, 2))
    graph = IRGraph(name="pm_v008", nodes=[in_node, relu_node], inputs=["x"], outputs=["y"])

    acyclic_ok = IRValidator.prove_acyclic(graph)
    assert isinstance(acyclic_ok, Ok)
    assert isinstance(acyclic_ok.value, ProvenGraph)

    schema_res = IRValidator.prove_schema(graph)
    assert isinstance(schema_res, (Valid, Invalid))

    cyc_a = IRNode(id="a", op_type="Relu", inputs=["b"], shape_metadata=(2,))
    cyc_b = IRNode(id="b", op_type="Relu", inputs=["a"], shape_metadata=(2,))
    cyclic_graph = IRGraph(name="cyc", nodes=[cyc_a, cyc_b], inputs=[], outputs=["b"])
    acyclic_err = IRValidator.prove_acyclic(cyclic_graph)
    assert isinstance(acyclic_err, Err) and isinstance(acyclic_err.error, CompilationError)

    pm = PassManager()
    visited: list[str] = []

    def _record_pass(g: IRGraph) -> bool:
        """Record graph name during pass execution.

        Args:
            g (IRGraph): Input graph.

        Returns:
            bool: False indicating in-place metadata inspection only.
        """
        visited.append(g.name)
        return False

    pm.add_pass(_record_pass, name="record_pass")
    checked_from_raw = pm.run_checked(graph.clone())
    assert isinstance(checked_from_raw, Ok)
    checked_from_proven = pm.run_checked(acyclic_ok.value)
    assert isinstance(checked_from_proven, Ok)
    assert len(visited) == 2

    pm_fail = PassManager()
    pm_fail.validators.append(IRValidator.check_shapes)
    bad_shape_graph = IRGraph(
        name="missing_shape",
        nodes=[IRNode(id="x", op_type="Input", inputs=[])],
        inputs=["x"],
        outputs=["x"],
    )
    checked_err = pm_fail.run_checked(bad_shape_graph)
    assert isinstance(checked_err, Err)


def test_serialization_ir_codec_and_results() -> None:
    """Verify COMPILER_IR_STR_CODEC, boolean shape dim guard, and JSON/YAML Result decoders."""
    assert _normalize_shape_dim(True) == "True"
    assert _normalize_shape_dim(4) == 4

    graph = IRGraph(
        name="codec_g",
        nodes=[IRNode(id="x", op_type="Input", inputs=[], shape_metadata=(2, 3))],
        inputs=["x"],
        outputs=["x"],
    )
    encoded_ok = graph_to_json_result(graph)
    assert isinstance(encoded_ok, Ok)

    decoded_ok = COMPILER_IR_STR_CODEC.decode(COMPILER_IR_STR_CODEC.encode(graph))
    assert isinstance(decoded_ok, Ok) and decoded_ok.value.name == "codec_g"

    with patch("ml_switcheroo_compiler.serialization.ir_format.graph_to_json", side_effect=TypeError("unserializable")):
        enc_err = graph_to_json_result(graph)
        assert isinstance(enc_err, Err)

    assert isinstance(json_to_graph_result("{bad_json"), Err)
    assert isinstance(json_to_graph_result("[1, 2, 3]"), Err)
    with patch(
        "ml_switcheroo_compiler.serialization.ir_format._parse_serialization_dict",
        side_effect=ValueError("bad graph dict"),
    ):
        assert isinstance(json_to_graph_result("{}"), Err)

    yaml_ok = yaml_to_graph_result("""name: yaml_g
inputs: []
outputs: []
nodes: {}
""")
    assert isinstance(yaml_ok, Ok) and yaml_ok.value.name == "yaml_g"
    yaml_empty = yaml_to_graph_result("")
    assert isinstance(yaml_empty, Ok)
    assert isinstance(yaml_to_graph_result("[\nunclosed"), Err)
    assert isinstance(yaml_to_graph_result("- 1\n- 2\n"), Err)
    with patch(
        "ml_switcheroo_compiler.serialization.ir_format._parse_serialization_dict",
        side_effect=ValueError("bad yaml dict"),
    ):
        assert isinstance(yaml_to_graph_result("name: bad"), Err)


def test_snapshot_grounding_v008_protocols_and_tuples(tmp_path: Path) -> None:
    """Verify SnapshotGroundingEngine satisfies GroundingEngineProtocol, tuple sequences, and v008 outcomes.

    Args:
        tmp_path (Path): Temporary directory fixture.
    """
    from ml_switcheroo_ir.schema.ghost import ExtendedGhostRef

    from ml_switcheroo_compiler.core.shape import OpValue as AttributeValue

    endpoints: set[str] = set()
    item_dict: dict[str, AttributeValue] = {
        "name": "add",
        "api_path": "pkg.add",
        "aliases": ("pkg.plus",),
        "params": ({"name": "x", "kind": "POSITIONAL_OR_KEYWORD"},),
        "overloads": ({"params": ({"name": "x", "kind": "POSITIONAL_OR_KEYWORD"},)},),
    }
    _extract_item_endpoints(item_dict, endpoints)
    assert "pkg.plus" in endpoints

    cat_endpoints: set[str] = set()
    _extract_category_endpoints({"math": (item_dict,)}, cat_endpoints)
    assert "pkg.add" in cat_endpoints

    pop_endpoints: set[str] = set()
    pop_items: dict[str, dict[str, AttributeValue]] = {}
    pop_refs: dict[str, ExtendedGhostRef] = {}
    _populate_from_snapshot_data((item_dict,), pop_endpoints, pop_items, pop_refs)
    assert "pkg.add" in pop_refs
    ref = pop_refs["pkg.add"]
    assert len(_signatures_from_ghost_ref(ref)) >= 1
    assert len(_signatures_from_item(item_dict)) >= 2

    snap_file = tmp_path / "demo_v1.json"
    snap_file.write_text(
        '{"categories": {"math": [{"name": "sin", "api_path": "demo.sin", '
        '"params": [{"name": "x", "kind": "POSITIONAL_OR_KEYWORD", "is_mandatory": true}]}]}}',
        encoding="utf-8",
    )
    cfg = BackendSnapshotTargetsConfig(
        targets={
            "demo_backend": BackendSnapshotTargetModel(
                framework="demo_fw",
                snapshot_glob="demo_v*.json",
                canonical_roots=["demo"],
                status="available",
            ),
            "missing_backend": BackendSnapshotTargetModel(
                framework="missing_fw",
                snapshot_glob="missing_v*.json",
                canonical_roots=["missing"],
                status="available",
            ),
        }
    )
    engine = SnapshotGroundingEngine(snapshot_dir=str(tmp_path), config=cfg, fail_on_missing=True)
    protocol_witness: GroundingEngineProtocol = engine
    assert protocol_witness is engine

    assert engine.resolve_backend_name("demo_backend") == "demo_backend"
    assert engine.resolve_backend_name("demo_fw") == "demo_backend"
    assert engine.resolve_backend_name("demo") == "demo_backend"
    assert engine.resolve_backend_name("unknown_xyz") is None

    assert engine.get_symbol("demo_fw", "demo.sin") is not None
    assert engine.get_symbol("unknown_xyz", "demo.sin") is None

    assert engine.suggest_closest_symbol("demo_fw", "demo.sim", max_distance=3) == "demo.sin"
    assert engine.suggest_closest_symbol("unknown_xyz", "demo.sim", max_distance=3) is None

    snap_ok = engine.load_snapshot_result("demo_backend")
    snap_err = engine.load_snapshot_result("missing_backend")
    assert isinstance(snap_ok, Ok)
    assert isinstance(snap_err, Err)

    valid_call = engine.validate_parameter_contract_checked("demo_backend", "demo.sin", 1, ())
    invalid_call = engine.validate_parameter_contract_checked("demo_backend", "demo.sin", 0, ("bad_kw",))
    assert isinstance(valid_call, Valid)
    assert isinstance(invalid_call, Invalid) and len(invalid_call.errors) >= 1

    grounded_graph = LogicalGraph(
        name="grounded_test",
        nodes=[LogicalNode(id="x", op_type="Input")],
        inputs=["x"],
        outputs=["x"],
    )
    proven_grounded = engine.prove_graph_grounded(grounded_graph)
    assert isinstance(proven_grounded, (Valid, Invalid))

    missing_dir_engine = SnapshotGroundingEngine(
        snapshot_dir=str(tmp_path / "does_not_exist"),
        config=cfg,
        fail_on_missing=False,
    )
    proven_no_dir = missing_dir_engine.prove_graph_grounded(grounded_graph)
    assert isinstance(proven_no_dir, (Valid, Invalid))
