"""Tests for test_spmd."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.spmd import (
    _create_all_gather_node,
    _create_all_reduce_node,
    _create_all_to_all_node,
    _create_reduce_scatter_node,
    _inject_all_gather,
    _inject_all_reduce,
    _inject_all_to_all,
    _inject_reduce_scatter,
    _is_boundary_transition,
    _process_spmd_input,
    _process_spmd_node,
    inject_spmd_communication_pass,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


@pytest.fixture(autouse=True)
def _reset_spmd_rules_fixture() -> Iterator[None]:
    """Ensure _SPMD_RULES is always cleared before and after each test."""
    from ml_switcheroo_compiler.transforms.passes import spmd

    spmd._SPMD_RULES = None
    yield
    spmd._SPMD_RULES = None


class DummySharding:
    """Test class DummySharding."""

    def __init__(self, mapping: list[EagerValue]) -> None:
        """Test function __init__."""
        self.mesh_mapping = mapping


def test_get_spmd_rules_yaml_fallback() -> None:
    """Test function test_get_spmd_rules_yaml_fallback."""
    from pathlib import Path
    from unittest.mock import mock_open, patch

    import yaml

    import ml_switcheroo_compiler.transforms.passes.spmd as spmd_mod
    from ml_switcheroo_compiler.transforms.passes.spmd import _get_spmd_rules

    spmd_mod._SPMD_RULES = None

    def mock_exists(self: EagerValue) -> EagerValue:
        """Test function mock_exists."""
        if cast("EagerTensorProtocol", getattr(self, "name")) == "spmd_mappings":
            return False
        if cast("EagerTensorProtocol", getattr(self, "name")) == "spmd_mappings.yaml":
            return True
        return False

    with patch.object(Path, "exists", mock_exists):
        with patch("builtins.open", mock_open(read_data=yaml.dump({"fallback": "rule"}))):
            rules = _get_spmd_rules()
            assert "fallback" in rules

    spmd_mod._SPMD_RULES = None


def test_is_boundary_transition() -> None:
    """Test function test_is_boundary_transition."""
    assert cast("Callable[..., EagerTensorProtocol]", _is_boundary_transition)(cast("Callable[..., EagerTensorProtocol]", DummySharding)([None]), cast("Callable[..., EagerTensorProtocol]", DummySharding)([None])) == (False, False)
    assert cast("Callable[..., EagerTensorProtocol]", _is_boundary_transition)(cast("Callable[..., EagerTensorProtocol]", DummySharding)(["x"]), cast("Callable[..., EagerTensorProtocol]", DummySharding)([None])) == (True, False)
    assert cast("Callable[..., EagerTensorProtocol]", _is_boundary_transition)(cast("Callable[..., EagerTensorProtocol]", DummySharding)([None]), cast("Callable[..., EagerTensorProtocol]", DummySharding)(["x"])) == (False, True)


def test_create_nodes() -> None:
    """Test function test_create_nodes."""
    sharding = DummySharding(["x"])

    n1 = cast("Callable[..., EagerTensorProtocol]", _create_all_gather_node)("inp1", sharding)
    assert n1.id == "inp1_all_gather"
    assert n1.op_type == "AllGather"
    assert bool(n1.attributes.get("dispatch_early")) is True

    n2 = cast("Callable[..., EagerTensorProtocol]", _create_reduce_scatter_node)("inp2", sharding)
    assert n2.id == "inp2_reduce_scatter"
    assert n2.op_type == "ReduceScatter"
    assert bool(n2.attributes.get("dispatch_early")) is True

    n3 = cast("Callable[..., EagerTensorProtocol]", _create_all_reduce_node)("inp3", sharding)
    assert n3.id == "inp3_all_reduce"
    assert n3.op_type == "AllReduce"
    assert bool(n3.attributes.get("dispatch_early")) is True

    n4 = cast("Callable[..., EagerTensorProtocol]", _create_all_to_all_node)("inp4", sharding)
    assert n4.id == "inp4_all_to_all"
    assert n4.op_type == "AllToAll"
    assert bool(n4.attributes.get("dispatch_early")) is True


def test_inject_nodes() -> None:
    """Test function test_inject_nodes."""
    sharding = DummySharding(["x"])

    node1 = IRNode(id="n1", op_type="Add", inputs=["in1", "in2"])
    res1 = cast("Callable[..., EagerTensorProtocol]", _inject_all_gather)(node1, 0, "in1", sharding)
    assert node1.inputs[0] == "in1_all_gather"
    assert res1.id == "in1_all_gather"

    node2 = IRNode(id="n2", op_type="Add", inputs=["in1", "in2"])
    res2 = cast("Callable[..., EagerTensorProtocol]", _inject_reduce_scatter)(node2, 1, "in2", sharding)
    assert node2.inputs[1] == "in2_reduce_scatter"
    assert res2.id == "in2_reduce_scatter"

    node3 = IRNode(id="n3", op_type="Add", inputs=["in3", "in2"])
    res3 = cast("Callable[..., EagerTensorProtocol]", _inject_all_reduce)(node3, 0, "in3", sharding)
    assert node3.inputs[0] == "in3_all_reduce"
    assert res3.id == "in3_all_reduce"

    node4 = IRNode(id="n4", op_type="Add", inputs=["in4", "in2"])
    res4 = cast("Callable[..., EagerTensorProtocol]", _inject_all_to_all)(node4, 0, "in4", sharding)
    assert node4.inputs[0] == "in4_all_to_all"
    assert res4.id == "in4_all_to_all"


def test_process_spmd_input() -> None:
    """Test function test_process_spmd_input."""
    sharding_unsharded = DummySharding([None])
    sharding_sharded_x = DummySharding(["x"])
    sharding_sharded_y = DummySharding(["y"])

    graph = IRGraph()
    in_node1 = cast("Callable[..., EagerTensorProtocol]", IRNode)(id="in1", op_type="Input", inputs=[], sharding=sharding_sharded_x)
    graph.nodes["in1"] = in_node1

    # 1. inp_sharded, not node_sharded, not grad/reduction => all_gather
    node1 = cast("Callable[..., EagerTensorProtocol]", IRNode)(id="n1", op_type="Add", inputs=["in1"], sharding=sharding_unsharded)
    res_ag = cast("Callable[..., EagerTensorProtocol]", _process_spmd_input)(node1, 0, "in1", graph, sharding_unsharded)
    # The default yaml has Add -> inject: AllGather
    assert res_ag is not None and res_ag.op_type == "AllGather"

    # 2. inp_sharded, not node_sharded, reduction => all_reduce
    node2 = cast("Callable[..., EagerTensorProtocol]", IRNode)(id="n2", op_type="ReduceSum", inputs=["in1"], sharding=sharding_unsharded)
    res_ar = cast("Callable[..., EagerTensorProtocol]", _process_spmd_input)(node2, 0, "in1", graph, sharding_unsharded)
    assert res_ar is not None and res_ar.op_type == "AllReduce"

    # 3. not inp_sharded, node_sharded, grad => reduce_scatter
    in_node_unsharded = cast("Callable[..., EagerTensorProtocol]", IRNode)(id="in_un", op_type="Input", inputs=[], sharding=sharding_unsharded)
    graph.nodes["in_un"] = in_node_unsharded
    node_grad = cast("Callable[..., EagerTensorProtocol]", IRNode)(id="grad_node", op_type="Grad", inputs=["in_un"], sharding=sharding_sharded_x)
    res_rs = cast("Callable[..., EagerTensorProtocol]", _process_spmd_input)(node_grad, 0, "in_un", graph, sharding_sharded_x)
    assert res_rs is not None and res_rs.op_type == "ReduceScatter"

    # 4. inp_sharded, node_sharded, different axes => all_to_all
    node3 = cast("Callable[..., EagerTensorProtocol]", IRNode)(id="n3", op_type="Add", inputs=["in1"], sharding=sharding_sharded_y)
    res_a2a = cast("Callable[..., EagerTensorProtocol]", _process_spmd_input)(node3, 0, "in1", graph, sharding_sharded_y)
    assert res_a2a is not None and res_a2a.op_type == "AllToAll"

    # Missing input node
    node4 = cast("Callable[..., EagerTensorProtocol]", IRNode)(id="n4", op_type="Add", inputs=["missing"], sharding=sharding_sharded_y)
    assert cast("Callable[..., EagerTensorProtocol]", _process_spmd_input)(node4, 0, "missing", graph, sharding_sharded_y) is None

    # Missing sharding on input node
    in_node_no_shard = IRNode(id="in_ns", op_type="Input", inputs=[])
    graph.nodes["in_ns"] = in_node_no_shard
    assert cast("Callable[..., EagerTensorProtocol]", _process_spmd_input)(node3, 0, "in_ns", graph, sharding_sharded_y) is None


def test_process_spmd_node() -> None:
    """Test function test_process_spmd_node."""
    node1 = IRNode(id="n1", op_type="Add", inputs=["in1", "in2"])
    graph = IRGraph(nodes={"in1": IRNode(id="in1", op_type="Input", inputs=[]), "in2": IRNode(id="in2", op_type="Input", inputs=[])})

    (mod, inj) = _process_spmd_node(node1, graph)
    assert not mod and not inj

    setattr(node1, "sharding", DummySharding([None]))
    graph.nodes["in1"].sharding = DummySharding(["x"])
    graph.nodes["in2"].sharding = DummySharding([None])
    (mod, inj) = _process_spmd_node(node1, graph)
    assert mod is True
    assert len(inj) == 1
    assert inj[0].op_type == "AllGather"


def test_inject_spmd_communication_pass() -> None:
    """Test function test_inject_spmd_communication_pass."""
    node1 = cast("Callable[..., EagerTensorProtocol]", IRNode)(id="n1", op_type="Add", inputs=["in1"], sharding=cast("Callable[..., EagerTensorProtocol]", DummySharding)([None]))
    graph = cast("Callable[..., EagerTensorProtocol]", IRGraph)(nodes={"in1": cast("Callable[..., EagerTensorProtocol]", IRNode)(id="in1", op_type="Input", inputs=[], sharding=cast("Callable[..., EagerTensorProtocol]", DummySharding)(["x"])), "n1": node1})
    res = cast("Callable[..., EagerTensorProtocol]", inject_spmd_communication_pass)(graph)
    assert bool(res) is True
    assert "in1_all_gather" in graph.nodes


def test_get_sharding_axes_none() -> None:
    """Test function test_get_sharding_axes_none."""
    from ml_switcheroo_compiler.transforms.passes.spmd import _get_sharding_axes

    assert _get_sharding_axes(None) == []


def test_spmd_insert_communications_not_is_grad() -> None:
    """Test function test_spmd_insert_communications_not_is_grad."""
    g = IRGraph()
    n1 = cast("Callable[..., EagerTensorProtocol]", IRNode)(id="n1", op_type="Add", inputs=["n2"], sharding=cast("Callable[..., EagerTensorProtocol]", DummySharding)([0]))
    n2 = IRNode(id="n2", op_type="Input", sharding=None)
    g.nodes["n1"] = n1
    g.nodes["n2"] = n2

    res = inject_spmd_communication_pass(g)
    assert res is False  # Not is_grad, so it doesn't insert reduce_scatter


def test_spmd_insert_communications_axes_len_mismatch() -> None:
    """Test function test_spmd_insert_communications_axes_len_mismatch."""
    g = IRGraph()
    n1 = cast("Callable[..., EagerTensorProtocol]", IRNode)(id="n1", op_type="Add", inputs=["n2"], sharding=cast("Callable[..., EagerTensorProtocol]", DummySharding)([0]))
    n2 = cast("Callable[..., EagerTensorProtocol]", IRNode)(id="n2", op_type="Input", sharding=cast("Callable[..., EagerTensorProtocol]", DummySharding)([0, 1]))
    g.nodes["n1"] = n1
    g.nodes["n2"] = n2

    res = inject_spmd_communication_pass(g)
    assert res is False  # Length of axes mismatch, no all_to_all


def test_spmd_insert_communications_not_inp_sharded_node_sharded_not_grad() -> None:
    """Test function test_spmd_insert_communications_not_inp_sharded_node_sharded_not_grad."""
    g = IRGraph()
    n1 = cast("Callable[..., EagerTensorProtocol]", IRNode)(id="n1", op_type="Add", inputs=["n2"], sharding=cast("Callable[..., EagerTensorProtocol]", DummySharding)([0]))
    n2 = cast("Callable[..., EagerTensorProtocol]", IRNode)(id="n2", op_type="Input", sharding=cast("Callable[..., EagerTensorProtocol]", DummySharding)([None]))
    g.nodes["n1"] = n1
    g.nodes["n2"] = n2
    # Wait, n1 has node_sharded True, n2 has inp_sharded False.
    # But n1 is NOT grad.
    # Actually wait. _is_boundary_transition(inp, node) gives False, True.
    # _process_spmd_input does:
    # elif not inp_sharded and node_sharded:
    #     if is_grad: ...
    # So if is_grad is False, it returns None.
    res = inject_spmd_communication_pass(g)
    assert res is False


def test_spmd_empty_conditions(mocker: EagerTensorProtocol) -> None:
    """Test function test_spmd_empty_conditions."""
    from ml_switcheroo_compiler.transforms.passes.spmd import _process_spmd_input

    # Create dummy rule with empty conditions
    mock_rules = {"reductions": [], "communication_matrix": [{"state": [True, True], "conditions": []}]}
    mocker.patch("ml_switcheroo_compiler.transforms.passes.spmd._get_spmd_rules", return_value=mock_rules)

    graph = IRGraph()
    in_node1 = cast("Callable[..., EagerTensorProtocol]", IRNode)(id="in1", op_type="Input", inputs=[], sharding=cast("Callable[..., EagerTensorProtocol]", DummySharding)(["x"]))
    graph.nodes["in1"] = in_node1

    node1 = cast("Callable[..., EagerTensorProtocol]", IRNode)(id="n1", op_type="Add", inputs=["in1"], sharding=cast("Callable[..., EagerTensorProtocol]", DummySharding)(["y"]))
    res = cast("Callable[..., EagerTensorProtocol]", _process_spmd_input)(node1, 0, "in1", graph, cast("Callable[..., EagerTensorProtocol]", DummySharding)(["y"]))
    assert res is None


def test_spmd_load_rules_branches() -> None:
    """Test function test_spmd_load_rules_branches."""
    from unittest.mock import mock_open, patch

    from ml_switcheroo_compiler.transforms.passes import spmd

    # reset rules
    spmd._SPMD_RULES = None

    class MockPathExistsDir:
        """Test class MockPathExistsDir."""

        def __init__(self, path: EagerValue) -> None:
            """Test function __init__."""
            self.path = path

        def exists(self) -> EagerValue:
            """Test function exists."""
            return True

        def __truediv__(self, other: EagerValue) -> EagerValue:
            """Test function __truediv__."""
            return cast("EagerTensorProtocol", self.path) / cast("EagerTensorProtocol", other)

    with patch("pathlib.Path.exists", return_value=True), patch("os.listdir", return_value=["test.txt", "valid.yaml", "invalid.yaml"]), patch("builtins.open", mock_open()), patch("yaml.safe_load", side_effect=[{"Add": {}}, ["not a dict"]]):
        # Test branches 184->183 (test.txt) and 187->183 (invalid.yaml returning a list)
        rules = spmd._get_spmd_rules()
        assert "Add" in rules

    # Test 191->194 (yaml_path does not exist when yaml_dir does not exist)
    spmd._SPMD_RULES = None
    with patch("pathlib.Path.exists", side_effect=[False, False]):  # first for yaml_dir, second for yaml_path
        rules = spmd._get_spmd_rules()
        assert rules == {}

    spmd._SPMD_RULES = None


def test_propagate_sharding_matmul_row_and_col_parallel() -> None:
    """Test function test_propagate_sharding_matmul_row_and_col_parallel."""
    from ml_switcheroo_compiler.transforms.passes.spmd import (
        propagate_sharding,
    )

    # Row parallel: LHS row-sharded, RHS replicated
    g = IRGraph()
    lhs = cast("Callable[..., EagerTensorProtocol]", IRNode)(id="lhs", op_type="Input", sharding=cast("Callable[..., EagerTensorProtocol]", DummySharding)(["x", None]))
    rhs = cast("Callable[..., EagerTensorProtocol]", IRNode)(id="rhs", op_type="Input", sharding=cast("Callable[..., EagerTensorProtocol]", DummySharding)([None, None]))
    mm = IRNode(id="mm", op_type="MatMul", inputs=["lhs", "rhs"])
    g.nodes = {"lhs": lhs, "rhs": rhs, "mm": mm}

    modified = propagate_sharding(g)
    assert modified is True
    assert cast("EagerTensorProtocol", getattr(mm.sharding, "mesh_mapping")) == ("x", None)

    # Col parallel: LHS replicated, RHS col-sharded
    g2 = IRGraph()
    lhs2 = cast("Callable[..., EagerTensorProtocol]", IRNode)(id="lhs2", op_type="Input", sharding=cast("Callable[..., EagerTensorProtocol]", DummySharding)([None, None]))
    rhs2 = cast("Callable[..., EagerTensorProtocol]", IRNode)(id="rhs2", op_type="Input", sharding=cast("Callable[..., EagerTensorProtocol]", DummySharding)([None, "y"]))
    mm2 = IRNode(id="mm2", op_type="MatMul", inputs=["lhs2", "rhs2"])
    g2.nodes = {"lhs2": lhs2, "rhs2": rhs2, "mm2": mm2}

    assert propagate_sharding(g2) is True
    assert cast("EagerTensorProtocol", getattr(mm2.sharding, "mesh_mapping")) == (None, "y")


def test_propagate_sharding_matmul_contracting_parallel() -> None:
    """Test function test_propagate_sharding_matmul_contracting_parallel."""
    from ml_switcheroo_compiler.transforms.passes.spmd import (
        spmd_partitioning_pass,
    )

    # Contracting parallel: LHS contracting dim sharded, RHS contracting dim sharded
    g = IRGraph()
    lhs = cast("Callable[..., EagerTensorProtocol]", IRNode)(id="lhs", op_type="Input", sharding=cast("Callable[..., EagerTensorProtocol]", DummySharding)([None, "x"]))
    rhs = cast("Callable[..., EagerTensorProtocol]", IRNode)(id="rhs", op_type="Input", sharding=cast("Callable[..., EagerTensorProtocol]", DummySharding)(["x", None]))
    mm = IRNode(id="mm", op_type="MatMul", inputs=["lhs", "rhs"])
    out = IRNode(id="out", op_type="Relu", inputs=["mm"])
    g.nodes = {"lhs": lhs, "rhs": rhs, "mm": mm, "out": out}
    g.outputs = ["out"]

    assert spmd_partitioning_pass(g) is True
    assert cast("EagerTensorProtocol", getattr(mm.sharding, "mesh_mapping")) == (None, None)
    assert "mm_all_reduce" in g.nodes
    assert g.nodes["out"].inputs == ["mm_all_reduce"]


def test_propagate_sharding_matmul_fallback() -> None:
    """Test function test_propagate_sharding_matmul_fallback."""
    from ml_switcheroo_compiler.transforms.passes.spmd import propagate_sharding

    g = IRGraph()
    lhs = cast("Callable[..., EagerTensorProtocol]", IRNode)(id="lhs", op_type="Input", sharding=cast("Callable[..., EagerTensorProtocol]", DummySharding)(["z"]))
    rhs = cast("Callable[..., EagerTensorProtocol]", IRNode)(id="rhs", op_type="Input", sharding=cast("Callable[..., EagerTensorProtocol]", DummySharding)(["w"]))
    mm = IRNode(id="mm", op_type="MatMul", inputs=["lhs", "rhs"])
    g.nodes = {"lhs": lhs, "rhs": rhs, "mm": mm}

    assert propagate_sharding(g) is True
    assert mm.sharding is not None


def test_propagate_sharding_reductions() -> None:
    """Test function test_propagate_sharding_reductions."""
    from ml_switcheroo_compiler.transforms.passes.spmd import (
        propagate_sharding,
        spmd_partitioning_pass,
    )

    # Case 1: Reduced dimension is sharded -> inject AllReduce
    g1 = IRGraph()
    inp1 = cast("Callable[..., EagerTensorProtocol]", IRNode)(id="inp1", op_type="Input", sharding=cast("Callable[..., EagerTensorProtocol]", DummySharding)(["x", None]))
    red1 = IRNode(id="red1", op_type="ReduceSum", inputs=["inp1"], attributes={"axis": 0})
    g1.nodes = {"inp1": inp1, "red1": red1}
    g1.outputs = ["red1"]

    assert spmd_partitioning_pass(g1) is True
    assert cast("EagerTensorProtocol", getattr(red1.sharding, "mesh_mapping")) == (None,)
    assert "red1_all_reduce" in g1.nodes
    assert g1.outputs == ["red1_all_reduce"]

    # Case 2: Reduced dimension is replicated -> keep other sharded dimensions
    g2 = IRGraph()
    inp2 = cast("Callable[..., EagerTensorProtocol]", IRNode)(id="inp2", op_type="Input", sharding=cast("Callable[..., EagerTensorProtocol]", DummySharding)(["x", None]))
    red2 = IRNode(id="red2", op_type="ReduceSum", inputs=["inp2"], attributes={"axis": [1]})
    g2.nodes = {"inp2": inp2, "red2": red2}

    assert propagate_sharding(g2) is True
    assert cast("EagerTensorProtocol", getattr(red2.sharding, "mesh_mapping")) == ("x",)
    assert red2.attributes.get("inject_collective") is None


def test_propagate_sharding_convolutions_and_elementwise() -> None:
    """Test function test_propagate_sharding_convolutions_and_elementwise."""
    from ml_switcheroo_compiler.transforms.passes.spmd import (
        SPMDShardingAnnotation,
        propagate_sharding,
        spmd_partitioning_pass,
    )

    # Conv2D
    g = IRGraph()
    inp = cast("Callable[..., EagerTensorProtocol]", IRNode)(id="inp", op_type="Input", sharding=cast("Callable[..., EagerTensorProtocol]", DummySharding)(["x", None, None, None]))
    conv = IRNode(id="conv", op_type="Conv2D", inputs=["inp"])
    # Elementwise Add
    add = IRNode(id="add", op_type="Add", inputs=["conv"])
    g.nodes = {"inp": inp, "conv": conv, "add": add}

    assert propagate_sharding(g) is True
    assert cast("EagerTensorProtocol", tuple(cast("EagerTensorProtocol", getattr(conv.sharding, "mesh_mapping")))) == cast("EagerTensorProtocol", ("x", None, None, None))
    assert cast("EagerTensorProtocol", tuple(cast("EagerTensorProtocol", getattr(add.sharding, "mesh_mapping")))) == cast("EagerTensorProtocol", ("x", None, None, None))

    # SPMDShardingAnnotation repr
    annot = SPMDShardingAnnotation("mesh1", ["x", None])
    assert "SPMDShardingAnnotation" in repr(annot)

    # Branch coverage edge cases:
    # 1. Elementwise with unsharded input
    g_el = IRGraph()
    in_un = IRNode(id="in_un", op_type="Input")
    add_un = IRNode(id="add_un", op_type="Add", inputs=["in_un"])
    g_el.nodes = {"in_un": in_un, "add_un": add_un}
    assert propagate_sharding(g_el) is False

    # 2. Matmul with < 2 inputs or unsharded inputs
    g_mm_empty = IRGraph()
    mm_empty = IRNode(id="mm", op_type="MatMul", inputs=["in1"])
    g_mm_empty.nodes = {"in1": IRNode(id="in1", op_type="Input"), "mm": mm_empty}
    assert propagate_sharding(g_mm_empty) is False

    # Matmul where inputs have sharding with empty mesh_mapping
    g_mm_no_map = IRGraph()
    g_mm_no_map.nodes = {
        "in1": cast("Callable[..., EagerTensorProtocol]", IRNode)(id="in1", op_type="Input", sharding=cast("Callable[..., EagerTensorProtocol]", DummySharding)([])),
        "in2": cast("Callable[..., EagerTensorProtocol]", IRNode)(id="in2", op_type="Input", sharding=cast("Callable[..., EagerTensorProtocol]", DummySharding)([])),
        "mm": IRNode(id="mm", op_type="MatMul", inputs=["in1", "in2"]),
    }
    assert propagate_sharding(g_mm_no_map) is False

    # 3. Reduction with empty inputs or unsharded input
    g_red_empty = IRGraph()
    red_empty = IRNode(id="red", op_type="ReduceSum", inputs=[])
    red_un = IRNode(id="red_un", op_type="ReduceSum", inputs=["in1"])
    g_red_empty.nodes = {"in1": IRNode(id="in1", op_type="Input"), "red": red_empty, "red_un": red_un}
    assert propagate_sharding(g_red_empty) is False

    # 4. Reduction reducing all dims resulting in out_map empty fallback to [None]
    g_red_all = IRGraph()
    inp_all = cast("Callable[..., EagerTensorProtocol]", IRNode)(id="inp_all", op_type="Input", sharding=cast("Callable[..., EagerTensorProtocol]", DummySharding)([None]))
    red_all = IRNode(id="red_all", op_type="ReduceSum", inputs=["inp_all"], attributes={"axis": 0})
    g_red_all.nodes = {"inp_all": inp_all, "red_all": red_all}
    assert propagate_sharding(g_red_all) is True
    assert cast("EagerTensorProtocol", getattr(red_all.sharding, "mesh_mapping")) == (None,)

    # 5. Spatial conv with no inputs or unsharded input
    g_cv_empty = IRGraph()
    cv_empty = IRNode(id="cv", op_type="Conv2D", inputs=[])
    cv_un = IRNode(id="cv_un", op_type="Conv2D", inputs=["in1"])
    g_cv_empty.nodes = {"in1": IRNode(id="in1", op_type="Input"), "cv": cv_empty, "cv_un": cv_un}
    assert propagate_sharding(g_cv_empty) is False

    # 6. spmd_partitioning_pass without outputs attribute and with ar_id already existing
    g_no_out = IRGraph()
    g_no_out.nodes = {
        "lhs": cast("Callable[..., EagerTensorProtocol]", IRNode)(id="lhs", op_type="Input", sharding=cast("Callable[..., EagerTensorProtocol]", DummySharding)([None, "x"])),
        "rhs": cast("Callable[..., EagerTensorProtocol]", IRNode)(id="rhs", op_type="Input", sharding=cast("Callable[..., EagerTensorProtocol]", DummySharding)(["x", None])),
        "mm": IRNode(id="mm", op_type="MatMul", inputs=["lhs", "rhs"]),
        "mm_all_reduce": IRNode(id="mm_all_reduce", op_type="AllReduce", inputs=["mm"]),
    }
    if hasattr(g_no_out, "outputs"):
        delattr(g_no_out, "outputs")
    assert spmd_partitioning_pass(g_no_out) is True


def test_spmd_pass_inject_collectives() -> None:
    """Test function test_spmd_pass_inject_collectives."""
    from ml_switcheroo_compiler.transforms.passes.spmd import spmd_partitioning_pass

    for coll in ("AllReduce", "ReduceScatter", "AllGather", "AllToAll"):
        g = IRGraph()
        n1 = cast("Callable[..., EagerTensorProtocol]", IRNode)(id="n1", op_type="Add", inputs=[], attributes={"inject_collective": coll}, sharding=cast("Callable[..., EagerTensorProtocol]", DummySharding)(["x"]))
        n2 = IRNode(id="n2", op_type="Relu", inputs=["n1"])
        g.nodes = {"n1": n1, "n2": n2}
        g.outputs = ["n1"]
        res = spmd_partitioning_pass(g)
        assert res is True
        expected_coll_id = "n1_all_reduce" if coll == "AllReduce" else f"n1_{coll.lower()}"
        assert expected_coll_id in g.nodes
        assert g.outputs == [expected_coll_id]
        assert g.nodes["n2"].inputs == [expected_coll_id]
