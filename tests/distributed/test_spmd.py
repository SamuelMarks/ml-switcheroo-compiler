"""Tests for SPMD communication passes."""

from ml_switcheroo_compiler.distributed.strategy import MeshShardingStrategy
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.spmd import inject_spmd_communication_pass


class MockSharding:
    def __init__(self, axes):
        self.mesh_mapping = axes


def test_spmd_all_gather(monkeypatch):
    """Test AllGather injection."""
    graph = IRGraph()
    n1 = IRNode(id="n1", op_type="Linear", sharding=MockSharding([None, "x"]))
    n2 = IRNode(id="n2", op_type="Linear", inputs=["n1"], sharding=MockSharding(["x", "y"]))
    graph.nodes["n1"] = n1
    graph.nodes["n2"] = n2

    import ml_switcheroo_compiler.transforms.passes.spmd as spmd_mod

    monkeypatch.setattr(spmd_mod, "_SPMD_RULES", {"communication_matrix": [{"state": [True, True], "conditions": [{"default": True, "inject": "AllGather"}]}]})

    inject_spmd_communication_pass(graph)

    assert "n1_all_gather" in graph.nodes
    assert graph.nodes["n2"].inputs == ["n1_all_gather"]


def test_spmd_reduce_scatter(monkeypatch):
    """Test ReduceScatter injection."""
    graph = IRGraph()
    n1 = IRNode(id="n1", op_type="Linear", sharding=MockSharding(["x", "y"]))
    n2 = IRNode(id="n2", op_type="Linear", inputs=["n1"], sharding=MockSharding(["x", None]))
    graph.nodes["n1"] = n1
    graph.nodes["n2"] = n2

    import ml_switcheroo_compiler.transforms.passes.spmd as spmd_mod

    monkeypatch.setattr(spmd_mod, "_SPMD_RULES", {"communication_matrix": [{"state": [True, True], "conditions": [{"default": True, "inject": "ReduceScatter"}]}]})

    inject_spmd_communication_pass(graph)

    assert "n1_reduce_scatter" in graph.nodes
    assert graph.nodes["n2"].inputs == ["n1_reduce_scatter"]


def test_spmd_all_reduce():
    """Test AllReduce injection for reductions."""
    graph = IRGraph()
    n1 = IRNode(id="n1", op_type="Linear", sharding=MockSharding(["x"]))
    n2 = IRNode(id="n2", op_type="Sum", inputs=["n1"], sharding=MockSharding(["x"]))
    graph.nodes["n1"] = n1
    graph.nodes["n2"] = n2

    # We mock rules to ensure AllReduce is triggered for Sum
    import ml_switcheroo_compiler.transforms.passes.spmd as spmd_mod

    spmd_mod._SPMD_RULES = {"reductions": ["Sum"], "communication_matrix": [{"state": [True, True], "conditions": [{"is_reduction": True, "inject": "AllReduce"}]}]}

    inject_spmd_communication_pass(graph)

    assert "n1_all_reduce" in graph.nodes
    assert graph.nodes["n2"].inputs == ["n1_all_reduce"]


def test_spmd_all_to_all():
    """Test AllToAll injection."""
    graph = IRGraph()
    n1 = IRNode(id="n1", op_type="Linear", sharding=MockSharding(["x", None]))
    n2 = IRNode(id="n2", op_type="Linear", inputs=["n1"], sharding=MockSharding([None, "y"]))
    graph.nodes["n1"] = n1
    graph.nodes["n2"] = n2

    import ml_switcheroo_compiler.transforms.passes.spmd as spmd_mod

    spmd_mod._SPMD_RULES = {"communication_matrix": [{"state": [True, True], "conditions": [{"axes_match": False, "axes_length_match": True, "inject": "AllToAll"}]}]}

    inject_spmd_communication_pass(graph)

    assert "n1_all_to_all" in graph.nodes
    assert graph.nodes["n2"].inputs == ["n1_all_to_all"]


def test_mesh_sharding_strategy():
    """Test the MeshShardingStrategy."""
    graph = IRGraph()
    n1 = IRNode(id="n1", op_type="Linear", sharding=MockSharding(["x"]))
    n2 = IRNode(id="n2", op_type="Linear", inputs=["n1"])
    graph.nodes["n1"] = n1
    graph.nodes["n2"] = n2

    strategy = MeshShardingStrategy()
    strategy.lower_sharding(graph)

    # Sharding should be propagated
    assert graph.nodes["n2"].sharding.mesh_mapping == ["x"]
