"""Tests for SPMD communication passes."""

from __future__ import annotations

import typing

from ml_switcheroo_ir import PartitionSpec

from ml_switcheroo_compiler.distributed.strategy import MeshShardingStrategy
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.spmd import inject_spmd_communication_pass
from tests.testing_utils import as_callable, as_node

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class MockSharding(PartitionSpec):
    """Test class MockSharding."""

    def __init__(self, axes: list[EagerValue]) -> None:
        """Test function __init__.

        Args:
            axes: Test parameter `axes`.
        """
        super().__init__(axes=tuple(str(a) if a is not None else None for a in axes))
        object.__setattr__(self, "mesh_mapping", axes)


def test_spmd_all_gather(monkeypatch: EagerTensorProtocol) -> None:
    """Test AllGather injection.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    graph = IRGraph()
    n1 = as_callable(IRNode)(id="n1", op_type="Linear", sharding=as_callable(MockSharding)([None, "x"]))
    n2 = as_callable(IRNode)(id="n2", op_type="Linear", inputs=["n1"], sharding=as_callable(MockSharding)(["x", "y"]))
    graph.nodes["n1"] = as_node(n1)
    graph.nodes["n2"] = as_node(n2)

    import ml_switcheroo_compiler.transforms.passes.spmd as spmd_mod

    monkeypatch.setattr(spmd_mod, "_SPMD_RULES", {"communication_matrix": [{"state": [True, True], "conditions": [{"default": True, "inject": "AllGather"}]}]})

    inject_spmd_communication_pass(graph)

    assert "n1_all_gather" in graph.nodes
    assert graph.nodes["n2"].inputs == ["n1_all_gather"]


def test_spmd_reduce_scatter(monkeypatch: EagerTensorProtocol) -> None:
    """Test ReduceScatter injection.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    graph = IRGraph()
    n1 = as_callable(IRNode)(id="n1", op_type="Linear", sharding=as_callable(MockSharding)(["x", "y"]))
    n2 = as_callable(IRNode)(id="n2", op_type="Linear", inputs=["n1"], sharding=as_callable(MockSharding)(["x", None]))
    graph.nodes["n1"] = as_node(n1)
    graph.nodes["n2"] = as_node(n2)

    import ml_switcheroo_compiler.transforms.passes.spmd as spmd_mod

    monkeypatch.setattr(spmd_mod, "_SPMD_RULES", {"communication_matrix": [{"state": [True, True], "conditions": [{"default": True, "inject": "ReduceScatter"}]}]})

    inject_spmd_communication_pass(graph)

    assert "n1_reduce_scatter" in graph.nodes
    assert graph.nodes["n2"].inputs == ["n1_reduce_scatter"]


def test_spmd_all_reduce() -> None:
    """Test AllReduce injection for reductions."""
    graph = IRGraph()
    n1 = as_callable(IRNode)(id="n1", op_type="Linear", sharding=as_callable(MockSharding)(["x"]))
    n2 = as_callable(IRNode)(id="n2", op_type="Sum", inputs=["n1"], sharding=as_callable(MockSharding)(["x"]))
    graph.nodes["n1"] = as_node(n1)
    graph.nodes["n2"] = as_node(n2)

    # We mock rules to ensure AllReduce is triggered for Sum
    import ml_switcheroo_compiler.transforms.passes.spmd as spmd_mod

    spmd_mod._SPMD_RULES = {"reductions": ["Sum"], "communication_matrix": [{"state": [True, True], "conditions": [{"is_reduction": True, "inject": "AllReduce"}]}]}

    inject_spmd_communication_pass(graph)

    assert "n1_all_reduce" in graph.nodes
    assert graph.nodes["n2"].inputs == ["n1_all_reduce"]


def test_spmd_all_to_all() -> None:
    """Test AllToAll injection."""
    graph = IRGraph()
    n1 = as_callable(IRNode)(id="n1", op_type="Linear", sharding=as_callable(MockSharding)(["x", None]))
    n2 = as_callable(IRNode)(id="n2", op_type="Linear", inputs=["n1"], sharding=as_callable(MockSharding)([None, "y"]))
    graph.nodes["n1"] = as_node(n1)
    graph.nodes["n2"] = as_node(n2)

    import ml_switcheroo_compiler.transforms.passes.spmd as spmd_mod

    spmd_mod._SPMD_RULES = {"communication_matrix": [{"state": [True, True], "conditions": [{"axes_match": False, "axes_length_match": True, "inject": "AllToAll"}]}]}

    inject_spmd_communication_pass(graph)

    assert "n1_all_to_all" in graph.nodes
    assert graph.nodes["n2"].inputs == ["n1_all_to_all"]


def test_mesh_sharding_strategy() -> None:
    """Test the MeshShardingStrategy."""
    graph = IRGraph()
    n1 = as_callable(IRNode)(id="n1", op_type="Linear", sharding=as_callable(MockSharding)(["x"]))
    n2 = IRNode(id="n2", op_type="Linear", inputs=["n1"])
    graph.nodes["n1"] = as_node(n1)
    graph.nodes["n2"] = n2

    strategy = MeshShardingStrategy()
    strategy.lower_sharding(graph)

    # Sharding should be propagated
    assert getattr(graph.nodes["n2"].sharding, "mesh_mapping") == ["x"]
