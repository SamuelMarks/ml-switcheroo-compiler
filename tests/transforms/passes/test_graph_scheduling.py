"""Tests for graph scheduling logic."""

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.graph_scheduling import (
    DefaultCostModel,
    graph_scheduling_pass,
)


def test_default_cost_model_memory() -> None:
    """Test memory cost calculation."""
    model = DefaultCostModel()

    node = IRNode(
        id="n1",
        op_type="Add",
        shape_metadata=(10, 20),
        attributes={"dtype": DType.Float32.value},
    )
    assert model.get_memory_cost(node) == 10 * 20 * 4

    node_f64 = IRNode(
        id="n2",
        op_type="Add",
        shape_metadata=(10,),
        attributes={"dtype": DType.Float64.value},
    )
    assert model.get_memory_cost(node_f64) == 10 * 8

    node_dyn = IRNode(
        id="n3",
        op_type="Add",
        shape_metadata=("B", 10),
        attributes={"dtype": DType.Float32.value},
    )
    assert model.get_memory_cost(node_dyn) == 4

    node_none = IRNode(
        id="n4",
        op_type="Add",
        shape_metadata=None,
        attributes={"dtype": DType.Float32.value},
    )
    assert model.get_memory_cost(node_none) == 4


def test_default_cost_model_compute() -> None:
    """Test compute cost calculation."""
    model = DefaultCostModel()

    node_matmul = IRNode(id="n1", op_type="MatMul", shape_metadata=(10, 10))
    assert model.get_compute_cost(node_matmul) == 1000

    node_add = IRNode(id="n2", op_type="Add", shape_metadata=(10,))
    assert model.get_compute_cost(node_add) == 10

    node_other = IRNode(id="n3", op_type="Relu", shape_metadata=(10,))
    assert model.get_compute_cost(node_other) == 50


def test_graph_scheduling_pass() -> None:
    """Test that graph_scheduling_pass modifies the graph order correctly."""
    graph = IRGraph()

    # Insert in worst order: n2, n1, m2, m1
    graph.nodes["n2"] = IRNode(id="n2", op_type="Add", inputs=["n1"], shape_metadata=(100, 100), attributes={"dtype": DType.Float32.value})
    graph.nodes["n1"] = IRNode(id="n1", op_type="MatMul", shape_metadata=(100, 100), attributes={"dtype": DType.Float32.value})
    graph.nodes["m2"] = IRNode(id="m2", op_type="Add", inputs=["m1"], shape_metadata=(1000,), attributes={"dtype": DType.Float32.value})
    graph.nodes["m1"] = IRNode(id="m1", op_type="Add", shape_metadata=(1000,), attributes={"dtype": DType.Float32.value})

    modified = graph_scheduling_pass(graph)
    assert len(graph.nodes) == 4

    assert list(graph.nodes.keys()) == ["m1", "m2", "n1", "n2"]

    assert not graph_scheduling_pass(graph)


def test_graph_scheduling_cycle() -> None:
    """Test graph with cycle to ensure it bails out safely."""
    graph = IRGraph()
    graph.nodes["n1"] = IRNode(id="n1", op_type="Add", inputs=["n2"])
    graph.nodes["n2"] = IRNode(id="n2", op_type="Add", inputs=["n1"])

    modified = graph_scheduling_pass(graph)
    assert not modified


def test_graph_scheduling_interleave() -> None:
    """Test compute heavy interleaving."""
    graph = IRGraph()
    graph.nodes["c1"] = IRNode(id="c1", op_type="MatMul", shape_metadata=(100, 100))
    graph.nodes["c2"] = IRNode(id="c2", op_type="MatMul", shape_metadata=(100, 100))
    modified = graph_scheduling_pass(graph)


def test_graph_scheduling_missing_branches():
    graph = IRGraph()
    # Node with non-existent input, tests 92->91 and 171->170
    graph.nodes["n1"] = IRNode(id="n1", op_type="Add", inputs=["non_existent"], shape_metadata=(100, 100), attributes={"dtype": DType.Float32.value})

    # Node that is used multiple times, so remaining_uses != 1 (tests 105->104)
    graph.nodes["n2"] = IRNode(id="n2", op_type="Add", inputs=["n1"], shape_metadata=(100, 100), attributes={"dtype": DType.Float32.value})
    graph.nodes["n3"] = IRNode(id="n3", op_type="Add", inputs=["n1"], shape_metadata=(100, 100), attributes={"dtype": DType.Float32.value})

    # Test 176->174 (in_degree[consumer] == 0 is false)
    # This naturally happens when a consumer has multiple dependencies and we pop one of them.
    # We add a consumer that depends on both n2 and n3.
    graph.nodes["n4"] = IRNode(id="n4", op_type="Add", inputs=["n2", "n3"], shape_metadata=(100, 100), attributes={"dtype": DType.Float32.value})

    modified = graph_scheduling_pass(graph)
