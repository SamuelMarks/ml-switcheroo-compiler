"""Docstring module."""

import pytest
from ml_switcheroo.ir.core import IRGraph, IRNode
from ml_switcheroo.transforms.pass_manager import (
    DAGTopologicalSorter,
    IRValidator,
    PassManager,
    _graph_hash,
)
from ml_switcheroo.core.errors import CompilationError


def test_dag_cycle() -> None:
    """Docstring."""
    graph = IRGraph()
    # A -> B -> A
    node_a = IRNode(id="A", op_type="dummy", inputs=["B"])
    node_b = IRNode(id="B", op_type="dummy", inputs=["A"])
    graph.nodes["A"] = node_a
    graph.nodes["B"] = node_b

    with pytest.raises(CompilationError, match="Cycle detected in graph."):
        DAGTopologicalSorter.sort(graph)


def test_check_cycles() -> None:
    """Docstring."""
    graph = IRGraph()
    node_a = IRNode(id="A", op_type="dummy", inputs=["B"])
    node_b = IRNode(id="B", op_type="dummy", inputs=["A"])
    graph.nodes["A"] = node_a
    graph.nodes["B"] = node_b

    with pytest.raises(CompilationError, match="Cycle detected in graph."):
        IRValidator.check_cycles(graph)


def test_check_shapes() -> None:
    """Docstring."""
    graph = IRGraph()
    node_a = IRNode(id="A", op_type="dummy", inputs=[])
    node_a.shape_metadata = None
    graph.nodes["A"] = node_a

    with pytest.raises(CompilationError, match="Node A is missing shape_metadata."):
        IRValidator.check_shapes(graph)

    node_a.shape_metadata = ()
    IRValidator.check_shapes(graph)


def test_graph_hash() -> None:
    """Docstring."""
    graph = IRGraph()
    node_a = IRNode(id="A", op_type="dummy", inputs=[])
    graph.nodes["A"] = node_a
    hash1 = _graph_hash(graph)

    graph2 = IRGraph()
    node_a2 = IRNode(id="A", op_type="dummy", inputs=[])
    graph2.nodes["A"] = node_a2
    hash2 = _graph_hash(graph2)

    assert hash1 == hash2

    node_a2.inputs = ["B"]
    hash3 = _graph_hash(graph2)
    assert hash1 != hash3


def test_pass_manager_run() -> None:
    """Docstring."""
    pm = PassManager()

    graph = IRGraph()
    node_a = IRNode(id="A", op_type="dummy", inputs=[])
    node_a.shape_metadata = ()
    graph.nodes["A"] = node_a

    called = False

    def dummy_pass(g: IRGraph) -> bool:
        """Docstring."""
        nonlocal called
        called = True
        return False

    pm.add_pass(dummy_pass)
    pm.run(graph)

    assert called


def test_pass_manager_run_until_converged() -> None:
    """Docstring."""
    pm = PassManager()

    graph = IRGraph()
    node_a = IRNode(id="A", op_type="dummy", inputs=[])
    node_a.shape_metadata = ()
    graph.nodes["A"] = node_a

    counter = 0

    def dummy_pass(g: IRGraph) -> bool:
        """Docstring."""
        nonlocal counter
        if counter < 2:
            counter += 1
            # modify graph to change hash
            g.nodes[f"B{counter}"] = IRNode(
                id=f"B{counter}", op_type="dummy", inputs=[]
            )  # noqa: E501
            g.nodes[f"B{counter}"].shape_metadata = ()
            return True
        return False

    pm.add_pass(dummy_pass)
    pm.run_until_converged(graph, max_iterations=5)

    assert counter == 2


def test_pass_manager_run_until_converged_max_iters() -> None:
    """Docstring."""
    pm = PassManager()

    graph = IRGraph()
    node_a = IRNode(id="A", op_type="dummy", inputs=[])
    node_a.shape_metadata = ()
    graph.nodes["A"] = node_a

    counter = 0

    def dummy_pass(g: IRGraph) -> bool:
        """Docstring."""
        nonlocal counter
        counter += 1
        g.nodes[f"B{counter}"] = IRNode(id=f"B{counter}", op_type="dummy", inputs=[])
        g.nodes[f"B{counter}"].shape_metadata = ()
        return True

    pm.add_pass(dummy_pass)
    pm.run_until_converged(graph, max_iterations=3)

    assert counter == 3
