"""Unit tests for the IR pass manager, validators, and topological sorter.

This module verifies the correctness of graph validation passes, cycle detection, graph
hashing, and the iterative execution of optimization passes within the PassManager.
"""

import pytest

from ml_switcheroo.core.errors import CompilationError
from ml_switcheroo.ir.core import IRGraph, IRNode
from ml_switcheroo.transforms.pass_manager import (
    DAGTopologicalSorter,
    IRValidator,
    PassManager,
    _graph_hash,
)


def test_dag_cycle() -> None:
    """Verifies that DAGTopologicalSorter detects cycles and raises a CompilationError.

    This test constructs a cyclic graph (A -> B -> A) and asserts that attempting
    to topologically sort it results in a CompilationError with the appropriate
    error message

    Returns:
    None
    """
    graph = IRGraph()
    # A -> B -> A
    node_a = IRNode(id="A", op_type="dummy", inputs=["B"])
    node_b = IRNode(id="B", op_type="dummy", inputs=["A"])
    graph.nodes["A"] = node_a
    graph.nodes["B"] = node_b

    with pytest.raises(CompilationError, match="Cycle detected in graph."):
        DAGTopologicalSorter.sort(graph)


def test_check_cycles() -> None:
    """Verifies that IRValidator.check_cycles detects cycles in the IRGraph.

    This test constructs a cyclic graph (A -> B -> A) and asserts that calling
    IRValidator.check_cycles raises a CompilationError with the appropriate error
    message

    Returns:
    None
    """
    graph = IRGraph()
    node_a = IRNode(id="A", op_type="dummy", inputs=["B"])
    node_b = IRNode(id="B", op_type="dummy", inputs=["A"])
    graph.nodes["A"] = node_a
    graph.nodes["B"] = node_b

    with pytest.raises(CompilationError, match="Cycle detected in graph."):
        IRValidator.check_cycles(graph)


def test_check_shapes() -> None:
    """Verifies that IRValidator.check_shapes validates the presence of shape metadata.

    This test ensures that a CompilationError is raised when a node is missing
    shape metadata, and that validation passes successfully once valid shape
    metadata is provided

    Returns:
    None
    """
    graph = IRGraph()
    node_a = IRNode(id="A", op_type="dummy", inputs=[])
    node_a.shape_metadata = None
    graph.nodes["A"] = node_a

    with pytest.raises(CompilationError, match="Node A is missing shape_metadata."):
        IRValidator.check_shapes(graph)

    node_a.shape_metadata = ()
    IRValidator.check_shapes(graph)


def test_graph_hash() -> None:
    """Verifies the consistency and sensitivity of the _graph_hash utility function.

    This test ensures that structurally identical graphs produce the same hash,
    and that modifying a node's inputs results in a different hash

    Returns:
    None
    """
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
    """Verifies that PassManager.run executes registered passes on an IRGraph.

    This test registers a simple dummy pass with the PassManager, runs it on
    a valid graph, and asserts that the pass was successfully executed

    Returns:
    None
    """
    pm = PassManager()

    graph = IRGraph()
    node_a = IRNode(id="A", op_type="dummy", inputs=[])
    node_a.shape_metadata = ()
    graph.nodes["A"] = node_a

    called = False

    def dummy_pass(g: IRGraph) -> bool:
        """A dummy optimization pass used for testing.

        Args:
        g (IRGraph): The intermediate representation graph to process

        Returns:
        bool: True if the graph was modified, False otherwise.
        """
        nonlocal called
        called = True
        return False

    pm.add_pass(dummy_pass)
    pm.run(graph)

    assert called


def test_pass_manager_run_until_converged() -> None:
    """Verifies that PassManager.run_until_converged executes passes until no changes.

    occur

    This test registers a pass that modifies the graph a finite number of times
    and verifies that the PassManager continues execution until the graph converges
    (i.e., no further modifications are made and the graph hash remains stable)

    Returns:
    None
    """
    pm = PassManager()

    graph = IRGraph()
    node_a = IRNode(id="A", op_type="dummy", inputs=[])
    node_a.shape_metadata = ()
    graph.nodes["A"] = node_a

    counter = 0

    def dummy_pass(g: IRGraph) -> bool:
        """A dummy optimization pass used for testing.

        Args:
        g (IRGraph): The intermediate representation graph to process

        Returns:
        bool: True if the graph was modified, False otherwise.
        """
        nonlocal counter
        if counter < 2:
            counter += 1
            # modify graph to change hash
            g.nodes[f"B{counter}"] = IRNode(
                id=f"B{counter}",
                op_type="dummy",
                inputs=[],
            )
            g.nodes[f"B{counter}"].shape_metadata = ()
            return True
        return False

    pm.add_pass(dummy_pass)
    pm.run_until_converged(graph, max_iterations=5)

    assert counter == 2


def test_pass_manager_run_until_converged_max_iters() -> None:
    """Verifies that PassManager.run_until_converged respects the max_iterations limit.

    This test registers a pass that continuously modifies the graph and verifies
    that the PassManager terminates execution once the specified maximum number
    of iterations is reached, preventing infinite loops

    Returns:
    None
    """
    pm = PassManager()

    graph = IRGraph()
    node_a = IRNode(id="A", op_type="dummy", inputs=[])
    node_a.shape_metadata = ()
    graph.nodes["A"] = node_a

    counter = 0

    def dummy_pass(g: IRGraph) -> bool:
        """A dummy optimization pass used for testing.

        Args:
        g (IRGraph): The intermediate representation graph to process

        Returns:
        bool: True if the graph was modified, False otherwise.
        """
        nonlocal counter
        counter += 1
        g.nodes[f"B{counter}"] = IRNode(id=f"B{counter}", op_type="dummy", inputs=[])
        g.nodes[f"B{counter}"].shape_metadata = ()
        return True

    pm.add_pass(dummy_pass)
    pm.run_until_converged(graph, max_iterations=3)

    assert counter == 3
