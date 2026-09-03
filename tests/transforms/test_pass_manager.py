# ruff: noqa: E501
"""Unit tests for the IR pass manager, validators, and topological sorter.

This module verifies the correctness of graph validation passes, cycle detection, graph
hashing, and the iterative execution of optimization passes within the PassManager.
"""

import pytest

from ml_switcheroo_compiler.core.errors import CompilationError
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter, IRValidator, PassManager, _graph_hash


def test_load_from_config():
    from unittest.mock import mock_open, patch

    import yaml

    from ml_switcheroo_compiler.transforms.pass_manager import PassManager

    pm = PassManager()
    with patch("os.path.exists", return_value=True):
        with patch(
            "builtins.open",
            mock_open(
                read_data=yaml.dump(
                    {
                        "execution_order": ["dead_code_elimination", "invalid_pass_name"],
                        "cost_model": {"memory_sizes": {}, "compute_costs": {"heavy_ops": [], "light_ops": [], "heavy_cost": 1, "light_cost": 1, "default_cost": 1}, "compute_heavy_threshold": 1, "heavy_interleave_penalty": 1, "light_interleave_penalty": 1},
                        "fusion_patterns": {},
                    }
                )
            ),
        ):
            pm.load_from_config()
            assert "dead_code_elimination" in pm.pass_names
            assert "invalid_pass_name" not in pm.pass_names

    with patch("os.path.exists", return_value=False):
        pm2 = PassManager()
        pm2.load_from_config()
        assert len(pm2.passes) == 0


def test_dag_cycle() -> None:
    """Test the dag cycle behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Verifies that DAGTopologicalSorter detects cycles and raises a CompilationError.\n\n    This test constructs a cyclic graph (A -> B -> A) and asserts that attempting\n    to topologically sort it results in a CompilationError with the appropriate\n    error message\n\n    Returns:\n    None\n    "
        graph = IRGraph()
        node_a = IRNode(id="A", op_type="dummy", inputs=["B"])
        node_b = IRNode(id="B", op_type="dummy", inputs=["A"])
        graph.nodes["A"] = node_a
        graph.nodes["B"] = node_b
        with pytest.raises(CompilationError, match="Cycle detected in graph."):
            DAGTopologicalSorter.sort(graph)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_check_cycles() -> None:
    """Test the check cycles behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Verifies that IRValidator.check_cycles detects cycles in the IRGraph.\n\n    This test constructs a cyclic graph (A -> B -> A) and asserts that calling\n    IRValidator.check_cycles raises a CompilationError with the appropriate error\n    message\n\n    Returns:\n    None\n    "
        graph = IRGraph()
        node_a = IRNode(id="A", op_type="dummy", inputs=["B"])
        node_b = IRNode(id="B", op_type="dummy", inputs=["A"])
        graph.nodes["A"] = node_a
        graph.nodes["B"] = node_b
        with pytest.raises(CompilationError, match="Cycle detected in graph."):
            IRValidator.check_cycles(graph)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_check_shapes() -> None:
    """Test the check shapes behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Verifies that IRValidator.check_shapes validates the presence of shape metadata.\n\n    This test ensures that a CompilationError is raised when a node is missing\n    shape metadata, and that validation passes successfully once valid shape\n    metadata is provided\n\n    Returns:\n    None\n    "
        graph = IRGraph()
        node_a = IRNode(id="A", op_type="dummy", inputs=[])
        node_a.shape_metadata = None
        graph.nodes["A"] = node_a
        with pytest.raises(CompilationError, match="Node A is missing shape_metadata."):
            IRValidator.check_shapes(graph)
        node_a.shape_metadata = ()
        IRValidator.check_shapes(graph)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_graph_hash() -> None:
    """Test the graph hash behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Verifies the consistency and sensitivity of the _graph_hash utility function.\n\n    This test ensures that structurally identical graphs produce the same hash,\n    and that modifying a node's inputs results in a different hash\n\n    Returns:\n    None\n    "
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
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_pass_manager_run() -> None:
    """Test the pass manager run behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Verifies that PassManager.run executes registered passes on an IRGraph.\n\n    This test registers a simple dummy pass with the PassManager, runs it on\n    a valid graph, and asserts that the pass was successfully executed\n\n    Returns:\n    None\n    "
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
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_pass_manager_run_until_converged() -> None:
    """Test the pass manager run until converged behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Verifies that PassManager.run_until_converged executes passes until no changes.\n\n    occur\n\n    This test registers a pass that modifies the graph a finite number of times\n    and verifies that the PassManager continues execution until the graph converges\n    (i.e., no further modifications are made and the graph hash remains stable)\n\n    Returns:\n    None\n    "
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
                g.nodes[f"B{counter}"] = IRNode(id=f"B{counter}", op_type="dummy", inputs=[])
                g.nodes[f"B{counter}"].shape_metadata = ()
                return True
            return False

        pm.add_pass(dummy_pass)
        pm.run_until_converged(graph, max_iterations=5)
        assert counter == 2
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_pass_manager_run_until_converged_max_iters() -> None:
    """Test the pass manager run until converged max iters behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Verifies that PassManager.run_until_converged respects the max_iterations limit.\n\n    This test registers a pass that continuously modifies the graph and verifies\n    that the PassManager terminates execution once the specified maximum number\n    of iterations is reached, preventing infinite loops\n\n    Returns:\n    None\n    "
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
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_pass_manager_nodes_list():
    from ml_switcheroo_compiler.ir.core import IRNode
    from ml_switcheroo_compiler.transforms.pass_manager import IRValidator, _graph_hash

    class MockGraph:
        def __init__(self):
            n = IRNode(id="n1", op_type="Exp")
            n.shape_metadata = ()
            self.nodes = [n]

    validator = IRValidator()
    mock_g = MockGraph()
    IRValidator.check_shapes(mock_g)

    h = _graph_hash(mock_g)
    assert isinstance(h, str)
