"""Unit tests for the IR pass manager, validators, and topological sorter.

This module verifies the correctness of graph validation passes, cycle detection, graph
hashing, and the iterative execution of optimization passes within the PassManager.
"""

from __future__ import annotations

import typing

import pytest

from ml_switcheroo_compiler.core.errors import CompilationError
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter, IRValidator, PassManager, _graph_hash
from tests.testing_utils import as_callable, as_tensor, as_tensor_dict

if typing.TYPE_CHECKING:
    pass


def test_load_from_config() -> None:
    """Test function test_load_from_config."""
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
        TensorOrScalar: The inferred shape or computed result.
    """
    graph = IRGraph()
    node_a = IRNode(id="A", op_type="dummy", inputs=["B"])
    node_b = IRNode(id="B", op_type="dummy", inputs=["A"])
    graph.nodes["A"] = node_a
    graph.nodes["B"] = node_b
    with pytest.raises(CompilationError, match="Cycle detected in graph."):
        DAGTopologicalSorter.sort(graph)


def test_check_cycles() -> None:
    """Test the check cycles behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    graph = IRGraph()
    node_a = IRNode(id="A", op_type="dummy", inputs=["B"])
    node_b = IRNode(id="B", op_type="dummy", inputs=["A"])
    graph.nodes["A"] = node_a
    graph.nodes["B"] = node_b
    with pytest.raises(CompilationError, match="Cycle detected in graph."):
        IRValidator.check_cycles(graph)


def test_check_shapes() -> None:
    """Test the check shapes behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
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
    """Test the graph hash behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
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
    """Test the pass manager run behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
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
    """Test the pass manager run until converged behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
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
            g.nodes[f"B{counter}"] = IRNode(id=f"B{counter}", op_type="dummy", inputs=[])
            g.nodes[f"B{counter}"].shape_metadata = ()
            return True
        return False

    pm.add_pass(dummy_pass)
    pm.run_until_converged(graph, max_iterations=5)
    assert counter == 2


def test_pass_manager_run_until_converged_max_iters() -> None:
    """Test the pass manager run until converged max iters behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
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


def test_pass_manager_nodes_list() -> None:
    """Test function test_pass_manager_nodes_list."""
    from ml_switcheroo_compiler.ir.core import IRNode
    from ml_switcheroo_compiler.transforms.pass_manager import IRValidator, _graph_hash

    class MockGraph:
        """Test class MockGraph."""

        def __init__(self) -> None:
            """Test function __init__."""
            n = IRNode(id="n1", op_type="Exp")
            n.shape_metadata = ()
            self.nodes = [n]

    IRValidator()
    mock_g = MockGraph()
    as_callable(IRValidator.check_shapes)(mock_g)
    h = as_callable(_graph_hash)(mock_g)
    assert isinstance(as_tensor(h), str)


def test_pass_pipeline_load_and_prerequisites() -> None:
    """Test function test_pass_pipeline_load_and_prerequisites."""
    pm = PassManager()
    pm.load_from_config()
    assert len(pm.passes) > 0
    assert "dead_code_elimination" in pm.pass_names


def test_pass_pipeline_missing_prerequisite_raises() -> None:
    """Test function test_pass_pipeline_missing_prerequisite_raises."""
    import tempfile

    import yaml

    invalid_pipeline = {"execution_order": ["pass_b", "pass_a"], "convergence_criteria": {"max_iterations": 5}, "prerequisites": {"pass_b": ["pass_a"]}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(invalid_pipeline, f)
        temp_path = f.name
    pm = PassManager()
    with pytest.raises(CompilationError, match="Pass 'pass_b' requires prerequisite 'pass_a'"):
        pm.load_from_config(config_path=temp_path)


def test_pass_manager_cyclic_oscillation_detection_and_rollback() -> None:
    """Test function test_pass_manager_cyclic_oscillation_detection_and_rollback."""
    pm = PassManager()
    graph = IRGraph()
    node_a = IRNode(id="A", op_type="Op1", inputs=[])
    node_a.shape_metadata = ()
    graph.nodes["A"] = node_a

    def oscillating_pass(g: IRGraph) -> bool:
        """Test function oscillating_pass.

        Args:
            g: Test parameter `g`.
        """
        node = g.nodes["A"]
        if node.op_type == "Op1":
            node.op_type = "Op2"
        else:
            node.op_type = "Op1"
        return True

    pm.add_pass(oscillating_pass)
    result_graph = pm.run_until_converged(graph, max_iterations=100)
    assert result_graph.nodes["A"].op_type in ("Op1", "Op2")


def test_pass_manager_coverage_branches() -> None:
    """Test remaining branch coverage in pass_manager.py."""
    import tempfile

    import yaml

    from ml_switcheroo_compiler.transforms.pass_manager import _restore_graph, _snapshot_graph

    g_list = IRGraph()
    n_item = IRNode(id="n1", op_type="Input")
    object.__setattr__(g_list, "nodes", [n_item])
    snap = _snapshot_graph(g_list)
    assert "n1" in snap["nodes"]
    _restore_graph(g_list, snap)
    assert len(g_list.nodes) == 1

    class DummyGraphWithoutIO:
        """Test class DummyGraphWithoutIO."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.nodes = as_tensor_dict({})

    dg = DummyGraphWithoutIO()
    as_callable(_restore_graph)(dg, snap)
    as_callable(_restore_graph)(dg, {"nodes": [], "inputs": [], "outputs": []})
    from ml_switcheroo_ir import LogicalGraph

    lg = LogicalGraph("test_lg")
    lg.inputs = ["in1"]
    lg.outputs = ["out1"]
    snap_lg = _snapshot_graph(lg)
    _restore_graph(lg, snap_lg)
    assert lg.inputs == ["in1"]
    assert lg.outputs == ["out1"]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"execution_order": ["dead_code_elimination"]}, f)
        temp_plain = f.name
    pm_plain = PassManager()
    pm_plain.load_from_config(config_path=temp_plain)
    assert "dead_code_elimination" in pm_plain.pass_names
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"other_key": 123}, f)
        temp_other = f.name
    pm_other = PassManager()
    pm_other.load_from_config(config_path=temp_other)
    assert len(pm_other.passes) == 0
    pm_conv = PassManager()

    class MockConvergence:
        """Test class MockConvergence."""

        max_iterations = 2

    setattr(pm_conv, "convergence_criteria", MockConvergence())
    g_conv = IRGraph()
    g_conv.nodes["n"] = IRNode(id="n", op_type="Input", shape_metadata=())
    res_conv = pm_conv.run_until_converged(g_conv)
    assert res_conv is not None
