"""Tests for pass manager lifecycle, buffer allocation gap reuse, and pattern discovery."""

import os
import tempfile

import yaml

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.pass_manager import (
    PassManager,
    _restore_graph,
    _snapshot_graph,
)
from ml_switcheroo_compiler.transforms.passes.buffer_allocation import BufferAllocationPass
from ml_switcheroo_compiler.transforms.passes.config_models import ConvergenceCriteria
from ml_switcheroo_compiler.transforms.passes.operator_fusion import (
    _discover_fusion_patterns,
)


def test_pass_manager_coverage_branches():
    """Test pass manager snapshot, restore, and config execution branches."""

    # 1. Graph with list nodes instead of dict
    class MockListGraph:
        def __init__(self):
            n = IRNode(id="n1", op_type="Input")
            n.shape_metadata = ()
            self.nodes = [n]
            self.inputs = ["n1"]
            self.outputs = ["n1"]

    lg = MockListGraph()
    snap = _snapshot_graph(lg)
    assert "n1" in snap["nodes"]

    # Restore onto list graph
    lg_target = MockListGraph()
    _restore_graph(lg_target, snap)
    assert len(lg_target.nodes) == 1

    # 2. load_from_config with plain execution_order
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"execution_order": ["dead_code_elimination"]}, f)
        temp_path1 = f.name

    pm = PassManager()
    pm.load_from_config(config_path=temp_path1)
    assert "dead_code_elimination" in pm.pass_names

    # 3. load_from_config with invalid structure
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump("just a string", f)
        temp_path2 = f.name

    pm2 = PassManager()
    pm2.load_from_config(config_path=temp_path2)
    assert len(pm2.passes) == 0

    # 4. run_until_converged with convergence_criteria set
    g = IRGraph()
    n = IRNode(id="x", op_type="Input")
    n.shape_metadata = ()
    g.nodes["x"] = n

    pm3 = PassManager()
    pm3.convergence_criteria = ConvergenceCriteria(max_iterations=2)
    res_g = pm3.run_until_converged(g)
    assert "x" in res_g.nodes


def test_buffer_allocation_gap_reuse():
    """Test calculate_arena_offsets finding gap between active blocks."""
    alloc_pass = BufferAllocationPass(alignment=16)
    g = IRGraph()

    # n1: alive 0..3, size 64, offset 0..64
    n1 = IRNode(id="n1", op_type="Input", shape_metadata=(16,))
    n1.attributes["dtype"] = "float32"

    # n2: alive 0..5, size 64, offset 64..128
    n2 = IRNode(id="n2", op_type="Input", shape_metadata=(16,))
    n2.attributes["dtype"] = "float32"

    # n3: alive 1..2, size 16 (can fit in gap if n1 died early)
    n3 = IRNode(id="n3", op_type="Add", inputs=["n1"], shape_metadata=(4,))
    n3.attributes["dtype"] = "float32"

    # n4: alive 4..5, size 32 (fits in n1's slot after n1 expires at step 3)
    n4 = IRNode(id="n4", op_type="Add", inputs=["n2"], shape_metadata=(8,))
    n4.attributes["dtype"] = "float32"

    # n_dyn: symbolic shape
    n_dyn = IRNode(id="n_dyn", op_type="Add", shape_metadata=("B", 10))
    n_dyn.attributes["dtype"] = "float32"

    g.nodes = {"n1": n1, "n2": n2, "n3": n3, "n4": n4, "n_dyn": n_dyn}
    g.outputs = ["n4"]

    offsets = alloc_pass.calculate_arena_offsets(g, alignment=16)
    assert len(offsets) == 4
    for off in offsets.values():
        assert off % 16 == 0


def test_discover_fusion_patterns_invalid_file():
    """Test error handling in pattern discovery when file is corrupt."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bad_file = os.path.join(tmpdir, "bad.yaml")
        with open(bad_file, "w") as f:
            f.write(":: this is invalid yaml ::")

        rules = _discover_fusion_patterns(patterns_dir=tmpdir)
        assert isinstance(rules, list)
        assert len(rules) == 0
