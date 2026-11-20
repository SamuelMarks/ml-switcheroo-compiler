"""Tests for architectural and ecosystem Tier-1/Tier-2 boundaries."""

from __future__ import annotations

import os
import subprocess

from ml_switcheroo_ir import LogicalGraph, LogicalNode

import ml_switcheroo_compiler
from ml_switcheroo_compiler.backends.registry import BackendRegistry


def test_no_numpy_leak() -> None:
    """Verify that the codebase does not leak numpy into non-numpy backends."""
    result = subprocess.run(["python3", "scripts/lint_numpy_leak.py"], capture_output=True, text=True)
    assert result.returncode == 0


def test_tier1_ir_not_vendored_and_unmutated() -> None:
    """Verify Tier-1 ml_switcheroo_ir is not vendored in compiler.ir and retains extra='forbid'."""
    ir_pkg_dir = os.path.join(os.path.dirname(ml_switcheroo_compiler.__file__), "ir")
    for forbidden_file in ("graph.py", "schema.py", "types.py", "wgsl_ops.json"):
        assert not os.path.exists(os.path.join(ir_pkg_dir, forbidden_file)), (
            f"Tier-1 ml-switcheroo-ir artifact '{forbidden_file}' must not be vendored in {ir_pkg_dir}"
        )

    assert LogicalGraph.model_config.get("extra") == "forbid"
    assert LogicalNode.model_config.get("extra") == "forbid"

    in_node = LogicalNode(id="x", op_type="Input", shape_metadata=(2, 2), dtype="float32")
    relu_node = LogicalNode(id="y", op_type="Relu", inputs=["x"], shape_metadata=(2, 2), dtype="float32")
    graph = LogicalGraph(name="bridge_parity", nodes=[in_node, relu_node], inputs=["x"], outputs=["y"])

    pure_py_cls = BackendRegistry.get("pure_python")
    generated_code = pure_py_cls(graph).generate()
    assert "def " in generated_code
