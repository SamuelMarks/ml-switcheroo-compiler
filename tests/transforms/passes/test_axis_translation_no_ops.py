"""Tests for test_axis_translation_no_ops."""

from __future__ import annotations

from ml_switcheroo_ir import NodeDict

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.axis_translation import axis_translation_pass


def test_axis_translation_empty_inputs() -> None:
    """Test function test_axis_translation_empty_inputs."""
    g = IRGraph()
    n = IRNode("conv", "Conv2D", inputs=[])
    n.attributes["layout"] = "NCHW"
    g.nodes = NodeDict({"conv": n})
    g.outputs = ["conv"]

    axis_translation_pass(g)

    assert g.nodes["conv"].attributes["layout"] == "NHWC"
    assert len(g.nodes["conv"].inputs) == 0
