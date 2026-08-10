from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.axis_translation import axis_translation_pass


def test_axis_translation_empty_inputs():
    g = IRGraph()
    n = IRNode("conv", "Conv2D", inputs=[])
    n.attributes["layout"] = "NCHW"
    g.nodes = {"conv": n}
    g.outputs = ["conv"]

    axis_translation_pass(g)

    assert g.nodes["conv"].attributes["layout"] == "NHWC"
    assert len(g.nodes["conv"].inputs) == 0
