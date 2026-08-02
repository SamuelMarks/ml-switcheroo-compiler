from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.axis_translation import axis_translation_pass


def test_axis_translation_pass():
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Conv2D", attributes={"layout": "NCHW"})
    n2 = IRNode(id="n2", op_type="Conv2D", attributes={"layout": "NHWC"})
    n3 = IRNode(id="n3", op_type="Add")
    g.nodes["n1"] = n1
    g.nodes["n2"] = n2
    g.nodes["n3"] = n3

    modified = axis_translation_pass(g)
    assert modified is True

    assert n1.attributes.get("layout") == "NHWC"
    assert n2.attributes.get("layout") == "NHWC"
    assert "layout" not in n3.attributes

    # Run again, should not modify
    modified_again = axis_translation_pass(g)
    assert modified_again is False


def test_axis_translation_empty_graph():
    g = IRGraph()
    assert axis_translation_pass(g) is False
