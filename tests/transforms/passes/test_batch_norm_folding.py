from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.batch_norm_folding import batch_norm_folding_pass


def test_batch_norm_folding_pass():
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Conv2D", inputs=["in1"])
    n2 = IRNode(id="n2", op_type="BatchNorm", inputs=["n1"])
    n3 = IRNode(id="n3", op_type="BatchNorm", inputs=["non_existent"])
    g.nodes["n1"] = n1
    g.nodes["n2"] = n2
    g.nodes["n3"] = n3

    modified = batch_norm_folding_pass(g)
    assert modified is True

    assert n1.attributes.get("folded_batch_norm") is True
    assert n2.attributes.get("folded") is True
    assert "folded" not in n3.attributes

    # Run again, should not modify (actually it might modify again if we don't check)
    # Wait, our simple pass doesn't check if already folded, so it will return True.
    # We should update it to check.


def test_batch_norm_folding_empty_graph():
    g = IRGraph()
    assert batch_norm_folding_pass(g) is False


def test_batch_norm_folding_again():
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Conv2D", inputs=["in1"])
    n2 = IRNode(id="n2", op_type="BatchNorm", inputs=["n1"])
    g.nodes["n1"] = n1
    g.nodes["n2"] = n2

    batch_norm_folding_pass(g)
    modified_again = batch_norm_folding_pass(g)
    assert modified_again is False


def test_batch_norm_folding_no_inputs():
    g = IRGraph()
    n2 = IRNode(id="n2", op_type="BatchNorm", inputs=[])
    g.nodes["n2"] = n2

    modified = batch_norm_folding_pass(g)
    assert modified is False
