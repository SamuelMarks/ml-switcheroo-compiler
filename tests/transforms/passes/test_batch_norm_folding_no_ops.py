from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.batch_norm_folding import batch_norm_folding_pass


def test_batch_norm_folding_empty():
    g = IRGraph()
    bn = IRNode("bn", "BatchNorm", inputs=[])
    g.nodes = {"bn": bn}
    g.outputs = ["bn"]

    batch_norm_folding_pass(g)

    # Should be a no-op
    assert g.nodes["bn"].op_type == "BatchNorm"
