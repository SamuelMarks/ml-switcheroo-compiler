"""Test test_poly_lower.py."""

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.poly_lower import polyfill_lowering_pass


def test_in_top_k_lowering():
    """Test test_in_top_k_lowering."""
    g = IRGraph()
    # Predictions: [batch, classes]
    g.nodes["preds"] = IRNode(id="preds", op_type="Input", shape_metadata=(10, 5))
    # Targets: [batch]
    g.nodes["targets"] = IRNode(id="targets", op_type="Input", shape_metadata=(10,))

    n = IRNode(id="n1", op_type="InTopK", inputs=["targets", "preds"], attributes={"k": 2})
    g.nodes["n1"] = n

    modified = polyfill_lowering_pass(g)
    assert modified

    # n1 should now be ReduceAny
    assert g.nodes["n1"].op_type == "ReduceAny"

    # Should contain TopK, TupleGetItem, ExpandDims, Equal
    op_types = [node.op_type for node in g.nodes.values()]
    assert "TopK" in op_types
    assert "TupleGetItem" in op_types
    assert "ExpandDims" in op_types
    assert "Equal" in op_types


def test_ctc_greedy_decoder_lowering():
    """Test test_ctc_greedy_decoder_lowering."""
    g = IRGraph()
    g.nodes["inputs"] = IRNode(id="inputs", op_type="Input", shape_metadata=(50, 10, 32))
    g.nodes["seq_len"] = IRNode(id="seq_len", op_type="Input", shape_metadata=(10,))

    n = IRNode(id="n1", op_type="CtcGreedyDecoder", inputs=["inputs", "seq_len"])
    g.nodes["n1"] = n

    modified = polyfill_lowering_pass(g)
    assert modified

    assert g.nodes["n1"].op_type == "CollapseRepeated"
    op_types = [node.op_type for node in g.nodes.values()]
    assert "Argmax" in op_types


def test_no_poly_lower():
    """Test test_no_poly_lower."""
    g = IRGraph()
    n = IRNode(id="n1", op_type="Add", inputs=["inputs", "seq_len"])
    g.nodes["n1"] = n
    modified = polyfill_lowering_pass(g)
    assert not modified


def test_rest_poly_lowering():
    """Test test_rest_poly_lowering."""
    g = IRGraph()
    n_iso = IRNode(id="iso", op_type="IsotonicRegression", inputs=[])
    n_ct = IRNode(id="ct", op_type="ConvTranspose", inputs=[], attributes={})
    n_dwf = IRNode(id="dwf", op_type="DepthwiseConv2dBackpropFilter", inputs=[], attributes={})
    n_dwi = IRNode(id="dwi", op_type="DepthwiseConv2dBackpropInput", inputs=[], attributes={})
    n_dil = IRNode(id="dil", op_type="Dilation2d", inputs=[], attributes={})
    n_ero = IRNode(id="ero", op_type="Erosion2d", inputs=[], attributes={})

    g.nodes = {"iso": n_iso, "ct": n_ct, "dwf": n_dwf, "dwi": n_dwi, "dil": n_dil, "ero": n_ero}

    modified = polyfill_lowering_pass(g)
    assert modified

    assert g.nodes["iso"].op_type == "WhileLoop"
    assert g.nodes["ct"].op_type == "Conv2D"
    assert g.nodes["dwf"].op_type == "Conv2D"
    assert g.nodes["dwi"].op_type == "Conv2D"
    assert g.nodes["dil"].op_type == "MaxPool2D"
    assert g.nodes["ero"].op_type == "MinPool2D"
