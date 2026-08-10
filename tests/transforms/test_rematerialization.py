from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.rematerialization import _estimate_compute, _estimate_memory, rematerialization_pass


def test_rematerialization_pass():
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Input")

    # Large low compute
    n2 = IRNode(id="n2", op_type="Add", inputs=["n1"])
    n2.shape_metadata = (1024, 1024)  # > 1MB

    # Small low compute
    n3 = IRNode(id="n3", op_type="Relu", inputs=["n2"])
    n3.shape_metadata = (10, 10)

    # High compute
    n4 = IRNode(id="n4", op_type="MatMul", inputs=["n3", "n1"])
    n4.shape_metadata = (1024, 1024)

    g.nodes = {"n1": n1, "n2": n2, "n3": n3, "n4": n4}

    modified = rematerialization_pass(g)
    assert modified
    assert n2.attributes.get("rematerialize") is True
    assert not n3.attributes.get("rematerialize")
    assert not n4.attributes.get("rematerialize")

    # Also coverage for the internal functions:
    assert _estimate_memory(n1) == 4.0
    n1.shape_metadata = 5
    assert _estimate_memory(n1) == 4.0
    assert _estimate_compute(n1) == 1.0
    n4.op_type = "Conv2D"
    n4.shape_metadata = [10, 10]
    assert _estimate_compute(n4) == 10000.0
    n4.shape_metadata = None
    assert _estimate_compute(n4) == 1.0
    assert _estimate_memory(n4) == 4.0

    n1.shape_metadata = [5, 5]
    assert _estimate_compute(n1) == 25.0
    n4.shape_metadata = [10, 10]
    assert _estimate_compute(n4) == 10000.0
    n4.shape_metadata = None
    assert _estimate_compute(n4) == 1.0
    assert _estimate_memory(n4) == 4.0
