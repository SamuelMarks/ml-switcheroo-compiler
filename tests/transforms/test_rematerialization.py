from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.rematerialization import _estimate_compute, _estimate_memory, _load_rules, rematerialization_pass


def test_rematerialization_pass():
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Input")

    # Large low compute
    n2 = IRNode(id="n2", op_type="Add", inputs=["n1"])
    n2.shape_metadata = (1024, 1024)  # > 1MB

    # 15 dummy nodes to create distance
    nodes = {"n1": n1, "n2": n2}
    prev = "n2"
    for i in range(15):
        nid = f"dummy_{i}"
        node = IRNode(id=nid, op_type="Relu", inputs=[prev])
        nodes[nid] = node
        prev = nid

    # Distant consumer
    n3 = IRNode(id="n3", op_type="MatMul", inputs=[prev, "n2"])
    nodes["n3"] = n3

    g.nodes = nodes

    modified = rematerialization_pass(g)
    assert modified
    assert n2.attributes.get("rematerialize") is True
    assert g.nodes["n3"].inputs[1] == "n2_remat"

    rules = _load_rules()

    # Also coverage for the internal functions:
    assert _estimate_memory(n1) == 4.0
    n1.shape_metadata = 5
    assert _estimate_memory(n1) == 4.0
    assert _estimate_compute(n1, rules) == 1.0
    n3.op_type = "Conv2D"
    n3.shape_metadata = [10, 10]
    assert _estimate_compute(n3, rules) == 10000.0
    n3.shape_metadata = None
    assert _estimate_compute(n3, rules) == 1.0
    assert _estimate_memory(n3) == 4.0

    n1.shape_metadata = [5, 5]
    assert _estimate_compute(n1, rules) == 25.0
    n3.shape_metadata = [10, 10]
    assert _estimate_compute(n3, rules) == 10000.0
    n3.shape_metadata = None
    assert _estimate_compute(n3, rules) == 1.0
    assert _estimate_memory(n3) == 4.0


def test_rematerialization_already_exists():
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.passes.rematerialization import rematerialization_pass

    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Input")
    n2 = IRNode(id="n2", op_type="Add", inputs=["n1"])
    n2.shape_metadata = (1024, 1024)

    nodes = {"n1": n1, "n2": n2}
    prev = "n2"
    for i in range(15):
        nid = f"dummy_{i}"
        node = IRNode(id=nid, op_type="Relu", inputs=[prev])
        nodes[nid] = node
        prev = nid

    n3 = IRNode(id="n3", op_type="MatMul", inputs=[prev, "n2"])
    nodes["n3"] = n3

    # simulate what happens if we already have the recompute node in graph
    n2_rec = IRNode(id="n2_remat", op_type="Add")
    nodes["n2_remat"] = n2_rec

    g.nodes = nodes

    modified = rematerialization_pass(g)
    # the cloned node is already in graph, but the inputs to consumers will still be updated if they weren't
    assert modified
    assert n2.attributes.get("rematerialize") is True
