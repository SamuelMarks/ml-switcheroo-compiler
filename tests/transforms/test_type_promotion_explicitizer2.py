from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.type_promotion_explicitizer import _needs_cast, type_promotion_explicitizer_pass


def test_type_promotion_explicitizer2():
    graph = IRGraph()
    n1 = IRNode(id="n1", op_type="Input", inputs=[], attributes={"dtype": DType.Float32.value})
    n2 = IRNode(id="n2", op_type="Input", inputs=[], attributes={"dtype": DType.Float64.value})
    n3 = IRNode(id="n3", op_type="Input", inputs=[], attributes={"dtype": DType.Int32.value})

    # Needs cast for Float32 and Float64 -> target is Float64
    n_add1 = IRNode(id="n_add1", op_type="Add", inputs=["n1", "n2"], attributes={})
    # Needs cast for Int32 and Float32 -> target is Float32
    n_add2 = IRNode(id="n_add2", op_type="Add", inputs=["n3", "n1"], attributes={})
    # Same types, no cast
    n_add3 = IRNode(id="n_add3", op_type="Add", inputs=["n1", "n1"], attributes={})
    # Error in type promotion -> invalid types
    n_add4 = IRNode(id="n_add4", op_type="Add", inputs=["n1", "invalid"], attributes={})
    n_invalid = IRNode(id="invalid", op_type="Input", inputs=[], attributes={"dtype": "invalid_type"})

    graph.nodes["n1"] = n1
    graph.nodes["n2"] = n2
    graph.nodes["n3"] = n3
    graph.nodes["n_add1"] = n_add1
    graph.nodes["n_add2"] = n_add2
    graph.nodes["n_add3"] = n_add3
    graph.nodes["invalid"] = n_invalid
    graph.nodes["n_add4"] = n_add4

    res = type_promotion_explicitizer_pass(graph)
    assert res is True

    # check if needs_cast returns None for invalid
    assert _needs_cast("invalid1", "invalid2") is None
    assert _needs_cast(None, "float32") is None
