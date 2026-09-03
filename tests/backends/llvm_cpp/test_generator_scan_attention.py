def test_scan_and_attention():
    """Test Scan and Attention."""
    from ml_switcheroo_compiler.backends.llvm_cpp.generator import CppGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    graph = IRGraph()
    n1 = IRNode(id="n1", op_type="Scan", inputs=["in_val"], shape_metadata=[10])
    n2 = IRNode(id="n2", op_type="Attention", inputs=["in_val", "in_val", "in_val"], shape_metadata=[10, 10])
    graph.nodes = {"n1": n1, "n2": n2}

    gen = CppGenerator(graph)
    code = gen.generate()
    assert "NDArrayView<float> n1({10});" in code
    assert "NDArrayView<float> n2({10,10});" in code
