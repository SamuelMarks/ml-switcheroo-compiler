def test_webgl_exhaustive_extra():
    """Test line coverage for WebGL generator."""
    import pytest

    from ml_switcheroo_compiler.backends.edge.webgl import WebGLCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    # Test missing template
    n_missing = IRNode(id="n_missing", op_type="MissingOp", inputs=[])
    g.nodes[n_missing.id] = n_missing
    g.sorted_nodes = [n_missing]

    gen = WebGLCodeGenerator(g)
    with pytest.raises(ValueError, match="Missing WebGL shader template"):
        gen.generate()


def test_webgl_shapes_and_inputs():
    """Test WebGL generation with shapes and inputs."""
    from ml_switcheroo_compiler.backends.edge.webgl import WebGLCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()

    n_in1 = IRNode(id="in1", op_type="Input", inputs=[], shape_metadata=[4, 8])
    n_in2 = IRNode(id="in2", op_type="Input", inputs=[], shape_metadata=[4])
    n_in3 = IRNode(id="in3", op_type="Input", inputs=[], shape_metadata=[2, "dim"])

    n_add = IRNode(id="n_add", op_type="Add", inputs=["in1", "in2", "in3", "in1", "in2"], shape_metadata=[4, 8])
    n_matmul = IRNode(id="n_matmul", op_type="MatMul", inputs=["n_add"], shape_metadata=[8])

    for n in [n_in1, n_in2, n_in3, n_add, n_matmul]:
        g.nodes[n.id] = n

    g.inputs = ["in1", "in2", "in3"]

    gen = WebGLCodeGenerator(g)
    code = gen.generate()

    assert "shader_n_add" in code
    assert "shader_n_matmul" in code
