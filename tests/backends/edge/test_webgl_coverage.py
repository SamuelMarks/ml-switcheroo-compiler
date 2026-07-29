"""Test WebGL backend edge cases coverage."""

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.edge.webgl import WebGLCodeGenerator


def test_webgl_compilation_and_orchestration():
    g = LogicalGraph(outputs=["out"])

    n_multi = LogicalNode(id="in1", op_type="Input", shape_metadata=(2, 3, 4))
    n_neg = LogicalNode(id="n_neg", op_type="Negative", inputs=["in1"], shape_metadata=(2, 3, 4))
    n_generic = LogicalNode(id="out", op_type="Tan", inputs=["n_neg"], shape_metadata=(2, 3, 4))

    g.nodes = {"in1": n_multi, "n_neg": n_neg, "out": n_generic}

    gen = WebGLCodeGenerator(g)
    code = gen.generate()

    assert "tan" in code


def test_webgl_compilation_1d():
    g = LogicalGraph(outputs=["out"])

    n1 = LogicalNode(id="in1", op_type="Input", shape_metadata=(4,))
    n2 = LogicalNode(id="in2", op_type="Input", shape_metadata=(4,))
    n_add = LogicalNode(id="out", op_type="Add", inputs=["in1", "in2"], shape_metadata=(4,))

    g.nodes = {"in1": n1, "in2": n2, "out": n_add}

    gen = WebGLCodeGenerator(g)
    code = gen.generate()
    assert "+" in code

    n_exp = LogicalNode(id="out", op_type="Exp", inputs=["in1"], shape_metadata=(4,))
    g.nodes["out"] = n_exp
    gen = WebGLCodeGenerator(g)
    assert "exp(" in gen.generate()

    # Also test empty shape
    n_empty = LogicalNode(id="out2", op_type="Input", shape_metadata=())
    g.nodes["out2"] = n_empty
    gen = WebGLCodeGenerator(g)
    # _get_shape_and_strides
    assert gen._get_shape_and_strides(n_empty) == ([], [])


def test_webgl_no_inputs():
    """Test webgl without inputs."""
    g = LogicalGraph(outputs=["n1"])
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Constant", attributes={"value": 1.0}, shape_metadata=(5, 5))
    gen = WebGLCodeGenerator(g)
    code = gen.generate()

    assert "vec2 uv = vec2(0.5, 0.5);" in code
    assert "int idx = 0;" in code


def test_webgl_input_missing_shape():
    """Test shape and stride getter when shape is missing."""
    g = LogicalGraph(outputs=["n1"])
    n1 = LogicalNode(id="n1", op_type="Input")
    g.nodes["n1"] = n1
    gen = WebGLCodeGenerator(g)
    shape, strides = gen._get_shape_and_strides(n1)
    assert shape == []
    assert strides == []

    n_scalar = LogicalNode(id="n_scalar", op_type="Input", shape_metadata=1)
    shape, strides = gen._get_shape_and_strides(n_scalar)
    assert shape == [1]


def test_webgl_generic_visit_none():
    """Test generic_visit with None."""
    gen = WebGLCodeGenerator(LogicalGraph())
    assert gen.generic_visit(None, []) == "glsl_op"


def test_webgl_evaluate_input_directly():
    """Explicitly test generic_visit on Input nodes to cover lines 116-123."""
    gen = WebGLCodeGenerator(LogicalGraph())

    n_2d = LogicalNode(id="n_2d", op_type="Input", shape_metadata=(2, 2))
    assert gen.generic_visit(n_2d, []) == "get_val_in_0(idx)"

    n_1d = LogicalNode(id="n_1d", op_type="Input", shape_metadata=(2,))
    assert gen.generic_visit(n_1d, []) == "texture(in_1, uv).r"


def test_webgl_orchestration_branches():
    # To hit 212->216 (has_ndim_gt_1 = True, but input_nodes is empty)
    # How to make has_ndim_gt_1 = True if input_nodes is empty?
    # Make a non-input node have shape > 1!
    g = LogicalGraph(outputs=[])
    g.nodes["const1"] = LogicalNode(id="const1", op_type="Constant", shape_metadata=(2, 2))
    gen = WebGLCodeGenerator(g)
    code = gen.generate()
    assert "int idx = 0;" in code  # has_ndim_gt_1 is True, no inputs

    # 237->245 (output_ids is True but out_node is None or no shape dims)
    # Give a fake output id
    g2 = LogicalGraph(outputs=["fake_id"])
    gen2 = WebGLCodeGenerator(g2)
    code2 = gen2.generate()
    assert "vec4(fake_id" in code2
