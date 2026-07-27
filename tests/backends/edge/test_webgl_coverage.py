"""Test WebGL backend edge cases coverage."""

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.edge.webgl import WebGLCodeGenerator


def test_webgl_coverage():
    """Test webgl code generation edge cases."""
    g = LogicalGraph(outputs=["n2"])

    # 1. Test scalar/no-shape input (returns [], [])
    n_scalar = LogicalNode(id="n_scalar", op_type="Input")  # No shape

    # 2. Test multi-dimensional input (triggers helper generation)
    n_multi = LogicalNode(id="n_multi", op_type="Input", shape_metadata=(2, 3, 4))

    # 3. Test Negative and generic operation execution
    n_neg = LogicalNode(id="n_neg", op_type="Negative", inputs=["n_multi"])
    n_generic = LogicalNode(id="n_generic", op_type="Tan", inputs=["n_neg"])

    g.nodes = {"n_scalar": n_scalar, "n_multi": n_multi, "n_neg": n_neg, "n2": n_generic}

    gen = WebGLCodeGenerator(g)

    # Trigger generic_visit properly to hit logic branches before building orchestration
    # This correctly exercises the Input evaluating paths (116-123)
    gen.generic_visit(n_scalar, [])
    gen.generic_visit(n_multi, [])
    gen.generic_visit(n_neg, ["in_multi"])
    gen.generic_visit(n_generic, ["in_neg"])

    code = gen.generate()

    # Assert WebGL generator produced multi-dimensional tex helpers
    assert "int c_" in code
    assert "offset =" in code

    # Assert Negative and Tan operations
    assert " -get_val_in_1(idx)" in code or " -texture(" in code or " -get_val_in" in code or " -v_" in code
    assert " tan(" in code

    # Assert gl_FragCoord branching
    assert "int idx = int(gl_FragCoord.x) + int(gl_FragCoord.y) *" in code


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

    n2 = LogicalNode(id="n2", op_type="Input", shape_metadata=())
    g.nodes["n2"] = n2
    shape, strides = gen._get_shape_and_strides(n2)
    assert shape == []
    assert strides == []

    n3 = LogicalNode(id="n3", op_type="Input", shape_metadata=5)
    g.nodes["n3"] = n3
    shape, strides = gen._get_shape_and_strides(n3)
    assert shape == [5]
    assert strides == [1]


def test_webgl_generic_visit_none():
    """Test generic_visit with None."""
    gen = WebGLCodeGenerator(LogicalGraph())
    assert gen.generic_visit(None, []) == "glsl_op"


def test_webgl_evaluate_input_directly():
    """Explicitly test generic_visit on Input nodes to cover lines 116-123."""
    gen = WebGLCodeGenerator(LogicalGraph())

    # Input with 2D shape
    n_2d = LogicalNode(id="n_2d", op_type="Input", shape_metadata=(2, 2))
    assert gen.generic_visit(n_2d, []) == "get_val_in_0(idx)"

    # Input with 1D shape
    n_1d = LogicalNode(id="n_1d", op_type="Input", shape_metadata=(2,))
    assert gen.generic_visit(n_1d, []) == "texture(in_1, uv).r"
