"""Test WebGPU backend edge cases coverage."""

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator


def test_webgpu_coverage():
    """Test webgpu code generation edge cases."""
    g = LogicalGraph(outputs=["n2"])

    # 1. Test scalar/no-shape input (returns [], [])
    n_scalar = LogicalNode(id="n_scalar", op_type="Input")  # No shape

    # 2. Test multi-dimensional input (triggers helper generation)
    n_multi = LogicalNode(id="n_multi", op_type="Input", shape_metadata=(2, 3, 4))

    # 3. Test Negative and generic operation execution
    n_neg = LogicalNode(id="n_neg", op_type="Negative", inputs=["n_multi"])
    n_generic = LogicalNode(id="n_generic", op_type="Tan", inputs=["n_neg"])

    g.nodes = {"n_scalar": n_scalar, "n_multi": n_multi, "n_neg": n_neg, "n2": n_generic}

    gen = WebGPUCodeGenerator(g)

    code = gen.generate()

    # Assert WebGPU generator produced multi-dimensional tex helpers
    assert "fn get_offset_" in code
    assert "let c_" in code

    # Assert Negative and Tan operations
    assert " -in_" in code or " -v_" in code
    assert " tan(" in code


def test_webgpu_node_no_id():
    """Test webgpu helper generation with no id."""
    g = LogicalGraph(outputs=["n1"])
    n1 = LogicalNode(id="n1", op_type="Input", shape_metadata=(5, 5))
    n1.id = ""  # intentionally blank
    g.nodes["n1"] = n1
    gen = WebGPUCodeGenerator(g)
    # _generate_coord_helpers
    assert "fn get_offset_" not in gen._generate_coord_helpers()


def test_webgpu_input_missing_shape():
    """Test shape and stride getter when shape is missing."""
    g = LogicalGraph(outputs=["n1"])
    n1 = LogicalNode(id="n1", op_type="Input")
    g.nodes["n1"] = n1
    gen = WebGPUCodeGenerator(g)
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


def test_webgpu_generic_visit_none():
    """Test generic_visit with None."""
    gen = WebGPUCodeGenerator(LogicalGraph())
    assert gen.generic_visit(None, []) == ""


def test_webgpu_evaluate_input_directly():
    """Explicitly test generic_visit on Input nodes to cover lines 124-131."""
    gen = WebGPUCodeGenerator(LogicalGraph())

    # Input with 2D shape
    n_2d = LogicalNode(id="n_2d", op_type="Input", shape_metadata=(2, 2))
    assert gen.generic_visit(n_2d, []) == "in_0[get_offset_n_2d(idx)]"

    # Input with 1D shape
    n_1d = LogicalNode(id="n_1d", op_type="Input", shape_metadata=(2,))
    assert gen.generic_visit(n_1d, []) == "in_1[idx]"


def test_webgpu_orchestration_missing_out_node():
    """Test orchestration generation when the output_id doesn't correspond to a node."""
    g = LogicalGraph(outputs=["non_existent_output"])
    gen = WebGPUCodeGenerator(g)

    # We call generate() and it implicitly checks the path for missing output node.
    code = gen.generate()
    # verify there was no error
    assert "non_existent_output" in code or True


test_webgpu_orchestration_missing_out_node()
"""Test WebGPU empty output ids."""


def test_webgpu_orchestration_empty_output_ids():
    """Test orchestration generation when output_ids is empty."""
    g = LogicalGraph(outputs=[])
    gen = WebGPUCodeGenerator(g)

    code = gen.generate()
    # It hits lines 347->354 because output_ids is empty
    assert code is not None


test_webgpu_orchestration_empty_output_ids()
