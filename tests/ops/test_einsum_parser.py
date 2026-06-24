import pytest
from ml_switcheroo_compiler.ops.linalg.einsum import EinsumEquationParser, Einsum


def test_parse_equation_sides():
    # Explicit ->
    in_subs, out_sub = EinsumEquationParser.parse_equation_sides("ij, jk -> ik")
    assert in_subs == "ij,jk"
    assert out_sub == "ik"

    # Implicit -> with counts
    in_subs, out_sub = EinsumEquationParser.parse_equation_sides("ij, jk")
    assert in_subs == "ij,jk"
    assert out_sub == "ik"

    # Implicit -> with ellipses
    in_subs, out_sub = EinsumEquationParser.parse_equation_sides("...ij, jk")
    assert in_subs == "...ij,jk"
    assert out_sub == "...ik"


def test_build_axis_size_map():
    shapes = [(2, 3), (3, 4)]
    dim_map, ell = EinsumEquationParser.build_axis_size_map("ij,jk", shapes)
    assert dim_map == {"i": 2, "j": 3, "k": 4}
    assert ell is None

    # Mismatch operand count
    with pytest.raises(ValueError, match="Equation expected 3 inputs but got 2"):
        EinsumEquationParser.build_axis_size_map("ij,jk,kl", shapes)

    # Ellipsis processing
    shapes_ell = [(5, 2, 3), (3, 4)]
    dim_map, ell = EinsumEquationParser.build_axis_size_map("...ij,jk", shapes_ell)
    assert dim_map == {"i": 2, "j": 3, "k": 4}
    assert ell == (5,)

    # Ellipsis with mismatch
    shapes_ell2 = [(5, 2, 3), (10, 3, 4)]
    with pytest.raises(ValueError, match="Ellipsis shapes cannot be broadcast"):
        EinsumEquationParser.build_axis_size_map("...ij,...jk", shapes_ell2)

    # Shape \(3, 4\) cannot match subscript jkl in regular processing
    with pytest.raises(ValueError, match="Shape \(3, 4\) cannot match subscript jkl"):
        EinsumEquationParser.build_axis_size_map("ij,jkl", shapes)


def test_compute_output_shape():
    dim_map = {"i": 2, "j": 3, "k": 4}
    # Regular
    assert EinsumEquationParser.compute_output_shape("ik", dim_map, None) == (2, 4)

    # Missing subscript
    with pytest.raises(ValueError, match="Output character z not found in inputs"):
        EinsumEquationParser.compute_output_shape("iz", dim_map, None)

    # Ellipsis multiple
    with pytest.raises(ValueError, match="Multiple ellipses in output subscript"):
        EinsumEquationParser.compute_output_shape("i...j...", dim_map, (5,))

    # Ellipsis out
    assert EinsumEquationParser.compute_output_shape("i...k", dim_map, (5,)) == (2, 5, 4)


def test_einsum_infer_shape():
    e = Einsum()

    # Missing args and equation
    with pytest.raises(ValueError, match="Einsum requires an 'equation' string attribute."):
        e.infer_shape()

    # Empty shapes
    assert e.infer_shape(equation="i->i") == ()

    # Not a tuple shape
    assert e.infer_shape("i->i", None) == ()

    # Invalid tuple
    assert e.infer_shape("i->i", "non-tuple-shape") == ()

    # Correct shape
    assert e.infer_shape((2, 3), (3, 4), equation="ij,jk->ik") == (2, 4)

    # Implicit equation in args
    assert e.infer_shape("ij,jk->ik", (2, 3), (3, 4)) == (2, 4)


def test_compute_output_shape_ellipsis_none():
    dim_map = {"i": 2, "k": 4}
    # Ellipsis out but ellipsis_shape is None
    assert EinsumEquationParser.compute_output_shape("i...k", dim_map, None) == (2, 4)
