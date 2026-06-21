import pytest
from ml_switcheroo_compiler.ops.linalg.basic import EinsumEquationParser, Einsum


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
    with pytest.raises(ValueError, match="Equation has 3 operands, but 2 shapes were provided"):
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

    # Shape length mismatch in regular processing
    with pytest.raises(ValueError, match="Shape length mismatch"):
        EinsumEquationParser.build_axis_size_map("ij,jkl", shapes)


def test_process_ellipsis_subscript():
    # Multiple ellipses
    with pytest.raises(ValueError, match="Multiple ellipses in operand subscript"):
        EinsumEquationParser._process_ellipsis_subscript("i...j...", (2, 3), {}, None)

    # Shape too small
    with pytest.raises(ValueError, match="Shape too small for subscripts"):
        EinsumEquationParser._process_ellipsis_subscript("...ij", (2,), {}, None)

    # Dimension mismatch in left
    dim_map = {"i": 5}
    with pytest.raises(ValueError, match="Dimension mismatch for subscript i"):
        EinsumEquationParser._process_ellipsis_subscript("i...j", (2, 3), dim_map, None)

    # Dimension mismatch in right
    dim_map = {"j": 5}
    with pytest.raises(ValueError, match="Dimension mismatch for subscript j"):
        EinsumEquationParser._process_ellipsis_subscript("i...j", (2, 3), dim_map, None)

    # Shape not a tuple
    with pytest.raises(ValueError, match="Shape must be a tuple"):
        EinsumEquationParser.build_axis_size_map("ij", [2])  # type: ignore


def test_process_regular_subscript():
    # Dimension mismatch
    dim_map = {"i": 5}
    with pytest.raises(ValueError, match="Dimension mismatch for subscript i"):
        EinsumEquationParser._process_regular_subscript("ij", (2, 3), dim_map)


def test_resolve_ellipses():
    # Broadcast
    assert EinsumEquationParser._resolve_ellipses((1, 5), (3, 1)) == (3, 5)
    assert EinsumEquationParser._resolve_ellipses((5,), (3, 5)) == (3, 5)


def test_compute_output_shape():
    dim_map = {"i": 2, "j": 3, "k": 4}
    # Regular
    assert EinsumEquationParser.compute_output_shape("ik", dim_map, None) == (2, 4)

    # Missing subscript
    with pytest.raises(ValueError, match="Output subscript z not in input"):
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
