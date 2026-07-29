import pytest

from ml_switcheroo_compiler.ops.linalg.einsum import Einsum, EinsumEquationParser, EinsumLexer, EinsumPlanner, EinsumValidator, ParsedEquationPart


def test_einsum_lexer():
    in_sub, out_sub = EinsumLexer.parse_equation_sides("ij,jk->ik")
    assert in_sub == "ij,jk"
    assert out_sub == "ik"

    in_sub, out_sub = EinsumLexer.parse_equation_sides("ij,jk")
    assert in_sub == "ij,jk"
    assert out_sub == "ik"

    in_sub, out_sub = EinsumLexer.parse_equation_sides("...ij,jk")
    assert in_sub == "...ij,jk"
    assert out_sub == "...ik"


def test_einsum_validator():
    EinsumValidator.validate_inputs("ij,jk", [(2, 3), (3, 4)])
    with pytest.raises(ValueError):
        EinsumValidator.validate_inputs("ij,jk", [(2, 3)])


def test_einsum_planner_calculate_output_shape_missing_char():
    from ml_switcheroo_compiler.ops.linalg.einsum import EinsumPlanner

    axis_map = {"a": 2, "b": 3}
    out_sub = ["a", "c", "b"]  # 'c' is not in axis_map

    assert EinsumPlanner._resolve_chars(out_sub, axis_map) == [2, 3]


def test_parsed_equation_part():
    part = ParsedEquationPart("ij", (2, 3))
    part.validate_length()
    part.validate_characters()

    amap = {}
    part.process_axis_map(amap)
    assert amap["i"] == 2
    assert amap["j"] == 3

    with pytest.raises(ValueError):
        ParsedEquationPart("ijk", (2, 3)).validate_length()

    with pytest.raises(ValueError):
        ParsedEquationPart("i1", (2, 3)).validate_characters()

    amap = {"i": 5}
    with pytest.raises(ValueError):
        part.process_axis_map(amap)

    # Allow matching duplicates or 1
    amap = {"i": 2}
    part.process_axis_map(amap)

    amap = {"i": 1}
    part.process_axis_map(amap)


def test_einsum_planner():
    # _validate_ellipsis_count
    with pytest.raises(ValueError):
        EinsumPlanner._validate_ellipsis_count("...i...", (2, 3))

    # _count_hidden_dims
    with pytest.raises(ValueError):
        EinsumPlanner._count_hidden_dims(2, 2, 3, "...ij...", (2, 3, 4))
    assert EinsumPlanner._count_hidden_dims(1, 1, 4, "i...j", (2, 3, 4, 5)) == 2

    # _combine_broadcast_shapes
    assert EinsumPlanner._combine_broadcast_shapes(None, (2, 3)) == (2, 3)
    assert EinsumPlanner._combine_broadcast_shapes((1, 3), (2, 1)) == (2, 3)
    with pytest.raises(ValueError):
        EinsumPlanner._combine_broadcast_shapes((2, 3), (4, 3))

    # _handle_ellipsis
    chars, named, bcast = EinsumPlanner._handle_ellipsis("i...j", (2, 3, 4, 5), None)
    assert chars == "ij"
    assert named == (2, 5)
    assert bcast == (3, 4)

    # _parse_named_part
    amap = {}
    EinsumPlanner._parse_named_part("ij", (2, 3), amap)
    assert amap == {"i": 2, "j": 3}

    # _parse_ellipsis_part
    amap = {}
    bcast = EinsumPlanner._parse_ellipsis_part("i...j", (2, 3, 4, 5), amap, None)
    assert amap == {"i": 2, "j": 5}
    assert bcast == (3, 4)

    # build_axis_size_map
    amap, bcast = EinsumPlanner.build_axis_size_map("...ij,jk", [(3, 4, 2, 3), (3, 4)])
    assert amap == {"i": 2, "j": 3, "k": 4}
    assert bcast == (3, 4)

    # _compute_output_shape_with_ellipsis
    out = EinsumPlanner._compute_output_shape_with_ellipsis(["i", "k"], {"i": 2, "k": 4}, (3, 4))
    assert out == [2, 3, 4, 4]

    # Missing char in axis map
    try:
        out = EinsumPlanner._compute_output_shape_with_ellipsis(["i", "x"], {"i": 2, "k": 4}, None)
    except KeyError:
        pass

    try:
        out = EinsumPlanner._compute_output_shape_with_ellipsis(["x", "k"], {"i": 2, "k": 4}, None)
    except KeyError:
        pass

    try:
        EinsumEquationParser._resolve_chars("x", {"i": 2})
    except KeyError:
        pass

    # compute_output_shape
    out = EinsumPlanner.compute_output_shape("...ik", {"i": 2, "j": 3, "k": 4}, (3, 4))
    assert out == (3, 4, 2, 4)
    with pytest.raises(ValueError):
        EinsumPlanner.compute_output_shape("...i...k", {}, None)


def test_einsum_equation_parser():
    in_sub, out_sub = EinsumEquationParser.parse_equation_sides("ij,jk->ik")
    assert in_sub == "ij,jk"

    amap, bcast = EinsumEquationParser.build_axis_size_map("ij,jk", [(2, 3), (3, 4)])
    assert amap["i"] == 2

    out = EinsumEquationParser._resolve_chars("ik", amap)
    assert out == [2, 4]

    out = EinsumEquationParser.compute_output_shape("ik", amap, None)
    assert out == (2, 4)

    out = EinsumEquationParser.parse_and_infer_shape("ij,jk->ik", [(2, 3), (3, 4)])
    assert out == (2, 4)


def test_einsum_op():
    op = Einsum()

    # Missing args
    with pytest.raises(ValueError):
        op.infer_shape()

    # Test correct
    assert op.infer_shape((2, 3), (3, 4), equation="ij,jk->ik") == (2, 4)
    assert op.infer_shape((2, 3), (3, 4), subscripts="ij,jk->ik") == (2, 4)
    assert op.infer_shape("ij,jk->ik", (2, 3), (3, 4)) == (2, 4)

    with pytest.raises(ValueError):
        op.infer_shape((2, 3))

    try:
        op.infer_shape((2, 3), None, equation="ij,jk")
    except ValueError:
        pass

    assert op.infer_shape((2, 3), "string", equation="ij,jk") == ()
    assert op.infer_shape(equation="ij,jk") == ()
