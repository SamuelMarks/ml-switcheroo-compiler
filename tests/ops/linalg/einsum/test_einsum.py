"""Tests for test_einsum."""

from __future__ import annotations

import typing

import pytest

from ml_switcheroo_compiler.ops.linalg.einsum import Einsum, EinsumEquationParser, EinsumLexer, EinsumPlanner, EinsumValidator, ParsedEquationPart

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol


def test_einsum_lexer() -> None:
    """Test function test_einsum_lexer."""
    (in_sub, out_sub) = EinsumLexer.parse_equation_sides("ij,jk->ik")
    assert in_sub == "ij,jk"
    assert out_sub == "ik"
    (in_sub, out_sub) = EinsumLexer.parse_equation_sides("ij,jk")
    assert in_sub == "ij,jk"
    assert out_sub == "ik"
    (in_sub, out_sub) = EinsumLexer.parse_equation_sides("...ij,jk")
    assert in_sub == "...ij,jk"
    assert out_sub == "...ik"


def test_einsum_validator() -> None:
    """Test function test_einsum_validator."""
    EinsumValidator.validate_inputs("ij,jk", [(2, 3), (3, 4)])
    with pytest.raises(ValueError):
        EinsumValidator.validate_inputs("ij,jk", [(2, 3)])


def test_einsum_planner_calculate_output_shape_missing_char() -> None:
    """Test function test_einsum_planner_calculate_output_shape_missing_char."""
    from ml_switcheroo_compiler.ops.linalg.einsum import EinsumPlanner

    axis_map = {"a": 2, "b": 3}
    out_sub = ["a", "c", "b"]
    assert typing.cast("EagerCallableProtocol", EinsumPlanner._resolve_chars)(out_sub, axis_map) == [2, 3]


def test_parsed_equation_part() -> None:
    """Test function test_parsed_equation_part."""
    part = ParsedEquationPart(chars="ij", shape=(2, 3))
    part.validate_length()
    part.validate_characters()
    amap = typing.cast("dict[str, EagerTensorProtocol]", {})
    typing.cast("EagerCallableProtocol", part.process_axis_map)(amap)
    assert amap["i"] == 2
    assert amap["j"] == 3
    with pytest.raises(ValueError):
        ParsedEquationPart(chars="ijk", shape=(2, 3)).validate_length()
    with pytest.raises(ValueError):
        ParsedEquationPart(chars="i1", shape=(2, 3)).validate_characters()
    amap = typing.cast("dict[str, EagerTensorProtocol]", {"i": 5})
    with pytest.raises(ValueError):
        typing.cast("EagerCallableProtocol", part.process_axis_map)(amap)
    amap = typing.cast("dict[str, EagerTensorProtocol]", {"i": 2})
    typing.cast("EagerCallableProtocol", part.process_axis_map)(amap)
    amap = typing.cast("dict[str, EagerTensorProtocol]", {"i": 1})
    typing.cast("EagerCallableProtocol", part.process_axis_map)(amap)


def test_einsum_planner() -> None:
    """Test function test_einsum_planner."""
    with pytest.raises(ValueError):
        EinsumPlanner._validate_ellipsis_count("...i...", (2, 3))
    with pytest.raises(ValueError):
        EinsumPlanner._count_hidden_dims(2, 2, 3, "...ij...", (2, 3, 4))
    assert EinsumPlanner._count_hidden_dims(1, 1, 4, "i...j", (2, 3, 4, 5)) == 2
    assert EinsumPlanner._combine_broadcast_shapes(None, (2, 3)) == (2, 3)
    assert EinsumPlanner._combine_broadcast_shapes((1, 3), (2, 1)) == (2, 3)
    with pytest.raises(ValueError):
        EinsumPlanner._combine_broadcast_shapes((2, 3), (4, 3))
    (chars, named, bcast) = EinsumPlanner._handle_ellipsis("i...j", (2, 3, 4, 5), None)
    assert chars == "ij"
    assert named == (2, 5)
    assert bcast == (3, 4)
    amap = typing.cast("dict[str, EagerTensorProtocol]", {})
    typing.cast("EagerCallableProtocol", EinsumPlanner._parse_named_part)("ij", (2, 3), amap)
    assert amap == {"i": 2, "j": 3}
    amap = {}
    bcast = typing.cast("tuple[int, ...] | None", typing.cast("EagerCallableProtocol", EinsumPlanner._parse_ellipsis_part)("i...j", (2, 3, 4, 5), amap, None))
    assert amap == {"i": 2, "j": 5}
    assert bcast == (3, 4)
    (amap_2, bcast_2) = typing.cast("EagerTensorProtocol", EinsumPlanner.build_axis_size_map("...ij,jk", [(3, 4, 2, 3), (3, 4)]))
    assert amap_2 == {"i": 2, "j": 3, "k": 4}
    assert bcast_2 == (3, 4)
    out = EinsumPlanner._compute_output_shape_with_ellipsis(["i", "k"], {"i": 2, "k": 4}, (3, 4))
    assert out == [2, 3, 4, 4]
    out = EinsumPlanner._compute_output_shape_with_ellipsis(["i", "x"], {"i": 2, "k": 4}, None)
    out = EinsumPlanner._compute_output_shape_with_ellipsis(["x", "k"], {"i": 2, "k": 4}, None)
    EinsumEquationParser._resolve_chars("x", {"i": 2})
    out = typing.cast("list[int]", EinsumPlanner.compute_output_shape("...ik", {"i": 2, "j": 3, "k": 4}, (3, 4)))
    assert typing.cast("EagerTensorProtocol", out) == typing.cast("EagerTensorProtocol", (3, 4, 2, 4))
    with pytest.raises(ValueError):
        EinsumPlanner.compute_output_shape("...i...k", {}, None)


def test_einsum_equation_parser() -> None:
    """Test function test_einsum_equation_parser."""
    (in_sub, out_sub) = EinsumEquationParser.parse_equation_sides("ij,jk->ik")
    assert in_sub == "ij,jk"
    (amap, bcast) = EinsumEquationParser.build_axis_size_map("ij,jk", [(2, 3), (3, 4)])
    assert amap["i"] == 2
    out = EinsumEquationParser._resolve_chars("ik", amap)
    assert out == [2, 4]
    out = typing.cast("list[int]", EinsumEquationParser.compute_output_shape("ik", amap, None))
    assert typing.cast("EagerTensorProtocol", out) == typing.cast("EagerTensorProtocol", (2, 4))
    out = typing.cast("list[int]", EinsumEquationParser.parse_and_infer_shape("ij,jk->ik", [(2, 3), (3, 4)]))
    assert typing.cast("EagerTensorProtocol", out) == typing.cast("EagerTensorProtocol", (2, 4))


def test_einsum_op() -> None:
    """Test function test_einsum_op."""
    op = Einsum()
    with pytest.raises(ValueError):
        op.infer_shape()
    assert typing.cast("EagerCallableProtocol", op.infer_shape)((2, 3), (3, 4), equation="ij,jk->ik") == (2, 4)
    assert typing.cast("EagerCallableProtocol", op.infer_shape)((2, 3), (3, 4), subscripts="ij,jk->ik") == (2, 4)
    assert typing.cast("EagerCallableProtocol", op.infer_shape)("ij,jk->ik", (2, 3), (3, 4)) == (2, 4)
    with pytest.raises(ValueError):
        typing.cast("EagerCallableProtocol", op.infer_shape)((2, 3))
    with pytest.raises(ValueError):
        typing.cast("EagerCallableProtocol", op.infer_shape)((2, 3), None, equation="ij,jk")
    assert typing.cast("EagerCallableProtocol", op.infer_shape)((2, 3), "string", equation="ij,jk") == ()
    assert op.infer_shape(equation="ij,jk") == ()
