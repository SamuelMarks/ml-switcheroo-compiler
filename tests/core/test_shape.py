"""Unit tests verifying shape broadcasting and matrix multiplication validation."""

from __future__ import annotations

import types
import typing

import pytest

from ml_switcheroo_compiler.core.errors import ShapeMismatchError
from ml_switcheroo_compiler.core.shape import _broadcast_dim, broadcast_shapes, validate_matrix_multiply_shapes

if typing.TYPE_CHECKING:
    pass


def test_broadcast_dim_equal() -> None:
    """Test broadcasting identical dimensions."""
    assert _broadcast_dim(4, 4, (4,), (4,)) == 4
    assert _broadcast_dim("?", "?", ("?",), ("?",)) == "?"


def test_broadcast_dim_ones_and_symbolic() -> None:
    """Test broadcasting unit and symbolic wildcard dimensions."""
    assert _broadcast_dim(1, 8, (1,), (8,)) == 8
    assert _broadcast_dim(8, 1, (8,), (1,)) == 8
    assert _broadcast_dim("?", 16, ("?",), (16,)) == 16
    assert _broadcast_dim(16, "?", (16,), ("?",)) == 16
    assert _broadcast_dim(-1, 32, (-1,), (32,)) == 32
    assert _broadcast_dim(32, -1, (32,), (-1,)) == 32


def test_broadcast_dim_mismatch() -> None:
    """Test error raised when dimensions cannot be broadcast together."""
    with pytest.raises(ShapeMismatchError):
        _broadcast_dim(4, 8, (4,), (8,))


def test_broadcast_shapes_success() -> None:
    """Test broadcasting various ranks, unit expansions, and symbolic shapes."""
    assert broadcast_shapes((), (3, 4)) == (3, 4)
    assert broadcast_shapes((1, 4), (5, 1)) == (5, 4)
    assert broadcast_shapes((2, 1, 6), (3, 6)) == (2, 3, 6)
    assert broadcast_shapes(("?", 1), (10, 4)) == (10, 4)


def test_broadcast_shapes_failure() -> None:
    """Test error raised when complete shape tuples are incompatible."""
    with pytest.raises(ShapeMismatchError):
        broadcast_shapes((3, 4), (3, 5))


def test_validate_matmul_shapes_strict_2d() -> None:
    """Test strict 2-D matrix multiplication validation rules."""
    assert validate_matrix_multiply_shapes((2, 3), (3, 4), strict_2d=True) == (2, 4)
    assert validate_matrix_multiply_shapes(("?", 3), (3, "?"), strict_2d=True) == ("?", "?")

    # Incompatible inner dimension
    with pytest.raises(ShapeMismatchError):
        validate_matrix_multiply_shapes((2, 3), (4, 5), strict_2d=True)

    # Incompatible rank (< 2 or > 2)
    with pytest.raises(ShapeMismatchError):
        validate_matrix_multiply_shapes((3,), (3, 4), strict_2d=True)

    with pytest.raises(ShapeMismatchError):
        validate_matrix_multiply_shapes((2, 3, 4), (2, 4, 5), strict_2d=True)


def test_validate_matmul_shapes_zero_rank() -> None:
    """Test error raised when an operand has rank 0."""
    with pytest.raises(ShapeMismatchError):
        validate_matrix_multiply_shapes((), (3, 4))
    with pytest.raises(ShapeMismatchError):
        validate_matrix_multiply_shapes((3, 4), ())


def test_validate_matmul_shapes_1d_dot() -> None:
    """Test 1-D vector dot product shape inference."""
    assert validate_matrix_multiply_shapes((4,), (4,)) == ()
    assert validate_matrix_multiply_shapes(("?",), (4,)) == ()

    with pytest.raises(ShapeMismatchError):
        validate_matrix_multiply_shapes((4,), (5,))


def test_validate_matmul_shapes_vector_matrix() -> None:
    """Test 1-D vector with N-D matrix multiplication."""
    assert validate_matrix_multiply_shapes((3,), (3, 4)) == (4,)
    assert validate_matrix_multiply_shapes((3,), (2, 3, 4)) == (2, 4)

    with pytest.raises(ShapeMismatchError):
        validate_matrix_multiply_shapes((3,), (4, 5))


def test_validate_matmul_shapes_matrix_vector() -> None:
    """Test N-D matrix with 1-D vector multiplication."""
    assert validate_matrix_multiply_shapes((2, 3), (3,)) == (2,)
    assert validate_matrix_multiply_shapes((5, 2, 3), (3,)) == (5, 2)

    with pytest.raises(ShapeMismatchError):
        validate_matrix_multiply_shapes((2, 3), (4,))


def test_validate_matmul_shapes_batched() -> None:
    """Test batched matrix multiplication with broadcasted batch dimensions."""
    assert validate_matrix_multiply_shapes((2, 3, 4), (4, 5)) == (2, 3, 5)
    assert validate_matrix_multiply_shapes((1, 3, 4), (2, 4, 5)) == (2, 3, 5)

    with pytest.raises(ShapeMismatchError):
        # Mismatched contracting dimension
        validate_matrix_multiply_shapes((2, 3, 4), (2, 5, 6))

    with pytest.raises(ShapeMismatchError):
        # Mismatched batch dimension
        validate_matrix_multiply_shapes((3, 2, 4), (4, 4, 5))


def test_shape_ellipsis_version_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify both Python <3.10 and >=3.10 EllipsisType resolution branches in ``core.shape``.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """
    import importlib
    import sys

    import ml_switcheroo_compiler.core._shape_types as shape_types_mod
    import ml_switcheroo_compiler.core.shape as shape_mod

    monkeypatch.setattr(types, "EllipsisType", type(Ellipsis), raising=False)
    monkeypatch.setattr(typing, "TypeAlias", type, raising=False)
    monkeypatch.setattr(sys, "version_info", (3, 10, 0, "final", 0))
    importlib.reload(shape_types_mod)
    importlib.reload(shape_mod)
    assert shape_mod.broadcast_shapes((2, 1), (1, 3)) == (2, 3)
    monkeypatch.undo()
    monkeypatch.setattr(sys, "version_info", (3, 9, 0, "final", 0))
    importlib.reload(shape_types_mod)
    importlib.reload(shape_mod)
    monkeypatch.undo()
    importlib.reload(shape_types_mod)
    importlib.reload(shape_mod)
    assert shape_mod.broadcast_shapes((4, 1), (1, 5)) == (4, 5)

    from collections.abc import Sequence

    class MetaOnlyStub:
        """Stub exposing ``shape_metadata`` without ``shape``."""

        shape_metadata: Sequence[shape_mod.ShapeDim] | None = (2, "N")

    class ShapeOnlyStub:
        """Stub exposing ``shape`` directly."""

        shape: Sequence[shape_mod.ShapeDim] = (3, "B")

    assert shape_mod.extract_symbolic_shape(ShapeOnlyStub()) == (3, "B")
    assert shape_mod.extract_symbolic_shape(MetaOnlyStub()) == (2, "N")
    assert shape_mod.extract_symbolic_shape(42, default=(1,)) == (1,)
    assert shape_mod.extract_shape(MetaOnlyStub()) == (2, -1)
    assert shape_mod.extract_shape(42, default=(1,)) == (1,)
    assert shape_mod.extract_dtype(42, default="int32") == "int32"
    assert shape_mod.extract_device(42, default="cpu") == "cpu"
    assert shape_mod.extract_data(42) == 42

    from tests.fixtures import build_test_graph, build_test_node

    n_fix = build_test_node("n0", "Relu", ["x"], {"alpha": 0.0})
    g_fix = build_test_graph("g_fix", {"n0": n_fix}, ["x"], ["n0"])
    assert g_fix.nodes["n0"].op_type == "Relu"


def test_mock_protocol_and_broken_data_holders() -> None:
    """Verify is_mock_with_attrs and readable-data helpers across Python 3.9 and 3.12+."""
    from unittest.mock import MagicMock

    from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs
    from ml_switcheroo_compiler.core._shape_extractors import _has_readable_data_attr
    from ml_switcheroo_compiler.core.shape import OpValue
    from ml_switcheroo_compiler.ops.creation.api_basic import _can_read_data_holder

    assert not is_mock_with_attrs(42, "shape")
    configured_mock = MagicMock(shape=(2, 3))
    assert is_mock_with_attrs(configured_mock, "shape")
    assert is_mock_with_attrs(MagicMock(), "unconfigured_attr")
    specced_mock = MagicMock(spec=["only_attr"])
    assert not is_mock_with_attrs(specced_mock, "missing_attr")

    class _BrokenDataHolder:
        """DataHolderProtocol witness whose .data property raises AttributeError."""

        @property
        def data(self) -> OpValue:
            """Raise AttributeError when inspected.

            Returns:
                OpValue: Unreachable; raises ``AttributeError``.

            Raises:
                AttributeError: Always raised to exercise broken descriptor handling.
            """
            raise AttributeError("broken descriptor")

    class _ValidDataHolder:
        """DataHolderProtocol witness whose .data property returns an integer."""

        @property
        def data(self) -> OpValue:
            """Return scalar data payload.

            Returns:
                OpValue: Constant integer payload.
            """
            return 1

    assert not _has_readable_data_attr(_BrokenDataHolder())
    assert _has_readable_data_attr(_ValidDataHolder())
    assert not _can_read_data_holder(_BrokenDataHolder())
    assert _can_read_data_holder(_ValidDataHolder())
