"""Test module."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.backends.backend_utils import format_shape_metadata, resolve_input_vars

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class DummyNode:
    """Test class DummyNode."""

    def __init__(self, inputs: EagerValue) -> None:
        """Test function __init__."""
        self.inputs = inputs


class DummyDim:
    """Test class DummyDim."""

    def __init__(self, id_val: EagerValue) -> None:
        """Test function __init__."""
        self.id = id_val


def test_resolve_input_vars() -> None:
    """Test function test_resolve_input_vars."""
    node = DummyNode(inputs=["a", "b", "c"])
    var_names = {"a": "var_a", "b": "var_b"}
    res = cast("Callable[..., EagerTensorProtocol]", resolve_input_vars)(node, var_names)
    assert res == ["var_a", "var_b", "c"]


def test_format_shape_metadata() -> None:
    """Test function test_format_shape_metadata."""
    node = DummyNode(inputs=[])

    # 1. No shape metadata
    assert cast("Callable[..., EagerTensorProtocol]", format_shape_metadata)(node, {}) is None

    # 2. empty shape metadata
    setattr(node, "shape_metadata", ())
    assert cast("Callable[..., EagerTensorProtocol]", format_shape_metadata)(node, {}) is None

    # 3. various dimension types
    setattr(node, "shape_metadata", (DummyDim("dim1"), "str_dim", 42))
    var_names = {"dim1": "resolved_dim1"}
    res = cast("Callable[..., EagerTensorProtocol]", format_shape_metadata)(node, var_names)
    assert res == "(resolved_dim1, 'str_dim', 42)"

    # 4. single element tuple formatting
    setattr(node, "shape_metadata", (42,))
    res = cast("Callable[..., EagerTensorProtocol]", format_shape_metadata)(node, {})
    assert res == "(42,)"

    # 5. dummy dim missing from var_names
    setattr(node, "shape_metadata", (DummyDim("missing"),))
    res = cast("Callable[..., EagerTensorProtocol]", format_shape_metadata)(node, {})
    assert res == "(missing,)"
