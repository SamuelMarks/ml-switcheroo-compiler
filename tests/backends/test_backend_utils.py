"""Test module."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.backend_utils import format_shape_metadata, resolve_input_vars
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_mutable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


class DummyNode:
    """Test class DummyNode."""

    def __init__(self, inputs: EagerValue) -> None:
        """Test function __init__.

        Args:
            inputs: Test parameter `inputs`.
        """
        self.inputs = inputs


class DummyDim:
    """Test class DummyDim."""

    def __init__(self, id_val: EagerValue) -> None:
        """Test function __init__.

        Args:
            id_val: Test parameter `id_val`.
        """
        self.id = id_val


def test_resolve_input_vars() -> None:
    """Test function test_resolve_input_vars."""
    node = DummyNode(inputs=["a", "b", "c"])
    var_names = {"a": "var_a", "b": "var_b"}
    res = ensure_callable(resolve_input_vars)(node, var_names)
    assert res == ["var_a", "var_b", "c"]


def test_format_shape_metadata() -> None:
    """Test function test_format_shape_metadata."""
    node = DummyNode(inputs=[])

    # 1. No shape metadata
    assert ensure_callable(format_shape_metadata)(node, {}) is None

    # 2. empty shape metadata
    ensure_mutable(node).shape_metadata = ensure_eager_tensor(())
    assert ensure_callable(format_shape_metadata)(node, {}) is None

    # 3. various dimension types
    ensure_mutable(node).shape_metadata = ensure_eager_tensor((DummyDim("dim1"), "str_dim", 42))
    var_names = {"dim1": "resolved_dim1"}
    res = ensure_callable(format_shape_metadata)(node, var_names)
    assert res == "(resolved_dim1, 'str_dim', 42)"

    # 4. single element tuple formatting
    ensure_mutable(node).shape_metadata = ensure_eager_tensor((42,))
    res = ensure_callable(format_shape_metadata)(node, {})
    assert res == "(42,)"

    # 5. dummy dim missing from var_names
    ensure_mutable(node).shape_metadata = ensure_eager_tensor((DummyDim("missing"),))
    res = ensure_callable(format_shape_metadata)(node, {})
    assert res == "(missing,)"
