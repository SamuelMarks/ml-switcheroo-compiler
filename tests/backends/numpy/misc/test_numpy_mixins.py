"""Test module."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.backends.numpy.numpy_mixins import NumpyScatterVisitor

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class DummyNode:
    """Test class DummyNode."""

    def __init__(self, attrs: EagerValue = None) -> None:
        """Test function __init__."""
        self.attributes = attrs or {}


def test_numpy_scatter_visitor() -> None:
    """Test function test_numpy_scatter_visitor."""
    vis = NumpyScatterVisitor()
    node = DummyNode()
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_TensorScatterUpdate)(node, ["a", "i", "u"]) == "(lambda c, i, u: [c.__setitem__(tuple(np.moveaxis(np.asarray(i), -1, 0)), u), c][1])(np.copy(a), i, u)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_TensorScatterAdd)(node, ["a", "i", "u"]) == "(lambda c, i, u: [np.add.at(c, tuple(np.moveaxis(np.asarray(i), -1, 0)), u), c][1])(np.copy(a), i, u)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_TensorScatterMax)(node, ["a", "i", "u"]) == "(lambda c, i, u: [np.maximum.at(c, tuple(np.moveaxis(np.asarray(i), -1, 0)), u), c][1])(np.copy(a), i, u)"
    assert cast("Callable[..., EagerTensorProtocol]", vis.visit_TensorScatterMin)(node, ["a", "i", "u"]) == "(lambda c, i, u: [np.minimum.at(c, tuple(np.moveaxis(np.asarray(i), -1, 0)), u), c][1])(np.copy(a), i, u)"
