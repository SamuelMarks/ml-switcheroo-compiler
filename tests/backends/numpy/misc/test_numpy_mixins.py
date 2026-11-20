"""Test module."""

from __future__ import annotations

from ml_switcheroo_ir import AttributeValue

from ml_switcheroo_compiler.backends.numpy.numpy_mixins import NumpyScatterVisitor
from tests.testing_utils import MockNodeStub


class DummyNode(MockNodeStub):
    """Test class DummyNode."""

    def __init__(self, attrs: dict[str, AttributeValue] | None = None) -> None:
        """Test function __init__.

        Args:
            attrs: Test parameter `attrs`.
        """
        self.attributes = attrs if attrs is not None else {}


def test_numpy_scatter_visitor() -> None:
    """Test function test_numpy_scatter_visitor."""
    vis = NumpyScatterVisitor()
    node = DummyNode()
    assert (
        vis.visit_TensorScatterUpdate(node, ["a", "i", "u"])
        == "(lambda c, i, u: [c.__setitem__(tuple(np.moveaxis(np.asarray(i), -1, 0)), u), c][1])(np.copy(a), i, u)"
    )
    assert (
        vis.visit_TensorScatterAdd(node, ["a", "i", "u"])
        == "(lambda c, i, u: [np.add.at(c, tuple(np.moveaxis(np.asarray(i), -1, 0)), u), c][1])(np.copy(a), i, u)"
    )
    assert (
        vis.visit_TensorScatterMax(node, ["a", "i", "u"])
        == "(lambda c, i, u: [np.maximum.at(c, tuple(np.moveaxis(np.asarray(i), -1, 0)), u), c][1])(np.copy(a), i, u)"
    )
    assert (
        vis.visit_TensorScatterMin(node, ["a", "i", "u"])
        == "(lambda c, i, u: [np.minimum.at(c, tuple(np.moveaxis(np.asarray(i), -1, 0)), u), c][1])(np.copy(a), i, u)"
    )
