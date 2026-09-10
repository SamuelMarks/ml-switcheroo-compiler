"""Test module."""

from ml_switcheroo_compiler.backends.numpy.numpy_mixins import NumpyScatterVisitor


class DummyNode:
    def __init__(self, attrs=None):
        self.attributes = attrs or {}


def test_numpy_scatter_visitor():
    vis = NumpyScatterVisitor()
    node = DummyNode()
    assert vis.visit_TensorScatterUpdate(node, ["a", "i", "u"]) == "(lambda c, i, u: [c.__setitem__(tuple(np.moveaxis(np.asarray(i), -1, 0)), u), c][1])(np.copy(a), i, u)"
    assert vis.visit_TensorScatterAdd(node, ["a", "i", "u"]) == "(lambda c, i, u: [np.add.at(c, tuple(np.moveaxis(np.asarray(i), -1, 0)), u), c][1])(np.copy(a), i, u)"
    assert vis.visit_TensorScatterMax(node, ["a", "i", "u"]) == "(lambda c, i, u: [np.maximum.at(c, tuple(np.moveaxis(np.asarray(i), -1, 0)), u), c][1])(np.copy(a), i, u)"
    assert vis.visit_TensorScatterMin(node, ["a", "i", "u"]) == "(lambda c, i, u: [np.minimum.at(c, tuple(np.moveaxis(np.asarray(i), -1, 0)), u), c][1])(np.copy(a), i, u)"
