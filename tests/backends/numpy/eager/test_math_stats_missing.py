import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.math_stats import _np_confusion_matrix, _np_randomcategorical, _np_randompermutation, _np_randomshuffle


class DummyBackend:
    pass


def test_math_stats_missing_branches():
    _np_randomcategorical(DummyBackend(), logits=np.ones(2))
    _np_randomcategorical(DummyBackend(), logits=np.ones(2), shape=(1,))

    class MockData:
        def __init__(self, data):
            self.data = data

    _np_randompermutation(DummyBackend(), "ignored", np.array([1, 2]))
    _np_randompermutation(DummyBackend(), "ignored", MockData(np.array([1, 2])))

    _np_confusion_matrix(DummyBackend(), MockData(np.array([0, 1])), MockData(np.array([0, 1])))
    _np_confusion_matrix(DummyBackend(), np.array([0, 1]), np.array([0, 1]))
    _np_confusion_matrix(DummyBackend(), np.array([0, 1]), np.array([0, 1]), num_classes=2)

    try:
        _np_confusion_matrix(DummyBackend())
    except ValueError:
        pass

    _np_randomshuffle(DummyBackend(), [0, 1])
    _np_randomshuffle(DummyBackend(), MockData(np.array([0, 1])))
