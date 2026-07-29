from unittest.mock import patch

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.math_misc import numpy_eager_registry


class MockCallableClass:
    def __init__(self, *args, **kwargs):
        self.called = True


def test_math_misc_missing_branches():
    import ml_switcheroo_compiler.ops as ops

    # Save original attributes
    attrs = ["RawMatMul", "SparseDenseMatMul", "rem", "confusion_matrix", "descriptive", "distributions"]
    originals = {}
    for attr in attrs:
        if hasattr(ops, attr):
            originals[attr] = getattr(ops, attr)
            delattr(ops, attr)

    try:
        res = numpy_eager_registry._registry["RawMatMul"](None, np.array([[1]]), np.array([[2]]))
        assert res.shape == (1, 1)

        res = numpy_eager_registry._registry["SparseDenseMatMul"](None, np.array([[1]]), np.array([[2]]))
        assert res.shape == (1, 1)

        res = numpy_eager_registry._registry["rem"](None, np.array([5]), np.array([2]))
        assert res[0] == 1

        res = numpy_eager_registry._registry["confusion_matrix"](None, np.array([0]), np.array([0]), num_classes=2)
        assert res.shape == (2, 2)

        res = numpy_eager_registry._registry["descriptive"](None, np.array([1, 2]))
        assert len(res) == 3

        res = numpy_eager_registry._registry["distributions"](None, np.array([1, 2]))
        assert len(res) == 2
    finally:
        # Restore
        for attr, val in originals.items():
            setattr(ops, attr, val)

    # Provide num_classes to Descriptive
    res = numpy_eager_registry._registry["Descriptive"](None, np.array([0]), num_classes=2)
    assert "mean" in res


def test_math_misc_callable_classes():
    import ml_switcheroo_compiler

    class DummyOps:
        descriptive = MockCallableClass
        distributions = MockCallableClass
        OpDef = type("DummyOpDef", (), {})

    with patch.object(ml_switcheroo_compiler, "ops", DummyOps()):
        res = numpy_eager_registry._registry["descriptive"](None, np.array([1]))
        assert getattr(res, "called", False) == True

        res = numpy_eager_registry._registry["distributions"](None, np.array([1]))
        assert getattr(res, "called", False) == True


def test_math_misc_confusion_matrix_none_classes():
    # test with num_classes=None to cover True branch
    res = numpy_eager_registry._registry["confusion_matrix"](None, np.array([0]), np.array([0]))
    assert res.shape == (1, 1)

    # test Descriptive with num_classes=None (Descriptive doesn't return shape, it returns a dict)
    res = numpy_eager_registry._registry["Descriptive"](None, np.array([0]))
    assert "mean" in res
