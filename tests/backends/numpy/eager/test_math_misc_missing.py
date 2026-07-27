import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.math_misc import _np_confusion_matrix, _np_descriptive, _np_distributions, _np_rawmatmul, _np_rem, _np_sparsedensematmul


class DummyBackend:
    pass


def test_math_misc_missing_branches():
    import builtins
    import sys

    # 1. Provide an empty DummyBackend
    # These operations check if there's an ops version, else default fallback
    original_import = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name == "ml_switcheroo_compiler.ops":

            class MockOps:
                pass

            return MockOps()
        return original_import(name, *args, **kwargs)

    builtins.__import__ = mocked_import
    try:
        if "ml_switcheroo_compiler.ops" in sys.modules:
            del sys.modules["ml_switcheroo_compiler.ops"]

        # 2438->2446, 2441: rawmatmul
        _np_rawmatmul(DummyBackend(), np.eye(2), np.eye(2))

        # 2524->2532: sparsedensematmul
        _np_sparsedensematmul(DummyBackend(), np.eye(2), np.eye(2))

        # 2756->2764: rem
        _np_rem(DummyBackend(), np.ones(2), np.ones(2))

        # 2813->2821: confusion_matrix
        _np_confusion_matrix(DummyBackend(), np.array([1, 1], dtype=np.int32), np.array([1, 1], dtype=np.int32))

        # 2827->2829: confusion matrix missing num_classes
        _np_confusion_matrix(DummyBackend(), np.array([1, 1], dtype=np.int32), np.array([1, 1], dtype=np.int32), num_classes=None)

        # 2851->2857, 2854-2856: descriptive
        _np_descriptive(DummyBackend(), np.ones(2))
        _np_descriptive(DummyBackend())

        # 2882->2888, 2885-2887, 2890-2891: distributions
        _np_distributions(DummyBackend(), np.ones(2))
        _np_distributions(DummyBackend())

    finally:
        builtins.__import__ = original_import
        if "ml_switcheroo_compiler.ops" in sys.modules:
            del sys.modules["ml_switcheroo_compiler.ops"]
