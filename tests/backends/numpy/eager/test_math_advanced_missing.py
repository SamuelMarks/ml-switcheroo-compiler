import sys
from unittest.mock import patch

import pytest

import ml_switcheroo_compiler.ops as ops


def setup_mock_opdef():
    if not hasattr(ops, "OpDef"):

        class DummyOpDef:
            pass

        ops.OpDef = DummyOpDef


def test_math_matrix_utils_distributions_fallback():
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_matrix_utils import _np_distributions

    setup_mock_opdef()

    class DummyBackend:
        def distributions(self, *args, **kwargs):
            return "hit_dist"

    class MockDistributions:
        def __new__(cls, *args, **kwargs):
            return "hit_mock_dist"

    ops.distributions = MockDistributions

    # Test success path
    assert _np_distributions(DummyBackend(), 1) == "hit_mock_dist"

    # Test exception path
    with patch("builtins.issubclass", side_effect=Exception("Test Error")):
        assert _np_distributions(DummyBackend(), 1) == "hit_dist"

    del ops.distributions


def test_math_matrix_utils_confusion_matrix_fallback():
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_matrix_utils import _np_confusion_matrix

    setup_mock_opdef()

    class DummyBackend:
        def confusion_matrix(self, *args, **kwargs):
            return "hit_cm"

    class MockCM:
        def __new__(cls, *args, **kwargs):
            return "hit_mock_cm"

    ops.confusion_matrix = MockCM

    assert _np_confusion_matrix(DummyBackend(), [0], [0], num_classes=1) == "hit_mock_cm"

    with patch("builtins.issubclass", side_effect=Exception("Test Error")):
        assert _np_confusion_matrix(DummyBackend(), [0], [0], num_classes=1) == "hit_cm"

    del ops.confusion_matrix


def test_math_misc_ext_descriptive_fallback():
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_misc_ext import _np_descriptive_2

    setup_mock_opdef()

    class DummyBackend:
        def descriptive(self, *args, **kwargs):
            return "hit_desc"

        def asarray(self, a):
            return a

        def zeros(self, a):
            return [1]

        def min(self, a):
            return 1

        def max(self, a):
            return 1

        def mean(self, a):
            return 1.0

        def std(self, a):
            return 0.0

    class MockDesc:
        def __new__(cls, *args, **kwargs):
            return "hit_mock_desc"

    ops.descriptive = MockDesc

    assert _np_descriptive_2(DummyBackend(), 1) == "hit_mock_desc"

    with patch("builtins.issubclass", side_effect=Exception("Test Error")):
        assert _np_descriptive_2(DummyBackend(), 1) == "hit_desc"

    del ops.descriptive


def test_math_misc_ext_rem_fallback():
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_misc_ext import _np_rem_2

    setup_mock_opdef()

    class DummyBackend:
        def rem(self, *args, **kwargs):
            return "hit_rem"

    class MockRem:
        def __new__(cls, *args, **kwargs):
            return "hit_mock_rem"

    ops.rem = MockRem

    assert _np_rem_2(DummyBackend(), 1, 2) == "hit_mock_rem"

    with patch("builtins.issubclass", side_effect=Exception("Test Error")):
        assert _np_rem_2(DummyBackend(), 1, 2) == "hit_rem"

    del ops.rem


def test_math_misc_ext_rem_3():
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_misc_ext import _np_rem_3

    class DummyBackend:
        pass

    assert _np_rem_3(DummyBackend(), 5, 2) == 1
    assert _np_rem_3(DummyBackend()) is None


def test_math_misc_ext_scipy_import_error():
    from ml_switcheroo_compiler.backends.numpy.eager.math_advanced.math_misc_ext import _np_betainc

    class DummyBackend:
        pass

    orig_scipy = sys.modules.get("scipy.special")
    sys.modules["scipy.special"] = None
    try:
        with pytest.raises(ImportError):
            _np_betainc(DummyBackend(), 1)
    finally:
        if orig_scipy:
            sys.modules["scipy.special"] = orig_scipy
        else:
            del sys.modules["scipy.special"]
