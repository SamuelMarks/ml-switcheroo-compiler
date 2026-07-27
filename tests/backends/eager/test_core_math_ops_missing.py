import sys

import numpy as np

from ml_switcheroo_compiler.backends.eager.core_math_ops import _mock_bandpart, _mock_triangular, _mock_triangularsolve, _mock_xlog1py, _mock_xlogy


class DummyBackend:
    pass


class DummyBackendWithRandom:
    class random:
        @staticmethod
        def triangular(*args, **kwargs):
            return "random.triangular"

        @staticmethod
        def triangularsolve(*args, **kwargs):
            return "random.triangularsolve"

        @staticmethod
        def xlog1py(*args, **kwargs):
            return "random.xlog1py"

        @staticmethod
        def xlogy(*args, **kwargs):
            return "random.xlogy"


def test_missing_branches():
    x = np.ones((2, 2))

    # bandpart num_lower < 0 and num_upper < 0
    _mock_bandpart(DummyBackend(), x, -1, -1)

    # triangular with and without random module
    assert _mock_triangular(DummyBackendWithRandom(), 1, 2, 3) == "random.triangular"
    _mock_triangular(DummyBackend(), 1, 2, 3)

    # triangularsolve with and without random module
    assert _mock_triangularsolve(DummyBackendWithRandom(), np.eye(2), np.ones(2)) == "random.triangularsolve"
    _mock_triangularsolve(DummyBackend(), np.eye(2), np.ones(2))

    # xlog1py and xlogy with random module
    assert _mock_xlog1py(DummyBackendWithRandom(), 1, 2) == "random.xlog1py"
    assert _mock_xlogy(DummyBackendWithRandom(), 1, 2) == "random.xlogy"

    # Temporarily hide scipy to test the fallback branch
    original_scipy = sys.modules.get("scipy", None)
    sys.modules["scipy"] = None
    try:
        _mock_xlog1py(DummyBackend(), np.array([0.0, 1.0]), np.array([1.0, 2.0]))
        _mock_xlogy(DummyBackend(), np.array([0.0, 1.0]), np.array([1.0, 2.0]))
    finally:
        if original_scipy is not None:
            sys.modules["scipy"] = original_scipy
        else:
            del sys.modules["scipy"]

    # test np.random doesn't exist for triangular
    import builtins

    original_hasattr = builtins.hasattr

    def mocked_hasattr(obj, name):
        if obj is np and name == "random":
            return False
        return original_hasattr(obj, name)

    builtins.hasattr = mocked_hasattr
    try:
        _mock_triangular(DummyBackend(), 1, 2, 3)
    finally:
        builtins.hasattr = original_hasattr
