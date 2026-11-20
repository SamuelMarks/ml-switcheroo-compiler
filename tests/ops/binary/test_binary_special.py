"""Tests for test_binary_special."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.ops.binary.special import Allclose, Atan2, Divmod, Isclose

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_binary_special() -> None:
    """Test function test_binary_special."""
    a = Atan2()
    assert a.infer_shape((1,), (1,)) == (1,)
    pass
    pass

    d = Divmod()
    assert d.infer_shape((1,), (1,)) == (1,)
    pass
    pass

    ac = Allclose()
    assert ac.infer_shape() == ()

    i = Isclose()
    assert i.infer_shape((1,), (1,)) == (1,)
    pass
    pass

    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True

    class DummyEval:
        """Test class DummyEval."""

        @staticmethod
        def evaluate(*a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function evaluate.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `evaluate`.
            """
            return "eager_eval"

    import unittest.mock as mock

    import ml_switcheroo_compiler.ops.binary.special as sp

    with mock.patch.object(sp, "EagerEvaluator", DummyEval()):
        assert d(1, 2) == "eager_eval"

    config.eager_mode = False

    class DummyBinary:
        """Test class DummyBinary."""

        def floor_divide(self, *a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function floor_divide.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `floor_divide`.
            """
            return "fd"

        def remainder(self, *a: BackendArg, **k: BackendArg) -> BackendArg:
            """Test function remainder.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                BackendArg: Computed output value from `remainder`.
            """
            return "r"

    import sys
    from unittest.mock import patch

    import ml_switcheroo_compiler.ops.binary as _real_binary

    assert _real_binary is not None
    with patch.dict(sys.modules, {"ml_switcheroo_compiler.ops.binary": DummyBinary()}):
        assert d(1, 2) == ("fd", "r")
    config.eager_mode = True


def test_binary_special_coverage() -> None:
    """Test function test_binary_special_coverage."""
    a = Atan2()
    assert a.infer_shape(1) == (1,)

    d = Divmod()
    assert d.infer_shape(1) == (1,)

    i = Isclose()
    assert i.infer_shape(1) == (1,)
