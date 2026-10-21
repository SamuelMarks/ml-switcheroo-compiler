"""Tests for test_binary_special."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.ops.binary.special import Allclose, Atan2, Divmod, Isclose
from tests.testing_utils import as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


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
        def evaluate(*a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function evaluate.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.
            """
            return "eager_eval"

    import ml_switcheroo_compiler.ops.binary.special as sp

    orig_evaluator = as_tensor(getattr(sp, "EagerEvaluator"))

    try:
        setattr(sp, "EagerEvaluator", DummyEval())
        assert as_tensor(d(1, 2)) == as_tensor("eager_eval")
    finally:
        setattr(sp, "EagerEvaluator", orig_evaluator)

    config.eager_mode = False

    class DummyBinary:
        """Test class DummyBinary."""

        def floor_divide(self, *a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function floor_divide.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.
            """
            return "fd"

        def remainder(self, *a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function remainder.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.
            """
            return "r"

    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"ml_switcheroo_compiler.ops.binary": DummyBinary()}):
        assert as_tensor(d(1, 2)) == as_tensor(("fd", "r"))
    config.eager_mode = True


def test_binary_special_coverage() -> None:
    """Test function test_binary_special_coverage."""
    a = Atan2()
    assert as_tensor(a.infer_shape(1)) == as_tensor(1)

    d = Divmod()
    assert as_tensor(d.infer_shape(1)) == as_tensor(1)

    i = Isclose()
    assert as_tensor(i.infer_shape(1)) == as_tensor(1)
