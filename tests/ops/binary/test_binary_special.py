"""Tests for test_binary_special."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.ops.binary.special import Allclose, Atan2, Divmod, Isclose
from tests.testing_utils import ensure_eager_tensor, ensure_mutable

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

            Returns:
                EagerValue: Computed output value from `evaluate`.
            """
            return "eager_eval"

    import ml_switcheroo_compiler.ops.binary.special as sp

    orig_evaluator = ensure_eager_tensor(ensure_eager_tensor(sp).EagerEvaluator)

    try:
        ensure_mutable(sp).EagerEvaluator = ensure_eager_tensor(DummyEval())
        assert ensure_eager_tensor(d(1, 2)) == ensure_eager_tensor("eager_eval")
    finally:
        ensure_mutable(sp).EagerEvaluator = ensure_eager_tensor(orig_evaluator)

    config.eager_mode = False

    class DummyBinary:
        """Test class DummyBinary."""

        def floor_divide(self, *a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function floor_divide.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `floor_divide`.
            """
            return "fd"

        def remainder(self, *a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function remainder.

            Args:
                a: Test parameter `a`.
                k: Test parameter `k`.

            Returns:
                EagerValue: Computed output value from `remainder`.
            """
            return "r"

    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"ml_switcheroo_compiler.ops.binary": DummyBinary()}):
        assert ensure_eager_tensor(d(1, 2)) == ensure_eager_tensor(("fd", "r"))
    config.eager_mode = True


def test_binary_special_coverage() -> None:
    """Test function test_binary_special_coverage."""
    a = Atan2()
    assert ensure_eager_tensor(a.infer_shape(1)) == ensure_eager_tensor(1)

    d = Divmod()
    assert ensure_eager_tensor(d.infer_shape(1)) == ensure_eager_tensor(1)

    i = Isclose()
    assert ensure_eager_tensor(i.infer_shape(1)) == ensure_eager_tensor(1)
