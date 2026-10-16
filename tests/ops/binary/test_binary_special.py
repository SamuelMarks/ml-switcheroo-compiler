"""Tests for test_binary_special."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.ops.binary.special import Allclose, Atan2, Divmod, Isclose

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


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
            """Test function evaluate."""
            return "eager_eval"

    import ml_switcheroo_compiler.ops.binary.special as sp

    orig_evaluator = cast("EagerTensorProtocol", getattr(sp, "EagerEvaluator"))

    try:
        setattr(sp, "EagerEvaluator", DummyEval())
        assert cast("EagerTensorProtocol", d(1, 2)) == cast("EagerTensorProtocol", "eager_eval")
    finally:
        setattr(sp, "EagerEvaluator", orig_evaluator)

    config.eager_mode = False

    class DummyBinary:
        """Test class DummyBinary."""

        def floor_divide(self, *a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function floor_divide."""
            return "fd"

        def remainder(self, *a: EagerValue, **k: EagerValue) -> EagerValue:
            """Test function remainder."""
            return "r"

    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"ml_switcheroo_compiler.ops.binary": DummyBinary()}):
        assert cast("EagerTensorProtocol", d(1, 2)) == cast("EagerTensorProtocol", ("fd", "r"))
    config.eager_mode = True


def test_binary_special_coverage() -> None:
    """Test function test_binary_special_coverage."""
    a = Atan2()
    assert cast("EagerTensorProtocol", a.infer_shape(1)) == cast("EagerTensorProtocol", 1)

    d = Divmod()
    assert cast("EagerTensorProtocol", d.infer_shape(1)) == cast("EagerTensorProtocol", 1)

    i = Isclose()
    assert cast("EagerTensorProtocol", i.infer_shape(1)) == cast("EagerTensorProtocol", 1)
