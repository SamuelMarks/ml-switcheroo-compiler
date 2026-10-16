"""Test module."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from ml_switcheroo_compiler.core.mixins import TensorArithmeticMixin, TensorBitwiseMixin, TensorLogicalMixin

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue
    from ml_switcheroo_compiler.ops.base import OpDef


class DummyTensor(TensorArithmeticMixin, TensorBitwiseMixin, TensorLogicalMixin):
    """Test class DummyTensor."""

    def _get_op(self, name: EagerValue) -> OpDef:
        """Test function _get_op."""
        return cast("OpDef", lambda *args, **kwargs: (name, args))


def test_mixins() -> None:
    """Test function test_mixins."""
    t = DummyTensor()

    assert cast("EagerTensorProtocol", (t + 1)[0]) == cast("EagerTensorProtocol", "Add")
    assert cast("EagerTensorProtocol", (1 + t)[0]) == cast("EagerTensorProtocol", "Add")
    assert cast("EagerTensorProtocol", (t - 1)[0]) == cast("EagerTensorProtocol", "Subtract")
    assert cast("EagerTensorProtocol", (1 - t)[0]) == cast("EagerTensorProtocol", "Subtract")
    assert cast("EagerTensorProtocol", (t * 1)[0]) == cast("EagerTensorProtocol", "Multiply")
    assert cast("EagerTensorProtocol", (1 * t)[0]) == cast("EagerTensorProtocol", "Multiply")
    assert cast("EagerTensorProtocol", (t / 1)[0]) == cast("EagerTensorProtocol", "TrueDivide")
    assert cast("EagerTensorProtocol", (1 / t)[0]) == cast("EagerTensorProtocol", "TrueDivide")
    assert cast("EagerTensorProtocol", (t**1)[0]) == cast("EagerTensorProtocol", "Power")
    assert cast("EagerTensorProtocol", (1**t)[0]) == cast("EagerTensorProtocol", "Power")
    assert cast("EagerTensorProtocol", (t // 1)[0]) == cast("EagerTensorProtocol", "FloorDivide")
    assert cast("EagerTensorProtocol", (1 // t)[0]) == cast("EagerTensorProtocol", "FloorDivide")
    assert cast("EagerTensorProtocol", (t % 1)[0]) == cast("EagerTensorProtocol", "Mod")
    assert cast("EagerTensorProtocol", (1 % t)[0]) == cast("EagerTensorProtocol", "Mod")

    assert cast("EagerTensorProtocol", (t & 1)[0]) == cast("EagerTensorProtocol", "BitwiseAnd")
    assert cast("EagerTensorProtocol", (1 & t)[0]) == cast("EagerTensorProtocol", "BitwiseAnd")
    assert cast("EagerTensorProtocol", (t | 1)[0]) == cast("EagerTensorProtocol", "BitwiseOr")
    assert cast("EagerTensorProtocol", (1 | t)[0]) == cast("EagerTensorProtocol", "BitwiseOr")
    assert cast("EagerTensorProtocol", (t ^ 1)[0]) == cast("EagerTensorProtocol", "BitwiseXor")
    assert cast("EagerTensorProtocol", (1 ^ t)[0]) == cast("EagerTensorProtocol", "BitwiseXor")
    assert cast("EagerTensorProtocol", (t << 1)[0]) == cast("EagerTensorProtocol", "LeftShift")
    assert cast("EagerTensorProtocol", (1 << t)[0]) == cast("EagerTensorProtocol", "LeftShift")
    assert cast("EagerTensorProtocol", (t >> 1)[0]) == cast("EagerTensorProtocol", "RightShift")
    assert cast("EagerTensorProtocol", (1 >> t)[0]) == cast("EagerTensorProtocol", "RightShift")

    assert cast("EagerTensorProtocol", (-t)[0]) == cast("EagerTensorProtocol", "Negative")
    assert cast("EagerTensorProtocol", (+t)[0]) == cast("EagerTensorProtocol", "Positive")
    assert cast("EagerTensorProtocol", abs(t)[0]) == cast("EagerTensorProtocol", "Abs")
    assert cast("EagerTensorProtocol", (~t)[0]) == cast("EagerTensorProtocol", "BitwiseNot")

    assert cast("EagerTensorProtocol", (t < 1)[0]) == "Less"
    assert cast("EagerTensorProtocol", (t > 1)[0]) == "Greater"
    assert cast("EagerTensorProtocol", (t <= 1)[0]) == "LessEqual"
    assert cast("EagerTensorProtocol", (t >= 1)[0]) == "GreaterEqual"
    assert cast("EagerTensorProtocol", getattr((cast("EagerTensorProtocol", t) == cast("EagerTensorProtocol", 1)), "__getitem__")(0)) == cast("EagerTensorProtocol", "Equal")
    assert cast("EagerTensorProtocol", getattr((cast("EagerTensorProtocol", t) != cast("EagerTensorProtocol", 1)), "__getitem__")(0)) == cast("EagerTensorProtocol", "NotEqual")

    assert hash(t) == id(t)


def test_mixins_real_op() -> None:
    """Test function test_mixins_real_op."""

    class DummyTensorRealOp(TensorArithmeticMixin):
        """Test class DummyTensorRealOp."""

        pass

    with pytest.raises(Exception, match=r".*"):
        DummyTensorRealOp() + 1
