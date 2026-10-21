"""Test module."""

from __future__ import annotations

import typing

import pytest

from ml_switcheroo_compiler.core.mixins import TensorArithmeticMixin, TensorBitwiseMixin, TensorLogicalMixin
from tests.testing_utils import as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue
    from ml_switcheroo_compiler.ops.base import OpDef


class DummyTensor(TensorArithmeticMixin, TensorBitwiseMixin, TensorLogicalMixin):
    """Test class DummyTensor."""

    def _get_op(self, name: EagerValue) -> OpDef:
        """Test function _get_op.

        Args:
            name: Test parameter `name`.
        """
        return typing.cast("OpDef", lambda *args, **kwargs: (name, args))


def test_mixins() -> None:
    """Test function test_mixins."""
    t = DummyTensor()

    assert as_tensor((t + 1)[0]) == as_tensor("Add")
    assert as_tensor((1 + t)[0]) == as_tensor("Add")
    assert as_tensor((t - 1)[0]) == as_tensor("Subtract")
    assert as_tensor((1 - t)[0]) == as_tensor("Subtract")
    assert as_tensor((t * 1)[0]) == as_tensor("Multiply")
    assert as_tensor((1 * t)[0]) == as_tensor("Multiply")
    assert as_tensor((t / 1)[0]) == as_tensor("TrueDivide")
    assert as_tensor((1 / t)[0]) == as_tensor("TrueDivide")
    assert as_tensor((t**1)[0]) == as_tensor("Power")
    assert as_tensor((1**t)[0]) == as_tensor("Power")
    assert as_tensor((t // 1)[0]) == as_tensor("FloorDivide")
    assert as_tensor((1 // t)[0]) == as_tensor("FloorDivide")
    assert as_tensor((t % 1)[0]) == as_tensor("Mod")
    assert as_tensor((1 % t)[0]) == as_tensor("Mod")

    assert as_tensor((t & 1)[0]) == as_tensor("BitwiseAnd")
    assert as_tensor((1 & t)[0]) == as_tensor("BitwiseAnd")
    assert as_tensor((t | 1)[0]) == as_tensor("BitwiseOr")
    assert as_tensor((1 | t)[0]) == as_tensor("BitwiseOr")
    assert as_tensor((t ^ 1)[0]) == as_tensor("BitwiseXor")
    assert as_tensor((1 ^ t)[0]) == as_tensor("BitwiseXor")
    assert as_tensor((t << 1)[0]) == as_tensor("LeftShift")
    assert as_tensor((1 << t)[0]) == as_tensor("LeftShift")
    assert as_tensor((t >> 1)[0]) == as_tensor("RightShift")
    assert as_tensor((1 >> t)[0]) == as_tensor("RightShift")

    assert as_tensor((-t)[0]) == as_tensor("Negative")
    assert as_tensor((+t)[0]) == as_tensor("Positive")
    assert as_tensor(abs(t)[0]) == as_tensor("Abs")
    assert as_tensor((~t)[0]) == as_tensor("BitwiseNot")

    assert as_tensor((t < 1)[0]) == "Less"
    assert as_tensor((t > 1)[0]) == "Greater"
    assert as_tensor((t <= 1)[0]) == "LessEqual"
    assert as_tensor((t >= 1)[0]) == "GreaterEqual"
    assert as_tensor(getattr((as_tensor(t) == as_tensor(1)), "__getitem__")(0)) == as_tensor("Equal")
    assert as_tensor(getattr((as_tensor(t) != as_tensor(1)), "__getitem__")(0)) == as_tensor("NotEqual")

    assert hash(t) == id(t)


def test_mixins_real_op() -> None:
    """Test function test_mixins_real_op."""

    class DummyTensorRealOp(TensorArithmeticMixin):
        """Test class DummyTensorRealOp."""

        pass

    with pytest.raises(Exception, match=r".*"):
        DummyTensorRealOp() + 1
