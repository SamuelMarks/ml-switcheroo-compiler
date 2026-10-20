"""Test module."""

from __future__ import annotations

import typing

import pytest

from ml_switcheroo_compiler.core.mixins import TensorArithmeticMixin, TensorBitwiseMixin, TensorLogicalMixin

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue
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

    assert typing.cast("EagerTensorProtocol", (t + 1)[0]) == typing.cast("EagerTensorProtocol", "Add")
    assert typing.cast("EagerTensorProtocol", (1 + t)[0]) == typing.cast("EagerTensorProtocol", "Add")
    assert typing.cast("EagerTensorProtocol", (t - 1)[0]) == typing.cast("EagerTensorProtocol", "Subtract")
    assert typing.cast("EagerTensorProtocol", (1 - t)[0]) == typing.cast("EagerTensorProtocol", "Subtract")
    assert typing.cast("EagerTensorProtocol", (t * 1)[0]) == typing.cast("EagerTensorProtocol", "Multiply")
    assert typing.cast("EagerTensorProtocol", (1 * t)[0]) == typing.cast("EagerTensorProtocol", "Multiply")
    assert typing.cast("EagerTensorProtocol", (t / 1)[0]) == typing.cast("EagerTensorProtocol", "TrueDivide")
    assert typing.cast("EagerTensorProtocol", (1 / t)[0]) == typing.cast("EagerTensorProtocol", "TrueDivide")
    assert typing.cast("EagerTensorProtocol", (t**1)[0]) == typing.cast("EagerTensorProtocol", "Power")
    assert typing.cast("EagerTensorProtocol", (1**t)[0]) == typing.cast("EagerTensorProtocol", "Power")
    assert typing.cast("EagerTensorProtocol", (t // 1)[0]) == typing.cast("EagerTensorProtocol", "FloorDivide")
    assert typing.cast("EagerTensorProtocol", (1 // t)[0]) == typing.cast("EagerTensorProtocol", "FloorDivide")
    assert typing.cast("EagerTensorProtocol", (t % 1)[0]) == typing.cast("EagerTensorProtocol", "Mod")
    assert typing.cast("EagerTensorProtocol", (1 % t)[0]) == typing.cast("EagerTensorProtocol", "Mod")

    assert typing.cast("EagerTensorProtocol", (t & 1)[0]) == typing.cast("EagerTensorProtocol", "BitwiseAnd")
    assert typing.cast("EagerTensorProtocol", (1 & t)[0]) == typing.cast("EagerTensorProtocol", "BitwiseAnd")
    assert typing.cast("EagerTensorProtocol", (t | 1)[0]) == typing.cast("EagerTensorProtocol", "BitwiseOr")
    assert typing.cast("EagerTensorProtocol", (1 | t)[0]) == typing.cast("EagerTensorProtocol", "BitwiseOr")
    assert typing.cast("EagerTensorProtocol", (t ^ 1)[0]) == typing.cast("EagerTensorProtocol", "BitwiseXor")
    assert typing.cast("EagerTensorProtocol", (1 ^ t)[0]) == typing.cast("EagerTensorProtocol", "BitwiseXor")
    assert typing.cast("EagerTensorProtocol", (t << 1)[0]) == typing.cast("EagerTensorProtocol", "LeftShift")
    assert typing.cast("EagerTensorProtocol", (1 << t)[0]) == typing.cast("EagerTensorProtocol", "LeftShift")
    assert typing.cast("EagerTensorProtocol", (t >> 1)[0]) == typing.cast("EagerTensorProtocol", "RightShift")
    assert typing.cast("EagerTensorProtocol", (1 >> t)[0]) == typing.cast("EagerTensorProtocol", "RightShift")

    assert typing.cast("EagerTensorProtocol", (-t)[0]) == typing.cast("EagerTensorProtocol", "Negative")
    assert typing.cast("EagerTensorProtocol", (+t)[0]) == typing.cast("EagerTensorProtocol", "Positive")
    assert typing.cast("EagerTensorProtocol", abs(t)[0]) == typing.cast("EagerTensorProtocol", "Abs")
    assert typing.cast("EagerTensorProtocol", (~t)[0]) == typing.cast("EagerTensorProtocol", "BitwiseNot")

    assert typing.cast("EagerTensorProtocol", (t < 1)[0]) == "Less"
    assert typing.cast("EagerTensorProtocol", (t > 1)[0]) == "Greater"
    assert typing.cast("EagerTensorProtocol", (t <= 1)[0]) == "LessEqual"
    assert typing.cast("EagerTensorProtocol", (t >= 1)[0]) == "GreaterEqual"
    assert typing.cast("EagerTensorProtocol", getattr((typing.cast("EagerTensorProtocol", t) == typing.cast("EagerTensorProtocol", 1)), "__getitem__")(0)) == typing.cast("EagerTensorProtocol", "Equal")
    assert typing.cast("EagerTensorProtocol", getattr((typing.cast("EagerTensorProtocol", t) != typing.cast("EagerTensorProtocol", 1)), "__getitem__")(0)) == typing.cast("EagerTensorProtocol", "NotEqual")

    assert hash(t) == id(t)


def test_mixins_real_op() -> None:
    """Test function test_mixins_real_op."""

    class DummyTensorRealOp(TensorArithmeticMixin):
        """Test class DummyTensorRealOp."""

        pass

    with pytest.raises(Exception, match=r".*"):
        DummyTensorRealOp() + 1
