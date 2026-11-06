"""Test module."""

from __future__ import annotations

import typing

import pytest

from ml_switcheroo_compiler.core.mixins import TensorArithmeticMixin, TensorBitwiseMixin, TensorLogicalMixin
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef
from tests.testing_utils import ensure_eager_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


class DummyTensor(TensorArithmeticMixin, TensorBitwiseMixin, TensorLogicalMixin):
    """Test class DummyTensor."""

    def _get_op(self, name: EagerValue) -> OpDef:
        """Test function _get_op.

        Args:
            name: Test parameter `name`.
        """

        class _MockOp(OpDef):
            """Mock OpDef returning Tensor wrapping (name, args)."""

            op_name = str(name)

            def __call__(self, *args: EagerValue, **kwargs: EagerValue) -> Tensor:
                """Return Tensor wrapping (name, args) tuple."""
                return Tensor((name, args), TensorConfig())

        return _MockOp()


def test_mixins() -> None:
    """Test function test_mixins."""
    t = DummyTensor()

    assert ensure_eager_tensor((t + 1).data)[0] == ensure_eager_tensor("Add")
    assert ensure_eager_tensor((1 + t).data)[0] == ensure_eager_tensor("Add")
    assert ensure_eager_tensor((t - 1).data)[0] == ensure_eager_tensor("Subtract")
    assert ensure_eager_tensor((1 - t).data)[0] == ensure_eager_tensor("Subtract")
    assert ensure_eager_tensor((t * 1).data)[0] == ensure_eager_tensor("Multiply")
    assert ensure_eager_tensor((1 * t).data)[0] == ensure_eager_tensor("Multiply")
    assert ensure_eager_tensor((t / 1).data)[0] == ensure_eager_tensor("TrueDivide")
    assert ensure_eager_tensor((1 / t).data)[0] == ensure_eager_tensor("TrueDivide")
    assert ensure_eager_tensor((t**1).data)[0] == ensure_eager_tensor("Power")
    assert ensure_eager_tensor((1**t).data)[0] == ensure_eager_tensor("Power")
    assert ensure_eager_tensor((t // 1).data)[0] == ensure_eager_tensor("FloorDivide")
    assert ensure_eager_tensor((1 // t).data)[0] == ensure_eager_tensor("FloorDivide")
    assert ensure_eager_tensor((t % 1).data)[0] == ensure_eager_tensor("Mod")
    assert ensure_eager_tensor((1 % t).data)[0] == ensure_eager_tensor("Mod")

    assert ensure_eager_tensor((t & 1).data)[0] == ensure_eager_tensor("BitwiseAnd")
    assert ensure_eager_tensor((1 & t).data)[0] == ensure_eager_tensor("BitwiseAnd")
    assert ensure_eager_tensor((t | 1).data)[0] == ensure_eager_tensor("BitwiseOr")
    assert ensure_eager_tensor((1 | t).data)[0] == ensure_eager_tensor("BitwiseOr")
    assert ensure_eager_tensor((t ^ 1).data)[0] == ensure_eager_tensor("BitwiseXor")
    assert ensure_eager_tensor((1 ^ t).data)[0] == ensure_eager_tensor("BitwiseXor")
    assert ensure_eager_tensor((t << 1).data)[0] == ensure_eager_tensor("LeftShift")
    assert ensure_eager_tensor((1 << t).data)[0] == ensure_eager_tensor("LeftShift")
    assert ensure_eager_tensor((t >> 1).data)[0] == ensure_eager_tensor("RightShift")
    assert ensure_eager_tensor((1 >> t).data)[0] == ensure_eager_tensor("RightShift")

    assert ensure_eager_tensor((-t).data)[0] == ensure_eager_tensor("Negative")
    assert ensure_eager_tensor((+t).data)[0] == ensure_eager_tensor("Positive")
    assert ensure_eager_tensor(abs(t).data)[0] == ensure_eager_tensor("Abs")
    assert ensure_eager_tensor((~t).data)[0] == ensure_eager_tensor("BitwiseNot")

    assert ensure_eager_tensor((t < 1).data)[0] == "Less"
    assert ensure_eager_tensor((t > 1).data)[0] == "Greater"
    assert ensure_eager_tensor((t <= 1).data)[0] == "LessEqual"
    assert ensure_eager_tensor((t >= 1).data)[0] == "GreaterEqual"
    assert ensure_eager_tensor(ensure_eager_tensor(ensure_eager_tensor(t) == ensure_eager_tensor(1)).data)[0] == ensure_eager_tensor("Equal")
    assert ensure_eager_tensor(ensure_eager_tensor(ensure_eager_tensor(t) != ensure_eager_tensor(1)).data)[0] == ensure_eager_tensor("NotEqual")

    assert hash(t) == id(t)


def test_mixins_real_op() -> None:
    """Test function test_mixins_real_op."""

    class DummyTensorRealOp(TensorArithmeticMixin):
        """Test class DummyTensorRealOp."""

        pass

    with pytest.raises(Exception, match=r".*"):
        DummyTensorRealOp() + 1
