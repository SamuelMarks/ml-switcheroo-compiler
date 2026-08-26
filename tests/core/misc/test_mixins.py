"""Test module."""

import pytest

from ml_switcheroo_compiler.core.mixins import TensorArithmeticMixin, TensorBitwiseMixin, TensorLogicalMixin


class DummyTensor(TensorArithmeticMixin, TensorBitwiseMixin, TensorLogicalMixin):
    def _get_op(self, name):
        return lambda *args, **kwargs: (name, args)


def test_mixins() -> None:
    t = DummyTensor()

    assert (t + 1)[0] == "Add"
    assert (1 + t)[0] == "Add"
    assert (t - 1)[0] == "Subtract"
    assert (1 - t)[0] == "Subtract"
    assert (t * 1)[0] == "Multiply"
    assert (1 * t)[0] == "Multiply"
    assert (t / 1)[0] == "TrueDivide"
    assert (1 / t)[0] == "TrueDivide"
    assert (t**1)[0] == "Power"
    assert (1**t)[0] == "Power"
    assert (t // 1)[0] == "FloorDivide"
    assert (1 // t)[0] == "FloorDivide"
    assert (t % 1)[0] == "Mod"
    assert (1 % t)[0] == "Mod"

    assert (t & 1)[0] == "BitwiseAnd"
    assert (1 & t)[0] == "BitwiseAnd"
    assert (t | 1)[0] == "BitwiseOr"
    assert (1 | t)[0] == "BitwiseOr"
    assert (t ^ 1)[0] == "BitwiseXor"
    assert (1 ^ t)[0] == "BitwiseXor"
    assert (t << 1)[0] == "LeftShift"
    assert (1 << t)[0] == "LeftShift"
    assert (t >> 1)[0] == "RightShift"
    assert (1 >> t)[0] == "RightShift"

    assert (-t)[0] == "Negative"
    assert (+t)[0] == "Positive"
    assert abs(t)[0] == "Abs"
    assert (~t)[0] == "BitwiseNot"

    assert (t < 1)[0] == "Less"
    assert (t > 1)[0] == "Greater"
    assert (t <= 1)[0] == "LessEqual"
    assert (t >= 1)[0] == "GreaterEqual"
    assert (t == 1)[0] == "Equal"
    assert (t != 1)[0] == "NotEqual"

    assert hash(t) == id(t)


def test_mixins_real_op():
    class DummyTensorRealOp(TensorArithmeticMixin):
        pass

    with pytest.raises(Exception):
        DummyTensorRealOp() + 1
