"""Test module."""

import pytest

from ml_switcheroo_compiler.core.mixins import TensorArithmeticMixin, TensorBitwiseMixin, TensorLogicalMixin


class DummyTensor(TensorArithmeticMixin, TensorBitwiseMixin, TensorLogicalMixin):
    def _get_op(self, name):
        return lambda *args, **kwargs: (name, args)


def test_mixins():
    t = DummyTensor()

    assert (t + 1) == ("Add", (t, 1))
    assert (1 + t) == ("Add", (1, t))
    assert (t - 1) == ("Subtract", (t, 1))
    assert (1 - t) == ("Subtract", (1, t))
    assert (t * 1) == ("Multiply", (t, 1))
    assert (1 * t) == ("Multiply", (1, t))
    assert (t / 1) == ("TrueDivide", (t, 1))
    assert (1 / t) == ("TrueDivide", (1, t))
    assert (t**1) == ("Power", (t, 1))
    assert (1**t) == ("Power", (1, t))
    assert (t // 1) == ("FloorDivide", (t, 1))
    assert (1 // t) == ("FloorDivide", (1, t))
    assert (t % 1) == ("Mod", (t, 1))
    assert (1 % t) == ("Mod", (1, t))

    assert (t & 1) == ("BitwiseAnd", (t, 1))
    assert (1 & t) == ("BitwiseAnd", (1, t))
    assert (t | 1) == ("BitwiseOr", (t, 1))
    assert (1 | t) == ("BitwiseOr", (1, t))
    assert (t ^ 1) == ("BitwiseXor", (t, 1))
    assert (1 ^ t) == ("BitwiseXor", (1, t))
    assert (t << 1) == ("LeftShift", (t, 1))
    assert (1 << t) == ("LeftShift", (1, t))
    assert (t >> 1) == ("RightShift", (t, 1))
    assert (1 >> t) == ("RightShift", (1, t))

    assert -t == ("Negative", (t,))
    assert +t == ("Positive", (t,))
    assert abs(t) == ("Abs", (t,))
    assert ~t == ("BitwiseNot", (t,))

    assert (t < 1) == ("Less", (t, 1))
    assert (t > 1) == ("Greater", (t, 1))
    assert (t <= 1) == ("LessEqual", (t, 1))
    assert (t >= 1) == ("GreaterEqual", (t, 1))
    assert (t == 1) == ("Equal", (t, 1))
    assert (t != 1) == ("NotEqual", (t, 1))

    assert hash(t) == id(t)


def test_mixins_real_op():
    class DummyTensorRealOp(TensorArithmeticMixin):
        pass

    with pytest.raises(Exception):
        DummyTensorRealOp() + 1
