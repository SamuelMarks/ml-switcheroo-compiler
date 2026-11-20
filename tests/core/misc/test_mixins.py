"""Test module."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.core.mixins import TensorArithmeticMixin, TensorBitwiseMixin, TensorLogicalMixin
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


class DummyTensor(Tensor):
    """Test ``Tensor`` subclass intercepting ``_get_op`` calls."""

    def __init__(self) -> None:
        """Initialize ``DummyTensor`` with empty default ``TensorConfig``."""
        super().__init__(None, TensorConfig())

    def _get_op(self, name: BackendArg) -> OpDef:
        """Test function _get_op.

        Args:
            name: Test parameter `name`.

        Returns:
            OpDef: Computed output value from `_get_op`.
        """

        class _MockOp(OpDef):
            """Mock OpDef returning Tensor wrapping (name, args)."""

            op_name = str(name)

            def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> Tensor:
                """Return Tensor wrapping (name, args) tuple.

                Args:
                    *args (BackendArg): Positional arguments forwarded to `__call__`.
                    **kwargs (BackendArg): Keyword arguments forwarded to `__call__`.

                Returns:
                    Tensor: Computed output value from `__call__`.
                """
                return Tensor((name, args), TensorConfig())

        return _MockOp()


def test_mixins() -> None:
    """Test function test_mixins."""
    from ml_switcheroo_compiler.backends.eager_registry import coerce_operand_list

    t = DummyTensor()

    assert coerce_operand_list((t + 1).data)[0] == "Add"
    assert coerce_operand_list((1 + t).data)[0] == "Add"
    assert coerce_operand_list((t - 1).data)[0] == "Subtract"
    assert coerce_operand_list((1 - t).data)[0] == "Subtract"
    assert coerce_operand_list((t * 1).data)[0] == "Multiply"
    assert coerce_operand_list((1 * t).data)[0] == "Multiply"
    assert coerce_operand_list((t / 1).data)[0] == "TrueDivide"
    assert coerce_operand_list((1 / t).data)[0] == "TrueDivide"
    assert coerce_operand_list((t**1).data)[0] == "Power"
    assert coerce_operand_list((1**t).data)[0] == "Power"
    assert coerce_operand_list((t // 1).data)[0] == "FloorDivide"
    assert coerce_operand_list((1 // t).data)[0] == "FloorDivide"
    assert coerce_operand_list((t % 1).data)[0] == "Mod"
    assert coerce_operand_list((1 % t).data)[0] == "Mod"

    assert coerce_operand_list((t & 1).data)[0] == "BitwiseAnd"
    assert coerce_operand_list((1 & t).data)[0] == "BitwiseAnd"
    assert coerce_operand_list((t | 1).data)[0] == "BitwiseOr"
    assert coerce_operand_list((1 | t).data)[0] == "BitwiseOr"
    assert coerce_operand_list((t ^ 1).data)[0] == "BitwiseXor"
    assert coerce_operand_list((1 ^ t).data)[0] == "BitwiseXor"
    assert coerce_operand_list((t << 1).data)[0] == "LeftShift"
    assert coerce_operand_list((1 << t).data)[0] == "LeftShift"
    assert coerce_operand_list((t >> 1).data)[0] == "RightShift"
    assert coerce_operand_list((1 >> t).data)[0] == "RightShift"

    assert coerce_operand_list((-t).data)[0] == "Negative"
    assert coerce_operand_list((+t).data)[0] == "Positive"
    assert coerce_operand_list(abs(t).data)[0] == "Abs"
    assert coerce_operand_list((~t).data)[0] == "BitwiseNot"

    assert coerce_operand_list((t < 1).data)[0] == "Less"
    assert coerce_operand_list((t > 1).data)[0] == "Greater"
    assert coerce_operand_list((t <= 1).data)[0] == "LessEqual"
    assert coerce_operand_list((t >= 1).data)[0] == "GreaterEqual"
    assert coerce_operand_list((t == 1).data)[0] == "Equal"
    assert coerce_operand_list((t != 1).data)[0] == "NotEqual"

    assert hash(t) == id(t)


def test_mixins_real_op() -> None:
    """Verify ``TensorArithmeticMixin._get_op`` dispatches real registered operations."""
    import numpy as np

    from ml_switcheroo_compiler.core.config import ConfigContext

    _ = (TensorArithmeticMixin, TensorBitwiseMixin, TensorLogicalMixin)
    with ConfigContext(eager_mode=True):
        t = Tensor(np.array([2, 4], dtype=np.int32), TensorConfig((2,), "int32", "cpu"))
        out = t + 3
        assert np.asarray(out).tolist() == [5, 7]
