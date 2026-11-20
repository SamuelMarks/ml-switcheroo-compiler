"""Core Mixins."""

from __future__ import annotations

import typing
from collections.abc import Callable

from ml_switcheroo_compiler.core.tensor_comparison import TensorComparisonMixin

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.core.shape import OpValue
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.core.tensor_mixins import TensorIndexingMixin
    from ml_switcheroo_compiler.ops.base import OpDef
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

    _MixinSelf: typing.TypeAlias = (
        "TensorArithmeticMixin | TensorBitwiseMixin | TensorLogicalMixin | TensorIndexingMixin | OpValue"
    )


def _as_tensor_self(val: _MixinSelf) -> Tensor | ProxyTensor:
    """Narrow ``self`` on a ``Tensor`` or ``ProxyTensor`` mixin via runtime ``isinstance`` check.

    Args:
        val (_MixinSelf): Mixin ``self`` instance.

    Returns:
        Tensor | ProxyTensor: Verified ``Tensor`` or ``ProxyTensor`` instance.
    """
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

    assert isinstance(val, (Tensor, ProxyTensor))
    return val


class TensorArithmeticMixin:
    """Arithmetic mixin."""

    def _get_op(self, name: str) -> OpDef:
        """Get the requested operation from the registry.

        Args:
            name (str): The name of the operation to retrieve.

        Returns:
            OpDef: The operation instance.
        """
        from ml_switcheroo_compiler.ops.registry import get_op

        return get_op(name)()

    def __add__(self, other: OpValue) -> Tensor:
        """Compute elementwise addition `self + other`.

        Args:
            other (OpValue): Right-hand operand.

        Returns:
            Tensor: A tensor containing the sum.
        """
        return self._get_op("Add")(_as_tensor_self(self), other)

    def __radd__(self, other: OpValue) -> Tensor:
        """Compute reverse elementwise addition `other + self`.

        Args:
            other (OpValue): Left-hand operand.

        Returns:
            Tensor: A tensor containing the sum.
        """
        return self._get_op("Add")(other, _as_tensor_self(self))

    def __sub__(self, other: OpValue) -> Tensor:
        """Compute elementwise subtraction `self - other`.

        Args:
            other (OpValue): Right-hand operand.

        Returns:
            Tensor: A tensor containing the difference.
        """
        return self._get_op("Subtract")(_as_tensor_self(self), other)

    def __rsub__(self, other: OpValue) -> Tensor:
        """Compute reverse elementwise subtraction `other - self`.

        Args:
            other (OpValue): Left-hand operand.

        Returns:
            Tensor: A tensor containing the difference.
        """
        return self._get_op("Subtract")(other, _as_tensor_self(self))

    def __mul__(self, other: OpValue) -> Tensor:
        """Compute elementwise multiplication `self * other`.

        Args:
            other (OpValue): Right-hand operand.

        Returns:
            Tensor: A tensor containing the product.
        """
        return self._get_op("Multiply")(_as_tensor_self(self), other)

    def __rmul__(self, other: OpValue) -> Tensor:
        """Compute reverse elementwise multiplication `other * self`.

        Args:
            other (OpValue): Left-hand operand.

        Returns:
            Tensor: A tensor containing the product.
        """
        return self._get_op("Multiply")(other, _as_tensor_self(self))

    def __truediv__(self, other: OpValue) -> Tensor:
        """Compute elementwise true division `self / other`.

        Args:
            other (OpValue): Right-hand divisor operand.

        Returns:
            Tensor: A tensor containing the quotient.
        """
        return self._get_op("TrueDivide")(_as_tensor_self(self), other)

    def __rtruediv__(self, other: OpValue) -> Tensor:
        """Compute reverse elementwise true division `other / self`.

        Args:
            other (OpValue): Left-hand dividend operand.

        Returns:
            Tensor: A tensor containing the quotient.
        """
        return self._get_op("TrueDivide")(other, _as_tensor_self(self))

    def __pow__(self, other: OpValue) -> Tensor:
        """Compute elementwise exponentiation `self ** other`.

        Args:
            other (OpValue): Exponent operand.

        Returns:
            Tensor: A tensor containing the power result.
        """
        return self._get_op("Power")(_as_tensor_self(self), other)

    def __rpow__(self, other: OpValue) -> Tensor:
        """Compute reverse elementwise exponentiation `other ** self`.

        Args:
            other (OpValue): Base operand.

        Returns:
            Tensor: A tensor containing the power result.
        """
        return self._get_op("Power")(other, _as_tensor_self(self))

    def __floordiv__(self, other: OpValue) -> Tensor:
        """Compute elementwise floor division `self // other`.

        Args:
            other (OpValue): Divisor operand.

        Returns:
            Tensor: A tensor containing the floor quotient.
        """
        return self._get_op("FloorDivide")(_as_tensor_self(self), other)

    def __rfloordiv__(self, other: OpValue) -> Tensor:
        """Compute reverse elementwise floor division `other // self`.

        Args:
            other (OpValue): Dividend operand.

        Returns:
            Tensor: A tensor containing the floor quotient.
        """
        return self._get_op("FloorDivide")(other, _as_tensor_self(self))

    def __mod__(self, other: OpValue) -> Tensor:
        """Compute elementwise modulo `self % other`.

        Args:
            other (OpValue): Divisor operand.

        Returns:
            Tensor: A tensor containing the remainder.
        """
        return self._get_op("Mod")(_as_tensor_self(self), other)

    def __rmod__(self, other: OpValue) -> Tensor:
        """Compute reverse elementwise modulo `other % self`.

        Args:
            other (OpValue): Dividend operand.

        Returns:
            Tensor: A tensor containing the remainder.
        """
        return self._get_op("Mod")(other, _as_tensor_self(self))


class TensorBitwiseMixin:
    """Apply bitwise mixin."""

    _get_op: Callable[[str], OpDef]

    def __and__(self, other: OpValue) -> Tensor:
        """Compute elementwise bitwise AND `self & other`.

        Args:
            other (OpValue): Right-hand operand.

        Returns:
            Tensor: A tensor containing the bitwise AND result.
        """
        return self._get_op("BitwiseAnd")(_as_tensor_self(self), other)

    def __rand__(self, other: OpValue) -> Tensor:
        """Compute reverse elementwise bitwise AND `other & self`.

        Args:
            other (OpValue): Left-hand operand.

        Returns:
            Tensor: A tensor containing the bitwise AND result.
        """
        return self._get_op("BitwiseAnd")(other, _as_tensor_self(self))

    def __or__(self, other: OpValue) -> Tensor:
        """Compute elementwise bitwise OR `self | other`.

        Args:
            other (OpValue): Right-hand operand.

        Returns:
            Tensor: A tensor containing the bitwise OR result.
        """
        return self._get_op("BitwiseOr")(_as_tensor_self(self), other)

    def __ror__(self, other: OpValue) -> Tensor:
        """Compute reverse elementwise bitwise OR `other | self`.

        Args:
            other (OpValue): Left-hand operand.

        Returns:
            Tensor: A tensor containing the bitwise OR result.
        """
        return self._get_op("BitwiseOr")(other, _as_tensor_self(self))

    def __xor__(self, other: OpValue) -> Tensor:
        """Compute elementwise bitwise XOR `self ^ other`.

        Args:
            other (OpValue): Right-hand operand.

        Returns:
            Tensor: A tensor containing the bitwise XOR result.
        """
        return self._get_op("BitwiseXor")(_as_tensor_self(self), other)

    def __rxor__(self, other: OpValue) -> Tensor:
        """Compute reverse elementwise bitwise XOR `other ^ self`.

        Args:
            other (OpValue): Left-hand operand.

        Returns:
            Tensor: A tensor containing the bitwise XOR result.
        """
        return self._get_op("BitwiseXor")(other, _as_tensor_self(self))

    def __lshift__(self, other: OpValue) -> Tensor:
        """Compute elementwise left bit shift `self << other`.

        Args:
            other (OpValue): Shift amount operand.

        Returns:
            Tensor: A tensor containing the left-shifted result.
        """
        return self._get_op("LeftShift")(_as_tensor_self(self), other)

    def __rlshift__(self, other: OpValue) -> Tensor:
        """Compute reverse elementwise left bit shift `other << self`.

        Args:
            other (OpValue): Value to shift.

        Returns:
            Tensor: A tensor containing the left-shifted result.
        """
        return self._get_op("LeftShift")(other, _as_tensor_self(self))

    def __rshift__(self, other: OpValue) -> Tensor:
        """Compute elementwise right bit shift `self >> other`.

        Args:
            other (OpValue): Shift amount operand.

        Returns:
            Tensor: A tensor containing the right-shifted result.
        """
        return self._get_op("RightShift")(_as_tensor_self(self), other)

    def __rrshift__(self, other: OpValue) -> Tensor:
        """Compute reverse elementwise right bit shift `other >> self`.

        Args:
            other (OpValue): Value to shift.

        Returns:
            Tensor: A tensor containing the right-shifted result.
        """
        return self._get_op("RightShift")(other, _as_tensor_self(self))

    def __neg__(self) -> Tensor:
        """Compute elementwise negation `-self`.

        Returns:
            Tensor: A tensor containing the negated values.
        """
        return self._get_op("Negative")(_as_tensor_self(self))

    def __pos__(self) -> Tensor:
        """Compute elementwise unary positive `+self`.

        Returns:
            Tensor: A tensor containing the positive values.
        """
        return self._get_op("Positive")(_as_tensor_self(self))

    def __abs__(self) -> Tensor:
        """Compute elementwise absolute value `abs(self)`.

        Returns:
            Tensor: A tensor containing the absolute values.
        """
        return self._get_op("Abs")(_as_tensor_self(self))

    def __invert__(self) -> Tensor:
        """Compute elementwise bitwise inversion `~self`.

        Returns:
            Tensor: A tensor containing the inverted values.
        """
        return self._get_op("BitwiseNot")(_as_tensor_self(self))


class TensorLogicalMixin(TensorComparisonMixin):
    """Apply logical mixin."""

    _get_op: Callable[[str], OpDef]

    def __lt__(self, other: OpValue) -> Tensor:
        """Compare elementwise strictly less than `self < other`.

        Args:
            other (OpValue): Right-hand operand.

        Returns:
            Tensor: A boolean tensor of comparison results.
        """
        return self._get_op("Less")(_as_tensor_self(self), other)

    def __gt__(self, other: OpValue) -> Tensor:
        """Compare elementwise strictly greater than `self > other`.

        Args:
            other (OpValue): Right-hand operand.

        Returns:
            Tensor: A boolean tensor of comparison results.
        """
        return self._get_op("Greater")(_as_tensor_self(self), other)

    def __le__(self, other: OpValue) -> Tensor:
        """Compare elementwise less than or equal `self <= other`.

        Args:
            other (OpValue): Right-hand operand.

        Returns:
            Tensor: A boolean tensor of comparison results.
        """
        return self._get_op("LessEqual")(_as_tensor_self(self), other)

    def __ge__(self, other: OpValue) -> Tensor:
        """Compare elementwise greater than or equal `self >= other`.

        Args:
            other (OpValue): Right-hand operand.

        Returns:
            Tensor: A boolean tensor of comparison results.
        """
        return self._get_op("GreaterEqual")(_as_tensor_self(self), other)

    def _eq(self, other: OpValue) -> Tensor:
        """Compare elementwise equality `self == other`.

        Args:
            other (OpValue): Right-hand operand.

        Returns:
            Tensor: A boolean tensor of equality results.
        """
        return self._get_op("Equal")(_as_tensor_self(self), other)

    def _ne(self, other: OpValue) -> Tensor:
        """Compare elementwise inequality `self != other`.

        Args:
            other (OpValue): Right-hand operand.

        Returns:
            Tensor: A boolean tensor of inequality results.
        """
        return self._get_op("NotEqual")(_as_tensor_self(self), other)
