"""Core Mixins."""

from __future__ import annotations

import typing
from collections.abc import Callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.core.shape import TensorOrScalar
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.core.tensor_mixins import TensorIndexingMixin
    from ml_switcheroo_compiler.ops.base import OpDef

    _MixinSelf: typing.TypeAlias = "TensorArithmeticMixin | TensorBitwiseMixin | TensorLogicalMixin | TensorIndexingMixin | TensorOrScalar"


def _is_tensor(val: _MixinSelf) -> typing.TypeGuard[Tensor]:
    """Type guard narrowing `self` on Tensor mixins to `Tensor`.

    Args:
        val (_MixinSelf): Mixin `self` instance.

    Returns:
        bool: True if instance is non-None.
    """
    return val is not None


def _as_tensor_self(val: _MixinSelf) -> Tensor:
    """Narrow `self` on a Tensor mixin to `Tensor` without `typing.cast`.

    Args:
        val (_MixinSelf): Mixin `self` instance.

    Returns:
        Tensor: Narrowed Tensor instance.
    """
    assert _is_tensor(val)
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

    def __add__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise addition `self + other`.

        Args:
            other (TensorOrScalar): Right-hand operand.

        Returns:
            Tensor: A tensor containing the sum.
        """
        return self._get_op("Add")(_as_tensor_self(self), other)

    def __radd__(self, other: TensorOrScalar) -> Tensor:
        """Compute reverse elementwise addition `other + self`.

        Args:
            other (TensorOrScalar): Left-hand operand.

        Returns:
            Tensor: A tensor containing the sum.
        """
        return self._get_op("Add")(other, _as_tensor_self(self))

    def __sub__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise subtraction `self - other`.

        Args:
            other (TensorOrScalar): Right-hand operand.

        Returns:
            Tensor: A tensor containing the difference.
        """
        return self._get_op("Subtract")(_as_tensor_self(self), other)

    def __rsub__(self, other: TensorOrScalar) -> Tensor:
        """Compute reverse elementwise subtraction `other - self`.

        Args:
            other (TensorOrScalar): Left-hand operand.

        Returns:
            Tensor: A tensor containing the difference.
        """
        return self._get_op("Subtract")(other, _as_tensor_self(self))

    def __mul__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise multiplication `self * other`.

        Args:
            other (TensorOrScalar): Right-hand operand.

        Returns:
            Tensor: A tensor containing the product.
        """
        return self._get_op("Multiply")(_as_tensor_self(self), other)

    def __rmul__(self, other: TensorOrScalar) -> Tensor:
        """Compute reverse elementwise multiplication `other * self`.

        Args:
            other (TensorOrScalar): Left-hand operand.

        Returns:
            Tensor: A tensor containing the product.
        """
        return self._get_op("Multiply")(other, _as_tensor_self(self))

    def __truediv__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise true division `self / other`.

        Args:
            other (TensorOrScalar): Right-hand divisor operand.

        Returns:
            Tensor: A tensor containing the quotient.
        """
        return self._get_op("TrueDivide")(_as_tensor_self(self), other)

    def __rtruediv__(self, other: TensorOrScalar) -> Tensor:
        """Compute reverse elementwise true division `other / self`.

        Args:
            other (TensorOrScalar): Left-hand dividend operand.

        Returns:
            Tensor: A tensor containing the quotient.
        """
        return self._get_op("TrueDivide")(other, _as_tensor_self(self))

    def __pow__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise exponentiation `self ** other`.

        Args:
            other (TensorOrScalar): Exponent operand.

        Returns:
            Tensor: A tensor containing the power result.
        """
        return self._get_op("Power")(_as_tensor_self(self), other)

    def __rpow__(self, other: TensorOrScalar) -> Tensor:
        """Compute reverse elementwise exponentiation `other ** self`.

        Args:
            other (TensorOrScalar): Base operand.

        Returns:
            Tensor: A tensor containing the power result.
        """
        return self._get_op("Power")(other, _as_tensor_self(self))

    def __floordiv__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise floor division `self // other`.

        Args:
            other (TensorOrScalar): Divisor operand.

        Returns:
            Tensor: A tensor containing the floor quotient.
        """
        return self._get_op("FloorDivide")(_as_tensor_self(self), other)

    def __rfloordiv__(self, other: TensorOrScalar) -> Tensor:
        """Compute reverse elementwise floor division `other // self`.

        Args:
            other (TensorOrScalar): Dividend operand.

        Returns:
            Tensor: A tensor containing the floor quotient.
        """
        return self._get_op("FloorDivide")(other, _as_tensor_self(self))

    def __mod__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise modulo `self % other`.

        Args:
            other (TensorOrScalar): Divisor operand.

        Returns:
            Tensor: A tensor containing the remainder.
        """
        return self._get_op("Mod")(_as_tensor_self(self), other)

    def __rmod__(self, other: TensorOrScalar) -> Tensor:
        """Compute reverse elementwise modulo `other % self`.

        Args:
            other (TensorOrScalar): Dividend operand.

        Returns:
            Tensor: A tensor containing the remainder.
        """
        return self._get_op("Mod")(other, _as_tensor_self(self))


class TensorBitwiseMixin:
    """Apply bitwise mixin."""

    _get_op: Callable[[str], OpDef]

    def __and__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise bitwise AND `self & other`.

        Args:
            other (TensorOrScalar): Right-hand operand.

        Returns:
            Tensor: A tensor containing the bitwise AND result.
        """
        return self._get_op("BitwiseAnd")(_as_tensor_self(self), other)

    def __rand__(self, other: TensorOrScalar) -> Tensor:
        """Compute reverse elementwise bitwise AND `other & self`.

        Args:
            other (TensorOrScalar): Left-hand operand.

        Returns:
            Tensor: A tensor containing the bitwise AND result.
        """
        return self._get_op("BitwiseAnd")(other, _as_tensor_self(self))

    def __or__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise bitwise OR `self | other`.

        Args:
            other (TensorOrScalar): Right-hand operand.

        Returns:
            Tensor: A tensor containing the bitwise OR result.
        """
        return self._get_op("BitwiseOr")(_as_tensor_self(self), other)

    def __ror__(self, other: TensorOrScalar) -> Tensor:
        """Compute reverse elementwise bitwise OR `other | self`.

        Args:
            other (TensorOrScalar): Left-hand operand.

        Returns:
            Tensor: A tensor containing the bitwise OR result.
        """
        return self._get_op("BitwiseOr")(other, _as_tensor_self(self))

    def __xor__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise bitwise XOR `self ^ other`.

        Args:
            other (TensorOrScalar): Right-hand operand.

        Returns:
            Tensor: A tensor containing the bitwise XOR result.
        """
        return self._get_op("BitwiseXor")(_as_tensor_self(self), other)

    def __rxor__(self, other: TensorOrScalar) -> Tensor:
        """Compute reverse elementwise bitwise XOR `other ^ self`.

        Args:
            other (TensorOrScalar): Left-hand operand.

        Returns:
            Tensor: A tensor containing the bitwise XOR result.
        """
        return self._get_op("BitwiseXor")(other, _as_tensor_self(self))

    def __lshift__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise left bit shift `self << other`.

        Args:
            other (TensorOrScalar): Shift amount operand.

        Returns:
            Tensor: A tensor containing the left-shifted result.
        """
        return self._get_op("LeftShift")(_as_tensor_self(self), other)

    def __rlshift__(self, other: TensorOrScalar) -> Tensor:
        """Compute reverse elementwise left bit shift `other << self`.

        Args:
            other (TensorOrScalar): Value to shift.

        Returns:
            Tensor: A tensor containing the left-shifted result.
        """
        return self._get_op("LeftShift")(other, _as_tensor_self(self))

    def __rshift__(self, other: TensorOrScalar) -> Tensor:
        """Compute elementwise right bit shift `self >> other`.

        Args:
            other (TensorOrScalar): Shift amount operand.

        Returns:
            Tensor: A tensor containing the right-shifted result.
        """
        return self._get_op("RightShift")(_as_tensor_self(self), other)

    def __rrshift__(self, other: TensorOrScalar) -> Tensor:
        """Compute reverse elementwise right bit shift `other >> self`.

        Args:
            other (TensorOrScalar): Value to shift.

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


def _is_bool_like(val: Tensor) -> typing.TypeGuard[bool]:
    """Type guard narrowing tensor comparison result to `bool` for `object.__eq__` signature compatibility.

    Args:
        val (Tensor): Tensor comparison result.

    Returns:
        bool: True if instance is non-None.
    """
    return val is not None


class TensorLogicalMixin:
    """Apply logical mixin."""

    _get_op: Callable[[str], OpDef]

    def __lt__(self, other: TensorOrScalar) -> Tensor:
        """Compare elementwise strictly less than `self < other`.

        Args:
            other (TensorOrScalar): Right-hand operand.

        Returns:
            Tensor: A boolean tensor of comparison results.
        """
        return self._get_op("Less")(_as_tensor_self(self), other)

    def __gt__(self, other: TensorOrScalar) -> Tensor:
        """Compare elementwise strictly greater than `self > other`.

        Args:
            other (TensorOrScalar): Right-hand operand.

        Returns:
            Tensor: A boolean tensor of comparison results.
        """
        return self._get_op("Greater")(_as_tensor_self(self), other)

    def __le__(self, other: TensorOrScalar) -> Tensor:
        """Compare elementwise less than or equal `self <= other`.

        Args:
            other (TensorOrScalar): Right-hand operand.

        Returns:
            Tensor: A boolean tensor of comparison results.
        """
        return self._get_op("LessEqual")(_as_tensor_self(self), other)

    def __ge__(self, other: TensorOrScalar) -> Tensor:
        """Compare elementwise greater than or equal `self >= other`.

        Args:
            other (TensorOrScalar): Right-hand operand.

        Returns:
            Tensor: A boolean tensor of comparison results.
        """
        return self._get_op("GreaterEqual")(_as_tensor_self(self), other)

    def __hash__(self) -> int:
        """Compute identity-based hash of the tensor instance.

        Returns:
            int: Identity hash value.
        """
        return id(self)

    def _eq(self, other: TensorOrScalar) -> Tensor:
        """Compare elementwise equality `self == other`.

        Args:
            other (TensorOrScalar): Right-hand operand.

        Returns:
            Tensor: A boolean tensor of equality results.
        """
        return self._get_op("Equal")(_as_tensor_self(self), other)

    def _ne(self, other: TensorOrScalar) -> Tensor:
        """Compare elementwise inequality `self != other`.

        Args:
            other (TensorOrScalar): Right-hand operand.

        Returns:
            Tensor: A boolean tensor of inequality results.
        """
        return self._get_op("NotEqual")(_as_tensor_self(self), other)

    def __eq__(self, other: object) -> bool:
        """Compare elementwise equality `self == other`.

        Args:
            other (TensorOrScalar): Right-hand operand.

        Returns:
            bool: A boolean tensor of equality results.
        """
        res = self._eq(typing.cast("TensorOrScalar", other))
        assert _is_bool_like(res)
        return res

    def __ne__(self, other: object) -> bool:
        """Compare elementwise inequality `self != other`.

        Args:
            other (TensorOrScalar): Right-hand operand.

        Returns:
            bool: A boolean tensor of inequality results.
        """
        res = self._ne(typing.cast("TensorOrScalar", other))
        assert _is_bool_like(res)
        return res
