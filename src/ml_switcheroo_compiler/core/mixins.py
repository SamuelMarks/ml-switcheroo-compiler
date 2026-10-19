"""Core Mixins."""

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ml_switcheroo_compiler.core.shape import TensorOrScalar
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.ops.base import OpDef


class TensorArithmeticMixin:
    """Arithmetic mixin."""

    def _get_op(self, name: str) -> "OpDef":
        """Get the requested operation from the registry.

        Args:
            name (str): The name of the operation to retrieve.

        Returns:
            OpDef: The operation instance.
        """
        from ml_switcheroo_compiler.ops.registry import get_op

        return get_op(name)()

    def __add__(self, other: "TensorOrScalar") -> "Tensor":
        """Compute elementwise addition `self + other`.

        Args:
            other (TensorOrScalar): Right-hand operand.

        Returns:
            Tensor: A tensor containing the sum.
        """
        return self._get_op("Add")(cast("Tensor", self), other)

    def __radd__(self, other: "TensorOrScalar") -> "Tensor":
        """Compute reverse elementwise addition `other + self`.

        Args:
            other (TensorOrScalar): Left-hand operand.

        Returns:
            Tensor: A tensor containing the sum.
        """
        return self._get_op("Add")(other, cast("Tensor", self))

    def __sub__(self, other: "TensorOrScalar") -> "Tensor":
        """Compute elementwise subtraction `self - other`.

        Args:
            other (TensorOrScalar): Right-hand operand.

        Returns:
            Tensor: A tensor containing the difference.
        """
        return self._get_op("Subtract")(cast("Tensor", self), other)

    def __rsub__(self, other: "TensorOrScalar") -> "Tensor":
        """Compute reverse elementwise subtraction `other - self`.

        Args:
            other (TensorOrScalar): Left-hand operand.

        Returns:
            Tensor: A tensor containing the difference.
        """
        return self._get_op("Subtract")(other, cast("Tensor", self))

    def __mul__(self, other: "TensorOrScalar") -> "Tensor":
        """Compute elementwise multiplication `self * other`.

        Args:
            other (TensorOrScalar): Right-hand operand.

        Returns:
            Tensor: A tensor containing the product.
        """
        return self._get_op("Multiply")(cast("Tensor", self), other)

    def __rmul__(self, other: "TensorOrScalar") -> "Tensor":
        """Compute reverse elementwise multiplication `other * self`.

        Args:
            other (TensorOrScalar): Left-hand operand.

        Returns:
            Tensor: A tensor containing the product.
        """
        return self._get_op("Multiply")(other, cast("Tensor", self))

    def __truediv__(self, other: "TensorOrScalar") -> "Tensor":
        """Compute elementwise true division `self / other`.

        Args:
            other (TensorOrScalar): Right-hand divisor operand.

        Returns:
            Tensor: A tensor containing the quotient.
        """
        return self._get_op("TrueDivide")(cast("Tensor", self), other)

    def __rtruediv__(self, other: "TensorOrScalar") -> "Tensor":
        """Compute reverse elementwise true division `other / self`.

        Args:
            other (TensorOrScalar): Left-hand dividend operand.

        Returns:
            Tensor: A tensor containing the quotient.
        """
        return self._get_op("TrueDivide")(other, cast("Tensor", self))

    def __pow__(self, other: "TensorOrScalar") -> "Tensor":
        """Compute elementwise exponentiation `self ** other`.

        Args:
            other (TensorOrScalar): Exponent operand.

        Returns:
            Tensor: A tensor containing the power result.
        """
        return self._get_op("Power")(cast("Tensor", self), other)

    def __rpow__(self, other: "TensorOrScalar") -> "Tensor":
        """Compute reverse elementwise exponentiation `other ** self`.

        Args:
            other (TensorOrScalar): Base operand.

        Returns:
            Tensor: A tensor containing the power result.
        """
        return self._get_op("Power")(other, cast("Tensor", self))

    def __floordiv__(self, other: "TensorOrScalar") -> "Tensor":
        """Compute elementwise floor division `self // other`.

        Args:
            other (TensorOrScalar): Divisor operand.

        Returns:
            Tensor: A tensor containing the floor quotient.
        """
        return self._get_op("FloorDivide")(cast("Tensor", self), other)

    def __rfloordiv__(self, other: "TensorOrScalar") -> "Tensor":
        """Compute reverse elementwise floor division `other // self`.

        Args:
            other (TensorOrScalar): Dividend operand.

        Returns:
            Tensor: A tensor containing the floor quotient.
        """
        return self._get_op("FloorDivide")(other, cast("Tensor", self))

    def __mod__(self, other: "TensorOrScalar") -> "Tensor":
        """Compute elementwise modulo `self % other`.

        Args:
            other (TensorOrScalar): Divisor operand.

        Returns:
            Tensor: A tensor containing the remainder.
        """
        return self._get_op("Mod")(cast("Tensor", self), other)

    def __rmod__(self, other: "TensorOrScalar") -> "Tensor":
        """Compute reverse elementwise modulo `other % self`.

        Args:
            other (TensorOrScalar): Dividend operand.

        Returns:
            Tensor: A tensor containing the remainder.
        """
        return self._get_op("Mod")(other, cast("Tensor", self))


class TensorBitwiseMixin:
    """Apply bitwise mixin."""

    _get_op: Callable[[str], "OpDef"]

    def __and__(self, other: "TensorOrScalar") -> "Tensor":
        """Compute elementwise bitwise AND `self & other`.

        Args:
            other (TensorOrScalar): Right-hand operand.

        Returns:
            Tensor: A tensor containing the bitwise AND result.
        """
        return self._get_op("BitwiseAnd")(cast("Tensor", self), other)

    def __rand__(self, other: "TensorOrScalar") -> "Tensor":
        """Compute reverse elementwise bitwise AND `other & self`.

        Args:
            other (TensorOrScalar): Left-hand operand.

        Returns:
            Tensor: A tensor containing the bitwise AND result.
        """
        return self._get_op("BitwiseAnd")(other, cast("Tensor", self))

    def __or__(self, other: "TensorOrScalar") -> "Tensor":
        """Compute elementwise bitwise OR `self | other`.

        Args:
            other (TensorOrScalar): Right-hand operand.

        Returns:
            Tensor: A tensor containing the bitwise OR result.
        """
        return self._get_op("BitwiseOr")(cast("Tensor", self), other)

    def __ror__(self, other: "TensorOrScalar") -> "Tensor":
        """Compute reverse elementwise bitwise OR `other | self`.

        Args:
            other (TensorOrScalar): Left-hand operand.

        Returns:
            Tensor: A tensor containing the bitwise OR result.
        """
        return self._get_op("BitwiseOr")(other, cast("Tensor", self))

    def __xor__(self, other: "TensorOrScalar") -> "Tensor":
        """Compute elementwise bitwise XOR `self ^ other`.

        Args:
            other (TensorOrScalar): Right-hand operand.

        Returns:
            Tensor: A tensor containing the bitwise XOR result.
        """
        return self._get_op("BitwiseXor")(cast("Tensor", self), other)

    def __rxor__(self, other: "TensorOrScalar") -> "Tensor":
        """Compute reverse elementwise bitwise XOR `other ^ self`.

        Args:
            other (TensorOrScalar): Left-hand operand.

        Returns:
            Tensor: A tensor containing the bitwise XOR result.
        """
        return self._get_op("BitwiseXor")(other, cast("Tensor", self))

    def __lshift__(self, other: "TensorOrScalar") -> "Tensor":
        """Compute elementwise left bit shift `self << other`.

        Args:
            other (TensorOrScalar): Shift amount operand.

        Returns:
            Tensor: A tensor containing the left-shifted result.
        """
        return self._get_op("LeftShift")(cast("Tensor", self), other)

    def __rlshift__(self, other: "TensorOrScalar") -> "Tensor":
        """Compute reverse elementwise left bit shift `other << self`.

        Args:
            other (TensorOrScalar): Value to shift.

        Returns:
            Tensor: A tensor containing the left-shifted result.
        """
        return self._get_op("LeftShift")(other, cast("Tensor", self))

    def __rshift__(self, other: "TensorOrScalar") -> "Tensor":
        """Compute elementwise right bit shift `self >> other`.

        Args:
            other (TensorOrScalar): Shift amount operand.

        Returns:
            Tensor: A tensor containing the right-shifted result.
        """
        return self._get_op("RightShift")(cast("Tensor", self), other)

    def __rrshift__(self, other: "TensorOrScalar") -> "Tensor":
        """Compute reverse elementwise right bit shift `other >> self`.

        Args:
            other (TensorOrScalar): Value to shift.

        Returns:
            Tensor: A tensor containing the right-shifted result.
        """
        return self._get_op("RightShift")(other, cast("Tensor", self))

    def __neg__(self) -> "Tensor":
        """Compute elementwise negation `-self`.

        Returns:
            Tensor: A tensor containing the negated values.
        """
        return self._get_op("Negative")(cast("Tensor", self))

    def __pos__(self) -> "Tensor":
        """Compute elementwise unary positive `+self`.

        Returns:
            Tensor: A tensor containing the positive values.
        """
        return self._get_op("Positive")(cast("Tensor", self))

    def __abs__(self) -> "Tensor":
        """Compute elementwise absolute value `abs(self)`.

        Returns:
            Tensor: A tensor containing the absolute values.
        """
        return self._get_op("Abs")(cast("Tensor", self))

    def __invert__(self) -> "Tensor":
        """Compute elementwise bitwise inversion `~self`.

        Returns:
            Tensor: A tensor containing the inverted values.
        """
        return self._get_op("BitwiseNot")(cast("Tensor", self))


class TensorLogicalMixin:
    """Apply logical mixin."""

    _get_op: Callable[[str], "OpDef"]

    def __lt__(self, other: "TensorOrScalar") -> "Tensor":
        """Compare elementwise strictly less than `self < other`.

        Args:
            other (TensorOrScalar): Right-hand operand.

        Returns:
            Tensor: A boolean tensor of comparison results.
        """
        return self._get_op("Less")(cast("Tensor", self), other)

    def __gt__(self, other: "TensorOrScalar") -> "Tensor":
        """Compare elementwise strictly greater than `self > other`.

        Args:
            other (TensorOrScalar): Right-hand operand.

        Returns:
            Tensor: A boolean tensor of comparison results.
        """
        return self._get_op("Greater")(cast("Tensor", self), other)

    def __le__(self, other: "TensorOrScalar") -> "Tensor":
        """Compare elementwise less than or equal `self <= other`.

        Args:
            other (TensorOrScalar): Right-hand operand.

        Returns:
            Tensor: A boolean tensor of comparison results.
        """
        return self._get_op("LessEqual")(cast("Tensor", self), other)

    def __ge__(self, other: "TensorOrScalar") -> "Tensor":
        """Compare elementwise greater than or equal `self >= other`.

        Args:
            other (TensorOrScalar): Right-hand operand.

        Returns:
            Tensor: A boolean tensor of comparison results.
        """
        return self._get_op("GreaterEqual")(cast("Tensor", self), other)

    def __hash__(self) -> int:
        """Compute identity-based hash of the tensor instance.

        Returns:
            int: Identity hash value.
        """
        return id(self)

    def _eq(self, other: "TensorOrScalar") -> "Tensor":
        """Compare elementwise equality `self == other`.

        Args:
            other (TensorOrScalar): Right-hand operand.

        Returns:
            Tensor: A boolean tensor of equality results.
        """
        return self._get_op("Equal")(cast("Tensor", self), other)

    def _ne(self, other: "TensorOrScalar") -> "Tensor":
        """Compare elementwise inequality `self != other`.

        Args:
            other (TensorOrScalar): Right-hand operand.

        Returns:
            Tensor: A boolean tensor of inequality results.
        """
        return self._get_op("NotEqual")(cast("Tensor", self), other)

    def __eq__(self, other: object) -> bool:
        """Compare elementwise equality `self == other`.

        Args:
            other (TensorOrScalar): Right-hand operand.

        Returns:
            bool: A boolean tensor of equality results.
        """
        return cast(bool, self._eq(cast("TensorOrScalar", other)))

    def __ne__(self, other: object) -> bool:
        """Compare elementwise inequality `self != other`.

        Args:
            other (TensorOrScalar): Right-hand operand.

        Returns:
            bool: A boolean tensor of inequality results.
        """
        return cast(bool, self._ne(cast("TensorOrScalar", other)))
