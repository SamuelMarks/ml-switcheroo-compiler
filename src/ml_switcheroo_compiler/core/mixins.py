"""Core Mixins."""

import typing  # noqa: F401
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.registry import EagerValue
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.ops.base import OpDef


class TensorArithmeticMixin:
    """Arithmetic mixin."""

    def _get_op(self, name: str) -> "OpDef":
        """Get the requested operation from the registry.

        Args:
            name (str): The name of the operation to retrieve.

        Returns: Tensor: The operation class or instance.
        """
        from ml_switcheroo_compiler.ops.registry import get_op

        return get_op(name)()

    def __add__(self, other: "EagerValue") -> "Tensor":
        """Add.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Add")(cast("Tensor", self), other)

    def __radd__(self, other: "EagerValue") -> "Tensor":
        """Radd.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Add")(other, cast("Tensor", self))

    def __sub__(self, other: "EagerValue") -> "Tensor":
        """Sub.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Subtract")(cast("Tensor", self), other)

    def __rsub__(self, other: "EagerValue") -> "Tensor":
        """Rsub.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Subtract")(other, cast("Tensor", self))

    def __mul__(self, other: "EagerValue") -> "Tensor":
        """Mul.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Multiply")(cast("Tensor", self), other)

    def __rmul__(self, other: "EagerValue") -> "Tensor":
        """Rmul.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Multiply")(other, cast("Tensor", self))

    def __truediv__(self, other: "EagerValue") -> "Tensor":
        """Truediv.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("TrueDivide")(cast("Tensor", self), other)

    def __rtruediv__(self, other: "EagerValue") -> "Tensor":
        """Rtruediv.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("TrueDivide")(other, cast("Tensor", self))

    def __pow__(self, other: "EagerValue") -> "Tensor":
        """Pow.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Power")(cast("Tensor", self), other)

    def __rpow__(self, other: "EagerValue") -> "Tensor":
        """Rpow.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Power")(other, cast("Tensor", self))

    def __floordiv__(self, other: "EagerValue") -> "Tensor":
        """Floordiv.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("FloorDivide")(cast("Tensor", self), other)

    def __rfloordiv__(self, other: "EagerValue") -> "Tensor":
        """Rfloordiv.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("FloorDivide")(other, cast("Tensor", self))

    def __mod__(self, other: "EagerValue") -> "Tensor":
        """Mod.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Mod")(cast("Tensor", self), other)

    def __rmod__(self, other: "EagerValue") -> "Tensor":
        """Rmod.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Mod")(other, cast("Tensor", self))


class TensorBitwiseMixin:
    """Apply bitwise mixin."""

    _get_op: Callable[[str], "OpDef"]

    def __and__(self, other: "EagerValue") -> "Tensor":
        """And.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("BitwiseAnd")(cast("Tensor", self), other)

    def __rand__(self, other: "EagerValue") -> "Tensor":
        """Rand.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("BitwiseAnd")(other, cast("Tensor", self))

    def __or__(self, other: "EagerValue") -> "Tensor":
        """Or.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("BitwiseOr")(cast("Tensor", self), other)

    def __ror__(self, other: "EagerValue") -> "Tensor":
        """Ror.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("BitwiseOr")(other, cast("Tensor", self))

    def __xor__(self, other: "EagerValue") -> "Tensor":
        """Xor.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("BitwiseXor")(cast("Tensor", self), other)

    def __rxor__(self, other: "EagerValue") -> "Tensor":
        """Rxor.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("BitwiseXor")(other, cast("Tensor", self))

    def __lshift__(self, other: "EagerValue") -> "Tensor":
        """Lshift.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("LeftShift")(cast("Tensor", self), other)

    def __rlshift__(self, other: "EagerValue") -> "Tensor":
        """Rlshift.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("LeftShift")(other, cast("Tensor", self))

    def __rshift__(self, other: "EagerValue") -> "Tensor":
        """Rshift.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("RightShift")(cast("Tensor", self), other)

    def __rrshift__(self, other: "EagerValue") -> "Tensor":
        """Rrshift.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("RightShift")(other, cast("Tensor", self))

    def __neg__(self) -> "Tensor":
        """Neg.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Negative")(cast("Tensor", self))

    def __pos__(self) -> "Tensor":
        """Pos.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Positive")(cast("Tensor", self))

    def __abs__(self) -> "Tensor":
        """Abs.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Abs")(cast("Tensor", self))

    def __invert__(self) -> "Tensor":
        """Invert.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("BitwiseNot")(cast("Tensor", self))


class TensorLogicalMixin:
    """Apply logical mixin."""

    _get_op: Callable[[str], "OpDef"]

    def __lt__(self, other: "EagerValue") -> "Tensor":
        """Lt.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Less")(cast("Tensor", self), other)

    def __gt__(self, other: "EagerValue") -> "Tensor":
        """Gt.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Greater")(cast("Tensor", self), other)

    def __le__(self, other: "EagerValue") -> "Tensor":
        """Le.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("LessEqual")(cast("Tensor", self), other)

    def __ge__(self, other: "EagerValue") -> "Tensor":
        """Ge.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("GreaterEqual")(cast("Tensor", self), other)

    def __hash__(self) -> int:
        """Evaluate __hash__ operation.

        Returns:
        int: Result.
        """
        return id(self)

    def _eq(self, other: "EagerValue") -> "Tensor":
        """Eq.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Equal")(cast("Tensor", self), other)

    def _ne(self, other: "EagerValue") -> "Tensor":
        """Ne.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("NotEqual")(cast("Tensor", self), other)


setattr(TensorLogicalMixin, "__eq__", TensorLogicalMixin._eq)
setattr(TensorLogicalMixin, "__ne__", TensorLogicalMixin._ne)
