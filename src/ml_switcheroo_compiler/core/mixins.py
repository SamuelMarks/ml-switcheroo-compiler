"""Core Mixins."""

import typing

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml_switcheroo_compiler.core.tensor import Tensor


if TYPE_CHECKING:
    pass


class TensorArithmeticMixin:
    """Arithmetic mixin."""

    def _get_op(self, name: str):
        """Get the requested operation from the registry.

        Args:
            name (str): The name of the operation to retrieve.

        Returns: Tensor: The operation class or instance.
        """
        from ml_switcheroo_compiler.ops.registry import get_op

        return get_op(name)()

    def __add__(self, other) -> "Tensor":
        """Add.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Add")(self, other)

    def __radd__(self, other) -> "Tensor":
        """Radd.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Add")(other, self)

    def __sub__(self, other) -> "Tensor":
        """Sub.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Subtract")(self, other)

    def __rsub__(self, other) -> "Tensor":
        """Rsub.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Subtract")(other, self)

    def __mul__(self, other) -> "Tensor":
        """Mul.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Multiply")(self, other)

    def __rmul__(self, other) -> "Tensor":
        """Rmul.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Multiply")(other, self)

    def __truediv__(self, other) -> "Tensor":
        """Truediv.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("TrueDivide")(self, other)

    def __rtruediv__(self, other) -> "Tensor":
        """Rtruediv.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("TrueDivide")(other, self)

    def __pow__(self, other) -> "Tensor":
        """Pow.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Power")(self, other)

    def __rpow__(self, other) -> "Tensor":
        """Rpow.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Power")(other, self)

    def __floordiv__(self, other) -> "Tensor":
        """Floordiv.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("FloorDivide")(self, other)

    def __rfloordiv__(self, other) -> "Tensor":
        """Rfloordiv.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("FloorDivide")(other, self)

    def __mod__(self, other) -> "Tensor":
        """Mod.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Mod")(self, other)

    def __rmod__(self, other) -> "Tensor":
        """Rmod.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Mod")(other, self)


class TensorBitwiseMixin:
    """Apply bitwise mixin."""

    def __and__(self, other) -> "Tensor":
        """And.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("BitwiseAnd")(self, other)

    def __rand__(self, other) -> "Tensor":
        """Rand.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("BitwiseAnd")(other, self)

    def __or__(self, other) -> "Tensor":
        """Or.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("BitwiseOr")(self, other)

    def __ror__(self, other) -> "Tensor":
        """Ror.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("BitwiseOr")(other, self)

    def __xor__(self, other) -> "Tensor":
        """Xor.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("BitwiseXor")(self, other)

    def __rxor__(self, other) -> "Tensor":
        """Rxor.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("BitwiseXor")(other, self)

    def __lshift__(self, other) -> "Tensor":
        """Lshift.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("LeftShift")(self, other)

    def __rlshift__(self, other) -> "Tensor":
        """Rlshift.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("LeftShift")(other, self)

    def __rshift__(self, other) -> "Tensor":
        """Rshift.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("RightShift")(self, other)

    def __rrshift__(self, other) -> "Tensor":
        """Rrshift.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("RightShift")(other, self)

    def __neg__(self) -> "Tensor":
        """Neg.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Negative")(self)

    def __pos__(self) -> "Tensor":
        """Pos.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Positive")(self)

    def __abs__(self) -> "Tensor":
        """Abs.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Abs")(self)

    def __invert__(self) -> "Tensor":
        """Invert.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("BitwiseNot")(self)


class TensorLogicalMixin:
    """Apply logical mixin."""

    def __lt__(self, other) -> "Tensor":
        """Lt.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Less")(self, other)

    def __gt__(self, other) -> "Tensor":
        """Gt.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Greater")(self, other)

    def __le__(self, other) -> "Tensor":
        """Le.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("LessEqual")(self, other)

    def __ge__(self, other) -> "Tensor":
        """Ge.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("GreaterEqual")(self, other)

    def __hash__(self) -> int:
        """Evaluate __hash__ operation.

        Returns:
        int: Result.
        """
        return id(self)

    def __eq__(self, other):
        """Eq.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Equal")(self, other)

    def __ne__(self, other):
        """Ne.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("NotEqual")(self, other)
