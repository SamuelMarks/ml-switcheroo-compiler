"""Math mixin for Tensor."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml_switcheroo_compiler.core.tensor import Tensor


class TensorArithmeticMixin:
    """Arithmetic mixin."""

    """Provides math dunder methods for Tensor."""

    def _get_op(self, name: str) -> object:
        """Execute _get_op.

        Args:
            name (Any): Argument name.

        Returns:
        Any: The result.
        """
        from ml_switcheroo_compiler.ops.base import get_op

        return get_op(name)()

    def __add__(self, other: object) -> "Tensor":
        """Add.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Add")(self, other)

    def __radd__(self, other: object) -> "Tensor":
        """Radd.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Add")(other, self)

    def __sub__(self, other: object) -> "Tensor":
        """Sub.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Subtract")(self, other)

    def __rsub__(self, other: object) -> "Tensor":
        """Rsub.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Subtract")(other, self)

    def __mul__(self, other: object) -> "Tensor":
        """Mul.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Multiply")(self, other)

    def __rmul__(self, other: object) -> "Tensor":
        """Rmul.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Multiply")(other, self)

    def __truediv__(self, other: object) -> "Tensor":
        """Truediv.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("TrueDivide")(self, other)

    def __rtruediv__(self, other: object) -> "Tensor":
        """Rtruediv.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("TrueDivide")(other, self)

    def __pow__(self, other: object) -> "Tensor":
        """Pow.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Power")(self, other)

    def __floordiv__(self, other: object) -> "Tensor":
        """Floordiv.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("FloorDivide")(self, other)

    def __mod__(self, other: object) -> "Tensor":
        """Mod.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Mod")(self, other)


class TensorBitwiseMixin:
    """Bitwise mixin."""

    def __and__(self, other: object) -> "Tensor":
        """And.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("BitwiseAnd")(self, other)

    def __or__(self, other: object) -> "Tensor":
        """Or.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("BitwiseOr")(self, other)

    def __xor__(self, other: object) -> "Tensor":
        """Xor.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("BitwiseXor")(self, other)

    def __lshift__(self, other: object) -> "Tensor":
        """Lshift.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("LeftShift")(self, other)

    def __rshift__(self, other: object) -> "Tensor":
        """Rshift.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("RightShift")(self, other)

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
    """Logical mixin."""

    def __lt__(self, other: object) -> "Tensor":
        """Lt.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Less")(self, other)

    def __gt__(self, other: object) -> "Tensor":
        """Gt.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Greater")(self, other)

    def __le__(self, other: object) -> "Tensor":
        """Le.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("LessEqual")(self, other)

    def __ge__(self, other: object) -> "Tensor":
        """Ge.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("GreaterEqual")(self, other)

    def __hash__(self) -> int:
        """Execute __hash__.

        Returns:
        Any: The result.
        """
        return id(self)

    def __eq__(self, other: object) -> "Tensor":
        """Eq.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Equal")(self, other)

    def __ne__(self, other: object) -> "Tensor":
        """Ne.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("NotEqual")(self, other)
