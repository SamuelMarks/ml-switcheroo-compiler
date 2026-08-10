# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
from typing import Any

"""Math mixin for Tensor."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ml_switcheroo_compiler.core.tensor import Tensor


if TYPE_CHECKING:
    pass


class TensorArithmeticMixin:
    """Arithmetic mixin."""

    def _get_op(self, name: str) -> Any:
        """Get the requested operation from the registry.

        Args:
            name (str): The name of the operation to retrieve.

        Returns: Any: The operation class or instance.
        """
        from ml_switcheroo_compiler.ops.registry import get_op

        return get_op(name)()

    def __add__(self, other: Any) -> "Tensor":
        """Add.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Add")(self, other)

    def __radd__(self, other: Any) -> "Tensor":
        """Radd.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Add")(other, self)

    def __sub__(self, other: Any) -> "Tensor":
        """Sub.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Subtract")(self, other)

    def __rsub__(self, other: Any) -> "Tensor":
        """Rsub.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Subtract")(other, self)

    def __mul__(self, other: Any) -> "Tensor":
        """Mul.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Multiply")(self, other)

    def __rmul__(self, other: Any) -> "Tensor":
        """Rmul.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Multiply")(other, self)

    def __truediv__(self, other: Any) -> "Tensor":
        """Truediv.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("TrueDivide")(self, other)

    def __rtruediv__(self, other: Any) -> "Tensor":
        """Rtruediv.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("TrueDivide")(other, self)

    def __pow__(self, other: Any) -> "Tensor":
        """Pow.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Power")(self, other)

    def __rpow__(self, other: Any) -> "Tensor":
        """Rpow.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Power")(other, self)

    def __floordiv__(self, other: Any) -> "Tensor":
        """Floordiv.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("FloorDivide")(self, other)

    def __rfloordiv__(self, other: Any) -> "Tensor":
        """Rfloordiv.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("FloorDivide")(other, self)

    def __mod__(self, other: Any) -> "Tensor":
        """Mod.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Mod")(self, other)

    def __rmod__(self, other: Any) -> "Tensor":
        """Rmod.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Mod")(other, self)


class TensorBitwiseMixin:
    """Apply bitwise mixin."""

    def __and__(self, other: Any) -> "Tensor":
        """And.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("BitwiseAnd")(self, other)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __rand__(self, other: Any) -> "Tensor":
        """Rand.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("BitwiseAnd")(other, self)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __or__(self, other: Any) -> "Tensor":
        """Or.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("BitwiseOr")(self, other)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __ror__(self, other: Any) -> "Tensor":
        """Ror.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("BitwiseOr")(other, self)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __xor__(self, other: Any) -> "Tensor":
        """Xor.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("BitwiseXor")(self, other)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __rxor__(self, other: Any) -> "Tensor":
        """Rxor.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("BitwiseXor")(other, self)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __lshift__(self, other: Any) -> "Tensor":
        """Lshift.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("LeftShift")(self, other)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __rlshift__(self, other: Any) -> "Tensor":
        """Rlshift.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("LeftShift")(other, self)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __rshift__(self, other: Any) -> "Tensor":
        """Rshift.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("RightShift")(self, other)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __rrshift__(self, other: Any) -> "Tensor":
        """Rrshift.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("RightShift")(other, self)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __neg__(self) -> "Tensor":
        """Neg.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Negative")(self)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __pos__(self) -> "Tensor":
        """Pos.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Positive")(self)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __abs__(self) -> "Tensor":
        """Abs.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Abs")(self)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __invert__(self) -> "Tensor":
        """Invert.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("BitwiseNot")(self)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism


class TensorLogicalMixin:
    """Apply logical mixin."""

    def __lt__(self, other: Any) -> "Tensor":
        """Lt.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Less")(self, other)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __gt__(self, other: Any) -> "Tensor":
        """Gt.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Greater")(self, other)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __le__(self, other: Any) -> "Tensor":
        """Le.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("LessEqual")(self, other)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __ge__(self, other: Any) -> "Tensor":
        """Ge.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("GreaterEqual")(self, other)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __hash__(self) -> int:
        """Evaluate __hash__ operation.

        Returns:
        int: Result.
        """
        return id(self)

    def __eq__(self, other: Any) -> Any:
        """Eq.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Equal")(self, other)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __ne__(self, other: Any) -> Any:
        """Ne.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("NotEqual")(self, other)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
