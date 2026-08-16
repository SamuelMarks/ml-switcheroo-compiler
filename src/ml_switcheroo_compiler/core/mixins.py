"""Core Mixins."""

import typing

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
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

    def __add__(self, other: Any) -> "Tensor":  # type: ignore
        """Add.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Add")(self, other)  # type: ignore

    def __radd__(self, other: Any) -> "Tensor":  # type: ignore
        """Radd.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Add")(other, self)  # type: ignore

    def __sub__(self, other: Any) -> "Tensor":  # type: ignore
        """Sub.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Subtract")(self, other)  # type: ignore

    def __rsub__(self, other: Any) -> "Tensor":  # type: ignore
        """Rsub.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Subtract")(other, self)  # type: ignore

    def __mul__(self, other: Any) -> "Tensor":  # type: ignore
        """Mul.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Multiply")(self, other)  # type: ignore

    def __rmul__(self, other: Any) -> "Tensor":  # type: ignore
        """Rmul.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Multiply")(other, self)  # type: ignore

    def __truediv__(self, other: Any) -> "Tensor":  # type: ignore
        """Truediv.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("TrueDivide")(self, other)  # type: ignore

    def __rtruediv__(self, other: Any) -> "Tensor":  # type: ignore
        """Rtruediv.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("TrueDivide")(other, self)  # type: ignore

    def __pow__(self, other: Any) -> "Tensor":  # type: ignore
        """Pow.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Power")(self, other)  # type: ignore

    def __rpow__(self, other: Any) -> "Tensor":  # type: ignore
        """Rpow.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Power")(other, self)  # type: ignore

    def __floordiv__(self, other: Any) -> "Tensor":  # type: ignore
        """Floordiv.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("FloorDivide")(self, other)  # type: ignore

    def __rfloordiv__(self, other: Any) -> "Tensor":  # type: ignore
        """Rfloordiv.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("FloorDivide")(other, self)  # type: ignore

    def __mod__(self, other: Any) -> "Tensor":  # type: ignore
        """Mod.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Mod")(self, other)  # type: ignore

    def __rmod__(self, other: Any) -> "Tensor":  # type: ignore
        """Rmod.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Mod")(other, self)  # type: ignore


class TensorBitwiseMixin:
    """Apply bitwise mixin."""

    def __and__(self, other: Any) -> "Tensor":  # type: ignore
        """And.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("BitwiseAnd")(self, other)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __rand__(self, other: Any) -> "Tensor":  # type: ignore
        """Rand.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("BitwiseAnd")(other, self)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __or__(self, other: Any) -> "Tensor":  # type: ignore
        """Or.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("BitwiseOr")(self, other)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __ror__(self, other: Any) -> "Tensor":  # type: ignore
        """Ror.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("BitwiseOr")(other, self)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __xor__(self, other: Any) -> "Tensor":  # type: ignore
        """Xor.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("BitwiseXor")(self, other)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __rxor__(self, other: Any) -> "Tensor":  # type: ignore
        """Rxor.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("BitwiseXor")(other, self)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __lshift__(self, other: Any) -> "Tensor":  # type: ignore
        """Lshift.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("LeftShift")(self, other)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __rlshift__(self, other: Any) -> "Tensor":  # type: ignore
        """Rlshift.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("LeftShift")(other, self)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __rshift__(self, other: Any) -> "Tensor":  # type: ignore
        """Rshift.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("RightShift")(self, other)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __rrshift__(self, other: Any) -> "Tensor":  # type: ignore
        """Rrshift.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("RightShift")(other, self)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __neg__(self) -> "Tensor":  # type: ignore
        """Neg.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Negative")(self)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __pos__(self) -> "Tensor":  # type: ignore
        """Pos.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Positive")(self)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __abs__(self) -> "Tensor":  # type: ignore
        """Abs.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Abs")(self)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __invert__(self) -> "Tensor":  # type: ignore
        """Invert.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("BitwiseNot")(self)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism


class TensorLogicalMixin:
    """Apply logical mixin."""

    def __lt__(self, other: Any) -> "Tensor":  # type: ignore
        """Lt.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Less")(self, other)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __gt__(self, other: Any) -> "Tensor":  # type: ignore
        """Gt.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("Greater")(self, other)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __le__(self, other: Any) -> "Tensor":  # type: ignore
        """Le.

        Args:
            other (object): The other to process.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self._get_op("LessEqual")(self, other)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __ge__(self, other: Any) -> "Tensor":  # type: ignore
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
