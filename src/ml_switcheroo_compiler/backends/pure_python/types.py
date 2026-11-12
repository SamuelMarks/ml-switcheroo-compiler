"""Type definitions and array wrappers for Pure Python execution backend."""

from __future__ import annotations

import math
import sys
import typing
from collections.abc import Callable, Sequence
from types import ModuleType

Numeric = typing.Union[int, float]  # noqa: UP007


class _PurePythonTypesModProtocol(typing.Protocol):
    """Protocol for binding `_NestedListSelf` onto `types` module."""

    _NestedListSelf: type


def _is_pp_mod(mod: ModuleType) -> typing.TypeGuard[_PurePythonTypesModProtocol]:
    """Type guard narrowing `ModuleType` to `_PurePythonTypesModProtocol`.

    Args:
        mod (ModuleType): Module instance.

    Returns:
        bool: True if `mod` is a `ModuleType`.
    """
    return isinstance(mod, ModuleType)


_pp_mod = sys.modules[__name__]
assert _is_pp_mod(_pp_mod)
_pp_mod._NestedListSelf = int

if typing.TYPE_CHECKING:
    _NestedListSelf: typing.TypeAlias = "NestedList"
NestedList = typing.Union[Numeric, Sequence[_NestedListSelf]]  # noqa: UP007


class PurePythonTensor:
    """Multi-dimensional tensor implementation in Pure Python standard library."""

    def __init__(
        self,
        data: Sequence[Numeric] | NestedList,
        shape: tuple[int, ...] | None = None,
        dtype: str = "float32",
    ) -> None:
        """Initialize PurePythonTensor.

        Args:
            data (Sequence[Numeric] | NestedList): Raw nested or flat numeric data.
            shape (tuple[int, ...] | None): Optional explicit shape tuple.
            dtype (str): Data type string representation.
        """
        self.dtype: str = dtype
        if isinstance(data, (int, float)):
            self.data: NestedList = float(data) if dtype.startswith("float") else int(data)
            self.shape: tuple[int, ...] = shape if shape is not None else ()
        elif isinstance(data, list):
            self.data = data
            self.shape = shape if shape is not None else self._infer_nested_shape(data)
        else:
            self.data = list(data)
            self.shape = shape if shape is not None else (len(self.data),)

    @staticmethod
    def _infer_nested_shape(val: NestedList) -> tuple[int, ...]:
        """Infer shape dimensions from nested lists recursively.

        Args:
            val (NestedList): Nested list or scalar element.

        Returns:
            tuple[int, ...]: Dimension sizes tuple.
        """
        dims: list[int] = []
        curr: NestedList = val
        while isinstance(curr, (list, tuple)):
            dims.append(len(curr))
            curr = curr[0] if curr else 0
        return tuple(dims)

    def to_flat_list(self) -> list[Numeric]:
        """Flatten nested list elements into a 1D list.

        Returns:
            list[Numeric]: 1D flat numeric elements list.
        """
        result: list[Numeric] = []

        def _flatten(item: NestedList) -> None:
            """Recursively flatten nested lists.

            Args:
                item (NestedList): Nested list or scalar number.
            """
            if isinstance(item, (int, float)):
                result.append(item)
            else:
                for sub in item:
                    _flatten(sub)

        _flatten(self.data)
        return result

    def apply_elementwise(self, func: Callable[[float], float]) -> PurePythonTensor:
        """Apply a unary scalar operation elementwise.

        Args:
            func (Callable[[float], float]): Unary math function to apply.

        Returns:
            PurePythonTensor: New tensor with transformed values.
        """

        def _apply(item: NestedList) -> NestedList:
            """Apply function recursively across nested list.

            Args:
                item (NestedList): Nested list or scalar.

            Returns:
                NestedList: Transformed element or nested list.
            """
            if isinstance(item, (int, float)):
                return func(float(item))
            return [_apply(sub) for sub in item]

        transformed: NestedList = _apply(self.data)
        return PurePythonTensor(transformed, shape=self.shape, dtype=self.dtype)

    def _combine_nested(
        self,
        a: NestedList,
        b: NestedList,
        fn: Callable[[float, float], float],
    ) -> NestedList:
        """Recursively combine two identically-shaped nested structures elementwise.

        Args:
            a (NestedList): Left nested operand.
            b (NestedList): Right nested operand.
            fn (Callable[[float, float], float]): Scalar binary operator.

        Returns:
            NestedList: Elementwise combined nested result.

        Raises:
            ShapeMismatchError: If `a` and `b` differ in nesting rank or sequence length.
        """
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return fn(float(a), float(b))
        if isinstance(a, Sequence) and isinstance(b, Sequence) and len(a) == len(b):
            return [self._combine_nested(ai, bi, fn) for ai, bi in zip(a, b)]
        from ml_switcheroo_compiler.core.errors import ShapeMismatchError

        raise ShapeMismatchError(f"Incompatible nested tensor structures for shapes {self.shape}.")

    def _binary_op(
        self,
        other: PurePythonTensor | Numeric,
        fn: Callable[[float, float], float],
    ) -> PurePythonTensor:
        """Apply a binary scalar or tensor operation elementwise.

        Args:
            other (PurePythonTensor | Numeric): Right-hand tensor or scalar operand.
            fn (Callable[[float, float], float]): Scalar binary operator.

        Returns:
            PurePythonTensor: Elementwise binary result tensor.

        Raises:
            ShapeMismatchError: If tensor shapes are incompatible.
        """
        if isinstance(other, (int, float)):
            rhs = float(other)
            return self.apply_elementwise(lambda x: fn(x, rhs))
        if self.shape != other.shape:
            from ml_switcheroo_compiler.core.errors import ShapeMismatchError

            raise ShapeMismatchError(f"Shapes {self.shape} and {other.shape} are incompatible.")
        return PurePythonTensor(self._combine_nested(self.data, other.data, fn), shape=self.shape, dtype=self.dtype)

    def __add__(self, other: PurePythonTensor | Numeric) -> PurePythonTensor:
        """Perform elementwise tensor addition.

        Args:
            other (PurePythonTensor | Numeric): Right-hand operand.

        Returns:
            PurePythonTensor: Result of addition.
        """
        return self._binary_op(other, lambda x, y: x + y)

    def __sub__(self, other: PurePythonTensor | Numeric) -> PurePythonTensor:
        """Perform elementwise tensor subtraction.

        Args:
            other (PurePythonTensor | Numeric): Right-hand operand.

        Returns:
            PurePythonTensor: Result of subtraction.
        """
        return self._binary_op(other, lambda x, y: x - y)

    def __mul__(self, other: PurePythonTensor | Numeric) -> PurePythonTensor:
        """Perform elementwise tensor multiplication.

        Args:
            other (PurePythonTensor | Numeric): Right-hand operand.

        Returns:
            PurePythonTensor: Result of multiplication.
        """
        return self._binary_op(other, lambda x, y: x * y)

    def __truediv__(self, other: PurePythonTensor | Numeric) -> PurePythonTensor:
        """Perform elementwise tensor true division.

        Args:
            other (PurePythonTensor | Numeric): Right-hand operand.

        Returns:
            PurePythonTensor: Result of division.
        """
        return self._binary_op(other, lambda x, y: x / y)

    def __neg__(self) -> PurePythonTensor:
        """Perform elementwise negation.

        Returns:
            PurePythonTensor: Negated tensor.
        """
        return self.apply_elementwise(lambda x: -x)

    def exp(self) -> PurePythonTensor:
        """Compute exponential elementwise.

        Returns:
            PurePythonTensor: Exponential tensor.
        """
        return self.apply_elementwise(math.exp)

    def log(self) -> PurePythonTensor:
        """Compute natural logarithm elementwise.

        Returns:
            PurePythonTensor: Natural logarithm tensor.
        """
        return self.apply_elementwise(math.log)

    def sqrt(self) -> PurePythonTensor:
        """Compute square root elementwise.

        Returns:
            PurePythonTensor: Square root tensor.
        """
        return self.apply_elementwise(math.sqrt)

    def sin(self) -> PurePythonTensor:
        """Compute sine elementwise.

        Returns:
            PurePythonTensor: Sine tensor.
        """
        return self.apply_elementwise(math.sin)

    def cos(self) -> PurePythonTensor:
        """Compute cosine elementwise.

        Returns:
            PurePythonTensor: Cosine tensor.
        """
        return self.apply_elementwise(math.cos)

    def tanh(self) -> PurePythonTensor:
        """Compute hyperbolic tangent elementwise.

        Returns:
            PurePythonTensor: Tanh tensor.
        """
        return self.apply_elementwise(math.tanh)

    def sum(self) -> float:
        """Sum all elements across all dimensions.

        Returns:
            float: Total scalar sum.
        """
        return sum(float(x) for x in self.to_flat_list())
