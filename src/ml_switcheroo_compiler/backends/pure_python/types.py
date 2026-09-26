"""Type definitions and array wrappers for Pure Python execution backend."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Callable, Union

Numeric = Union[int, float]
NestedList = Union[Numeric, list["NestedList"]]


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
        while isinstance(curr, list):
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
            if isinstance(item, list):
                for sub in item:
                    _flatten(sub)
            else:
                result.append(item)

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
            if isinstance(item, list):
                return [_apply(sub) for sub in item]
            return func(float(item))

        transformed: NestedList = _apply(self.data)
        return PurePythonTensor(transformed, shape=self.shape, dtype=self.dtype)

    def __add__(self, other: PurePythonTensor | Numeric) -> PurePythonTensor:
        """Perform elementwise tensor addition.

        Args:
            other (PurePythonTensor | Numeric): Right-hand operand.

        Returns:
            PurePythonTensor: Result of addition.
        """
        if isinstance(other, (int, float)):
            return self.apply_elementwise(lambda x: x + float(other))

        def _add(a: NestedList, b: NestedList) -> NestedList:
            """Recursively add two nested lists elementwise.

            Args:
                a (NestedList): Left nested operand.
                b (NestedList): Right nested operand.

            Returns:
                NestedList: Summed nested elements.
            """
            if isinstance(a, list) and isinstance(b, list):
                return [_add(ai, bi) for ai, bi in zip(a, b)]
            return float(a) + float(b)

        return PurePythonTensor(_add(self.data, other.data), shape=self.shape, dtype=self.dtype)

    def __sub__(self, other: PurePythonTensor | Numeric) -> PurePythonTensor:
        """Perform elementwise tensor subtraction.

        Args:
            other (PurePythonTensor | Numeric): Right-hand operand.

        Returns:
            PurePythonTensor: Result of subtraction.
        """
        if isinstance(other, (int, float)):
            return self.apply_elementwise(lambda x: x - float(other))

        def _sub(a: NestedList, b: NestedList) -> NestedList:
            """Recursively subtract two nested lists elementwise.

            Args:
                a (NestedList): Left nested operand.
                b (NestedList): Right nested operand.

            Returns:
                NestedList: Subtracted nested elements.
            """
            if isinstance(a, list) and isinstance(b, list):
                return [_sub(ai, bi) for ai, bi in zip(a, b)]
            return float(a) - float(b)

        return PurePythonTensor(_sub(self.data, other.data), shape=self.shape, dtype=self.dtype)

    def __mul__(self, other: PurePythonTensor | Numeric) -> PurePythonTensor:
        """Perform elementwise tensor multiplication.

        Args:
            other (PurePythonTensor | Numeric): Right-hand operand.

        Returns:
            PurePythonTensor: Result of multiplication.
        """
        if isinstance(other, (int, float)):
            return self.apply_elementwise(lambda x: x * float(other))

        def _mul(a: NestedList, b: NestedList) -> NestedList:
            """Recursively multiply two nested lists elementwise.

            Args:
                a (NestedList): Left nested operand.
                b (NestedList): Right nested operand.

            Returns:
                NestedList: Multiplied nested elements.
            """
            if isinstance(a, list) and isinstance(b, list):
                return [_mul(ai, bi) for ai, bi in zip(a, b)]
            return float(a) * float(b)

        return PurePythonTensor(_mul(self.data, other.data), shape=self.shape, dtype=self.dtype)

    def __truediv__(self, other: PurePythonTensor | Numeric) -> PurePythonTensor:
        """Perform elementwise tensor true division.

        Args:
            other (PurePythonTensor | Numeric): Right-hand operand.

        Returns:
            PurePythonTensor: Result of division.
        """
        if isinstance(other, (int, float)):
            divisor: float = float(other)
            return self.apply_elementwise(lambda x: x / divisor)

        def _div(a: NestedList, b: NestedList) -> NestedList:
            """Recursively divide two nested lists elementwise.

            Args:
                a (NestedList): Left nested operand.
                b (NestedList): Right nested operand.

            Returns:
                NestedList: Divided nested elements.
            """
            if isinstance(a, list) and isinstance(b, list):
                return [_div(ai, bi) for ai, bi in zip(a, b)]
            return float(a) / float(b)

        return PurePythonTensor(_div(self.data, other.data), shape=self.shape, dtype=self.dtype)

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
