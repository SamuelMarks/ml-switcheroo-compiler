"""Forward arithmetic, comparison, and indexing dunder protocol definitions."""

from __future__ import annotations

import typing

from .types_base import BackendArg

if typing.TYPE_CHECKING:
    from .protocols import EagerTensorProtocol


_EqT = typing.TypeVar("_EqT")


class _ArrayArithmeticAndIndexingProto(typing.Protocol):
    """Structural protocol for forward arithmetic, relational comparison, and indexing dunders."""

    def __eq__(self, other: _EqT) -> bool:
        """Compare equality against `other`.

        Args:
            other (_EqT): Right-hand comparison operand.

        Returns:
            bool: Equality result.
        """
        ...

    def __ne__(self, other: _EqT) -> bool:
        """Compare inequality against `other`.

        Args:
            other (_EqT): Right-hand comparison operand.

        Returns:
            bool: Inequality result.
        """
        ...

    def __getitem__(self, key: BackendArg) -> EagerTensorProtocol:
        """Index or slice into the array.

        Args:
            key (BackendArg): Index, slice, mask, or multi-axis tuple.

        Returns:
            EagerTensorProtocol: Indexed sub-tensor or scalar view.
        """
        ...

    def __setitem__(self, key: BackendArg, value: BackendArg) -> None:
        """Mutate an indexed element or slice in place.

        Args:
            key (BackendArg): Target index, slice, or mask.
            value (BackendArg): Assigned tensor or scalar value.
        """
        ...

    def __len__(self) -> int:
        """Return size of the leading axis.

        Returns:
            int: Leading dimension length.
        """
        ...

    def __neg__(self) -> EagerTensorProtocol:
        """Compute unary negation `-self`.

        Returns:
            EagerTensorProtocol: Negated tensor.
        """
        ...

    def __add__(self, other: BackendArg) -> EagerTensorProtocol:
        """Compute elementwise sum `self + other`.

        Args:
            other (BackendArg): Right-hand addend.

        Returns:
            EagerTensorProtocol: Sum tensor.
        """
        ...

    def __sub__(self, other: BackendArg) -> EagerTensorProtocol:
        """Compute elementwise difference `self - other`.

        Args:
            other (BackendArg): Right-hand subtrahend.

        Returns:
            EagerTensorProtocol: Difference tensor.
        """
        ...

    def __mul__(self, other: BackendArg) -> EagerTensorProtocol:
        """Compute elementwise product `self * other`.

        Args:
            other (BackendArg): Right-hand multiplier.

        Returns:
            EagerTensorProtocol: Product tensor.
        """
        ...

    def __truediv__(self, other: BackendArg) -> EagerTensorProtocol:
        """Compute elementwise true division `self / other`.

        Args:
            other (BackendArg): Right-hand divisor.

        Returns:
            EagerTensorProtocol: Quotient tensor.
        """
        ...

    def __floordiv__(self, other: BackendArg) -> EagerTensorProtocol:
        """Compute elementwise floor division `self // other`.

        Args:
            other (BackendArg): Right-hand divisor.

        Returns:
            EagerTensorProtocol: Floor-quotient tensor.
        """
        ...

    def __mod__(self, other: BackendArg) -> EagerTensorProtocol:
        """Compute elementwise remainder `self % other`.

        Args:
            other (BackendArg): Right-hand divisor.

        Returns:
            EagerTensorProtocol: Remainder tensor.
        """
        ...

    def __matmul__(self, other: BackendArg) -> EagerTensorProtocol:
        """Compute matrix multiplication `self @ other`.

        Args:
            other (BackendArg): Right-hand matrix/tensor operand.

        Returns:
            EagerTensorProtocol: Matrix product tensor.
        """
        ...

    def __pow__(self, other: BackendArg) -> EagerTensorProtocol:
        """Compute elementwise power `self ** other`.

        Args:
            other (BackendArg): Exponent operand.

        Returns:
            EagerTensorProtocol: Exponentiated tensor.
        """
        ...

    def __lt__(self, other: BackendArg) -> EagerTensorProtocol:
        """Compute elementwise strict less-than `self < other`.

        Args:
            other (BackendArg): Right-hand comparison operand.

        Returns:
            EagerTensorProtocol: Boolean comparison tensor.
        """
        ...

    def __le__(self, other: BackendArg) -> EagerTensorProtocol:
        """Compute elementwise less-than-or-equal `self <= other`.

        Args:
            other (BackendArg): Right-hand comparison operand.

        Returns:
            EagerTensorProtocol: Boolean comparison tensor.
        """
        ...

    def __gt__(self, other: BackendArg) -> EagerTensorProtocol:
        """Compute elementwise strict greater-than `self > other`.

        Args:
            other (BackendArg): Right-hand comparison operand.

        Returns:
            EagerTensorProtocol: Boolean comparison tensor.
        """
        ...

    def __ge__(self, other: BackendArg) -> EagerTensorProtocol:
        """Compute elementwise greater-than-or-equal `self >= other`.

        Args:
            other (BackendArg): Right-hand comparison operand.

        Returns:
            EagerTensorProtocol: Boolean comparison tensor.
        """
        ...
