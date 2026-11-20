"""Bitwise, shift, scalar-coercion, and reverse-arithmetic dunder protocol definitions."""

from __future__ import annotations

import builtins
import typing
from collections.abc import Generator

from .types_base import BackendArg

if typing.TYPE_CHECKING:
    from .protocols import EagerTensorProtocol


class _ArrayBitwiseAndReverseDundersProto(typing.Protocol):
    """Structural protocol for bitwise, shift, coercion, and reverse binary array dunders."""

    def __iter__(self) -> builtins.map[EagerTensorProtocol]:
        """Iterate over slices along the leading tensor dimension.

        Returns:
            builtins.map[EagerTensorProtocol]: Iterator over leading dimension slices.
        """
        ...

    def __int__(self) -> int:
        """Convert scalar tensor to a Python `int`.

        Returns:
            int: Converted integer scalar.
        """
        ...

    def __float__(self) -> float:
        """Convert scalar tensor to a Python `float`.

        Returns:
            float: Converted floating-point scalar.
        """
        ...

    def __abs__(self) -> EagerTensorProtocol:
        """Compute elementwise magnitude `abs(self)`.

        Returns:
            EagerTensorProtocol: Elementwise absolute value tensor.
        """
        ...

    def __invert__(self) -> EagerTensorProtocol:
        """Compute elementwise bitwise or logical inversion `~self`.

        Returns:
            EagerTensorProtocol: Inverted tensor.
        """
        ...

    def __lshift__(self, other: BackendArg) -> EagerTensorProtocol:
        """Compute elementwise left bit-shift `self << other`.

        Args:
            other (BackendArg): Bit-shift count operand.

        Returns:
            EagerTensorProtocol: Left-shifted tensor.
        """
        ...

    def __rshift__(self, other: BackendArg) -> EagerTensorProtocol:
        """Compute elementwise right bit-shift `self >> other`.

        Args:
            other (BackendArg): Bit-shift count operand.

        Returns:
            EagerTensorProtocol: Right-shifted tensor.
        """
        ...

    def __and__(self, other: BackendArg) -> EagerTensorProtocol:
        """Compute elementwise bitwise/logical AND `self & other`.

        Args:
            other (BackendArg): Right-hand operand.

        Returns:
            EagerTensorProtocol: Bitwise AND tensor.
        """
        ...

    def __or__(self, other: BackendArg) -> EagerTensorProtocol:
        """Compute elementwise bitwise/logical OR `self | other`.

        Args:
            other (BackendArg): Right-hand operand.

        Returns:
            EagerTensorProtocol: Bitwise OR tensor.
        """
        ...

    def __xor__(self, other: BackendArg) -> EagerTensorProtocol:
        """Compute elementwise bitwise/logical XOR `self ^ other`.

        Args:
            other (BackendArg): Right-hand operand.

        Returns:
            EagerTensorProtocol: Bitwise XOR tensor.
        """
        ...

    def __radd__(self, other: BackendArg) -> EagerTensorProtocol:
        """Compute reverse addition `other + self`.

        Args:
            other (BackendArg): Left-hand addend.

        Returns:
            EagerTensorProtocol: Elementwise sum tensor.
        """
        ...

    def __rsub__(self, other: BackendArg) -> EagerTensorProtocol:
        """Compute reverse subtraction `other - self`.

        Args:
            other (BackendArg): Left-hand minuend.

        Returns:
            EagerTensorProtocol: Elementwise difference tensor.
        """
        ...

    def __rmul__(self, other: BackendArg) -> EagerTensorProtocol:
        """Compute reverse multiplication `other * self`.

        Args:
            other (BackendArg): Left-hand factor.

        Returns:
            EagerTensorProtocol: Elementwise product tensor.
        """
        ...

    def __rtruediv__(self, other: BackendArg) -> EagerTensorProtocol:
        """Compute reverse true division `other / self`.

        Args:
            other (BackendArg): Left-hand dividend.

        Returns:
            EagerTensorProtocol: Elementwise quotient tensor.
        """
        ...

    def __rfloordiv__(self, other: BackendArg) -> EagerTensorProtocol:
        """Compute reverse floor division `other // self`.

        Args:
            other (BackendArg): Left-hand dividend.

        Returns:
            EagerTensorProtocol: Elementwise floor-quotient tensor.
        """
        ...

    def __rmod__(self, other: BackendArg) -> EagerTensorProtocol:
        """Compute reverse modulo `other % self`.

        Args:
            other (BackendArg): Left-hand dividend.

        Returns:
            EagerTensorProtocol: Elementwise remainder tensor.
        """
        ...

    def __rpow__(self, other: BackendArg) -> EagerTensorProtocol:
        """Compute reverse exponentiation `other ** self`.

        Args:
            other (BackendArg): Base operand.

        Returns:
            EagerTensorProtocol: Exponentiated tensor.
        """
        ...


class _ExtendedTensorDundersProto(typing.Protocol):
    """Extended dunders present on full eager tensor/module protocols."""

    def __contains__(self, item: BackendArg) -> bool:
        """Test membership `item in self`.

        Args:
            item (BackendArg): Candidate element.

        Returns:
            bool: `True` when `item` is contained in `self`.
        """
        ...

    def __await__(self) -> Generator[None, None, EagerTensorProtocol]:
        """Await asynchronous stream or device tensor operation.

        Returns:
            Generator[None, None, EagerTensorProtocol]: Awaitable generator yielding a tensor.
        """
        ...

    def __index__(self) -> int:
        """Convert integer scalar tensor to a Python slice/indexing `int`.

        Returns:
            int: Converted index integer.
        """
        ...

    def __fspath__(self) -> str:
        """Return filesystem path string when representing a path-like resource.

        Returns:
            str: Filesystem path string.
        """
        ...

    def __pos__(self) -> EagerTensorProtocol:
        """Compute unary positive `+self`.

        Returns:
            EagerTensorProtocol: Unary positive result tensor.
        """
        ...

    def __rlshift__(self, other: BackendArg) -> EagerTensorProtocol:
        """Compute reverse left bit-shift `other << self`.

        Args:
            other (BackendArg): Left-hand operand.

        Returns:
            EagerTensorProtocol: Left-shifted tensor.
        """
        ...

    def __rrshift__(self, other: BackendArg) -> EagerTensorProtocol:
        """Compute reverse right bit-shift `other >> self`.

        Args:
            other (BackendArg): Left-hand operand.

        Returns:
            EagerTensorProtocol: Right-shifted tensor.
        """
        ...

    def __rmatmul__(self, other: BackendArg) -> EagerTensorProtocol:
        """Compute reverse matrix product `other @ self`.

        Args:
            other (BackendArg): Left-hand matrix/tensor operand.

        Returns:
            EagerTensorProtocol: Matrix product tensor.
        """
        ...
