"""Elementwise tensor equality and inequality dunder mixin."""

from __future__ import annotations

import typing
from collections.abc import Callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.core.shape import OpValue
    from ml_switcheroo_compiler.core.tensor import Tensor


class TensorComparisonMixin:
    """Mixin providing Python Array API elementwise `__eq__` and `__ne__` tensor operators."""

    _eq: Callable[[OpValue], Tensor]
    _ne: Callable[[OpValue], Tensor]

    def __hash__(self) -> int:
        """Compute identity-based hash of the tensor instance.

        Returns:
            int: Identity hash value.
        """
        return id(self)

    def __eq__(self, other: OpValue) -> Tensor:
        """Compare elementwise equality `self == other`.

        Args:
            other (OpValue): Right-hand operand.

        Returns:
            Tensor: A boolean tensor of equality results.
        """
        return self._eq(other)

    def __ne__(self, other: OpValue) -> Tensor:
        """Compare elementwise inequality `self != other`.

        Args:
            other (OpValue): Right-hand operand.

        Returns:
            Tensor: A boolean tensor of inequality results.
        """
        return self._ne(other)
