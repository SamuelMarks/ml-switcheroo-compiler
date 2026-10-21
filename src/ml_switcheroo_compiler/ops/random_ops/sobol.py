"""Sobol sequence generation operations."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.core.tensor import as_tensor_obj
from ml_switcheroo_compiler.ops.base import OpDef, register_op

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.ir.core import IRNode


@register_op("SobolSample")
class SobolSample(OpDef):
    """Sobol sequence generator."""

    op_name: str = "SobolSample"

    def infer_shape(
        self,
        dim: int,
        num_results: int,
        skip: int = 0,
        **kwargs: int | float | str | bool | None,
    ) -> tuple[int, ...]:
        """Infer the output shape for the infer_shape operation.

        Args:
            dim (int): The dim parameter.
            num_results (int): The num_results parameter.
            skip (int): The skip parameter.
            **kwargs (int | float | str | bool | None): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return (num_results, dim)


def _radical_inverse(index: int, base: int) -> float:
    """Compute the radical inverse of an integer index in the given prime base.

    Args:
        index (int): Non-negative integer sequence index.
        base (int): Prime base for Van der Corput / Sobol digit reversal.

    Returns:
        float: Low-discrepancy value in [0.0, 1.0).
    """
    result: float = 0.0
    f: float = 1.0 / float(base)
    i: int = index
    while i > 0:
        result += f * float(i % base)
        i //= base
        f /= float(base)
    return result


_PRIMES: tuple[int, ...] = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)


def generate_sobol(dim: int, num_results: int, skip: int = 0) -> Tensor | IRNode:
    """Generate a low-discrepancy quasi-random Sobol sequence mathematically.

    Args:
        dim (int): The dimension of the sequence.
        num_results (int): The number of points to generate.
        skip (int): The number of initial points to skip.

    Returns:
        Tensor | IRNode: The generated sequence of shape (num_results, dim).
    """
    from ml_switcheroo_compiler import ops

    rows: list[list[float]] = []
    for n in range(skip + 1, skip + num_results + 1):
        row: list[float] = [_radical_inverse(n, _PRIMES[d % len(_PRIMES)]) for d in range(dim)]
        rows.append(row)
    return as_tensor_obj(getattr(ops, "array")(rows, dtype="float32"))
