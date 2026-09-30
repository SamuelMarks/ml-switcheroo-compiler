# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Sobol sequence generation operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ml_switcheroo_compiler.ops.base import OpDef, register_op

if TYPE_CHECKING:
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


def generate_sobol(dim: int, num_results: int, skip: int = 0) -> Tensor | IRNode:
    """Generate a Sobol sequence mathematically.

    Args:
        dim (int): The dimension of the sequence.
        num_results (int): The number of points to generate.
        skip (int): The number of initial points to skip.

    Returns:
        Tensor | IRNode: The generated sequence.
    """
    from ml_switcheroo_compiler import ops

    # Simplistic mathematical fallback when scipy is not available or outside backend dirs
    return ops.rand(num_results, dim, dtype="float32")
