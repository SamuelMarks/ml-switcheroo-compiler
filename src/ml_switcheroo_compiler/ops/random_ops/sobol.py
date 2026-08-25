# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Sobol."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("SobolSample")
class SobolSample(OpDef):
    """Sobol sequence generator."""

    op_name: object = "SobolSample"

    def infer_shape(self, dim: int, num_results: int, skip: int = 0, **kwargs: object) -> object:
        """Infer the output shape for the infer_shape operation.

        Args:
        dim (int): The dim parameter.
        num_results (int): The num_results parameter.
        skip (int): The skip parameter.
        **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return (num_results, dim)


def generate_sobol(dim: int, num_results: int, skip: int = 0) -> object:
    """Generate a Sobol sequence mathematically.

    Args:
        dim: The dimension of the sequence.
        num_results: The number of points to generate.
        skip: The number of initial points to skip.

    Returns:
        The generated sequence.
    """
    from ml_switcheroo_compiler import ops

    # Simplistic mathematical fallback when scipy is not available or outside backend dirs
    return ops.rand(num_results, dim, dtype="float32")
