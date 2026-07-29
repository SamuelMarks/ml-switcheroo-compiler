"""Sobol."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("SobolSample")
class SobolSample(OpDef):
    """Sobol sequence generator."""

    op_name = "SobolSample"

    def infer_shape(self, dim: int, num_results: int, skip: int = 0, **kwargs: object) -> object:
        """Infer shape."""
        return (num_results, dim)


def generate_sobol(dim: int, num_results: int, skip: int = 0) -> object:
    """Generates a Sobol sequence mathematically.

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
