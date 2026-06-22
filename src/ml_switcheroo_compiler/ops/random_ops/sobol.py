"""Sobol."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("SobolSample")
class SobolSample(OpDef):
    """Sobol sequence generator."""

    op_name = "SobolSample"

    def infer_shape(self, dim: int, num_results: int, skip: int = 0, **kwargs: object) -> object:
        """Infer shape."""
        return (num_results, dim)
