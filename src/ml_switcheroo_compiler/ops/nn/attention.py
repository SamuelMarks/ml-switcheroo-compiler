"""Attention operations."""

from typing import Any

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, scale_factor: Tensor) -> Any:  # type: ignore
    """Scaled Dot-Product Attention.

    Args:
        q (Tensor): Queries.
        k (Tensor): Keys.
        v (Tensor): Values.
        scale_factor (Tensor): Scale factor.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op(
            "ScaledDotProductAttention",
            (q.data if type(q).__name__ == "Tensor" else q),
            (k.data if type(k).__name__ == "Tensor" else k),
            (v.data if type(v).__name__ == "Tensor" else v),
            (scale_factor.data if type(scale_factor).__name__ == "Tensor" else scale_factor),
        )
        return Tensor(data, TensorConfig(getattr(data, "shape", q.shape), q.dtype, q.device))  # type: ignore
    return _emit_shape_node("ScaledDotProductAttention", [q, k, v, scale_factor], {}, q.shape, q.dtype)


@register_op("ScaledDotProductAttention")
class ScaledDotProductAttention(OpDef):
    """Scaled Dot-Product Attention op."""

    op_name = "ScaledDotProductAttention"

    def infer_shape(self, q: Any, k: Any, v: Any, scale_factor: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            q (object): The q parameter.
            k (object): The k parameter.
            v (object): The v parameter.
            scale_factor (object): The scale factor parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return getattr(q, "shape", ())
