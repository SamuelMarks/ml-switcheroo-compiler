"""Attention operations."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import OpValue, extract_shape
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, scale_factor: Tensor) -> OpValue:
    """Scaled Dot-Product Attention.

    Args:
        q (Tensor): Queries.
        k (Tensor): Keys.
        v (Tensor): Values.
        scale_factor (Tensor): Scale factor.

    Returns:
        Tensor: Computed `Tensor` output.
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
        return Tensor(data, TensorConfig(extract_shape(data, extract_shape(q)), q.dtype, q.device))
    return _emit_shape_node("ScaledDotProductAttention", [q, k, v, scale_factor], {}, q.shape, q.dtype)


@register_op("ScaledDotProductAttention")
class ScaledDotProductAttention(OpDef):
    """Scaled Dot-Product Attention op."""

    op_name = "ScaledDotProductAttention"

    def infer_shape(
        self,
        q: OpValue,
        k: OpValue,
        v: OpValue,
        scale_factor: OpValue,
        **kwargs: OpValue,
    ) -> OpValue:
        """Infer shape.

        Args:
            q (OpValue): Input `q` argument (q).
            k (OpValue): Input `k` argument (k).
            v (OpValue): Input `v` argument (v).
            scale_factor (OpValue): The scale factor parameter.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        _typed_ret_1: OpValue = extract_shape(q)
        return _typed_ret_1
