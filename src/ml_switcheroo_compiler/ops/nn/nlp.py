"""NLP operations."""

from ml_switcheroo_compiler.core.constants import MAGIC_VAL_0_5

from dataclasses import dataclass
from typing import Optional

from ml_switcheroo_compiler.core.tensor import Tensor


@dataclass
class AttentionInputs:
    """Inputs for attention.

    Attributes:
        query (Tensor): Query tensor.
        key (Tensor): Key tensor.
        value (Tensor): Value tensor.
    """

    query: Tensor
    key: Tensor
    value: Tensor


@dataclass
class AttentionConfig:
    """Configuration for attention.

    Attributes:
        mask (Optional[Tensor]): Attention mask. Defaults to None.
        dropout (float): Dropout rate. Defaults to 0.0.
        is_causal (bool): Whether to apply causal mask. Defaults to False.
    """

    mask: Optional[Tensor] = None
    dropout: float = 0.0
    is_causal: bool = False


def embedding(
    inputs: Tensor,
    weights: Tensor,
) -> Tensor:
    """Embedding lookup.

    Args:
        inputs (Tensor): The input indices.
        weights (Tensor): The embedding weights.

    Returns:
        Tensor: The embeddings.
    """
    from ml_switcheroo_compiler.ops.shape import gather

    return gather(weights, 0, inputs)


def _apply_causal_mask(query: Tensor, key: Tensor, scores: Tensor) -> Tensor:
    """Function docstring.

    Args:
        query: Arg.
        key: Arg.
        scores: Arg.
    """
    import math
    from ml_switcheroo_compiler.ops.creation import ones, full_like
    from ml_switcheroo_compiler.ops.shape import tril, where

    seq_len_q = query.shape[-2]
    seq_len_k = key.shape[-2]

    causal_mask = tril(ones((seq_len_q, seq_len_k), dtype=query.dtype))
    neg_inf = full_like(scores, -math.inf)
    return where(causal_mask > MAGIC_VAL_0_5, scores, neg_inf)


def _scaled_dot_product_attention_scores(
    query: Tensor, key: Tensor, is_causal: bool, mask: Optional[Tensor]
) -> Tensor:
    """Calculate scaled dot-product attention scores.

    Args:
        query (Tensor): Query tensor.
        key (Tensor): Key tensor.
        is_causal (bool): Whether to apply causal mask.
        mask (Optional[Tensor]): Attention mask.

    Returns:
        Tensor: Attention scores.
    """
    import math
    from ml_switcheroo_compiler.ops.binary import add, true_divide
    from ml_switcheroo_compiler.ops.linalg import matmul
    from ml_switcheroo_compiler.ops.shape import permute

    depth = query.shape[-1]
    dims = list(range(len(key.shape)))
    dims[-1], dims[-2] = dims[-2], dims[-1]
    key_t = permute(key, tuple(dims))

    scores = matmul(query, key_t)
    scores = true_divide(scores, math.sqrt(float(depth)))

    if is_causal:
        scores = _apply_causal_mask(query, key, scores)

    if mask is not None:
        scores = add(scores, mask)

    return scores


def attention(
    inputs: AttentionInputs,
    config: Optional[AttentionConfig] = None,
) -> Tensor:
    """Scaled dot-product attention.

    Args:
        inputs (AttentionInputs): The Q, K, V tensors.
        config (Optional[AttentionConfig]): Configuration object for attention.

    Returns:
        Tensor: The attention output.
    """
    if config is None:
        config = AttentionConfig()

    from ml_switcheroo_compiler.nn.activations import softmax
    from ml_switcheroo_compiler.ops.linalg import matmul

    scores = _scaled_dot_product_attention_scores(
        inputs.query, inputs.key, config.is_causal, config.mask
    )

    attn_weights = softmax(scores, axis=-1)

    if config.dropout > 0.0:
        pass

    return matmul(attn_weights, inputs.value)


@dataclass
class DotProductAttentionConfig:
    mask: object = None
    scale: float = None
    dropout_rate: float = 0.0
    seed: object = None
    training: bool = False


def dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    config: Optional[DotProductAttentionConfig] = None,
) -> Tensor:
    """Computes dot-product attention."""
    # Simplified dot product attention wrapper using the existing attention function
    # It assumes the default attention handles dot product.
    return attention(
        AttentionInputs(query=query, key=key, value=value),
        config=AttentionConfig(
            mask=config.mask if config else None,
            dropout=config.dropout_rate if config else 0.0,
            is_causal=False,
        ),
    )


__all__ = [
    "AttentionConfig",
    "AttentionInputs",
    "attention",
    "embedding",
]
