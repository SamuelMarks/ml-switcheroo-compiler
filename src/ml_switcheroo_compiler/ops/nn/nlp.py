"""NLP operations."""

from typing import Optional


from ml_switcheroo_compiler.core.tensor import Tensor


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


def attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Optional[Tensor] = None,
    dropout: float = 0.0,
    is_causal: bool = False,
) -> Tensor:
    """Scaled dot-product attention.

    Args:
        query (Tensor): Query tensor.
        key (Tensor): Key tensor.
        value (Tensor): Value tensor.
        mask (Optional[Tensor]): Attention mask.
        dropout (float): Dropout rate.
        is_causal (bool): Whether to apply causal mask.

    Returns:
        Tensor: The attention output.
    """
    import math
    from ml_switcheroo_compiler.ops.linalg import matmul
    from ml_switcheroo_compiler.ops.shape import permute
    from ml_switcheroo_compiler.ops.binary import true_divide, add
    from ml_switcheroo_compiler.nn.activations import softmax

    # Query: (..., seq_len_q, depth)
    # Key: (..., seq_len_k, depth)
    # Value: (..., seq_len_v, depth_v)

    depth = query.shape[-1]

    # Transpose key for matmul: (..., depth, seq_len_k)
    dims = list(range(len(key.shape)))
    dims[-1], dims[-2] = dims[-2], dims[-1]
    key_t = permute(key, tuple(dims))

    # scores: (..., seq_len_q, seq_len_k)
    scores = matmul(query, key_t)
    scores = true_divide(scores, math.sqrt(float(depth)))

    if is_causal:
        # Generate causal mask
        from ml_switcheroo_compiler.ops.creation import ones
        from ml_switcheroo_compiler.ops.shape import tril

        seq_len_q = query.shape[-2]
        seq_len_k = key.shape[-2]

        causal_mask = tril(ones((seq_len_q, seq_len_k), dtype=query.dtype))
        # Mask out upper triangle with -inf
        from ml_switcheroo_compiler.ops.shape import where
        from ml_switcheroo_compiler.ops.creation import full_like

        neg_inf = full_like(scores, -math.inf)
        scores = where(causal_mask > 0.5, scores, neg_inf)

    if mask is not None:
        scores = add(scores, mask)

    attn_weights = softmax(scores, axis=-1)

    if dropout > 0.0:
        # Dropout logic can be complex in pure ops. We skip the actual stochastic
        # dropout here unless we want to inject random logic. We'll leave it as a no-op
        # or simplified version if tracing.
        pass

    return matmul(attn_weights, value)


__all__ = ["embedding", "attention"]
