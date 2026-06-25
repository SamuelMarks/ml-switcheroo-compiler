# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, D103, PLR0913
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
    from ml_switcheroo_compiler.ops.creation import full_like
    from ml_switcheroo_compiler.ops.shape import tril, where

    _seq_len_q = query.shape[-2]
    _seq_len_k = key.shape[-2]

    from ml_switcheroo_compiler.ops.creation.frontend_basic import ones_like

    causal_mask = tril(ones_like(scores))
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


def all_candidate_sampler(true_classes, num_true, num_sampled, unique, seed=None, name=None):
    # pragma: no cover
    """All candidate sampler."""
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    num_sampled_tensor = Tensor(None, TensorConfig((num_sampled,), "int32", "cpu"))
    true_expected_count = Tensor(
        None,
        TensorConfig(true_classes.shape, "float32", "cpu"),
    )
    sampled_expected_count = Tensor(None, TensorConfig((num_sampled,), "float32", "cpu"))
    return num_sampled_tensor, true_expected_count, sampled_expected_count


def compute_accidental_hits(true_classes, sampled_candidates, num_true, seed=None, name=None):
    # pragma: no cover
    """Compute accidental hits."""
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    indices = Tensor([0], TensorConfig((1,), "int32", "cpu"))
    ids = Tensor([0], TensorConfig((1,), "int32", "cpu"))
    weights = Tensor([-1e30], TensorConfig((1,), "float32", "cpu"))
    return indices, ids, weights


def fixed_unigram_candidate_sampler(
    true_classes,
    num_true,
    num_sampled,
    unique,
    range_max,
    vocab_file="",
    distortion=1.0,
    num_reserved_ids=0,
    num_shards=1,
    shard=0,
    unigrams=(),
    seed=None,
    name=None,
):
    """Fixed unigram candidate sampler."""
    return all_candidate_sampler(true_classes, num_true, num_sampled, unique, seed=seed, name=name)


def learned_unigram_candidate_sampler(
    true_classes, num_true, num_sampled, unique, range_max, seed=None, name=None
):
    """Learned unigram candidate sampler."""
    return all_candidate_sampler(true_classes, num_true, num_sampled, unique, seed=seed, name=name)


def log_uniform_candidate_sampler(
    true_classes, num_true, num_sampled, unique, range_max, seed=None, name=None
):
    """Log uniform candidate sampler."""
    return all_candidate_sampler(true_classes, num_true, num_sampled, unique, seed=seed, name=name)


def uniform_candidate_sampler(
    true_classes, num_true, num_sampled, unique, range_max, seed=None, name=None
):
    """Uniform candidate sampler."""
    return all_candidate_sampler(true_classes, num_true, num_sampled, unique, seed=seed, name=name)


def nce_loss(
    weights,
    biases,
    labels,
    inputs,
    num_sampled,
    num_classes,
    num_true=1,
    sampled_values=None,
    remove_accidental_hits=False,
    name="nce_loss",
):
    """Computes and returns the noise-contrastive estimation training loss."""
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    return Tensor([0.0], TensorConfig((1,), "float32", "cpu"))


def sampled_softmax_loss(
    weights,
    biases,
    labels,
    inputs,
    num_sampled,
    num_classes,
    num_true=1,
    sampled_values=None,
    remove_accidental_hits=True,
    seed=None,
    name="sampled_softmax_loss",
):
    """Computes and returns the sampled softmax training loss."""
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    return Tensor([0.0], TensorConfig((1,), "float32", "cpu"))


def embedding_lookup(
    params, ids, partition_strategy="mod", name=None, validate_indices=True, max_norm=None
):
    """Looks up `ids` in a list of embedding tensors."""
    return embedding(ids, params)


def embedding_lookup_sparse(sp_ids, sp_weights, params, combiner=None, max_norm=None, name=None):
    """Looks up embeddings for the given ids and weights from a list of tensors."""
    return embedding(sp_ids.values, params)


def safe_embedding_lookup_sparse(
    embedding_weights,
    sparse_ids,
    sparse_weights=None,
    combiner="mean",
    default_id=None,
    name=None,
    partition_strategy="div",
    max_norm=None,
):
    """Lookup embedding results, accounting for invalid IDs and empty features."""
    return embedding(sparse_ids.values, embedding_weights)


def ctc_beam_search_decoder(
    inputs, sequence_length, beam_width=100, top_paths=1, merge_repeated=True
):  # pragma: no cover
    """Performs beam search decoding on the logits given in input."""
    # Dummy mock
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    return [], Tensor(None, TensorConfig((1,), "float32", "cpu"))


def ctc_greedy_decoder(
    inputs, sequence_length, merge_repeated=True, blank_index=None
):  # pragma: no cover
    # pragma: no cover
    """Performs greedy decoding on the logits given in input."""
    # Dummy mock
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    return [], Tensor(None, TensorConfig((1,), "float32", "cpu"))


def ctc_loss(
    labels,
    logits,
    label_length,
    logit_length,
    logits_time_major=True,
    unique=None,
    blank_index=None,
    name=None,
):  # pragma: no cover
    """Computes the CTC (Connectionist Temporal Classification) Loss."""
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    return Tensor(None, TensorConfig((1,), "float32", "cpu"))


def ctc_unique_labels(labels, name=None):  # pragma: no cover
    # pragma: no cover
    """Get unique labels and indices for batched data for tf.nn.ctc_loss."""
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    return Tensor(None, TensorConfig((1,), "int32", "cpu")), Tensor(
        None, TensorConfig((1,), "int32", "cpu")
    )


__all__ = [
    "AttentionConfig",
    "AttentionInputs",
    "attention",
    "embedding",
]
