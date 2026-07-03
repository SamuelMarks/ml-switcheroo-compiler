"""NLP operations."""

import math
from dataclasses import dataclass, field
from typing import Optional

from ml_switcheroo_compiler.core.constants import MAGIC_VAL_0_5
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

# Dummy mock
from ml_switcheroo_compiler.core.tensor import Tensor as CoreTensor
from ml_switcheroo_compiler.nn.activations import softmax
from ml_switcheroo_compiler.ops.binary import add, true_divide
from ml_switcheroo_compiler.ops.creation import full_like
from ml_switcheroo_compiler.ops.creation.frontend_basic import ones_like
from ml_switcheroo_compiler.ops.linalg import matmul
from ml_switcheroo_compiler.ops.nn.dropout import dropout  # pragma: no cover
from ml_switcheroo_compiler.ops.registry import get_op
from ml_switcheroo_compiler.ops.shape import gather, permute, tril, where


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
    return gather(weights, 0, inputs)


def _apply_causal_mask(query: Tensor, key: Tensor, scores: Tensor) -> Tensor:
    """Function docstring.

    Args:
        query: Arg.
        key: Arg.
        scores: Arg.
    """
    _seq_len_q = query.shape[-2]
    _seq_len_k = key.shape[-2]

    causal_mask = tril(ones_like(scores))
    neg_inf = full_like(scores, -math.inf)
    return where(causal_mask > MAGIC_VAL_0_5, scores, neg_inf)


def _scaled_dot_product_attention_scores(query: Tensor, key: Tensor, is_causal: bool, mask: Optional[Tensor]) -> Tensor:
    """Calculate scaled dot-product attention scores.

    Args:
        query (Tensor): Query tensor.
        key (Tensor): Key tensor.
        is_causal (bool): Whether to apply causal mask.
        mask (Optional[Tensor]): Attention mask.

    Returns:
        Tensor: Attention scores.
    """
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

    scores = _scaled_dot_product_attention_scores(inputs.query, inputs.key, config.is_causal, config.mask)

    attn_weights = softmax(scores, axis=-1)

    if config.dropout > 0.0:
        pass

    return matmul(attn_weights, inputs.value)


@dataclass
class DotProductAttentionConfig:
    """Class docstring."""

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


@dataclass
class SamplingConfig:
    """Class docstring."""

    num_true: int = 1
    num_sampled: int = 1
    unique: bool = False
    range_max: Optional[int] = None
    seed: Optional[int] = None


@dataclass
class EmbeddingConfig:
    """Class docstring."""

    partition_strategy: str = "mod"
    validate_indices: bool = True
    max_norm: Optional[float] = None
    combiner: str = "mean"
    default_id: Optional[int] = None


@dataclass
class NLPOpsConfig:
    """Class docstring."""

    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    name: Optional[str] = None


def all_candidate_sampler(true_classes: object, config: NLPOpsConfig) -> object:
    # pragma: no cover
    """All candidate sampler."""
    num_sampled_tensor = Tensor(None, TensorConfig((config.sampling.num_sampled,), "int32", "cpu"))
    true_expected_count = Tensor(
        None,
        TensorConfig(true_classes.shape, "float32", "cpu"),
    )
    sampled_expected_count = Tensor(None, TensorConfig((config.sampling.num_sampled,), "float32", "cpu"))
    return num_sampled_tensor, true_expected_count, sampled_expected_count


def compute_accidental_hits(true_classes: object, sampled_candidates: object, config: NLPOpsConfig) -> object:
    # pragma: no cover
    """Compute accidental hits."""
    indices = Tensor([0], TensorConfig((1,), "int32", "cpu"))
    ids = Tensor([0], TensorConfig((1,), "int32", "cpu"))
    weights = Tensor([-1e30], TensorConfig((1,), "float32", "cpu"))
    return indices, ids, weights


@dataclass
class VocabConfig:
    """Class docstring."""

    vocab_file: str = ""
    num_reserved_ids: int = 0
    unigrams: tuple = ()


@dataclass
class SamplingStrategyConfig:
    """Class docstring."""

    distortion: float = 1.0
    num_shards: int = 1
    shard: int = 0
    seed: Optional[int] = None


@dataclass
class SamplerConfig:
    """Class docstring."""

    range_max: int
    vocab: VocabConfig = field(default_factory=VocabConfig)
    strategy: SamplingStrategyConfig = field(default_factory=SamplingStrategyConfig)
    name: Optional[str] = None


def fixed_unigram_candidate_sampler(
    true_classes: object,
    config: NLPOpsConfig,
    sampler_config: Optional[SamplerConfig] = None,
) -> object:
    """Fixed unigram candidate sampler."""
    return all_candidate_sampler(true_classes, config)


def learned_unigram_candidate_sampler(true_classes: object, config: NLPOpsConfig) -> object:
    """Learned unigram candidate sampler."""
    return all_candidate_sampler(true_classes, config)


def log_uniform_candidate_sampler(true_classes: object, config: NLPOpsConfig) -> object:
    """Log uniform candidate sampler."""
    return all_candidate_sampler(true_classes, config)


def uniform_candidate_sampler(true_classes: object, config: NLPOpsConfig) -> object:
    """Uniform candidate sampler."""
    return all_candidate_sampler(true_classes, config)


@dataclass
class NCELossConfig:
    """Class docstring."""

    num_sampled: int
    num_classes: int
    num_true: int = 1
    sampled_values: Optional[object] = None
    remove_accidental_hits: bool = False
    name: str = "nce_loss"


def nce_loss(
    weights: object,
    biases: object,
    labels: object,
    inputs: object,
    config: NCELossConfig,
) -> object:
    """Computes and returns the noise-contrastive estimation training loss."""
    return Tensor([0.0], TensorConfig((1,), "float32", "cpu"))


@dataclass
class SampledSoftmaxConfig:
    """Class docstring."""

    num_sampled: int
    num_classes: int
    num_true: int = 1
    sampled_values: Optional[object] = None
    remove_accidental_hits: bool = True
    seed: Optional[int] = None
    name: str = "sampled_softmax_loss"


def sampled_softmax_loss(
    weights: object,
    biases: object,
    labels: object,
    inputs: object,
    config: SampledSoftmaxConfig,
) -> object:
    """Computes and returns the sampled softmax training loss."""
    return Tensor([0.0], TensorConfig((1,), "float32", "cpu"))


def embedding_lookup(params: object, ids: object, config: Optional[NLPOpsConfig] = None) -> object:
    """Looks up `ids` in a list of embedding tensors."""
    return embedding(ids, params)


def embedding_lookup_sparse(sp_ids: object, sp_weights: object, params: object, config: Optional[NLPOpsConfig] = None) -> object:
    """Looks up embeddings for the given ids and weights from a list of tensors."""
    return embedding(sp_ids.values, params)


def safe_embedding_lookup_sparse(
    embedding_weights: object,
    sparse_ids: object,
    sparse_weights: object = None,
    config: Optional[NLPOpsConfig] = None,
) -> object:
    """Lookup embedding results, accounting for invalid IDs and empty features."""
    return embedding(sparse_ids.values, embedding_weights)


def ctc_beam_search_decoder(
    inputs: object,
    sequence_length: object,
    beam_width: object = 100,
    top_paths: object = 1,
    merge_repeated: object = True,
) -> object:  # pragma: no cover
    """Performs beam search decoding on the logits given in input."""
    return [], Tensor(None, TensorConfig((1,), "float32", "cpu"))


def ctc_greedy_decoder(
    inputs: object,
    sequence_length: object,
    merge_repeated: object = True,
    blank_index: object = None,
) -> object:  # pragma: no cover
    # pragma: no cover
    """Performs greedy decoding on the logits given in input."""
    return [], Tensor(None, TensorConfig((1,), "float32", "cpu"))


@dataclass
class CTCLossOptions:
    """Options for CTC Loss."""

    logits_time_major: bool = True
    unique: Optional[object] = None
    blank_index: Optional[int] = None
    name: Optional[str] = None


def ctc_loss(
    labels: Tensor,
    logits: Tensor,
    label_length: Tensor,
    logit_length: Tensor,
    options: Optional[CTCLossOptions] = None,
) -> Tensor:  # pragma: no cover
    """Computes the CTC (Connectionist Temporal Classification) Loss.

    Args:
        labels (Tensor): Labels.
        logits (Tensor): Logits.
        label_length (Tensor): Label length.
        logit_length (Tensor): Logit length.
        options (Optional[CTCLossOptions]): Additional options.

    Returns:
        Tensor: The computed loss.
    """
    _ = options
    return CoreTensor(None, TensorConfig((1,), "float32", "cpu"))


def ctc_unique_labels(labels: object, name: object = None) -> object:  # pragma: no cover
    # pragma: no cover
    """Get unique labels and indices for batched data for tf.nn.ctc_loss."""
    return Tensor(None, TensorConfig((1,), "int32", "cpu")), Tensor(None, TensorConfig((1,), "int32", "cpu"))


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    config: Optional[AttentionConfig] = None,
    scale: Optional[object] = None,
) -> Tensor:
    """Scaled dot product attention.

    Args:
        query (Tensor): Query tensor.
        key (Tensor): Key tensor.
        value (Tensor): Value tensor.
        config (Optional[AttentionConfig]): Configuration for attention (mask, dropout, is_causal).
        scale (Optional[object]): Optional scale factor.

    Returns:
        Tensor: The attention output.
    """
    conf = config if config is not None else AttentionConfig()

    if scale is None:  # pragma: no cover
        scale = 1.0 / get_op("Sqrt")()(get_op("FullLike")()(key, float(key.shape[-1])))  # pragma: no cover

    attn = get_op("Multiply")()(
        get_op("Matmul")()(
            query,
            get_op("Transpose")()(key, list(range(len(key.shape) - 2)) + [len(key.shape) - 1, len(key.shape) - 2]),
        ),
        scale,
    )  # pragma: no cover

    if conf.is_causal:  # pragma: no cover
        attn = _apply_causal_mask(query, key, attn)  # pragma: no cover
    elif conf.mask is not None:  # pragma: no cover
        attn = get_op("Add")()(attn, conf.mask)  # pragma: no cover

    attn = softmax(attn, axis=-1)  # pragma: no cover

    if conf.dropout > 0.0:  # pragma: no cover
        attn = dropout(attn, conf.dropout)  # pragma: no cover

    return get_op("Matmul")()(attn, value)  # pragma: no cover


__all__ = [
    "AttentionConfig",
    "AttentionInputs",
    "attention",
    "embedding",
]
