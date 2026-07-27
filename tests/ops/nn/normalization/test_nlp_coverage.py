# ruff: noqa: E501
import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.nlp import (
    NLPOpsConfig,
    all_candidate_sampler,
    compute_accidental_hits,
    ctc_beam_search_decoder,
    ctc_greedy_decoder,
    ctc_loss,
    ctc_unique_labels,
    fixed_unigram_candidate_sampler,
    learned_unigram_candidate_sampler,
    log_uniform_candidate_sampler,
    nce_loss,
    sampled_softmax_loss,
    uniform_candidate_sampler,
)


def test_nlp_coverage():
    config.eager_mode = True
    labels = Tensor(np.array([[1]]), TensorConfig(shape=(1, 1), dtype=DType("int32"), device=Device("cpu")))
    logits = Tensor(np.array([[[1.0, 2.0, 3.0]]]), TensorConfig(shape=(1, 1, 3), dtype=DType("float32"), device=Device("cpu")))
    seq_len = Tensor(np.array([1]), TensorConfig(shape=(1,), dtype=DType("int32"), device=Device("cpu")))

    cfg = NLPOpsConfig()
    assert all_candidate_sampler(labels, cfg) is not None
    assert compute_accidental_hits(labels, labels, cfg) is not None
    assert fixed_unigram_candidate_sampler(labels, cfg) is not None
    assert learned_unigram_candidate_sampler(labels, cfg) is not None
    assert log_uniform_candidate_sampler(labels, cfg) is not None
    assert uniform_candidate_sampler(labels, cfg) is not None

    w = Tensor(np.array([[1.0, 2.0, 3.0]]), TensorConfig(shape=(1, 3), dtype=DType("float32"), device=Device("cpu")))
    b = Tensor(np.array([1.0]), TensorConfig(shape=(1,), dtype=DType("float32"), device=Device("cpu")))

    assert nce_loss(w, b, labels, logits, 1) is not None
    assert sampled_softmax_loss(w, b, labels, logits, 1) is not None
    assert ctc_beam_search_decoder(logits, seq_len) is not None
    assert ctc_greedy_decoder(logits, seq_len) is not None
    assert ctc_loss(labels, logits, seq_len, seq_len) is not None
    assert ctc_unique_labels(labels) is not None
