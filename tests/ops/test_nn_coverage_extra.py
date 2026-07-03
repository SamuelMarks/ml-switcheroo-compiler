"""Module docstring."""

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.ops.creation.frontend import array
from ml_switcheroo_compiler.ops.nn.nlp import (
    NCELossConfig,
    NLPOpsConfig,
    SampledSoftmaxConfig,
    SamplerConfig,
    SamplingConfig,
    all_candidate_sampler,
    compute_accidental_hits,
    embedding_lookup,
    embedding_lookup_sparse,
    fixed_unigram_candidate_sampler,
    learned_unigram_candidate_sampler,
    log_uniform_candidate_sampler,
    nce_loss,
    safe_embedding_lookup_sparse,
    sampled_softmax_loss,
    uniform_candidate_sampler,
)
from ml_switcheroo_compiler.ops.nn.rnn_utils import (
    DropoutWrapperConfig,
    RNNCellDeviceWrapper,
    RNNCellDropoutWrapper,
    RNNCellResidualWrapper,
)


def test_nn_wrappers() -> object:
    """Function docstring."""
    config.eager_mode = True

    class DummyCell:
        """Class docstring."""

        def __call__(self, x: object, state: object, **kwargs: object) -> object:
            """Function docstring."""
            return x, state

    cell = DummyCell()
    wrapped1 = RNNCellDeviceWrapper(cell, "cpu")
    out, st = wrapped1(array(1.0), array(0.0))

    wrapped2 = RNNCellDropoutWrapper(cell, config=DropoutWrapperConfig(input_keep_prob=0.5, output_keep_prob=0.5))
    out, st = wrapped2(array(1.0), array(0.0))

    wrapped3 = RNNCellResidualWrapper(cell)
    out, st = wrapped3(array(1.0), array(0.0))
    config.eager_mode = False


def test_nn_samplers_and_losses() -> object:
    """Function docstring."""
    config.eager_mode = True
    # Because these return dummy mock values, we just verify they execute correctly without raising
    t1 = array(np.array([1, 2]))
    t2 = array(np.array([0, 1]))

    nlp_conf = NLPOpsConfig(sampling=SamplingConfig(num_true=1, num_sampled=1, unique=True, range_max=10))

    _ = all_candidate_sampler(t1, config=nlp_conf)
    _ = compute_accidental_hits(t1, t2, config=nlp_conf)
    _ = fixed_unigram_candidate_sampler(t1, config=nlp_conf, sampler_config=SamplerConfig(range_max=10))
    _ = learned_unigram_candidate_sampler(t1, config=nlp_conf)
    _ = log_uniform_candidate_sampler(t1, config=nlp_conf)
    _ = uniform_candidate_sampler(t1, config=nlp_conf)

    # losses
    _ = nce_loss(t1, t1, t1, t1, config=NCELossConfig(num_sampled=1, num_classes=1))
    _ = sampled_softmax_loss(t1, t1, t1, t1, config=SampledSoftmaxConfig(num_sampled=1, num_classes=1))

    class SparseMock:
        """Class docstring."""

        def __init__(self, val: object) -> object:
            """Function docstring."""
            self.values = val

    _ = embedding_lookup(t1, t2)
    _ = embedding_lookup_sparse(SparseMock(t2), None, t1)
    _ = safe_embedding_lookup_sparse(t1, SparseMock(t2))
    config.eager_mode = False
