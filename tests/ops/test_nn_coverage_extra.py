import numpy as np
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.ops import (
    all_candidate_sampler,
    compute_accidental_hits,
    fixed_unigram_candidate_sampler,
    learned_unigram_candidate_sampler,
    log_uniform_candidate_sampler,
    uniform_candidate_sampler,
    nce_loss,
    sampled_softmax_loss,
    embedding_lookup,
    embedding_lookup_sparse,
    safe_embedding_lookup_sparse,
    RNNCellDeviceWrapper,
    RNNCellDropoutWrapper,
    RNNCellResidualWrapper,
    array,
)


def test_nn_wrappers():
    config.eager_mode = True

    class DummyCell:
        def __call__(self, x, state, **kwargs):
            return x, state

    cell = DummyCell()
    wrapped1 = RNNCellDeviceWrapper(cell, "cpu")
    out, st = wrapped1(array(1.0), array(0.0))

    wrapped2 = RNNCellDropoutWrapper(cell, input_keep_prob=0.5, output_keep_prob=0.5)
    out, st = wrapped2(array(1.0), array(0.0))

    wrapped3 = RNNCellResidualWrapper(cell)
    out, st = wrapped3(array(1.0), array(0.0))
    config.eager_mode = False


def test_nn_samplers_and_losses():
    config.eager_mode = True
    # Because these return dummy mock values, we just verify they execute correctly without raising
    t1 = array(np.array([1, 2]))
    t2 = array(np.array([0, 1]))

    _ = all_candidate_sampler(t1, 1, 1, True)
    _ = compute_accidental_hits(t1, t2, 1)
    _ = fixed_unigram_candidate_sampler(t1, 1, 1, True, 10)
    _ = learned_unigram_candidate_sampler(t1, 1, 1, True, 10)
    _ = log_uniform_candidate_sampler(t1, 1, 1, True, 10)
    _ = uniform_candidate_sampler(t1, 1, 1, True, 10)

    # losses
    _ = nce_loss(t1, t1, t1, t1, 1, 1)
    _ = sampled_softmax_loss(t1, t1, t1, t1, 1, 1)

    class SparseMock:
        def __init__(self, val):
            self.values = val

    _ = embedding_lookup(t1, t2)
    _ = embedding_lookup_sparse(SparseMock(t2), None, t1)
    _ = safe_embedding_lookup_sparse(t1, SparseMock(t2))
    config.eager_mode = False
