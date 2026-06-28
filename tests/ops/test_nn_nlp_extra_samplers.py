from ml_switcheroo_compiler.ops.nn.nlp import (
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
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
import numpy as np


def test_samplers_and_losses():
    device = Device("cpu")
    t = Tensor(np.ones((2,)), TensorConfig((2,), DType.Int32, device))
    t_f = Tensor(np.ones((2,)), TensorConfig((2,), DType.Float32, device))

    all_candidate_sampler(t, 1, 1, True)
    compute_accidental_hits(t, t, 1)

    fixed_unigram_candidate_sampler(t, 1, 1, True, 10)
    learned_unigram_candidate_sampler(t, 1, 1, True, 10)
    log_uniform_candidate_sampler(t, 1, 1, True, 10)
    uniform_candidate_sampler(t, 1, 1, True, 10)

    nce_loss(t_f, t_f, t, t_f, 1, 10)
    sampled_softmax_loss(t_f, t_f, t, t_f, 1, 10)


def test_embeddings():
    device = Device("cpu")
    t = Tensor(np.ones((2,)), TensorConfig((2,), DType.Int32, device))
    t_f = Tensor(np.ones((2,)), TensorConfig((2,), DType.Float32, device))

    from ml_switcheroo_compiler.core.config import ConfigContext
    from unittest.mock import patch

    with ConfigContext(eager_mode=True):
        with patch("ml_switcheroo_compiler.ops.shape.gather") as mock_gather:
            mock_gather.return_value = "res"
            embedding_lookup(t_f, t)

            class MockRagged:
                def __init__(self, values):
                    self.values = values

            r = MockRagged(t)
            embedding_lookup_sparse(r, t_f, t_f)
            safe_embedding_lookup_sparse(t_f, r)
