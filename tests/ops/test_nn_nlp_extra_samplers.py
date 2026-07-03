"""Module docstring."""

from unittest.mock import patch

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
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


def test_samplers_and_losses() -> object:
    """Function docstring."""
    device = Device("cpu")
    t = Tensor(np.ones((2,)), TensorConfig((2,), DType.Int32, device))
    t_f = Tensor(np.ones((2,)), TensorConfig((2,), DType.Float32, device))

    config = NLPOpsConfig(sampling=SamplingConfig(num_true=1, num_sampled=1, unique=True, range_max=10))

    all_candidate_sampler(t, config)
    compute_accidental_hits(t, t, config)

    fixed_unigram_candidate_sampler(t, config=config, sampler_config=SamplerConfig(range_max=10))
    learned_unigram_candidate_sampler(t, config)
    log_uniform_candidate_sampler(t, config)
    uniform_candidate_sampler(t, config)

    nce_loss(t_f, t_f, t, t_f, config=NCELossConfig(num_sampled=1, num_classes=10))
    sampled_softmax_loss(t_f, t_f, t, t_f, config=SampledSoftmaxConfig(num_sampled=1, num_classes=10))


def test_embeddings() -> object:
    """Function docstring."""
    device = Device("cpu")
    t = Tensor(np.ones((2,)), TensorConfig((2,), DType.Int32, device))
    t_f = Tensor(np.ones((2,)), TensorConfig((2,), DType.Float32, device))

    with ConfigContext(eager_mode=True):
        with patch("ml_switcheroo_compiler.ops.shape.gather") as mock_gather:
            mock_gather.return_value = "res"
            embedding_lookup(t_f, t)

            class MockRagged:
                """Class docstring."""

                def __init__(self, values: object) -> object:
                    """Function docstring."""
                    self.values = values

            r = MockRagged(t)
            embedding_lookup_sparse(r, t_f, t_f)
            safe_embedding_lookup_sparse(t_f, r)
