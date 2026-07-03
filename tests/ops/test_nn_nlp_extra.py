"""Module docstring."""

from unittest.mock import patch

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.nlp import (
    AttentionConfig,
    AttentionInputs,
    NCELossConfig,
    NLPOpsConfig,
    SampledSoftmaxConfig,
    SamplerConfig,
    SamplingConfig,
    attention,
    dot_product_attention,
    embedding,
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


def test_dot_product_attention_inner_extra() -> object:
    """Function docstring."""
    device = Device("cpu")
    q = Tensor(np.ones((2, 4, 5)), TensorConfig((2, 4, 5), DType.Float32, device))
    k = Tensor(np.ones((2, 4, 5)), TensorConfig((2, 4, 5), DType.Float32, device))
    v = Tensor(np.ones((2, 4, 5)), TensorConfig((2, 4, 5), DType.Float32, device))

    with ConfigContext(eager_mode=True):
        with patch("ml_switcheroo_compiler.ops.shape.permute") as mock_permute:
            with patch("ml_switcheroo_compiler.ops.linalg.matmul") as mock_matmul:
                with patch("ml_switcheroo_compiler.ops.binary.true_divide") as mock_true_divide:
                    with patch("ml_switcheroo_compiler.ops.binary.add") as mock_add:
                        with patch("ml_switcheroo_compiler.nn.activations.softmax") as mock_softmax:
                            mock_permute.return_value = k
                            mock_matmul.return_value = q
                            mock_true_divide.return_value = q
                            mock_add.return_value = q
                            mock_softmax.return_value = q

                            # dot_product_attention (which is older interface calling _dot_product_attention)
                            res1 = dot_product_attention(q, k, v)
                            assert res1 is not None

                            mask = Tensor(np.zeros((2, 4, 4)), TensorConfig((2, 4, 4), DType.Float32, device))

                            inputs = AttentionInputs(query=q, key=k, value=v)
                            conf = AttentionConfig(is_causal=True, dropout=0.5, mask=mask)

                            with patch("ml_switcheroo_compiler.ops.nn.dropout") as mock_dropout:
                                mock_dropout.return_value = q
                                res2 = attention(inputs, conf)
                                assert res2 is not None

                                # also hit no config case
                                res3 = attention(inputs)
                                assert res3 is not None


def test_embedding_extra() -> object:
    """Function docstring."""
    device = Device("cpu")
    i = Tensor(np.array([1, 2]), TensorConfig((2,), DType.Int32, device))
    w = Tensor(np.ones((5, 10)), TensorConfig((5, 10), DType.Float32, device))
    with ConfigContext(eager_mode=True):
        with patch("ml_switcheroo_compiler.ops.shape.gather") as mock_gather:
            mock_gather.return_value = "res"
            res = embedding(i, w)
            assert res == "res"


def test_samplers_extra() -> object:
    """Function docstring."""
    device = Device("cpu")
    t = Tensor(np.ones((2,)), TensorConfig((2,), DType.Int32, device))

    nlp_conf = NLPOpsConfig(sampling=SamplingConfig(num_true=1, num_sampled=1, unique=True, range_max=10))

    with ConfigContext(eager_mode=True):
        with patch("ml_switcheroo_compiler.ops.nn.nlp.all_candidate_sampler") as mock_sampler:
            mock_sampler.return_value = (t, t, t)
            fixed_unigram_candidate_sampler(t, config=nlp_conf, sampler_config=SamplerConfig(range_max=10))
            learned_unigram_candidate_sampler(t, config=nlp_conf)
            log_uniform_candidate_sampler(t, config=nlp_conf)
            uniform_candidate_sampler(t, config=nlp_conf)

            # just hit the function
            nce_loss(t, t, t, t, config=NCELossConfig(num_sampled=1, num_classes=10))
            sampled_softmax_loss(t, t, t, t, config=SampledSoftmaxConfig(num_sampled=1, num_classes=10))

        with patch("ml_switcheroo_compiler.ops.nn.nlp.embedding") as mock_emb:
            mock_emb.return_value = "emb"

            class MockRagged:
                """Class docstring."""

                def __init__(self, values: object) -> object:
                    """Function docstring."""
                    self.values = values

            ragged = MockRagged(t)
            embedding_lookup(t, t)
            embedding_lookup_sparse(ragged, t, t)
            safe_embedding_lookup_sparse(t, ragged)
