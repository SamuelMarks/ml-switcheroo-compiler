"""Tests for test_nlp."""

from __future__ import annotations

import typing

# ruff: noqa: E501
import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Device, DType, Tensor, TensorConfig
from ml_switcheroo_compiler.ops.nn.nlp import (
    NCELossConfig,
    NLPOpsConfig,
    SampledSoftmaxConfig,
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
from tests.testing_utils import MockTensorStub, MockTestStub, coerce_operand_list, extract_test_shape

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_nlp_coverage() -> None:
    """Test function test_nlp_coverage."""
    config.eager_mode = True
    labels = Tensor(np.array([[1]]), TensorConfig(shape=(1, 1), dtype=DType("int32"), device=Device("cpu")))
    logits = Tensor(
        np.array([[[1.0, 2.0, 3.0]]]), TensorConfig(shape=(1, 1, 3), dtype=DType("float32"), device=Device("cpu"))
    )
    seq_len = Tensor(np.array([1]), TensorConfig(shape=(1,), dtype=DType("int32"), device=Device("cpu")))

    cfg = NLPOpsConfig()
    assert len(coerce_operand_list(all_candidate_sampler(labels, cfg))) == 3
    assert len(coerce_operand_list(compute_accidental_hits(labels, labels, cfg))) == 3
    assert len(coerce_operand_list(fixed_unigram_candidate_sampler(labels, cfg))) == 3
    assert len(coerce_operand_list(learned_unigram_candidate_sampler(labels, cfg))) == 3
    assert len(coerce_operand_list(log_uniform_candidate_sampler(labels, cfg))) == 3
    assert len(coerce_operand_list(uniform_candidate_sampler(labels, cfg))) == 3

    w = Tensor(np.array([[1.0, 2.0, 3.0]]), TensorConfig(shape=(1, 3), dtype=DType("float32"), device=Device("cpu")))
    b = Tensor(np.array([1.0]), TensorConfig(shape=(1,), dtype=DType("float32"), device=Device("cpu")))

    nce_cfg = NCELossConfig(num_sampled=1, num_classes=3)
    ss_cfg = SampledSoftmaxConfig(num_sampled=1, num_classes=3)
    assert tuple(extract_test_shape(nce_loss(w, b, labels, logits, nce_cfg))) == (1,)
    assert tuple(extract_test_shape(sampled_softmax_loss(w, b, labels, logits, ss_cfg))) == (1,)
    assert len(coerce_operand_list(ctc_beam_search_decoder(logits, seq_len))) == 2
    assert len(coerce_operand_list(ctc_greedy_decoder(logits, seq_len))) == 2
    assert tuple(extract_test_shape(ctc_loss(labels, logits, seq_len, seq_len))) == (1,)
    assert len(coerce_operand_list(ctc_unique_labels(labels))) == 2


def test_nlp_options_and_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test NLP options dataclasses, CtcLoss shape inference, and eager execution.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.nn.nlp import CtcLoss, CTCLossOptions, NLPOpsConfig, SamplingConfig, VocabConfig

    VocabConfig()
    ctc_decode_conf = CTCLossOptions()
    NLPOpsConfig()
    SamplingConfig()

    assert ctc_decode_conf.logits_time_major == 1

    class LogitsShape3(MockTestStub):
        """Test class LogitsShape3."""

        shape = (10, 32, 5)

    class LogitsShape2(MockTestStub):
        """Test class LogitsShape2."""

        shape = (10, 5)

    assert CtcLoss().infer_shape(None, LogitsShape3(), None, None) == (32,)
    assert CtcLoss().infer_shape(None, LogitsShape2(), None, None) == (1,)

    orig = config.eager_mode
    config.eager_mode = True

    class DummyBackend:
        """Test class DummyBackend."""

        def execute_op(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function execute_op.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `execute_op`.
            """
            return np.array([1.0])

    import ml_switcheroo_compiler.backends.registry as reg

    monkeypatch.setattr(reg, "get_active_backend", lambda: DummyBackend())
    try:
        t_labels = Tensor(np.array([1]), TensorConfig(shape=(1,), dtype=DType("int32"), device=Device("cpu")))
        t_logits = Tensor(np.array([[1.0]]), TensorConfig(shape=(1, 1), dtype=DType("float32"), device=Device("cpu")))
        t_len = Tensor(np.array([1]), TensorConfig(shape=(1,), dtype=DType("int32"), device=Device("cpu")))
        res = ctc_loss(t_labels, t_logits, t_len, t_len)
        assert extract_test_shape(res) == (1,)
    finally:
        config.eager_mode = orig


def test_nlp_tracing_candidate_samplers_and_losses() -> None:
    """Test candidate samplers and loss operations in symbolic tracing mode."""
    import ml_switcheroo_compiler.ops.nn.nlp as nlp
    import ml_switcheroo_compiler.tracing.state as state
    from ml_switcheroo_compiler.core.config import config

    orig = config.eager_mode
    orig_tracing = state.global_tracing_state.is_tracing
    config.eager_mode = False
    state.global_tracing_state.is_tracing = True
    orig_add_node = state.global_tracing_state.add_node
    pytest.MonkeyPatch().setattr(state.global_tracing_state, "add_node", lambda node: None, raising=False)

    try:

        class DummyTensor(MockTensorStub):
            """Test class DummyTensor."""

            def __init__(self) -> None:
                """Initialize stub attributes."""
                super().__init__()
                self.shape = (1, 1, 1)
                self.dtype = "float32"
                self.device = "cpu"
                self.data = "dummy"

        t = DummyTensor()
        c = nlp.NLPOpsConfig()

        nlp.fixed_unigram_candidate_sampler(t, c)
        nlp.compute_accidental_hits(t, t, c)
        nlp.ctc_loss(t, t, t, t)
        nlp.log_uniform_candidate_sampler(t, c)
        nlp.learned_unigram_candidate_sampler(t, c)
        nlp.all_candidate_sampler(t, c)
        nlp.uniform_candidate_sampler(t, c)

        nc = nlp.NCELossConfig(num_sampled=10, num_classes=100)
        nlp.nce_loss(t, t, t, t, nc)

        sc = nlp.SampledSoftmaxConfig(num_sampled=10, num_classes=100)
        nlp.sampled_softmax_loss(t, t, t, t, sc)

        nlp.ctc_beam_search_decoder(t, t)
        nlp.ctc_greedy_decoder(t, t)
        nlp.ctc_unique_labels(t)
    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        pytest.MonkeyPatch().setattr(state.global_tracing_state, "add_node", orig_add_node, raising=False)
    assert state.global_tracing_state.add_node == orig_add_node
