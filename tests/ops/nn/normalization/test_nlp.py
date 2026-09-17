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


def test_nlp_options_and_shapes(monkeypatch):
    """Test NLP options dataclasses, CtcLoss shape inference, and eager execution."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.nn.nlp import CtcLoss, CTCLossOptions, NLPOpsConfig, SamplingConfig, VocabConfig

    vocab_conf = VocabConfig()
    ctc_decode_conf = CTCLossOptions()
    ctc_loss_conf = NLPOpsConfig()
    sampling_conf = SamplingConfig()

    assert ctc_decode_conf.logits_time_major == 1

    class LogitsShape3:
        shape = (10, 32, 5)

    class LogitsShape2:
        shape = (10, 5)

    assert CtcLoss().infer_shape(None, LogitsShape3(), None, None) == (32,)
    assert CtcLoss().infer_shape(None, LogitsShape2(), None, None) == (1,)

    orig = config.eager_mode
    config.eager_mode = True

    class DummyBackend:
        def execute_op(self, *args, **kwargs):
            return np.array([1.0])

    import ml_switcheroo_compiler.backends.registry as reg

    monkeypatch.setattr(reg, "get_active_backend", lambda: DummyBackend())
    try:
        t_labels = Tensor(np.array([1]), TensorConfig(shape=(1,), dtype=DType("int32"), device=Device("cpu")))
        t_logits = Tensor(np.array([[1.0]]), TensorConfig(shape=(1, 1), dtype=DType("float32"), device=Device("cpu")))
        t_len = Tensor(np.array([1]), TensorConfig(shape=(1,), dtype=DType("int32"), device=Device("cpu")))
        res = ctc_loss(t_labels, t_logits, t_len, t_len)
        assert res.shape == (1,)
    finally:
        config.eager_mode = orig


def test_nlp_tracing_candidate_samplers_and_losses():
    """Test candidate samplers and loss operations in symbolic tracing mode."""
    import ml_switcheroo_compiler.ops.nn.nlp as nlp
    import ml_switcheroo_compiler.tracing.state as state
    from ml_switcheroo_compiler.core.config import config

    orig = config.eager_mode
    orig_tracing = state.global_tracing_state.is_tracing
    config.eager_mode = False
    state.global_tracing_state.is_tracing = True
    orig_add_node = state.global_tracing_state.add_node
    state.global_tracing_state.add_node = lambda node: None

    try:

        class DummyTensor:
            shape = (1, 1, 1)
            dtype = "float32"
            device = "cpu"
            data = "dummy"

        t = DummyTensor()
        c = nlp.NLPOpsConfig()

        nlp.fixed_unigram_candidate_sampler(t, c)
        nlp.compute_accidental_hits(t, t, c)
        nlp.ctc_loss(t, t, t, t)
        nlp.log_uniform_candidate_sampler(t, c)
        nlp.learned_unigram_candidate_sampler(t, c)
        nlp.all_candidate_sampler(t, c)
        nlp.uniform_candidate_sampler(t, c)

        nc = nlp.NCELossConfig(10, 100)
        nlp.nce_loss(t, t, t, t, nc)

        sc = nlp.SampledSoftmaxConfig(10, 100)
        nlp.sampled_softmax_loss(t, t, t, t, sc)

        nlp.ctc_beam_search_decoder(t, t)
        nlp.ctc_greedy_decoder(t, t)
        nlp.ctc_unique_labels(t)
    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        state.global_tracing_state.add_node = orig_add_node
