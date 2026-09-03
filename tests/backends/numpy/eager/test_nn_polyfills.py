import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.nn_polyfills import (
    _np_all_candidate_sampler,
    _np_collapse_repeated,
    _np_ctc_beam_search_decoder,
    _np_ctc_beam_step,
    _np_ctc_unique_labels,
    _np_log_poisson_loss,
    _np_max_pool_with_argmax,
    _np_normalize_moments,
    _np_quantized_conv,
    _np_sufficient_statistics,
    _np_weighted_moments,
)


def test_log_poisson_loss():
    targets = np.array([1.0, 2.0])
    log_input = np.array([0.5, -0.5])
    res = _np_log_poisson_loss(np, targets, log_input)
    assert res.shape == (2,)

    res_full = _np_log_poisson_loss(np, targets, log_input, compute_full_loss=True)
    assert res_full.shape == (2,)


def test_all_candidate_sampler():
    true_classes = np.array([[1], [2]])
    sampled, true_exp, sampled_exp = _np_all_candidate_sampler(np, true_classes, num_sampled=2, num_classes=5)
    assert sampled.shape == (2,)
    assert true_exp.shape == (2, 1)
    assert sampled_exp.shape == (2,)


def test_ctc_beam_search_decoder_extend_path():
    import numpy as np

    inputs = np.array(
        [
            [[0.1, 0.4, 0.5]],
            [[0.1, 0.8, 0.1]],
            [[0.1, 0.8, 0.1]],  # Duplicate
        ]
    )
    inputs = np.log(inputs)
    seq_len = np.array([3])
    sparse, log_probs = _np_ctc_beam_search_decoder(np, inputs, seq_len, beam_width=5)
    assert len(sparse) == 3
    inputs = np.random.rand(5, 2, 4)
    seq_len = np.array([5, 4])
    sparse, log_probs = _np_ctc_beam_search_decoder(np, inputs, seq_len, beam_width=2)
    assert len(sparse) == 3
    assert log_probs.shape == (2,)

    # Empty test
    sparse, log_probs = _np_ctc_beam_search_decoder(np, np.zeros((0, 0, 4)), np.array([]), beam_width=2)
    assert len(sparse) == 3


def test_ctc_unique_labels():
    labels = np.array([1, 1, 2, 3, 2])
    unique, indices = _np_ctc_unique_labels(np, labels)
    assert list(unique) == [1, 2, 3]


def test_normalize_moments():
    mean, var = _np_normalize_moments(np, 10, 5, 20, 1.0)
    assert mean == 1.5
    assert var > 0


def test_sufficient_statistics():
    x = np.random.rand(2, 3, 4)
    c, m, v, s = _np_sufficient_statistics(np, x, axes=(1, 2), keepdims=True)
    assert s.shape == (2, 1, 1)

    c, m, v, s = _np_sufficient_statistics(np, x, axes=None)
    assert s.shape == ()


def test_weighted_moments():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    fw = np.array([[1.0, 1.0], [2.0, 2.0]])
    m, v = _np_weighted_moments(np, x, axes=(1,), frequency_weights=fw, keepdims=True)
    assert m.shape == (2, 1)


def test_max_pool_with_argmax():
    x = np.random.rand(1, 4, 4, 1)
    m, a = _np_max_pool_with_argmax(np, x, pool_size=2)
    assert m.shape == (1, 4, 4, 1)

    m, a = _np_max_pool_with_argmax(np, x, pool_size=(1, 2, 2, 1))
    assert m.shape == (1, 4, 4, 1)


def test_collapse_repeated():
    x = np.array([1, 1, 2, 2, 3, 1])
    c, idx = _np_collapse_repeated(np, x)
    assert list(c) == [1, 2, 3, 1]

    c, idx = _np_collapse_repeated(np, np.array([]))
    assert len(c) == 0


def test_quantized_conv():
    # 2D input NHWC, 2D weight HWIO
    inp = np.random.rand(1, 4, 4, 2).astype(np.float32)
    wt = np.random.rand(2, 2, 2, 3).astype(np.float32)
    scales = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    biases = np.zeros(3, dtype=np.float32)

    res = _np_quantized_conv(np, inp, wt, scales, biases, stride=1, padding=0, dilation=1)
    assert res.shape == (1, 3, 3, 3)

    res = _np_quantized_conv(np, inp, wt, scales, None, stride=(1, 1), padding=1, dilation=(1, 1))
    assert res.shape == (1, 5, 5, 3)


def test_ctc_beam_step_edge_cases():
    beam = {(1,): (-1.0, -2.0)}
    log_p = np.array([-0.5, -0.6, -1.0])
    next_beam = _np_ctc_beam_step(beam, log_p, num_classes=3, blank=2, beam_width=2)
    assert (1,) in next_beam
    assert (1, 0) in next_beam


def test_ctc_beam_search_merge_paths():
    """Test merge paths in ctc beam search."""
    import numpy as np

    # inputs: [max_time, batch_size, num_classes]
    # To have () and (1,) in the beam, the first step should give prob to blank (class 0) and class 1.
    # The second step should give prob to class 1.

    # Let num_classes = 3, blank = 2
    # step 0:
    # prob(0) = 0.2
    # prob(1) = 0.4
    # prob(blank=2) = 0.4

    # step 1:
    # prob(1) = 0.9, others small

    inputs = np.array([[[0.2, 0.4, 0.4]], [[0.05, 0.9, 0.05]]])
    # log probs
    inputs = np.log(inputs)
    seq_len = np.array([2])
    sparse, log_probs = _np_ctc_beam_search_decoder(np, inputs, seq_len, beam_width=5)
    assert len(sparse) == 3
