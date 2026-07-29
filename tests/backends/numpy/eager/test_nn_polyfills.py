import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.nn_polyfills import (
    _np_all_candidate_sampler,
    _np_collapse_repeated,
    _np_conv_transpose,
    _np_ctc_beam_search_decoder,
    _np_ctc_greedy_decoder,
    _np_ctc_unique_labels,
    _np_depthwise_conv2d_backprop_filter,
    _np_depthwise_conv2d_backprop_input,
    _np_dilation2d,
    _np_erosion2d,
    _np_in_top_k,
    _np_isotonic_regression,
    _np_log_poisson_loss,
    _np_max_pool_with_argmax,
    _np_normalize_moments,
    _np_quantized_conv,
    _np_sufficient_statistics,
    _np_weighted_moments,
)


class DummyBackend:
    @staticmethod
    def asarray(x):
        return np.asarray(x)


def test_nn_polyfills():
    # Isotonic
    y = np.array([1, 0, 2])
    res, _ = _np_isotonic_regression(DummyBackend(), y)
    assert res.shape == (3,)

    # ConvTranspose
    res = _np_conv_transpose(DummyBackend(), np.ones((1, 2, 2, 1)), np.ones((2, 2, 1, 1)), output_shape=(1, 3, 3, 1))
    assert res.shape == (1, 3, 3, 1)
    res3 = _np_conv_transpose(DummyBackend(), np.ones((2, 2)), np.ones((2, 2)))
    assert res3.shape == (2, 2)
    # None output shape
    res_no_shape = _np_conv_transpose(DummyBackend(), np.ones((1, 2, 2, 1)), np.ones((2, 2, 1, 1)))
    assert res_no_shape.shape == (1, 3, 3, 1)

    # Depthwise Backprop Filter
    arr = np.ones((1, 2, 2, 1))
    ob = np.ones((1, 2, 2, 1))
    f_shape = (2, 2, 1, 1)
    out_bpf = _np_depthwise_conv2d_backprop_filter(DummyBackend(), arr, f_shape, ob)
    assert out_bpf.shape == f_shape

    # Depthwise Backprop Input
    out_bpi = _np_depthwise_conv2d_backprop_input(DummyBackend(), (1, 2, 2, 1), np.ones(f_shape), ob)
    assert out_bpi.shape == (1, 2, 2, 1)

    # Dilation/Erosion
    arr2d = np.ones((2, 2))
    assert _np_dilation2d(DummyBackend(), arr2d, np.ones((2, 2))).shape == (2, 2)
    assert _np_erosion2d(DummyBackend(), arr2d, np.ones((2, 2))).shape == (2, 2)
    arr4d = np.ones((1, 2, 2, 1))
    assert _np_dilation2d(DummyBackend(), arr4d, np.ones((2, 2, 1))).shape == (1, 2, 2, 1)
    assert _np_erosion2d(DummyBackend(), arr4d, np.ones((2, 2, 1))).shape == (1, 2, 2, 1)

    # InTopK
    t = np.array([0, 1])
    p = np.array([[0.1, 0.9], [0.9, 0.1]])
    assert _np_in_top_k(DummyBackend(), t, p, k=1).shape == (2,)
    assert _np_in_top_k(DummyBackend(), 0, np.array([0.1, 0.9]), k=1).shape == ()

    # LogPoisson
    assert _np_log_poisson_loss(DummyBackend(), [1], [0.1]).shape == (1,)
    assert _np_log_poisson_loss(DummyBackend(), [1], [0.1], compute_full_loss=True).shape == (1,)

    # AllCandidateSampler
    assert _np_all_candidate_sampler(DummyBackend(), [1])[0].shape == (1,)

    # CTC decoders
    logits = np.ones((5, 1, 3))
    seq_len = np.array([5])
    beam_sparse, beam_log = _np_ctc_beam_search_decoder(DummyBackend(), logits, seq_len)
    assert len(beam_sparse) == 3

    greedy_sparse, greedy_log = _np_ctc_greedy_decoder(DummyBackend(), logits, seq_len)
    assert len(greedy_sparse) == 3
    greedy_sparse_empty, _ = _np_ctc_greedy_decoder(DummyBackend(), np.zeros((0, 1, 3)), np.array([0]))
    assert greedy_sparse_empty[0].shape == (0, 2)

    # CtcUniqueLabels
    u, i = _np_ctc_unique_labels(DummyBackend(), [1, 1, 2])
    assert u.shape == (2,)

    # NormalizeMoments
    m, v = _np_normalize_moments(DummyBackend(), [1], [0], [1], [0])
    assert m.shape == (1,)

    # Sufficient Stats & Weighted Moments
    ss1 = _np_sufficient_statistics(DummyBackend(), np.ones((2, 2)), axes=None)
    ss2 = _np_sufficient_statistics(DummyBackend(), np.ones((2, 2)), axes=[0], keepdims=True)
    assert len(ss1) == 4

    wm = _np_weighted_moments(DummyBackend(), np.ones((2, 2)), axes=[0], frequency_weights=np.ones((2, 2)), keepdims=True)
    assert len(wm) == 2

    # MaxPoolWithArgmax
    mp_max, mp_arg = _np_max_pool_with_argmax(DummyBackend(), np.ones((1, 4, 4, 1)), pool_size=2)
    assert mp_max.shape == (1, 4, 4, 1)

    # CollapseRepeated
    c, i = _np_collapse_repeated(DummyBackend(), np.array([1, 1, 2]))
    assert c.shape == (2,)
    c_empty, _ = _np_collapse_repeated(DummyBackend(), np.array([]))
    assert c_empty.shape == (0,)

    # QuantizedConv
    q_res1 = _np_quantized_conv(DummyBackend(), np.ones((1, 2, 2, 1)), np.ones((2, 2, 1, 1)), np.ones(1), padding=0)
    assert q_res1.shape == (1, 1, 1, 1)
    q_res2 = _np_quantized_conv(DummyBackend(), np.ones((1, 2, 2, 1)), np.ones((2, 2, 1, 1)), np.ones(1), padding=1)
    assert q_res2.shape == (1, 3, 3, 1)
    q_res3 = _np_quantized_conv(DummyBackend(), np.ones((1, 2, 2, 1)), np.ones((2, 2, 1, 1)), np.ones(1), None, padding="SAME")
    assert q_res3.shape == (1, 2, 2, 1)

    # CtcBeamSearch empty
    beam_empty, _ = _np_ctc_beam_search_decoder(DummyBackend(), np.zeros((0, 1, 3)), np.array([0]))
    assert beam_empty[0].shape == (0, 2)
    # QuantizedConv with biases
    q_res4 = _np_quantized_conv(DummyBackend(), np.ones((1, 2, 2, 1)), np.ones((2, 2, 1, 1)), np.ones(1), np.ones(1), padding=0)
    assert q_res4.shape == (1, 1, 1, 1)
