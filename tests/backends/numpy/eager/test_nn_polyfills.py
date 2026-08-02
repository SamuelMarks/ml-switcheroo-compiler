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


"""Test Numpy eager nn stubs coverage."""


def test_depthwise_conv2d_backprop():

    # Needs 4D arrays to trigger loop
    inp = np.ones((1, 4, 4, 2))
    ob = np.ones((1, 4, 4, 4))
    # f_shape (H, W, in_channels, channel_multiplier)
    f_shape = (2, 2, 2, 2)

    res_f = _np_depthwise_conv2d_backprop_filter(None, inp, f_shape, ob)
    assert res_f.shape == f_shape

    f = np.ones((2, 2, 2, 2))
    input_sizes = (1, 4, 4, 2)
    res_inp = _np_depthwise_conv2d_backprop_input(None, input_sizes, f, ob)
    assert res_inp.shape == input_sizes


def test_dilation_erosion_2d():
    inp_4d = np.ones((1, 4, 4, 2))
    f = np.ones((2, 2, 2))

    res_dil = _np_dilation2d(None, inp_4d, f)
    assert res_dil.shape == (1, 4, 4, 2)

    res_ero = _np_erosion2d(None, inp_4d, f)
    assert res_ero.shape == (1, 4, 4, 2)


def test_in_top_k():
    t = np.array([1, 2])
    p = np.array([[0.1, 0.9, 0.2], [0.8, 0.2, 0.1]])

    # 2D case
    res = _np_in_top_k(None, t, p, k=2)
    assert res.shape == (2,)

    # 1D case
    t1 = np.array(1)
    p1 = np.array([0.1, 0.9, 0.2])
    res1 = _np_in_top_k(None, t1, p1, k=2)
    assert isinstance(res1, np.bool_) or isinstance(res1, bool) or res1.shape == ()


def test_log_poisson_loss():
    t = np.array([1.0, 2.0])
    l = np.array([0.5, 0.8])

    res = _np_log_poisson_loss(None, t, l, compute_full_loss=True)
    assert res.shape == (2,)


def test_all_candidate_sampler():
    t = np.array([[1], [2]])
    s, tec, sec = _np_all_candidate_sampler(None, t, num_sampled=2, num_classes=5)
    assert s.shape == (2,)
    assert tec.shape == (2, 1)
    assert sec.shape == (2,)


def test_ctc_greedy_decoder():
    inputs = np.array([[[0.1, 0.9], [0.8, 0.2]], [[0.9, 0.1], [0.9, 0.1]]])
    seq_len = np.array([2, 1])
    sparse, log_prob = _np_ctc_greedy_decoder(None, inputs, seq_len)
    assert len(sparse) == 3
    assert log_prob.shape == (2,)

    # Trigger empty sequence branch
    seq_len_zero = np.array([0, 0])
    sparse_empty, _ = _np_ctc_greedy_decoder(None, inputs, seq_len_zero)
    assert sparse_empty[0].shape == (0, 2)


def test_ctc_unique_labels():
    labels = np.array([1, 2, 1, 3])
    u, idx = _np_ctc_unique_labels(None, labels)
    assert u.shape == (3,)
    assert idx.shape == (4,)


def test_normalize_moments():
    c = np.array([10.0])
    m = np.array([0.5])
    v = np.array([1.2])
    # Shift must be zero/numeric, not None
    res = _np_normalize_moments(None, c, m, v, shift=0.0)
    assert res[0].shape == (1,)


def test_max_pool_with_argmax():
    inp = np.ones((1, 4, 4, 1))
    # Test 4D unpacking path
    m, a = _np_max_pool_with_argmax(None, inp, pool_size=2)
    assert m.shape == (1, 4, 4, 1)


def test_collapse_repeated():
    # Empty
    c, i = _np_collapse_repeated(None, np.array([]))
    assert c.size == 0
    # Duplicates
    c, i = _np_collapse_repeated(None, np.array([1, 1, 2, 2, 3]))
    assert len(c) == 3


def test_depthwise_conv2d_backprop_ndim_branch():
    """Test depthwise branches where input is not 4D."""
    inp = np.ones((4, 4))
    ob = np.ones((4, 4))
    f_shape = (2, 2, 2, 2)

    res_f = _np_depthwise_conv2d_backprop_filter(None, inp, f_shape, ob)
    assert np.all(res_f == 0)

    f = np.ones((2, 2, 2, 2))
    input_sizes = (4, 4)
    res_inp = _np_depthwise_conv2d_backprop_input(None, input_sizes, f, ob)
    assert np.all(res_inp == 0)


def test_dilation_erosion_2d_ndim_branch():
    """Test dilation/erosion branches where input is not 4D."""
    inp_2d = np.ones((4, 4))
    f = np.ones((2, 2))

    res_dil = _np_dilation2d(None, inp_2d, f)
    assert res_dil.shape == (4, 4)

    res_ero = _np_erosion2d(None, inp_2d, f)
    assert res_ero.shape == (4, 4)


def test_log_poisson_loss_no_full():
    """Test log poisson loss without compute_full_loss."""
    t = np.array([1.0])
    l = np.array([0.5])

    res = _np_log_poisson_loss(None, t, l, compute_full_loss=False)
    assert res.shape == (1,)


def test_ctc_beam_search_decoder():
    """Test beam search decoder stub."""
    inputs = np.array([[[0.1, 0.9], [0.8, 0.2]], [[0.9, 0.1], [0.9, 0.1]]])
    seq_len = np.array([2, 1])
    sparse, log_prob = _np_ctc_beam_search_decoder(None, inputs, seq_len)
    assert len(sparse) == 3


def test_max_pool_with_argmax_ndim():
    """Test max pool with argmax when input is not 4D."""
    inp = np.ones((1, 4, 4))
    # Provide a tuple for size so scipy doesn't fail on a single int with 3D
    m, a = _np_max_pool_with_argmax(None, inp, pool_size=(1, 2, 2))
    assert m.shape == (1, 4, 4)


def test_quantized_conv2d_coverage(monkeypatch):
    """Test quantized conv2d branches."""
    # We must mock _conv_general_dilated

    def mock_conv(*args, **kwargs):
        return "mock_conv"

    import ml_switcheroo_compiler.backends.numpy.eager.conv as conv_module

    monkeypatch.setattr(conv_module, "_conv_general_dilated", mock_conv)

    inp = np.ones((1, 4, 4, 1))
    weight = np.ones((2, 2, 1, 1))
    scales = np.ones((1,))

    # Without biases -> hits line 247
    res = _np_quantized_conv(None, inp, weight, scales, None, padding=1)
    assert res is not None

    # With biases -> hits padding != 0 path -> line 272
    biases = np.ones((1,))
    res2 = _np_quantized_conv(None, inp, weight, scales, biases, padding=1)
    assert res2 is not None

    # Padding = 0 -> hits line 270 (padding='VALID')
    res3 = _np_quantized_conv(None, inp, weight, scales, biases, padding=0)
    assert res3 is not None


def test_isotonic_regression_coverage():
    # v1 > v2 requires some adjacent elements where the first is greater than the second
    x = np.array([2.0, 1.0, 3.0], dtype=np.float32)
    res, segments = _np_isotonic_regression(None, x)
    assert np.allclose(res, np.array([1.5, 1.5, 3.0]))


def test_ctc_beam_search_empty():
    inputs = np.zeros((2, 1, 3))  # seq_len=2, batch=1, classes=3
    seq_len = np.array([0])
    res = _np_ctc_beam_search_decoder(None, inputs, seq_len)
    sparse, log_prob = res
    assert sparse[0].shape == (0, 2)
    assert len(sparse[1]) == 0


def test_ctc_beam_search_complex():
    # seq_len=3, batch=1, classes=3.
    # Try to make beam sorting reorder things to hit line 229.
    inputs = np.array(
        [
            [[10.0, 1.0, 0.0]],  # Step 0: class 0 is most likely
            [[1.0, 10.0, 0.0]],  # Step 1: class 1 is most likely
            [[10.0, 1.0, 0.0]],  # Step 2: class 0 is most likely
        ]
    )
    seq_len = np.array([3])
    res = _np_ctc_beam_search_decoder(None, inputs, seq_len, beam_width=5)
    sparse, log_prob = res
    assert len(sparse[0]) > 0
