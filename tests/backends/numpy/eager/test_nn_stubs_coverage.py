"""Test Numpy eager nn stubs coverage."""

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.nn_stubs import (
    _np_all_candidate_sampler,
    _np_collapse_repeated,
    _np_ctc_beam_search_decoder,
    _np_ctc_greedy_decoder,
    _np_ctc_unique_labels,
    _np_depthwise_conv2d_backprop_filter,
    _np_depthwise_conv2d_backprop_input,
    _np_dilation2d,
    _np_erosion2d,
    _np_in_top_k,
    _np_log_poisson_loss,
    _np_max_pool_with_argmax,
    _np_normalize_moments,
    _np_quantized_conv,
)


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
