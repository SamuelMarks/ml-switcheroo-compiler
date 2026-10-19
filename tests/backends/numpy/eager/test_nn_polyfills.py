"""Tests for test_nn_polyfills."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.nn_polyfills import (
    _np_all_candidate_sampler,
    _np_collapse_repeated,
    _np_ctc_beam_search_decoder,
    _np_ctc_beam_step,
    _np_ctc_unique_labels,
    _np_depthwise_conv2d_backprop_filter,
    _np_depthwise_conv2d_backprop_input,
    _np_log_poisson_loss,
    _np_max_pool_with_argmax,
    _np_normalize_moments,
    _np_quantized_conv,
    _np_sufficient_statistics,
    _np_weighted_moments,
)

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol


def test_log_poisson_loss() -> None:
    """Test function test_log_poisson_loss."""
    targets = np.array([1.0, 2.0])
    log_input = np.array([0.5, -0.5])
    res = _np_log_poisson_loss(np, targets, log_input)
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (2,)

    res_full = _np_log_poisson_loss(np, targets, log_input, compute_full_loss=True)
    assert cast("EagerTensorProtocol", getattr(res_full, "shape")) == (2,)

    # Test non-log predictions
    pred_non_log = np.array([1.5, 0.6])
    res_non_log = _np_log_poisson_loss(np, targets, pred_non_log, log_input=False)
    assert cast("EagerTensorProtocol", getattr(res_non_log, "shape")) == (2,)

    # Test Stirling approximation fallback without scipy
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"scipy.special": None}):
        res_stirling = _np_log_poisson_loss(np, np.array([0.5, 2.0]), np.array([0.1, 0.2]), compute_full_loss=True)
        assert cast("EagerTensorProtocol", getattr(res_stirling, "shape")) == (2,)

        # Also test with target < 0.5 so that t + 1.0 < 1.5 triggers small_mask in _stirling_gammaln
        res_stirling_small = _np_log_poisson_loss(np, np.array([0.2]), np.array([0.1]), compute_full_loss=True)
        assert cast("EagerTensorProtocol", getattr(res_stirling_small, "shape")) == (1,)


def test_all_candidate_sampler() -> None:
    """Test function test_all_candidate_sampler."""
    true_classes = np.array([[1], [2]])
    sampled, true_exp, sampled_exp = cast("EagerCallableProtocol", _np_all_candidate_sampler)(np, true_classes, num_sampled=2, num_classes=5)
    assert cast("EagerTensorProtocol", getattr(sampled, "shape")) == (2,)
    assert cast("EagerTensorProtocol", getattr(true_exp, "shape")) == (2, 1)
    assert cast("EagerTensorProtocol", getattr(sampled_exp, "shape")) == (2,)


def test_ctc_beam_search_decoder_extend_path() -> None:
    """Test function test_ctc_beam_search_decoder_extend_path."""
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
    sparse, log_probs = cast("EagerCallableProtocol", _np_ctc_beam_search_decoder)(np, inputs, seq_len, beam_width=5)
    assert cast("EagerCallableProtocol", len)(sparse) == 3
    inputs = np.random.rand(5, 2, 4)
    seq_len = np.array([5, 4])
    sparse, log_probs = cast("EagerCallableProtocol", _np_ctc_beam_search_decoder)(np, inputs, seq_len, beam_width=2)
    assert cast("EagerCallableProtocol", len)(sparse) == 3
    assert cast("EagerTensorProtocol", getattr(log_probs, "shape")) == (2,)

    # Empty test
    sparse, log_probs = cast("EagerCallableProtocol", _np_ctc_beam_search_decoder)(np, cast("EagerCallableProtocol", np.zeros)((0, 0, 4)), cast("EagerCallableProtocol", np.array)([]), beam_width=2)
    assert cast("EagerCallableProtocol", len)(sparse) == 3

    # Multi-path test
    sparse_multi, log_probs_multi = cast("EagerCallableProtocol", _np_ctc_beam_search_decoder)(np, inputs, seq_len, beam_width=5, top_paths=3)
    assert cast("EagerCallableProtocol", len)(sparse_multi) == 3
    assert cast("EagerTensorProtocol", getattr(log_probs_multi, "shape")) == (2, 3)
    for sp in sparse_multi:
        assert cast("EagerCallableProtocol", len)(sp) == 3
        assert cast("EagerTensorProtocol", getattr(cast("EagerTensorProtocol", getattr(sp, "__getitem__")(2)), "__getitem__")(0)) == 2


def test_ctc_unique_labels() -> None:
    """Test function test_ctc_unique_labels."""
    labels = np.array([1, 1, 2, 3, 2])
    unique, indices = cast("EagerCallableProtocol", _np_ctc_unique_labels)(np, labels)
    assert cast("EagerCallableProtocol", list)(unique) == [1, 2, 3]


def test_normalize_moments() -> None:
    """Test function test_normalize_moments."""
    mean, var = cast("EagerCallableProtocol", _np_normalize_moments)(np, 10, 5, 20, 1.0)
    assert mean == 1.5
    assert var > 0


def test_sufficient_statistics() -> None:
    """Test function test_sufficient_statistics."""
    x = np.random.rand(2, 3, 4)
    c, m, v, s = cast("EagerCallableProtocol", _np_sufficient_statistics)(np, x, axes=(1, 2), keepdims=True)
    assert cast("EagerTensorProtocol", getattr(s, "shape")) == (2, 1, 1)

    c, m, v, s = cast("EagerCallableProtocol", _np_sufficient_statistics)(np, x, axes=None)
    assert cast("EagerTensorProtocol", getattr(s, "shape")) == ()


def test_weighted_moments() -> None:
    """Test function test_weighted_moments."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    fw = np.array([[1.0, 1.0], [2.0, 2.0]])
    m, v = cast("EagerCallableProtocol", _np_weighted_moments)(np, x, axes=(1,), frequency_weights=fw, keepdims=True)
    assert cast("EagerTensorProtocol", getattr(m, "shape")) == (2, 1)


def test_max_pool_with_argmax() -> None:
    """Test function test_max_pool_with_argmax."""
    x = np.random.rand(1, 4, 4, 1)
    m, a = cast("EagerCallableProtocol", _np_max_pool_with_argmax)(np, x, pool_size=2)
    assert cast("EagerTensorProtocol", getattr(m, "shape")) == (1, 4, 4, 1)

    m, a = cast("EagerCallableProtocol", _np_max_pool_with_argmax)(np, x, pool_size=(1, 2, 2, 1))
    assert cast("EagerTensorProtocol", getattr(m, "shape")) == (1, 4, 4, 1)


def test_collapse_repeated() -> None:
    """Test function test_collapse_repeated."""
    x = np.array([1, 1, 2, 2, 3, 1])
    c, idx = cast("EagerCallableProtocol", _np_collapse_repeated)(np, x)
    assert cast("EagerCallableProtocol", list)(c) == [1, 2, 3, 1]

    c, idx = cast("EagerCallableProtocol", _np_collapse_repeated)(np, cast("EagerCallableProtocol", np.array)([]))
    assert cast("EagerCallableProtocol", len)(c) == 0


def test_quantized_conv() -> None:
    # 2D input NHWC, 2D weight HWIO
    """Test function test_quantized_conv."""
    inp = np.random.rand(1, 4, 4, 2).astype(np.float32)
    wt = np.random.rand(2, 2, 2, 3).astype(np.float32)
    scales = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    biases = np.zeros(3, dtype=np.float32)

    res = _np_quantized_conv(np, inp, wt, scales, biases, stride=1, padding=0, dilation=1)
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (1, 3, 3, 3)

    res = _np_quantized_conv(np, inp, wt, scales, None, stride=(1, 1), padding=1, dilation=(1, 1))
    assert cast("EagerTensorProtocol", getattr(res, "shape")) == (1, 5, 5, 3)

    # Test padding as string / non-int to cover 387->393 branch
    res_valid = _np_quantized_conv(np, inp, wt, scales, biases, stride=1, padding="VALID", dilation=1)
    assert cast("EagerTensorProtocol", getattr(res_valid, "shape")) == (1, 3, 3, 3)


def test_ctc_beam_step_edge_cases() -> None:
    """Test function test_ctc_beam_step_edge_cases."""
    beam = {(1,): (-1.0, -2.0)}
    log_p = np.array([-0.5, -0.6, -1.0])
    next_beam = cast("EagerCallableProtocol", _np_ctc_beam_step)(beam, log_p, num_classes=3, blank=2, beam_width=2)
    assert (1,) in next_beam
    assert (1, 0) in next_beam


def test_ctc_beam_search_merge_paths() -> None:
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
    sparse, log_probs = cast("EagerCallableProtocol", _np_ctc_beam_search_decoder)(np, inputs, seq_len, beam_width=5)
    assert cast("EagerCallableProtocol", len)(sparse) == 3

    # Test top_paths > 1
    sparse_multi, log_probs_multi = cast("EagerCallableProtocol", _np_ctc_beam_search_decoder)(np, inputs, seq_len, beam_width=5, top_paths=2)
    assert cast("EagerCallableProtocol", len)(sparse_multi) == 2
    assert cast("EagerTensorProtocol", getattr(log_probs_multi, "shape")) == (1, 2)

    # Test top_paths == 1 with top_paths < 1 in loop (forcing empty sparse_list)
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.backends.numpy.eager.nn_polyfills.range", return_value=[]):
        sparse_empty, _ = cast("EagerCallableProtocol", _np_ctc_beam_search_decoder)(np, inputs, seq_len, beam_width=5, top_paths=1)
        assert cast("EagerTensorProtocol", getattr(cast("EagerTensorProtocol", getattr(sparse_empty, "__getitem__")(0)), "shape")) == (0, 2)


def test_max_pool_with_argmax_fallback() -> None:
    """Test max_pool_with_argmax when scipy.ndimage is unavailable."""
    import sys
    from unittest.mock import patch

    x = np.random.rand(1, 4, 4, 1)
    with patch.dict(sys.modules, {"scipy.ndimage": None}):
        m, a = cast("EagerCallableProtocol", _np_max_pool_with_argmax)(np, x, pool_size=2)
        assert cast("EagerTensorProtocol", getattr(m, "shape")) == (1, 4, 4, 1)
        assert cast("EagerTensorProtocol", getattr(a, "shape")) == (1,)


def test_depthwise_conv2d_backprop_polyfills() -> None:
    """Verify NumPy eager polyfills for depthwise conv2d filter and input backprop."""
    df = _np_depthwise_conv2d_backprop_filter(np, np.zeros((1, 4, 4, 2)), (3, 3, 2, 1), np.zeros((1, 2, 2, 2)))
    assert df.shape == (3, 3, 2, 1)
    df_int = _np_depthwise_conv2d_backprop_filter(np, np.zeros((1, 4, 4, 2)), 4, np.zeros((1, 2, 2, 2)))
    assert df_int.shape == (4,)

    di = _np_depthwise_conv2d_backprop_input(np, (1, 4, 4, 2), np.zeros((3, 3, 2, 1)), np.zeros((1, 2, 2, 2)))
    assert di.shape == (1, 4, 4, 2)
    di_int = _np_depthwise_conv2d_backprop_input(np, 4, np.zeros((3, 3, 2, 1)), np.zeros((1, 2, 2, 2)))
    assert di_int.shape == (4,)
