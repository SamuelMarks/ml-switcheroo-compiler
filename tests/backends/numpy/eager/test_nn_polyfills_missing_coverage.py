import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.nn_polyfills import _np_ctc_beam_search_decoder, _np_isotonic_regression


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
