"""Test dummy ops and signal functions."""

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.numpy.eager.audio_extras import (
    _np_dct,
    _np_frame,
    _np_idct,
    _np_inverse_mdct,
    _np_mdct,
    _np_overlap_and_add,
)


def test_audio_extras_numpy() -> None:
    """Test audio extras numpy."""
    x = np.random.rand(2, 4)

    res = _np_dct(np, x, type=1, norm="ortho")
    res = _np_dct(np, x, type=1)
    res = _np_dct(np, x, type=2, norm="ortho")
    res = _np_dct(np, x, type=2)
    res = _np_dct(np, x, type=3, norm="ortho")
    res = _np_dct(np, x, type=3)
    res = _np_dct(np, x, type=4, norm="ortho")
    res = _np_dct(np, x, type=4)

    with pytest.raises(ValueError):
        _np_dct(np, x, type=5)

    res = _np_idct(np, x, type=1, norm="ortho")
    res = _np_idct(np, x, type=1)
    res = _np_idct(np, x, type=2, norm="ortho")
    res = _np_idct(np, x, type=2)
    res = _np_idct(np, x, type=3, norm="ortho")
    res = _np_idct(np, x, type=3)
    res = _np_idct(np, x, type=4, norm="ortho")
    res = _np_idct(np, x, type=4)

    with pytest.raises(ValueError):
        _np_idct(np, x, type=5)

    x2 = np.random.rand(2, 4)
    res_mdct = _np_mdct(np, x2)
    assert res_mdct.shape == (2, 2)
    res_imdct = _np_inverse_mdct(np, res_mdct)
    assert res_imdct.shape == (2, 4)

    with pytest.raises(ValueError):
        _np_mdct(np, np.random.rand(3))  # shape length not even or not matched to 2N default?
        # Actually default N is length // 2, so 3 // 2 is 1, 2*N is 2, length is 3, raises ValueError

    res_frame = _np_frame(np, x, frame_length=2, frame_step=1)
    assert res_frame.shape == (2, 3, 2)

    res_ola = _np_overlap_and_add(np, res_frame, frame_step=1)
    assert res_ola.shape == (2, 4)

    # test frame <= 0
    x_small = np.ones((2, 1))
    res = _np_frame(np, x_small, frame_length=5, frame_step=2)
    assert res.shape[-2] == 0
