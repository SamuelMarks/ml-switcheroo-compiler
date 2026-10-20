"""Tests for test_audio."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.numpy.eager.audio_features import _np_dct, _np_frame, _np_idct, _np_inverse_mdct, _np_mdct, _np_overlap_and_add

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol


def test_audio_missing() -> None:
    # 2,3,4 with and without ortho
    """Test function test_audio_missing."""
    _np_dct(np, np.ones((10,)), type=1, norm="ortho")
    _np_dct(np, np.ones((10,)), type=1, norm=None)
    _np_dct(np, np.ones((10,)), type=2, norm="ortho")
    _np_dct(np, np.ones((10,)), type=3, norm="ortho")
    _np_dct(np, np.ones((10,)), type=4, norm="ortho")
    _np_dct(np, np.ones((10,)), type=4, norm=None)

    with pytest.raises(ValueError):
        _np_dct(np, np.ones((10,)), type=5)

    _np_idct(np, np.ones((10,)), type=1, norm="ortho")
    _np_idct(np, np.ones((10,)), type=1, norm=None)
    _np_idct(np, np.ones((10,)), type=2, norm=None)
    _np_idct(np, np.ones((10,)), type=3, norm=None)
    _np_idct(np, np.ones((10,)), type=4, norm=None)

    with pytest.raises(ValueError):
        _np_idct(np, np.ones((10,)), type=5)

    res3 = _np_mdct(np, np.ones((20,)))
    assert typing.cast("EagerCallableProtocol", len)(res3) == 10

    with pytest.raises(ValueError):
        _np_mdct(np, np.ones((20,)), N=8)

    res4 = _np_inverse_mdct(np, np.ones((10,)))
    assert typing.cast("EagerCallableProtocol", len)(res4) == 20

    res5 = _np_frame(np, np.ones((100,)), frame_length=10, frame_step=5)
    assert res5 is not None

    res5b = _np_frame(np, np.ones((5,)), frame_length=10, frame_step=5)
    assert typing.cast("EagerTensorProtocol", getattr(res5b, "shape")) == (0, 10)

    res6 = _np_overlap_and_add(np, np.ones((10, 10)), frame_step=5)
    assert res6 is not None
