"""Tests for test_audio_ops."""

from __future__ import annotations

import typing
from collections.abc import Sequence

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.audio.ops import (
    Dct,
    Frame,
    HammingWindow,
    HannWindow,
    Idct,
    InverseMdct,
    Istft,
    KaiserWindow,
    Mdct,
    MelFilterbank,
    MelSpectrogram,
    Mfcc,
    MfccsFromLogMelSpectrograms,
    OverlapAndAdd,
    Stft,
)
from ml_switcheroo_compiler.ops.base import OpDef
from tests.testing_utils import MockTestStub, coerce_int, coerce_operand_list

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_audio_ops_transformations() -> None:
    """Test function test_audio_ops_transformations."""
    import unittest.mock as mock

    t_audio = Tensor(np.random.randn(1, 16000).astype("float32"), TensorConfig((1, 16000), "float32", "cpu"))
    with mock.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.is_tracing", True):
        with mock.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.active_graph"):
            stft = Stft()
            assert isinstance(stft.infer_shape(t_audio), tuple)
            istft = Istft()
            assert isinstance(istft.infer_shape(t_audio), tuple)
            mel = MelSpectrogram()
            assert isinstance(mel.infer_shape(t_audio), tuple)
            mfcc = Mfcc()
            assert isinstance(mfcc.infer_shape(t_audio), tuple)
            hann = HannWindow()
            assert isinstance(hann.infer_shape(128), tuple)


class MockArray(MockTestStub):
    """Test class MockArray."""

    shape: tuple[int, ...]

    def __init__(self, shape: Sequence[int]) -> None:
        """Test function __init__.

        Args:
            shape (Sequence[int]): Test parameter `shape`.
        """
        raw_t = tuple(shape)
        self.shape = tuple(coerce_int(x) for x in raw_t)


class MockArg(MockTestStub):
    """Test class MockArg."""

    def __init__(self, value: BackendArg) -> None:
        """Test function __init__.

        Args:
            value: Test parameter `value`.
        """
        self.value = value


class _TestAudioOpsInferShapeState(typing.TypedDict, total=False):
    """Typed shared state for `_test_audio_ops_infer_shape` split helpers."""

    NoShape: type[MockTestStub]
    op: OpDef


def _test_audio_ops_infer_shape_part_1(_state: _TestAudioOpsInferShapeState) -> None:
    """Execute part 1 of test_audio_ops_infer_shape.

    Args:
        _state: Shared test state dictionary.
    """
    stft_op = Stft()
    assert stft_op.infer_shape(MockArray((2, 100)), frame_step=10, frame_length=20) == (
        2,
        9,
        11,
    )

    class NoShape(MockTestStub):
        """Test class NoShape."""

        pass

    assert stft_op.infer_shape(NoShape()) == ()
    assert stft_op.dispatch_infer_shape() == ()
    assert stft_op.infer_shape(MockArray(())) == ()
    mel_op = MelSpectrogram()
    assert mel_op.infer_shape(MockArray((2, 100)), frame_step=10, frame_length=20, num_mel_bins=8) == (2, 9, 8)
    assert mel_op.infer_shape(NoShape()) == ()
    assert mel_op.dispatch_infer_shape() == ()
    assert mel_op.infer_shape(MockArray(())) == ()
    istft_op = Istft()
    assert istft_op.infer_shape(MockArray((2, 9, 11)), frame_step=10, frame_length=20) == (
        2,
        100,
    )
    assert istft_op.infer_shape(NoShape()) == ()
    assert istft_op.dispatch_infer_shape() == ()
    assert istft_op.infer_shape(MockArray(())) == ()
    assert istft_op.infer_shape(MockArray((10,))) == (10,)
    fb_op = MelFilterbank()
    assert fb_op.dispatch_infer_shape(num_mel_bins=12, fft_length=20) == (11, 12)
    assert fb_op.dispatch_infer_shape() == (129, 128)
    _state.update({"NoShape": NoShape, "op": fb_op})


def _test_audio_ops_infer_shape_part_2(_state: _TestAudioOpsInferShapeState) -> None:
    """Execute part 2 of test_audio_ops_infer_shape.

    Args:
        _state: Shared test state dictionary.
    """
    NoShape = _state["NoShape"]
    mfcc_op = Mfcc()
    assert mfcc_op.infer_shape(MockArray((2, 100)), frame_step=10, frame_length=20, num_mel_bins=16, num_mfccs=4) == (
        2,
        9,
        4,
    )
    assert mfcc_op.dispatch_infer_shape(NoShape()) == ()
    assert mfcc_op.dispatch_infer_shape() == ()
    assert mfcc_op.infer_shape(MockArray(())) == ()
    mfccs_op = MfccsFromLogMelSpectrograms()
    assert mfccs_op.infer_shape(MockArray((2, 9, 16)), num_mfccs=4) == (2, 9, 4)
    assert mfccs_op.dispatch_infer_shape(NoShape()) == ()
    assert mfccs_op.dispatch_infer_shape() == ()
    assert mfccs_op.infer_shape(MockArray(())) == ()
    op1 = HannWindow()
    assert op1.infer_shape(window_length=5) == (5,)
    assert op1.infer_shape(MockArg(6), window_length=None) == (6,)
    assert op1.infer_shape(MockArray(()), window_length=256) == (256,)
    op2 = HammingWindow()
    assert op2.infer_shape(window_length=6) == (6,)
    assert op2.infer_shape(MockArg(7), window_length=None) == (7,)
    assert op2.infer_shape(MockArray(()), window_length=256) == (256,)
    op3 = KaiserWindow()
    assert op3.infer_shape(window_length=7) == (7,)
    assert op3.infer_shape(MockArg(8), window_length=None) == (8,)
    assert op3.infer_shape(MockArray(()), window_length=256) == (256,)
    _state.update({"op": mfccs_op})
    _state.update({"NoShape": NoShape})


def _test_audio_ops_infer_shape_part_3(_state: _TestAudioOpsInferShapeState) -> None:
    """Execute part 3 of test_audio_ops_infer_shape.

    Args:
        _state: Shared test state dictionary.
    """
    dct_op = Dct()
    assert dct_op.infer_shape(MockArray((10, 20))) == (10, 20)
    idct_op = Idct()
    assert idct_op.infer_shape(MockArray((10, 20))) == (10, 20)
    mdct_op = Mdct()
    res = mdct_op.infer_shape(MockArray((2, 100)), frame_length=20, frame_step=10)
    assert len(coerce_operand_list(res)) == 3
    assert ensure_ndarray(res)[0] == 2
    assert ensure_ndarray(res)[2] == 10
    res_pad = mdct_op.infer_shape(MockArray((2, 100)), frame_length=20, frame_step=10, pad_end=True)
    assert len(coerce_operand_list(res_pad)) == 3
    res_pre = mdct_op.infer_shape(MockArray((2, 5, 20)), frame_length=20)
    assert res_pre == (2, 5, 10)
    res_short = mdct_op.infer_shape(MockArray((2, 10)), frame_length=20)
    assert res_short == (2, 5)
    assert mdct_op.infer_shape(MockArray(())) == ()
    assert mdct_op.infer_shape() == ()
    imdct_op = InverseMdct()
    res = imdct_op.infer_shape(MockArray((2, 5, 10)), frame_length=20)
    assert len(coerce_operand_list(res)) == 3
    assert res == (2, 5, 20)
    assert imdct_op.infer_shape(MockArray(())) == ()
    _state.update({"op": imdct_op})


def _test_audio_ops_infer_shape_part_4(_state: _TestAudioOpsInferShapeState) -> None:
    """Execute part 4 of test_audio_ops_infer_shape.

    Args:
        _state: Shared test state dictionary.
    """
    (NoShape, op) = (_state["NoShape"], _state["op"])
    assert op.dispatch_infer_shape(NoShape()) == ()
    assert op.dispatch_infer_shape() == ()
    frame_op = Frame()
    assert frame_op.infer_shape(MockArray((2, 100)), frame_length=20, frame_step=10) == (
        2,
        9,
        20,
    )
    assert frame_op.infer_shape(MockArray((2, 100)), frame_length=20, frame_step=10, pad_end=True) == (2, 10, 20)
    assert frame_op.infer_shape(MockArray(())) == ()
    assert frame_op.dispatch_infer_shape(NoShape()) == ()
    assert frame_op.dispatch_infer_shape() == ()
    oaa_op = OverlapAndAdd()
    assert oaa_op.infer_shape(MockArray((2, 9, 20)), frame_step=10) == (2, 100)
    assert oaa_op.infer_shape(MockArray(())) == ()
    assert oaa_op.dispatch_infer_shape(NoShape()) == ()
    with pytest.raises(TypeError):
        oaa_op.dispatch_infer_shape()
    _state.update({"op": oaa_op})


def test_audio_ops_infer_shape() -> None:
    """Test function test_audio_ops_infer_shape."""
    _state: _TestAudioOpsInferShapeState = {}
    _test_audio_ops_infer_shape_part_1(_state)
    _test_audio_ops_infer_shape_part_2(_state)
    _test_audio_ops_infer_shape_part_3(_state)
    _test_audio_ops_infer_shape_part_4(_state)
    assert len(_state) > 0
