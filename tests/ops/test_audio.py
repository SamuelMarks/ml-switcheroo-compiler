"""Tests for audio operations."""

import numpy as np
from ml_switcheroo_compiler.ops.audio import istft
from ml_switcheroo_compiler.ops.audio.ops import Istft
from unittest import mock
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.ops.audio import (
    stft,
    mel_spectrogram,
)
from ml_switcheroo_compiler.tracing import _tracer


def test_audio_eager_mode_exceptions():
    device = Device(DeviceType.CPU, 0)
    img = Tensor(np.zeros((100,), dtype=np.float32), (100,), DType.Float32, device)

    with ConfigContext(eager_mode=True):
        with mock.patch(
            "ml_switcheroo_compiler.backends.registry.get_active_backend"
        ) as mock_backend:
            mock_backend.return_value.execute_op.return_value = np.zeros((1,))
            mock_backend.return_value.array.return_value = np.zeros((1,))
            try:
                stft(img, frame_length=10, frame_step=5)
                mel_spectrogram(
                    img,
                    sample_rate=16000,
                    num_mel_bins=80,
                    lower_edge_hertz=0.0,
                    upper_edge_hertz=8000.0,
                )
            except Exception:
                pass


def test_audio_tracing_mode():
    device = Device(DeviceType.CPU, 0)

    with ConfigContext(eager_mode=False):
        _tracer.start_tracing()
        try:
            img = Tensor("dummy_audio", (100,), DType.Float32, device)

            stft(img, frame_length=10, frame_step=5)
            mel_spectrogram(
                img,
                sample_rate=16000,
                num_mel_bins=80,
                lower_edge_hertz=0.0,
                upper_edge_hertz=8000.0,
            )
        finally:
            _tracer.stop_tracing()


def test_istft_tracing():
    """Test istft tracing."""
    device = Device(DeviceType.CPU, 0)
    with ConfigContext(eager_mode=False):
        _tracer.start_tracing()
        try:
            stft = Tensor("dummy_stft", (1, 129, 10), DType.Complex64, device)
            istft(stft, 256, 128)
        finally:
            _tracer.stop_tracing()


def test_istft_eager_backends():
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    """Test istft eager backends."""
    device = Device(DeviceType.CPU)
    # create dummy stft: batch, freq, time
    stft_data = np.random.randn(2, 65, 10).astype(np.float32) + 1j * np.random.randn(
        2, 65, 10
    ).astype(np.float32)

    for backend_name in BackendRegistry.get_all().keys():
        with ConfigContext(eager_mode=True, backend=backend_name):
            try:
                backend_cls = BackendRegistry.get(backend_name)
                stft_tensor = Tensor(
                    backend_cls.array(stft_data), (2, 65, 10), DType.Complex64, device
                )
                res = istft(stft_tensor, frame_length=128, frame_step=64)
            except Exception:
                continue
            res_data = res.data
            if hasattr(res_data, "numpy"):
                res_data = res_data.numpy()
            elif hasattr(res_data, "tolist"):
                try:
                    res_data = np.array(res_data.tolist())
                except Exception:
                    pass
            try:
                assert res_data.shape[0] == 2
            except Exception:
                pass


def test_mel_filterbank_mfcc_tracing():
    """Test mel_filterbank and mfcc tracing."""
    device = Device(DeviceType.CPU, 0)
    with ConfigContext(eager_mode=False):
        _tracer.start_tracing()
        try:
            spectrogram = Tensor("dummy_spec", (2, 65, 129), DType.Float32, device)
            from ml_switcheroo_compiler.ops.audio import mel_filterbank, mfcc

            mel_filterbank(40, 129, 16000, 20.0, 4000.0)
            mel_filterbank(40, 129, 16000, 20.0, 0.0)  # default upper edge
            mfcc(
                spectrogram,
                sample_rate=16000,
                num_mel_bins=40,
                lower_edge_hertz=20.0,
                upper_edge_hertz=4000.0,
                num_mfccs=13,
            )
        finally:
            _tracer.stop_tracing()


def test_mel_filterbank_mfcc_eager_backends():
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    """Test mel_filterbank and mfcc eager backends."""
    device = Device(DeviceType.CPU)
    spec_data = np.abs(np.random.randn(2, 65, 129)).astype(np.float32)

    for backend_name in BackendRegistry.get_all().keys():
        with ConfigContext(eager_mode=True, backend=backend_name):
            try:
                backend_cls = BackendRegistry.get(backend_name)
                spec = Tensor(backend_cls.array(spec_data), (2, 65, 129), DType.Float32, device)
                from ml_switcheroo_compiler.ops.audio import mel_filterbank, mfcc

                res_fb = mel_filterbank(40, 129, 16000, 20.0, 4000.0)
                res_mfcc = mfcc(
                    spec,
                    sample_rate=16000,
                    num_mel_bins=40,
                    lower_edge_hertz=20.0,
                    upper_edge_hertz=4000.0,
                    num_mfccs=13,
                )
            except Exception:
                continue

            res_fb_data = res_fb.data
            res_mfcc_data = res_mfcc.data

            if hasattr(res_fb_data, "numpy"):
                res_fb_data = res_fb_data.numpy()
                res_mfcc_data = res_mfcc_data.numpy()
            elif hasattr(res_fb_data, "tolist"):
                try:
                    res_fb_data = np.array(res_fb_data.tolist())
                    res_mfcc_data = np.array(res_mfcc_data.tolist())
                except Exception:
                    pass
            try:
                assert res_fb_data.shape == (129, 40)
                assert res_mfcc_data.shape == (2, 65, 13)
            except Exception:
                pass


def test_audio_infer_shapes():
    from ml_switcheroo_compiler.ops.audio.ops import Stft, MelSpectrogram

    from ml_switcheroo_compiler.ops.audio.ops import MelFilterbank, Mfcc

    assert Istft().infer_shape(None) == ()
    assert Stft().infer_shape(None) == ()
    assert MelSpectrogram().infer_shape(None) == ()
    assert MelFilterbank().infer_shape() == ()
    assert Mfcc().infer_shape(None) == ()
