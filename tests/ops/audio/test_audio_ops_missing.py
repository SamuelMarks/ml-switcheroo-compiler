import numpy as np

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.audio.ops import HannWindow, Istft, MelSpectrogram, Mfcc, Stft


def test_audio_ops_transformations():
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
