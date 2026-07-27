"""Tests for STFT and ISTFT operations."""

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.signal import Istft, Stft, istft, stft
from ml_switcheroo_compiler.tracing.state import global_tracing_state


def test_stft_istft_eager() -> None:
    """Test stft and istft in eager mode."""
    config.eager_mode = True
    x_data = np.random.randn(10, 100).astype(np.float32)
    x = Tensor(x_data, TensorConfig(x_data.shape, "float32", None))

    # stft
    Zxx = stft(x, nfft=16, noverlap=8)
    assert Zxx.shape == (10, 9, 11)

    # istft
    xrec = istft(Zxx, nfft=16, noverlap=8)
    assert xrec.shape == (10, 96)  # (11 - 1) * 8 + 16 = 80 + 16 = 96


def test_stft_istft_tracing() -> None:
    """Test stft and istft in tracing mode."""
    config.eager_mode = False
    x_data = np.random.randn(10, 100).astype(np.float32)
    x = Tensor(x_data, TensorConfig(x_data.shape, "float32", None))

    global_tracing_state.start_tracing()
    # stft
    Zxx = stft(x, nfft=16, noverlap=8)
    assert Zxx.shape == (10, 9, 11)

    # istft
    xrec = istft(Zxx, nfft=16, noverlap=8)
    assert xrec.shape == (10, 96)
    global_tracing_state.stop_tracing()


def test_stft_istft_infer_shape_invalid() -> None:
    """Test coverage for invalid shapes in infer_shape."""
    s = Stft()
    assert s.infer_shape(None, nfft=16, noverlap=8) == ()

    class Dummy:
        pass

    assert s.infer_shape(Dummy(), nfft=16, noverlap=8) == ()

    class DummyShape:
        shape = ()

    assert s.infer_shape(DummyShape(), nfft=16, noverlap=8) == ()

    i = Istft()
    assert i.infer_shape(None, nfft=16, noverlap=8) == ()
    assert i.infer_shape(Dummy(), nfft=16, noverlap=8) == ()
    assert i.infer_shape(DummyShape(), nfft=16, noverlap=8) == ()

    class DummyShape1D:
        shape = (10,)

    assert i.infer_shape(DummyShape1D(), nfft=16, noverlap=8) == ()


def test_stft_istft_invalid_overlap() -> None:
    """Test ValueError is raised for invalid noverlap."""
    config.eager_mode = True
    x_data = np.random.randn(10, 100).astype(np.float32)
    x = Tensor(x_data, TensorConfig(x_data.shape, "float32", None))

    with pytest.raises(ValueError, match="noverlap must be less than nfft"):
        stft(x, nfft=16, noverlap=16)

    with pytest.raises(ValueError, match="noverlap must be less than nfft"):
        Zxx = stft(x, nfft=16, noverlap=8)
        istft(Zxx, nfft=16, noverlap=16)

    config.eager_mode = False
    global_tracing_state.start_tracing()
    with pytest.raises(ValueError, match="noverlap must be less than nfft"):
        stft(x, nfft=16, noverlap=16)

    with pytest.raises(ValueError, match="noverlap must be less than nfft"):
        Zxx = Tensor(np.zeros((10, 9, 11), dtype=np.complex64), TensorConfig((10, 9, 11), "complex64", None))
        istft(Zxx, nfft=16, noverlap=16)
    global_tracing_state.stop_tracing()
