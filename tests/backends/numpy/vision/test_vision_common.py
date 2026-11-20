"""Unit tests for NumPy eager vision and audio common operations."""

from __future__ import annotations

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager.vision_ops import common as vision_common_module
from ml_switcheroo_compiler.backends.eager.vision_ops.common import _np_mel_filterbank, _np_mfcc, _np_power_iteration
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


def test_vision_common_module_exports() -> None:
    """Verify all declared symbols in `vision_common.__all__` exist on the module."""
    exported = dir(vision_common_module)
    for symbol in vision_common_module.__all__:
        assert symbol in exported


def test_np_mel_filterbank_real_execution() -> None:
    """Verify `_np_mel_filterbank` computes non-negative triangular filterbank matrices."""
    mod = np
    default_fb = np.asarray(_np_mel_filterbank(mod, None, config={}))
    assert default_fb.shape == (129, 40)
    assert bool(np.all(default_fb >= 0.0))
    assert float(np.sum(default_fb)) > 0.0

    small_cfg = {
        "num_spectrogram_bins": 8,
        "num_mel_bins": 4,
        "sample_rate": 8000,
        "lower_edge_hertz": 0.0,
        "upper_edge_hertz": 4000.0,
    }
    small_fb = np.asarray(_np_mel_filterbank(mod, None, config=small_cfg))
    assert small_fb.shape == (8, 4)
    assert bool(np.all(small_fb >= 0.0))
    assert float(np.sum(small_fb)) > 0.0

    reg_mel = numpy_eager_registry.get("MelFilterbank")
    assert reg_mel is not None
    reg_fb = np.asarray(reg_mel(np, None, config=small_cfg))
    np.testing.assert_allclose(reg_fb, small_fb, rtol=1e-6)


def test_np_mfcc_real_execution_and_validation() -> None:
    """Verify `_np_mfcc` computes finite cepstral coefficients on valid spectrograms and rejects bad shapes."""
    mod = np
    mfcc_cfg = {
        "num_spectrogram_bins": 8,
        "num_mel_bins": 4,
        "num_mfccs": 3,
        "sample_rate": 8000,
        "lower_edge_hertz": 0.0,
        "upper_edge_hertz": 4000.0,
    }
    spectrogram = np.array(
        [
            [1.0, 2.0, 4.0, 8.0, 4.0, 2.0, 1.0, 0.5],
            [1.0, 2.0, 4.0, 8.0, 4.0, 2.0, 1.0, 0.5],
        ],
        dtype=np.float32,
    )
    mfcc_out = np.asarray(_np_mfcc(mod, spectrogram, config=mfcc_cfg))
    assert mfcc_out.shape == (2, 3)
    assert bool(np.all(np.isfinite(mfcc_out)))
    np.testing.assert_allclose(mfcc_out[0], mfcc_out[1], rtol=1e-6)

    reg_mfcc = numpy_eager_registry.get("Mfcc")
    assert reg_mfcc is not None
    reg_out = np.asarray(reg_mfcc(np, spectrogram, config=mfcc_cfg))
    np.testing.assert_allclose(reg_out, mfcc_out, rtol=1e-6)

    with pytest.raises(ValueError):
        _np_mfcc(mod, np.ones((2, 2), dtype=np.float32), config={})


def test_np_power_iteration_branches_and_registry() -> None:
    """Verify `_np_power_iteration` converges to the dominant singular value across `u` branches."""
    mod = np
    w = np.array([[3.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    res1 = _np_power_iteration(mod, w, num_iters=10)
    assert isinstance(res1, (tuple, list))
    v1, u1, sigma1 = res1
    assert np.asarray(v1).shape == (2,)
    assert np.asarray(u1).shape == (2,)
    assert np.isclose(float(np.asarray(sigma1)), 3.0, atol=1e-4)

    u_1d = np.array([1.0, 0.0], dtype=np.float32)
    res2 = _np_power_iteration(mod, w, num_iters=2, u=u_1d)
    assert isinstance(res2, (tuple, list))
    v2, u2, sigma2 = res2
    assert np.asarray(v2).shape == (2,)
    assert np.asarray(u2).shape == (2,)
    assert np.isclose(float(np.asarray(sigma2)), 3.0, atol=1e-5)

    u_2d = np.array([[1.0], [0.0]], dtype=np.float32)
    res3 = _np_power_iteration(mod, w, num_iters=2, u=u_2d)
    assert isinstance(res3, (tuple, list))
    v3, u3, sigma3 = res3
    assert np.asarray(v3).shape == (2,)
    assert np.asarray(u3).shape == (2,)
    assert np.isclose(float(np.asarray(sigma3)), 3.0, atol=1e-5)

    reg_pi = numpy_eager_registry.get("PowerIteration")
    assert reg_pi is not None
    w_eye = np.eye(3, dtype=np.float32)
    v_reg, u_reg, sigma_reg = reg_pi(np, w_eye, num_iters=2)
    assert np.asarray(v_reg).shape == (3,)
    assert np.asarray(u_reg).shape == (3,)
    assert np.isclose(float(np.asarray(sigma_reg)), 1.0, atol=1e-5)
