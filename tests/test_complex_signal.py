import numpy as np
from ml_switcheroo_compiler import ops


def test_complex_signal():
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True

    x = ops.array(np.random.randn(2, 4).astype(np.float32))
    fft_out = ops.fft(x)
    assert fft_out is not None

    irfft_out = ops.irfft(fft_out)
    assert irfft_out is not None

    x2 = ops.array(np.random.randn(2, 4, 4).astype(np.float32))
    fft2_out = ops.fft2(x2)
    assert fft2_out is not None

    ifft2_out = ops.ifft2(fft2_out)
    assert ifft2_out is not None

    abs_v = ops.array(np.random.randn(2, 4).astype(np.float32))
    angle = ops.array(np.random.randn(2, 4).astype(np.float32))
    polar_out = ops.polar(abs_v, angle)
    assert polar_out is not None

    real_view = ops.view_as_real(polar_out)
    assert real_view is not None

    # We should use numpy arrays or actual backend objects inside if ops.array doesn't support complex.
    # We'll see.
