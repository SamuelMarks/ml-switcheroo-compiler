"""Numpy FFT Ops."""

from dataclasses import dataclass

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


@dataclass
class FFTConfig:
    """FFT Configuration."""

    s: object = None
    axes: object = None
    norm: object = None


@numpy_eager_registry.register("Fft3d")
def _np_fft3d(backend_module: object, a: object, config: FFTConfig = None, **kwargs: object) -> object:
    """Evaluate the fft3d logic eagerly backed by NumPy.

    Args:
        backend_module: Required parameter for backend_module.
        a: Required parameter for a.
        config: FFT configuration.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        The evaluated or processed output.
    """
    config = config or FFTConfig(axes=(-3, -2, -1))
    return backend_module.fft.fftn(a, s=config.s, axes=config.axes, norm=config.norm)


@numpy_eager_registry.register("Ifft3d")
def _np_ifft3d(backend_module: object, a: object, config: FFTConfig = None, **kwargs: object) -> object:
    """Evaluate the ifft3d logic eagerly backed by NumPy."""
    config = config or FFTConfig(axes=(-3, -2, -1))
    return backend_module.fft.ifftn(a, s=config.s, axes=config.axes, norm=config.norm)


@numpy_eager_registry.register("Rfft2d")
def _np_rfft2d(backend_module: object, a: object, config: FFTConfig = None, **kwargs: object) -> object:
    """Evaluate the rfft2d logic eagerly backed by NumPy."""
    config = config or FFTConfig(axes=(-2, -1))
    return backend_module.fft.rfftn(a, s=config.s, axes=config.axes, norm=config.norm)


@numpy_eager_registry.register("Rfft3d")
def _np_rfft3d(backend_module: object, a: object, config: FFTConfig = None, **kwargs: object) -> object:
    """Evaluate the rfft3d logic eagerly backed by NumPy."""
    config = config or FFTConfig(axes=(-3, -2, -1))
    return backend_module.fft.rfftn(a, s=config.s, axes=config.axes, norm=config.norm)


@numpy_eager_registry.register("Irfft2d")
def _np_irfft2d(backend_module: object, a: object, config: FFTConfig = None, **kwargs: object) -> object:
    """Evaluate the irfft2d logic eagerly backed by NumPy."""
    config = config or FFTConfig(axes=(-2, -1))
    return backend_module.fft.irfftn(a, s=config.s, axes=config.axes, norm=config.norm)


@numpy_eager_registry.register("Irfft3d")
def _np_irfft3d(backend_module: object, a: object, config: FFTConfig = None, **kwargs: object) -> object:
    """Evaluate the irfft3d logic eagerly backed by NumPy."""
    config = config or FFTConfig(axes=(-3, -2, -1))
    return backend_module.fft.irfftn(a, s=config.s, axes=config.axes, norm=config.norm)


@numpy_eager_registry.register("Fftnd")
def _np_fftnd(backend_module: object, a: object, config: FFTConfig = None, **kwargs: object) -> object:
    """Evaluate the fftnd logic eagerly backed by NumPy."""
    config = config or FFTConfig(axes=None)
    return backend_module.fft.fftn(a, s=config.s, axes=config.axes, norm=config.norm)


@numpy_eager_registry.register("Ifftnd")
def _np_ifftnd(backend_module: object, a: object, config: FFTConfig = None, **kwargs: object) -> object:
    """Evaluate the ifftnd logic eagerly backed by NumPy."""
    config = config or FFTConfig(axes=None)
    return backend_module.fft.ifftn(a, s=config.s, axes=config.axes, norm=config.norm)


@numpy_eager_registry.register("Rfftnd")
def _np_rfftnd(backend_module: object, a: object, config: FFTConfig = None, **kwargs: object) -> object:
    """Evaluate the rfftnd logic eagerly backed by NumPy."""
    config = config or FFTConfig(axes=None)
    return backend_module.fft.rfftn(a, s=config.s, axes=config.axes, norm=config.norm)


@numpy_eager_registry.register("Irfftnd")
def _np_irfftnd(backend_module: object, a: object, config: FFTConfig = None, **kwargs: object) -> object:
    """Evaluate the irfftnd logic eagerly backed by NumPy."""
    config = config or FFTConfig(axes=None)
    return backend_module.fft.irfftn(a, s=config.s, axes=config.axes, norm=config.norm)


@numpy_eager_registry.register("WindowHann")
def _np_window_hann(backend_module: object, length: int, *args: object, **kwargs: object) -> object:
    """Evaluate the window hann logic eagerly backed by NumPy."""
    return backend_module.hanning(length)


@numpy_eager_registry.register("WindowHamming")
def _np_window_hamming(backend_module: object, length: int, *args: object, **kwargs: object) -> object:
    """Evaluate the window hamming logic eagerly backed by NumPy."""
    return backend_module.hamming(length)


@numpy_eager_registry.register("Stft")
def _np_stft(backend_module: object, x: object, nfft: int, noverlap: int = 0, *args: object, **kwargs: object) -> object:
    """Evaluate the stft logic eagerly backed by NumPy."""
    x_arr = backend_module.asarray(x)
    step = nfft - noverlap
    if step <= 0:
        raise ValueError("noverlap must be less than nfft")
    shape = x_arr.shape[:-1] + ((x_arr.shape[-1] - noverlap) // step, nfft)
    strides = x_arr.strides[:-1] + (step * x_arr.strides[-1], x_arr.strides[-1])
    frames = backend_module.lib.stride_tricks.as_strided(x_arr, shape=shape, strides=strides)
    stft_res = backend_module.fft.rfft(frames, n=nfft, axis=-1)
    return backend_module.swapaxes(stft_res, -1, -2)


@numpy_eager_registry.register("Istft")
def _np_istft(backend_module: object, x: object, nfft: int, noverlap: int = 0, *args: object, **kwargs: object) -> object:
    """Evaluate the istft logic eagerly backed by NumPy."""
    x_arr = backend_module.asarray(x)
    step = nfft - noverlap
    if step <= 0:
        raise ValueError("noverlap must be less than nfft")
    T = x_arr.shape[-1]
    L = (T - 1) * step + nfft
    x_frames = backend_module.swapaxes(x_arr, -1, -2)
    frames = backend_module.fft.irfft(x_frames, n=nfft, axis=-1)
    out_shape = x_arr.shape[:-2] + (L,)
    out = backend_module.zeros(out_shape, dtype=frames.dtype)
    for t in range(T):
        start = t * step
        out[..., start : start + nfft] += frames[..., t, :]
    return out
