"""Numpy FFT Ops."""

from __future__ import annotations

import threading  # noqa: F401

import numpy as np
import pydantic

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue, numpy_eager_registry
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol


class FFTConfig(pydantic.BaseModel):
    """FFT Configuration."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    s: list[int] | tuple[int, ...] | None = None
    axes: list[int] | tuple[int, ...] | None = None
    norm: str | None = None


@numpy_eager_registry.register("Fft3d")
def _np_fft3d(backend_module: BackendModuleProtocol, a: EagerValue, config: FFTConfig | None = None, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np fft3d operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        a (EagerValue): The a parameter.
        config (FFTConfig): The config parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    config = config or FFTConfig(axes=(-3, -2, -1))
    return backend_module.fft.fftn(a, s=config.s, axes=config.axes, norm=config.norm)


@numpy_eager_registry.register("Ifft3d")
def _np_ifft3d(backend_module: BackendModuleProtocol, a: EagerValue, config: FFTConfig | None = None, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np ifft3d operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        a (EagerValue): The a parameter.
        config (FFTConfig): The config parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    config = config or FFTConfig(axes=(-3, -2, -1))
    return backend_module.fft.ifftn(a, s=config.s, axes=config.axes, norm=config.norm)


@numpy_eager_registry.register("Rfft2d")
def _np_rfft2d(backend_module: BackendModuleProtocol, a: EagerValue, config: FFTConfig | None = None, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np rfft2d operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        a (EagerValue): The a parameter.
        config (FFTConfig): The config parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    config = config or FFTConfig(axes=(-2, -1))
    return backend_module.fft.rfftn(a, s=config.s, axes=config.axes, norm=config.norm)


@numpy_eager_registry.register("Rfft3d")
def _np_rfft3d(backend_module: BackendModuleProtocol, a: EagerValue, config: FFTConfig | None = None, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np rfft3d operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        a (EagerValue): The a parameter.
        config (FFTConfig): The config parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    config = config or FFTConfig(axes=(-3, -2, -1))
    return backend_module.fft.rfftn(a, s=config.s, axes=config.axes, norm=config.norm)


@numpy_eager_registry.register("Irfft2d")
def _np_irfft2d(backend_module: BackendModuleProtocol, a: EagerValue, config: FFTConfig | None = None, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np irfft2d operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        a (EagerValue): The a parameter.
        config (FFTConfig): The config parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    config = config or FFTConfig(axes=(-2, -1))
    return backend_module.fft.irfftn(a, s=config.s, axes=config.axes, norm=config.norm)


@numpy_eager_registry.register("Irfft3d")
def _np_irfft3d(backend_module: BackendModuleProtocol, a: EagerValue, config: FFTConfig | None = None, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np irfft3d operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        a (EagerValue): The a parameter.
        config (FFTConfig): The config parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    config = config or FFTConfig(axes=(-3, -2, -1))
    return backend_module.fft.irfftn(a, s=config.s, axes=config.axes, norm=config.norm)


@numpy_eager_registry.register("Fftnd")
def _np_fftnd(backend_module: BackendModuleProtocol, a: EagerValue, config: FFTConfig | None = None, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np fftnd operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        a (EagerValue): The a parameter.
        config (FFTConfig): The config parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    config = config or FFTConfig(axes=None)
    return backend_module.fft.fftn(a, s=config.s, axes=config.axes, norm=config.norm)


@numpy_eager_registry.register("Ifftnd")
def _np_ifftnd(backend_module: BackendModuleProtocol, a: EagerValue, config: FFTConfig | None = None, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np ifftnd operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        a (EagerValue): The a parameter.
        config (FFTConfig): The config parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    config = config or FFTConfig(axes=None)
    return backend_module.fft.ifftn(a, s=config.s, axes=config.axes, norm=config.norm)


@numpy_eager_registry.register("Rfftnd")
def _np_rfftnd(backend_module: BackendModuleProtocol, a: EagerValue, config: FFTConfig | None = None, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np rfftnd operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        a (EagerValue): The a parameter.
        config (FFTConfig): The config parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    config = config or FFTConfig(axes=None)
    return backend_module.fft.rfftn(a, s=config.s, axes=config.axes, norm=config.norm)


@numpy_eager_registry.register("Irfftnd")
def _np_irfftnd(backend_module: BackendModuleProtocol, a: EagerValue, config: FFTConfig | None = None, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np irfftnd operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        a (EagerValue): The a parameter.
        config (FFTConfig): The config parameter.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    config = config or FFTConfig(axes=None)
    return backend_module.fft.irfftn(a, s=config.s, axes=config.axes, norm=config.norm)


@numpy_eager_registry.register("WindowHann")
def _np_window_hann(backend_module: BackendModuleProtocol, length: int, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np window hann operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        length (int): The length parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.hanning(length)


@numpy_eager_registry.register("WindowHamming")
def _np_window_hamming(backend_module: BackendModuleProtocol, length: int, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np window hamming operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        length (int): The length parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return backend_module.hamming(length)


@numpy_eager_registry.register("Stft")
def _np_stft(backend_module: BackendModuleProtocol, x: EagerValue, nfft: int, noverlap: int = 0, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np stft operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        nfft (int): The nfft parameter.
        noverlap (int): The noverlap parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        ValueError: An exception.
    """
    x_arr: EagerTensorProtocol = backend_module.asarray(x)
    step: int = nfft - noverlap
    if step <= 0:
        raise ValueError("noverlap must be less than nfft")
    shape: tuple[int, ...] = x_arr.shape[:-1] + ((x_arr.shape[-1] - noverlap) // step, nfft)
    raw_strides: tuple[int, ...] = tuple(int(s) for s in x_arr.strides)
    strides: tuple[int, ...] = raw_strides[:-1] + (step * raw_strides[-1], raw_strides[-1])
    frames: EagerTensorProtocol = backend_module.lib.stride_tricks.as_strided(x_arr, shape=shape, strides=strides)
    stft_res: EagerTensorProtocol = backend_module.fft.rfft(frames, n=nfft, axis=-1)
    return backend_module.swapaxes(stft_res, -1, -2)


@numpy_eager_registry.register("Istft")
def _np_istft(backend_module: BackendModuleProtocol, x: EagerValue, nfft: int, noverlap: int = 0, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Execute the eager np istft operation on the active backend array.

    Args:
        backend_module (ModuleType): The backend_module parameter.
        x (EagerValue): The x parameter.
        nfft (int): The nfft parameter.
        noverlap (int): The noverlap parameter.
        *args (EagerValue): Positional args.
        **kwargs (EagerValue): Keyword args.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        ValueError: An exception.
    """
    x_arr: EagerTensorProtocol = backend_module.asarray(x)
    step: int = nfft - noverlap
    if step <= 0:
        raise ValueError("noverlap must be less than nfft")
    T = x_arr.shape[-1]
    L = (T - 1) * step + nfft
    x_frames: EagerTensorProtocol = backend_module.swapaxes(x_arr, -1, -2)
    frames: EagerTensorProtocol = backend_module.fft.irfft(x_frames, n=nfft, axis=-1)
    out_shape: tuple[int, ...] = x_arr.shape[:-2] + (L,)
    out: EagerTensorProtocol = backend_module.zeros(out_shape, dtype=frames.dtype)
    for t in range(T):
        start = t * step
        out[..., start : start + nfft] = out[..., start : start + nfft] + frames[..., t, :].astype(np.float64)
    return out
