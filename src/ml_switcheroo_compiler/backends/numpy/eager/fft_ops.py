"""Numpy FFT Ops."""

from __future__ import annotations

import threading  # noqa: F401
from dataclasses import dataclass
from types import ModuleType
from typing import cast

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, numpy_eager_registry


@dataclass
class FFTConfig:
    """FFT Configuration."""

    s: list[int] | tuple[int, ...] | None = None
    axes: list[int] | tuple[int, ...] | None = None
    norm: str | None = None


@numpy_eager_registry.register("Fft3d")
def _np_fft3d(backend_module: ModuleType, a: EagerValue, config: FFTConfig | None = None, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_fft3d operation.

    Args:
        backend_module (object): The backend_module parameter.
        a (object): The a parameter.
        config (FFTConfig): The config parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    config = config or FFTConfig(axes=(-3, -2, -1))
    return cast(EagerValue, backend_module.fft.fftn(a, s=config.s, axes=config.axes, norm=config.norm))


@numpy_eager_registry.register("Ifft3d")
def _np_ifft3d(backend_module: ModuleType, a: EagerValue, config: FFTConfig | None = None, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_ifft3d operation.

    Args:
        backend_module (object): The backend_module parameter.
        a (object): The a parameter.
        config (FFTConfig): The config parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    config = config or FFTConfig(axes=(-3, -2, -1))
    return cast(EagerValue, backend_module.fft.ifftn(a, s=config.s, axes=config.axes, norm=config.norm))


@numpy_eager_registry.register("Rfft2d")
def _np_rfft2d(backend_module: ModuleType, a: EagerValue, config: FFTConfig | None = None, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_rfft2d operation.

    Args:
        backend_module (object): The backend_module parameter.
        a (object): The a parameter.
        config (FFTConfig): The config parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    config = config or FFTConfig(axes=(-2, -1))
    return cast(EagerValue, backend_module.fft.rfftn(a, s=config.s, axes=config.axes, norm=config.norm))


@numpy_eager_registry.register("Rfft3d")
def _np_rfft3d(backend_module: ModuleType, a: EagerValue, config: FFTConfig | None = None, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_rfft3d operation.

    Args:
        backend_module (object): The backend_module parameter.
        a (object): The a parameter.
        config (FFTConfig): The config parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    config = config or FFTConfig(axes=(-3, -2, -1))
    return cast(EagerValue, backend_module.fft.rfftn(a, s=config.s, axes=config.axes, norm=config.norm))


@numpy_eager_registry.register("Irfft2d")
def _np_irfft2d(backend_module: ModuleType, a: EagerValue, config: FFTConfig | None = None, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_irfft2d operation.

    Args:
        backend_module (object): The backend_module parameter.
        a (object): The a parameter.
        config (FFTConfig): The config parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    config = config or FFTConfig(axes=(-2, -1))
    return cast(EagerValue, backend_module.fft.irfftn(a, s=config.s, axes=config.axes, norm=config.norm))


@numpy_eager_registry.register("Irfft3d")
def _np_irfft3d(backend_module: ModuleType, a: EagerValue, config: FFTConfig | None = None, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_irfft3d operation.

    Args:
        backend_module (object): The backend_module parameter.
        a (object): The a parameter.
        config (FFTConfig): The config parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    config = config or FFTConfig(axes=(-3, -2, -1))
    return cast(EagerValue, backend_module.fft.irfftn(a, s=config.s, axes=config.axes, norm=config.norm))


@numpy_eager_registry.register("Fftnd")
def _np_fftnd(backend_module: ModuleType, a: EagerValue, config: FFTConfig | None = None, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_fftnd operation.

    Args:
        backend_module (object): The backend_module parameter.
        a (object): The a parameter.
        config (FFTConfig): The config parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    config = config or FFTConfig(axes=None)
    return cast(EagerValue, backend_module.fft.fftn(a, s=config.s, axes=config.axes, norm=config.norm))


@numpy_eager_registry.register("Ifftnd")
def _np_ifftnd(backend_module: ModuleType, a: EagerValue, config: FFTConfig | None = None, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_ifftnd operation.

    Args:
        backend_module (object): The backend_module parameter.
        a (object): The a parameter.
        config (FFTConfig): The config parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    config = config or FFTConfig(axes=None)
    return cast(EagerValue, backend_module.fft.ifftn(a, s=config.s, axes=config.axes, norm=config.norm))


@numpy_eager_registry.register("Rfftnd")
def _np_rfftnd(backend_module: ModuleType, a: EagerValue, config: FFTConfig | None = None, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_rfftnd operation.

    Args:
        backend_module (object): The backend_module parameter.
        a (object): The a parameter.
        config (FFTConfig): The config parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    config = config or FFTConfig(axes=None)
    return cast(EagerValue, backend_module.fft.rfftn(a, s=config.s, axes=config.axes, norm=config.norm))


@numpy_eager_registry.register("Irfftnd")
def _np_irfftnd(backend_module: ModuleType, a: EagerValue, config: FFTConfig | None = None, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_irfftnd operation.

    Args:
        backend_module (object): The backend_module parameter.
        a (object): The a parameter.
        config (FFTConfig): The config parameter.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    config = config or FFTConfig(axes=None)
    return cast(EagerValue, backend_module.fft.irfftn(a, s=config.s, axes=config.axes, norm=config.norm))


@numpy_eager_registry.register("WindowHann")
def _np_window_hann(backend_module: ModuleType, length: int, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_window_hann operation.

    Args:
        backend_module (object): The backend_module parameter.
        length (int): The length parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.hanning(length))


@numpy_eager_registry.register("WindowHamming")
def _np_window_hamming(backend_module: ModuleType, length: int, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_window_hamming operation.

    Args:
        backend_module (object): The backend_module parameter.
        length (int): The length parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    return cast(EagerValue, backend_module.hamming(length))


@numpy_eager_registry.register("Stft")
def _np_stft(backend_module: ModuleType, x: EagerValue, nfft: int, noverlap: int = 0, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_stft operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        nfft (int): The nfft parameter.
        noverlap (int): The noverlap parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        ValueError: An exception.
    """
    x_arr: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = backend_module.asarray(x)
    step: int = nfft - noverlap
    if step <= 0:
        raise ValueError("noverlap must be less than nfft")
    shape: tuple[int, ...] = x_arr.shape[:-1] + ((x_arr.shape[-1] - noverlap) // step, nfft)
    strides: tuple[int, ...] = x_arr.strides[:-1] + (step * x_arr.strides[-1], x_arr.strides[-1])
    frames: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = backend_module.lib.stride_tricks.as_strided(x_arr, shape=shape, strides=strides)
    stft_res: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = backend_module.fft.rfft(frames, n=nfft, axis=-1)
    return cast(EagerValue, backend_module.swapaxes(stft_res, -1, -2))


@numpy_eager_registry.register("Istft")
def _np_istft(backend_module: ModuleType, x: EagerValue, nfft: int, noverlap: int = 0, *args: EagerValue, **kwargs: EagerValue) -> EagerValue:
    """Evaluate _np_istft operation.

    Args:
        backend_module (object): The backend_module parameter.
        x (object): The x parameter.
        nfft (int): The nfft parameter.
        noverlap (int): The noverlap parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.

    Raises:
        ValueError: An exception.
    """
    x_arr: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = backend_module.asarray(x)
    step: int = nfft - noverlap
    if step <= 0:
        raise ValueError("noverlap must be less than nfft")
    T = x_arr.shape[-1]
    L = (T - 1) * step + nfft
    x_frames: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = backend_module.swapaxes(x_arr, -1, -2)
    frames: np.ndarray[tuple[int, ...], np.dtype[np.generic]] = backend_module.fft.irfft(x_frames, n=nfft, axis=-1)
    out_shape: tuple[int, ...] = x_arr.shape[:-2] + (L,)
    out: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = backend_module.zeros(out_shape, dtype=frames.dtype)
    for t in range(T):
        start = t * step
        out[..., start : start + nfft] = out[..., start : start + nfft] + frames[..., t, :].astype(np.float64)
    return out
