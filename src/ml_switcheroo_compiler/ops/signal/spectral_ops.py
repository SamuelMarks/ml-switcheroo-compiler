"""Spectral analysis, window functions, and Short-Time Fourier Transform operations."""

from __future__ import annotations

import typing
from collections.abc import Sequence

import pydantic

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol, get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.interpreter.evaluator import TensorLike


@register_op("Welch")
class Welch(OpDef):
    """Welch power spectral density estimation operation definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer output shape for `Welch` from the input signal tensor.

        Args:
            *args (EagerTensorProtocol): Positional input signal arguments.
            **kwargs (TensorOrScalar): Keyword Welch configuration attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
        _ = kwargs
        return args[0].shape


class WindowConfig(pydantic.BaseModel):
    """Windowing configuration for Welch spectral density estimation."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    fs: float = 1.0
    window: str = "hann"
    nperseg: int | None = None
    noverlap: int | None = None
    nfft: int | None = None


class FilterState(pydantic.BaseModel):
    """Detrending, scaling, and averaging options for Welch spectral estimation."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    detrend: str = "constant"
    return_onesided: bool = True
    scaling: str = "density"
    axis: int = -1
    average: str = "mean"


class WelchConfig(pydantic.BaseModel):
    """Composite configuration for Welch's method (`WindowConfig` + `FilterState`)."""

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    window_config: WindowConfig = pydantic.Field(default_factory=WindowConfig)
    filter_state: FilterState = pydantic.Field(default_factory=FilterState)


def welch(
    x: Tensor,
    config_params: WelchConfig | None = None,
) -> TensorOrScalar:
    """Estimate power spectral density using Welch's overlapped periodogram method.

    Args:
        x (Tensor): Time-series input signal tensor.
        config_params (Optional[WelchConfig]): Optional window and filter configuration.

    Returns:
        TensorOrScalar: Tuple of `(sample_frequencies_tensor, power_spectral_density_tensor)`.
    """
    if config_params is None:
        config_params = WelchConfig()

    if config.eager_mode:
        backend = get_active_backend()
        f, Pxx = typing.cast(
            "tuple[TensorLike, TensorLike]",
            backend.execute_op(
                "Welch",
                x.data,
                fs=config_params.window_config.fs,
                window=config_params.window_config.window,
                nperseg=config_params.window_config.nperseg,
                noverlap=config_params.window_config.noverlap,
                nfft=config_params.window_config.nfft,
                detrend=config_params.filter_state.detrend,
                return_onesided=config_params.filter_state.return_onesided,
                scaling=config_params.filter_state.scaling,
                axis=config_params.filter_state.axis,
                average=config_params.filter_state.average,
            ),
        )
        return (
            Tensor(f, TensorConfig(f.shape, x.dtype, x.device)),
            Tensor(Pxx, TensorConfig(Pxx.shape, x.dtype, x.device)),
        )

    f_shape = (256,)
    Pxx_shape = (256,)

    f_out, Pxx_out = typing.cast(
        "tuple[Tensor, Tensor]",
        _emit_linalg_node(
            "Welch",
            [x],
            {
                "fs": config_params.window_config.fs,
                "window": config_params.window_config.window,
                "nperseg": config_params.window_config.nperseg,
                "noverlap": config_params.window_config.noverlap,
                "nfft": config_params.window_config.nfft,
                "detrend": config_params.filter_state.detrend,
                "return_onesided": config_params.filter_state.return_onesided,
                "scaling": config_params.filter_state.scaling,
                "axis": config_params.filter_state.axis,
                "average": config_params.filter_state.average,
            },
            [f_shape, Pxx_shape],
            [x.dtype, x.dtype],
        ),
    )
    return f_out, Pxx_out


@register_op("WindowHann")
class WindowHann(OpDef):
    """Hann window generation operation definition."""

    def infer_shape(self, length: int, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer 1D `(length,)` output shape for `WindowHann`.

        Args:
            length (int): Number of points in the output window.
            **kwargs (TensorOrScalar): Additional keyword attributes.

        Returns:
            TensorOrScalar: 1-tuple `(length,)`.
        """
        _ = kwargs
        return (length,)


@register_op("WindowHamming")
class WindowHamming(OpDef):
    """Hamming window generation operation definition."""

    def infer_shape(self, length: int, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer 1D `(length,)` output shape for `WindowHamming`.

        Args:
            length (int): Number of points in the output window.
            **kwargs (TensorOrScalar): Additional keyword attributes.

        Returns:
            TensorOrScalar: 1-tuple `(length,)`.
        """
        _ = kwargs
        return (length,)


@register_op("Stft")
class Stft(OpDef):
    """Short-Time Fourier Transform operation definition."""

    def infer_shape(
        self,
        x: Tensor | EagerTensorProtocol,
        nfft: int,
        noverlap: int = 0,
        **kwargs: TensorOrScalar,
    ) -> TensorOrScalar:
        """Infer output shape for Short-Time Fourier Transform.

        Args:
            x (Tensor | EagerTensorProtocol): The input signal tensor.
            nfft (int): The FFT window length.
            noverlap (int): The number of overlapping samples between frames.
            **kwargs (TensorOrScalar): Additional keyword arguments.

        Returns:
            TensorOrScalar: Inferred spectrogram output shape.

        Raises:
            ValueError: If `noverlap` is greater than or equal to `nfft`.
        """
        _ = kwargs
        if not hasattr(x, "shape") or not x.shape:
            return ()
        step = nfft - noverlap
        if step <= 0:
            raise ValueError("noverlap must be less than nfft")
        num_frames = (int(x.shape[-1]) - noverlap) // step
        return tuple(x.shape[:-1]) + (nfft // 2 + 1, num_frames)


@register_op("Istft")
class Istft(OpDef):
    """Inverse Short-Time Fourier Transform operation definition."""

    def infer_shape(
        self,
        x: Tensor | EagerTensorProtocol,
        nfft: int,
        noverlap: int = 0,
        **kwargs: TensorOrScalar,
    ) -> TensorOrScalar:
        """Infer output shape for Inverse Short-Time Fourier Transform.

        Args:
            x (Tensor | EagerTensorProtocol): The input spectrogram tensor.
            nfft (int): The FFT window length.
            noverlap (int): The number of overlapping samples between frames.
            **kwargs (TensorOrScalar): Additional keyword arguments.

        Returns:
            TensorOrScalar: Inferred time-domain signal shape.

        Raises:
            ValueError: If `noverlap` is greater than or equal to `nfft`.
        """
        _ = kwargs
        if not hasattr(x, "shape") or not x.shape or len(x.shape) < 2:
            return ()
        step = nfft - noverlap
        if step <= 0:
            raise ValueError("noverlap must be less than nfft")
        T = int(x.shape[-1])
        L = (T - 1) * step + nfft
        return tuple(x.shape[:-2]) + (L,)


def window_hann(length: int) -> TensorOrScalar:
    """Generate a symmetric Hann window of length `length`.

    Args:
        length (int): Number of points in the output window.

    Returns:
        TensorOrScalar: 1D `Tensor` of shape `(length,)` containing the Hann window.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("WindowHann", length)
        return Tensor(data, TensorConfig(getattr(data, "shape", (length,)), "float32", None))
    return _emit_shape_node("WindowHann", [], {"length": length}, (length,), "float32")


def window_hamming(length: int) -> TensorOrScalar:
    """Generate a symmetric Hamming window of length `length`.

    Args:
        length (int): Number of points in the output window.

    Returns:
        TensorOrScalar: 1D `Tensor` of shape `(length,)` containing the Hamming window.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("WindowHamming", length)
        return Tensor(data, TensorConfig(getattr(data, "shape", (length,)), "float32", None))
    return _emit_shape_node("WindowHamming", [], {"length": length}, (length,), "float32")


def stft(x: Tensor, nfft: int, noverlap: int = 0) -> TensorOrScalar:
    """Compute the Short-Time Fourier Transform (STFT) of signal `x`.

    Args:
        x (Tensor): Time-domain input signal tensor.
        nfft (int): FFT window length.
        noverlap (int): Number of overlapping samples between adjacent frames.

    Returns:
        TensorOrScalar: Complex STFT spectrogram `Tensor`.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Stft", (x.data if type(x).__name__ == "Tensor" else x), nfft=nfft, noverlap=noverlap)
        out_shape = typing.cast("tuple[int, ...]", getattr(data, "shape", Stft().infer_shape(x, nfft=nfft, noverlap=noverlap)))
        return Tensor(data, TensorConfig(out_shape, x.dtype, x.device))
    out_shape = typing.cast("tuple[int, ...]", Stft().infer_shape(x, nfft=nfft, noverlap=noverlap))
    return _emit_shape_node(
        "Stft",
        [x],
        {"nfft": nfft, "noverlap": noverlap},
        typing.cast("Sequence[int | str]", out_shape),
        "complex64",
    )


def istft(x: Tensor, nfft: int, noverlap: int = 0) -> TensorOrScalar:
    """Compute the Inverse Short-Time Fourier Transform (ISTFT) of spectrogram `x`.

    Args:
        x (Tensor): Frequency-domain STFT spectrogram tensor.
        nfft (int): FFT window length.
        noverlap (int): Number of overlapping samples between adjacent frames.

    Returns:
        TensorOrScalar: Reconstructed time-domain signal `Tensor`.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Istft", (x.data if type(x).__name__ == "Tensor" else x), nfft=nfft, noverlap=noverlap)
        out_shape = typing.cast("tuple[int, ...]", getattr(data, "shape", Istft().infer_shape(x, nfft=nfft, noverlap=noverlap)))
        return Tensor(data, TensorConfig(out_shape, x.dtype, x.device))
    out_shape = typing.cast("tuple[int, ...]", Istft().infer_shape(x, nfft=nfft, noverlap=noverlap))
    return _emit_shape_node(
        "Istft",
        [x],
        {"nfft": nfft, "noverlap": noverlap},
        typing.cast("Sequence[int | str]", out_shape),
        "float32",
    )
