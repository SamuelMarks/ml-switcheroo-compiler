"""Module spectral_ops.py."""

from .common_ops import _emit_signal_node

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Signal processing operations."""

from dataclasses import dataclass
from typing import Any, Optional

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


@register_op("Welch")
class Welch(OpDef):
    """Welch."""

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return args[0].shape


@dataclass
@dataclass
class WindowConfig:
    """Configuration class for window config."""

    fs: float = 1.0
    window: str = "hann"
    nperseg: Optional[int] = None
    noverlap: Optional[int] = None
    nfft: Optional[int] = None


@dataclass
class FilterState:
    """Configuration class for filter state."""

    detrend: str = "constant"
    return_onesided: bool = True
    scaling: str = "density"
    axis: int = -1
    average: str = "mean"


@dataclass
class WelchConfig:
    """Configuration class for welch config."""

    window_config: WindowConfig = WindowConfig()
    filter_state: FilterState = FilterState()


def welch(
    x: Tensor,  # type: ignore
    config_params: Optional[WelchConfig] = None,
) -> Any:
    """Evaluate welch operation.

    Args:
        x (Tensor): The x parameter.
        config_params (Optional): The config_params parameter.

    Returns:
        tuple: Result.
    """
    if config_params is None:
        config_params = WelchConfig()

    if config.eager_mode:
        backend = get_active_backend()
        f, Pxx = backend.execute_op(
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
        )
        return (
            Tensor(f, TensorConfig(f.shape, x.dtype, x.device)),
            Tensor(Pxx, TensorConfig(Pxx.shape, x.dtype, x.device)),
        )

    f_shape = (256,)
    Pxx_shape = (256,)

    f, Pxx = _emit_linalg_node(  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
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
    )
    return f, Pxx


@register_op("WindowHann")
class WindowHann(OpDef):
    """WindowHann class."""

    def infer_shape(self, length: int, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            length (int): The length parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return (length,)


@register_op("WindowHamming")
class WindowHamming(OpDef):
    """WindowHamming class."""

    def infer_shape(self, length: int, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            length (int): The length parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return (length,)


@register_op("Stft")
class Stft(OpDef):
    """Stft class."""

    def infer_shape(self, x: Any, nfft: int, noverlap: int = 0, **kwargs: Any) -> Any:
        """infer_shape function.

        Args:
            x (object): The x parameter.
            nfft (int): The nfft parameter.
            noverlap (int): The noverlap parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.

        Raises:
            ValueError: An exception.
        """
        if not hasattr(x, "shape") or not x.shape:
            return ()
        step = nfft - noverlap
        if step <= 0:
            raise ValueError("noverlap must be less than nfft")
        num_frames = (x.shape[-1] - noverlap) // step
        return x.shape[:-1] + (nfft // 2 + 1, num_frames)


@register_op("Istft")
class Istft(OpDef):
    """Istft class."""

    def infer_shape(self, x: Any, nfft: int, noverlap: int = 0, **kwargs: Any) -> Any:
        """infer_shape function.

        Args:
            x (object): The x parameter.
            nfft (int): The nfft parameter.
            noverlap (int): The noverlap parameter.
            **kwargs (object): Keyword args.

        Returns: Any: Result.

        Raises:
            ValueError: An exception.
        """
        if not hasattr(x, "shape") or not x.shape or len(x.shape) < 2:
            return ()
        step = nfft - noverlap
        if step <= 0:
            raise ValueError("noverlap must be less than nfft")
        T = x.shape[-1]
        L = (T - 1) * step + nfft
        return x.shape[:-2] + (L,)


def window_hann(length: int) -> Any:
    """Generate a Hann window.

    Args:
        length (int): The length parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("WindowHann", length)
        return Tensor(data, TensorConfig(getattr(data, "shape", (length,)), "float32", None))  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    # Note: Using float32 as default dtype for window
    return _emit_shape_node("WindowHann", [], {"length": length}, (length,), "float32")


def window_hamming(length: int) -> Any:
    """Generate a Hamming window.

    Args:
        length (int): The length parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("WindowHamming", length)
        return Tensor(data, TensorConfig(getattr(data, "shape", (length,)), "float32", None))  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    return _emit_shape_node("WindowHamming", [], {"length": length}, (length,), "float32")


def stft(x: Tensor, nfft: int, noverlap: int = 0) -> Any:  # type: ignore
    """Compute the Short Time Fourier Transform.

    Args:
        x (Tensor): The x parameter.
        nfft (int): The nfft parameter.
        noverlap (int): The noverlap parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Stft", (x.data if type(x).__name__ == "Tensor" else x), nfft=nfft, noverlap=noverlap)
        out_shape = getattr(data, "shape", Stft().infer_shape(x, nfft=nfft, noverlap=noverlap))
        return Tensor(data, TensorConfig(out_shape, x.dtype, x.device))
    out_shape = Stft().infer_shape(x, nfft=nfft, noverlap=noverlap)
    return _emit_shape_node("Stft", [x], {"nfft": nfft, "noverlap": noverlap}, out_shape, "complex64")


def istft(x: Tensor, nfft: int, noverlap: int = 0) -> Any:  # type: ignore
    """Compute the Inverse Short Time Fourier Transform.

    Args:
        x (Tensor): The x parameter.
        nfft (int): The nfft parameter.
        noverlap (int): The noverlap parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Istft", (x.data if type(x).__name__ == "Tensor" else x), nfft=nfft, noverlap=noverlap)
        out_shape = getattr(data, "shape", Istft().infer_shape(x, nfft=nfft, noverlap=noverlap))
        return Tensor(data, TensorConfig(out_shape, x.dtype, x.device))
    out_shape = Istft().infer_shape(x, nfft=nfft, noverlap=noverlap)
    return _emit_shape_node("Istft", [x], {"nfft": nfft, "noverlap": noverlap}, out_shape, "float32")
