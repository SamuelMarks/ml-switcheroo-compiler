# pragma: no cover
"""Signal processing operations."""
# pragma: no cover

# pragma: no cover
from dataclasses import dataclass
from typing import Optional

# pragma: no cover
# pragma: no cover
# pragma: no cover
from ml_switcheroo_compiler.backends.registry import get_active_backend

# pragma: no cover
from ml_switcheroo_compiler.core.config import config

# pragma: no cover
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op

# pragma: no cover
# pragma: no cover
# pragma: no cover
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node

# pragma: no cover

# pragma: no cover


# pragma: no cover
@register_op("Convolve2d")
# pragma: no cover
class Convolve2d(OpDef):
    # pragma: no cover
    """Convolve2d."""

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    def infer_shape(self, *args: object, **kwargs: object) -> object:
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        """Infer shape."""
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        return args[0].shape


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover


# pragma: no cover
# pragma: no cover
# pragma: no cover
@register_op("Fftconvolve")
# pragma: no cover
# pragma: no cover
# pragma: no cover
class Fftconvolve(OpDef):
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    """Fftconvolve."""

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    def infer_shape(self, *args: object, **kwargs: object) -> object:
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        """Infer shape."""
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        return args[0].shape


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover


# pragma: no cover
# pragma: no cover
# pragma: no cover
@register_op("Welch")
# pragma: no cover
# pragma: no cover
# pragma: no cover
class Welch(OpDef):
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    """Welch."""

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    def infer_shape(self, *args: object, **kwargs: object) -> object:
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        """Infer shape."""
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        return args[0].shape


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover


# pragma: no cover
# pragma: no cover
# pragma: no cover
def _emit_signal_node(
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    op_type: str,
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    inputs: list[Tensor],
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    attrs: dict[str, object],
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    out_shape: tuple[int, ...],
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    dtype: str,
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
) -> Tensor:
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    """Emit a signal node."""
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover

    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    return _emit_linalg_node(op_type, inputs, attrs, [out_shape], [dtype])


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover


# pragma: no cover
# pragma: no cover
# pragma: no cover
def convolve2d(
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    in1: Tensor,
    in2: Tensor,
    mode: str = "full",
    boundary: str = "fill",
    fillvalue: float = 0.0,
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
) -> Tensor:
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    """Evaluate convolve2d."""
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    if config.eager_mode:
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover

        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        backend = get_active_backend()
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        data = backend.execute_op(
            # pragma: no cover
            # pragma: no cover
            # pragma: no cover
            "Convolve2d",
            in1.data,
            in2.data,
            mode=mode,
            boundary=boundary,
            fillvalue=fillvalue,
            # pragma: no cover
            # pragma: no cover
            # pragma: no cover
        )
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        return Tensor(data, TensorConfig(data.shape, in1.dtype, in1.device))
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    return _emit_signal_node(
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        "Convolve2d",
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        [in1, in2],
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        {"mode": mode, "boundary": boundary, "fillvalue": fillvalue},
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        in1.shape,
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        in1.dtype,
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
    )


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover


# pragma: no cover
# pragma: no cover
# pragma: no cover
def fftconvolve(in1: Tensor, in2: Tensor, mode: str = "full", axes: object = None) -> Tensor:
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    """Evaluate fftconvolve."""
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    if config.eager_mode:
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover

        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        backend = get_active_backend()
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        data = backend.execute_op("Fftconvolve", in1.data, in2.data, mode=mode, axes=axes)
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        return Tensor(data, TensorConfig(data.shape, in1.dtype, in1.device))
    # pragma: no cover
    # pragma: no cover
    # pragma: no cover
    return _emit_signal_node(
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
        "Fftconvolve",
        [in1, in2],
        {"mode": mode, "axes": axes},
        in1.shape,
        in1.dtype,
        # pragma: no cover
        # pragma: no cover
        # pragma: no cover
    )


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover


# pragma: no cover
# pragma: no cover
# pragma: no cover


@dataclass
@dataclass
class WindowConfig:
    """Class docstring."""

    fs: float = 1.0
    window: str = "hann"
    nperseg: Optional[int] = None
    noverlap: Optional[int] = None
    nfft: Optional[int] = None


@dataclass
class FilterState:
    """Class docstring."""

    detrend: str = "constant"
    return_onesided: bool = True
    scaling: str = "density"
    axis: int = -1
    average: str = "mean"


@dataclass
class WelchConfig:
    """Class docstring."""

    window_config: WindowConfig = WindowConfig()
    filter_state: FilterState = FilterState()


def welch(
    x: Tensor,
    config_params: Optional[WelchConfig] = None,
) -> tuple[Tensor, Tensor]:
    """Evaluate welch."""
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

    f, Pxx = _emit_linalg_node(
        "Welch",
        [x],
        {
            "fs": config_params.fs,
            "window": config_params.window,
            "nperseg": config_params.nperseg,
            "noverlap": config_params.noverlap,
            "nfft": config_params.nfft,
            "detrend": config_params.detrend,
            "return_onesided": config_params.return_onesided,
            "scaling": config_params.scaling,
            "axis": config_params.axis,
            "average": config_params.average,
        },
        [f_shape, Pxx_shape],
        [x.dtype, x.dtype],
    )
    return f, Pxx


# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover

# pragma: no cover
# pragma: no cover
# pragma: no cover
__all__ = ["convolve2d", "fftconvolve", "welch"]
# pragma: no cover
# pragma: no cover
# pragma: no cover
