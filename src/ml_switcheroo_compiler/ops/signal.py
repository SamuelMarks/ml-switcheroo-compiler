"""Signal processing operations."""

from dataclasses import dataclass
from typing import Optional

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node


@register_op("Convolve2d")
class Convolve2d(OpDef):
    """Convolve2d."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
        object: Result.
        """
        return args[0].shape


@register_op("Fftconvolve")
class Fftconvolve(OpDef):
    """Fftconvolve."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return args[0].shape


@register_op("Welch")
class Welch(OpDef):
    """Welch."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return args[0].shape


def _emit_signal_node(
    op_type: str,
    inputs: list[Tensor],
    attrs: dict[str, object],
    out_shape: tuple[int, ...],
    dtype: str,
) -> Tensor:
    """Emit a signal node.

    Args:
        op_type (str): The op_type parameter.
        inputs (list): The inputs parameter.
        attrs (dict): The attrs parameter.
        out_shape (tuple): The out_shape parameter.
        dtype (str): The dtype parameter.

    Returns:
        Tensor: Result.
    """
    return _emit_linalg_node(op_type, inputs, attrs, [out_shape], [dtype])


def _validate_conv2d_args(in1: Tensor, in2: Tensor) -> None:
    """Validate arguments for convolve2d.

    Args:
        in1 (Tensor): First input.
        in2 (Tensor): Second input.

    Raises:
        ValueError: If shapes are not statically known.
    """
    if in1.shape is None or in2.shape is None:
        raise ValueError("Inputs to convolve2d must have statically known shapes.")


def _calculate_padding(mode: str, boundary: str, fillvalue: float) -> dict[str, object]:
    """Calculate padding configuration for convolve2d.

    Args:
        mode (str): Padding mode.
        boundary (str): Boundary condition.
        fillvalue (float): Fill value for 'fill' boundary.

    Returns:
        dict[str, object]: Padding configuration dictionary.
    """
    return {"mode": mode, "boundary": boundary, "fillvalue": fillvalue}


def convolve2d(
    in1: Tensor,
    in2: Tensor,
    mode: str = "full",
    boundary: str = "fill",
    fillvalue: float = 0.0,
) -> Tensor:
    """Evaluate convolve2d operation.

    Args:
        in1 (Tensor): The in1 parameter.
        in2 (Tensor): The in2 parameter.
        mode (str): The mode parameter.
        boundary (str): The boundary parameter.
        fillvalue (float): The fillvalue parameter.

    Returns:
        Tensor: Result.
    """
    _validate_conv2d_args(in1, in2)
    kwargs = _calculate_padding(mode, boundary, fillvalue)

    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Convolve2d", in1.data, in2.data, **kwargs)
        return Tensor(data, TensorConfig(data.shape, in1.dtype, in1.device))

    return _emit_signal_node(
        "Convolve2d",
        [in1, in2],
        kwargs,
        in1.shape,
        in1.dtype,
    )


def fftconvolve(in1: Tensor, in2: Tensor, mode: str = "full", axes: object = None) -> Tensor:
    """Evaluate fftconvolve operation.

    Args:
        in1 (Tensor): The in1 parameter.
        in2 (Tensor): The in2 parameter.
        mode (str): The mode parameter.
        axes (object): The axes parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()

        data = backend.execute_op("Fftconvolve", in1.data, in2.data, mode=mode, axes=axes)

        return Tensor(data, TensorConfig(data.shape, in1.dtype, in1.device))

    return _emit_signal_node(
        "Fftconvolve",
        [in1, in2],
        {"mode": mode, "axes": axes},
        in1.shape,
        in1.dtype,
    )


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
    x: Tensor,
    config_params: Optional[WelchConfig] = None,
) -> tuple[Tensor, Tensor]:
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

    f, Pxx = _emit_linalg_node(
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


@register_op("Fft")
class Fft(OpDef):
    """Fft class."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return args[0].shape


def fft(input: Tensor, *args: object, **kwargs: object) -> Tensor:
    """Evaluate fft operation.

    Args:
        input (Tensor): The input parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Fft", input.data, *args, **kwargs)
        return Tensor(data, TensorConfig(data.shape, input.dtype, input.device))
    return _emit_signal_node("Fft", [input], kwargs, input.shape, input.dtype)


__all__ = ["convolve2d", "fftconvolve", "welch", "fft", "window_hann", "window_hamming", "stft", "istft", "ifft", "fftn", "ifftn", "rfftn", "irfftn", "ifft2", "rfft2", "irfft2", "fftnd", "ifftnd", "rfftnd", "irfftnd", "fftshift", "ifftshift", "hfft", "rfftfreq"]


@register_op("Rfft")
class Rfft(OpDef):
    """Rfft class."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return args[0].shape


@register_op("Fft2")
class Fft2(OpDef):
    """Fft2 class."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return args[0].shape


@register_op("Fftfreq")
class Fftfreq(OpDef):
    """Fftfreq class."""

    def infer_shape(self, n: object, d: object = 1.0, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            n (object): The n parameter.
            d (object): The d parameter.
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return (n,) if isinstance(n, int) else ()


@register_op("Irfft")
class Irfft(OpDef):
    """Irfft class."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return args[0].shape


@register_op("Ihfft")
class Ihfft(OpDef):
    """Ihfft class."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return args[0].shape


@register_op("WindowHann")
class WindowHann(OpDef):
    """WindowHann class."""

    def infer_shape(self, length: int, **kwargs: object) -> object:
        """Infer shape.

        Args:
            length (int): The length parameter.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return (length,)


@register_op("WindowHamming")
class WindowHamming(OpDef):
    """WindowHamming class."""

    def infer_shape(self, length: int, **kwargs: object) -> object:
        """Infer shape.

        Args:
            length (int): The length parameter.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return (length,)


@register_op("Stft")
class Stft(OpDef):
    """Stft class."""

    def infer_shape(self, x: object, nfft: int, noverlap: int = 0, **kwargs: object) -> object:
        """infer_shape function.

        Args:
            x (object): The x parameter.
            nfft (int): The nfft parameter.
            noverlap (int): The noverlap parameter.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.

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

    def infer_shape(self, x: object, nfft: int, noverlap: int = 0, **kwargs: object) -> object:
        """infer_shape function.

        Args:
            x (object): The x parameter.
            nfft (int): The nfft parameter.
            noverlap (int): The noverlap parameter.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.

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


def window_hann(length: int) -> Tensor:
    """Generate a Hann window.

    Args:
        length (int): The length parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("WindowHann", length)
        return Tensor(data, TensorConfig(getattr(data, "shape", (length,)), "float32", None))
    # Note: Using float32 as default dtype for window
    return _emit_shape_node("WindowHann", [], {"length": length}, (length,), "float32")


def window_hamming(length: int) -> Tensor:
    """Generate a Hamming window.

    Args:
        length (int): The length parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("WindowHamming", length)
        return Tensor(data, TensorConfig(getattr(data, "shape", (length,)), "float32", None))
    return _emit_shape_node("WindowHamming", [], {"length": length}, (length,), "float32")


def stft(x: Tensor, nfft: int, noverlap: int = 0) -> Tensor:
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


def istft(x: Tensor, nfft: int, noverlap: int = 0) -> Tensor:
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


@register_op("Ifft")
class Ifft(OpDef):
    """Ifft class."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def ifft(input: Tensor, *args: object, **kwargs: object) -> Tensor:
    """Evaluate ifft operation.

    Args:
        input (Tensor): The input parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Ifft", input.data, *args, **kwargs)
        return Tensor(data, TensorConfig(getattr(data, "shape", getattr(input, "shape", ())), getattr(input, "dtype", "float32"), getattr(input, "device", None)))
    return _emit_signal_node("Ifft", [input], kwargs, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


@register_op("Fftn")
class Fftn(OpDef):
    """Fftn class."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def fftn(input: Tensor, *args: object, **kwargs: object) -> Tensor:
    """Evaluate fftn operation.

    Args:
        input (Tensor): The input parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Fftn", input.data, *args, **kwargs)
        return Tensor(data, TensorConfig(getattr(data, "shape", getattr(input, "shape", ())), getattr(input, "dtype", "float32"), getattr(input, "device", None)))
    return _emit_signal_node("Fftn", [input], kwargs, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


@register_op("Ifftn")
class Ifftn(OpDef):
    """Ifftn class."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def ifftn(input: Tensor, *args: object, **kwargs: object) -> Tensor:
    """Evaluate ifftn operation.

    Args:
        input (Tensor): The input parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Ifftn", input.data, *args, **kwargs)
        return Tensor(data, TensorConfig(getattr(data, "shape", getattr(input, "shape", ())), getattr(input, "dtype", "float32"), getattr(input, "device", None)))
    return _emit_signal_node("Ifftn", [input], kwargs, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


@register_op("Rfftn")
class Rfftn(OpDef):
    """Rfftn class."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def rfftn(input: Tensor, *args: object, **kwargs: object) -> Tensor:
    """Evaluate rfftn operation.

    Args:
        input (Tensor): The input parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Rfftn", input.data, *args, **kwargs)
        return Tensor(data, TensorConfig(getattr(data, "shape", getattr(input, "shape", ())), getattr(input, "dtype", "float32"), getattr(input, "device", None)))
    return _emit_signal_node("Rfftn", [input], kwargs, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


@register_op("Irfftn")
class Irfftn(OpDef):
    """Irfftn class."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def irfftn(input: Tensor, *args: object, **kwargs: object) -> Tensor:
    """Evaluate irfftn operation.

    Args:
        input (Tensor): The input parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Irfftn", input.data, *args, **kwargs)
        return Tensor(data, TensorConfig(getattr(data, "shape", getattr(input, "shape", ())), getattr(input, "dtype", "float32"), getattr(input, "device", None)))
    return _emit_signal_node("Irfftn", [input], kwargs, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


@register_op("Ifft2")
class Ifft2(OpDef):
    """Ifft2 class."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def ifft2(input: Tensor, *args: object, **kwargs: object) -> Tensor:
    """Evaluate ifft2 operation.

    Args:
        input (Tensor): The input parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Ifft2", input.data, *args, **kwargs)
        return Tensor(data, TensorConfig(getattr(data, "shape", getattr(input, "shape", ())), getattr(input, "dtype", "float32"), getattr(input, "device", None)))
    return _emit_signal_node("Ifft2", [input], kwargs, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


@register_op("Rfft2")
class Rfft2(OpDef):
    """Rfft2 class."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def rfft2(input: Tensor, *args: object, **kwargs: object) -> Tensor:
    """Evaluate rfft2 operation.

    Args:
        input (Tensor): The input parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Rfft2", input.data, *args, **kwargs)
        return Tensor(data, TensorConfig(getattr(data, "shape", getattr(input, "shape", ())), getattr(input, "dtype", "float32"), getattr(input, "device", None)))
    return _emit_signal_node("Rfft2", [input], kwargs, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


@register_op("Irfft2")
class Irfft2(OpDef):
    """Irfft2 class."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def irfft2(input: Tensor, *args: object, **kwargs: object) -> Tensor:
    """Evaluate irfft2 operation.

    Args:
        input (Tensor): The input parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Irfft2", input.data, *args, **kwargs)
        return Tensor(data, TensorConfig(getattr(data, "shape", getattr(input, "shape", ())), getattr(input, "dtype", "float32"), getattr(input, "device", None)))
    return _emit_signal_node("Irfft2", [input], kwargs, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


@register_op("Fftnd")
class Fftnd(OpDef):
    """Fftnd class."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def fftnd(input: Tensor, *args: object, **kwargs: object) -> Tensor:
    """Evaluate fftnd operation.

    Args:
        input (Tensor): The input parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Fftnd", input.data, *args, **kwargs)
        return Tensor(data, TensorConfig(getattr(data, "shape", getattr(input, "shape", ())), getattr(input, "dtype", "float32"), getattr(input, "device", None)))
    return _emit_signal_node("Fftnd", [input], kwargs, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


@register_op("Ifftnd")
class Ifftnd(OpDef):
    """Ifftnd class."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def ifftnd(input: Tensor, *args: object, **kwargs: object) -> Tensor:
    """Evaluate ifftnd operation.

    Args:
        input (Tensor): The input parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Ifftnd", input.data, *args, **kwargs)
        return Tensor(data, TensorConfig(getattr(data, "shape", getattr(input, "shape", ())), getattr(input, "dtype", "float32"), getattr(input, "device", None)))
    return _emit_signal_node("Ifftnd", [input], kwargs, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


@register_op("Rfftnd")
class Rfftnd(OpDef):
    """Rfftnd class."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def rfftnd(input: Tensor, *args: object, **kwargs: object) -> Tensor:
    """Evaluate rfftnd operation.

    Args:
        input (Tensor): The input parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Rfftnd", input.data, *args, **kwargs)
        return Tensor(data, TensorConfig(getattr(data, "shape", getattr(input, "shape", ())), getattr(input, "dtype", "float32"), getattr(input, "device", None)))
    return _emit_signal_node("Rfftnd", [input], kwargs, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


@register_op("Irfftnd")
class Irfftnd(OpDef):
    """Irfftnd class."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def irfftnd(input: Tensor, *args: object, **kwargs: object) -> Tensor:
    """Evaluate irfftnd operation.

    Args:
        input (Tensor): The input parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Irfftnd", input.data, *args, **kwargs)
        return Tensor(data, TensorConfig(getattr(data, "shape", getattr(input, "shape", ())), getattr(input, "dtype", "float32"), getattr(input, "device", None)))
    return _emit_signal_node("Irfftnd", [input], kwargs, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


@register_op("Fftshift")
class Fftshift(OpDef):
    """Fftshift class."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def fftshift(input: Tensor, *args: object, **kwargs: object) -> Tensor:
    """Evaluate fftshift operation.

    Args:
        input (Tensor): The input parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Fftshift", input.data, *args, **kwargs)
        return Tensor(data, TensorConfig(getattr(data, "shape", getattr(input, "shape", ())), getattr(input, "dtype", "float32"), getattr(input, "device", None)))
    return _emit_signal_node("Fftshift", [input], kwargs, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


@register_op("Ifftshift")
class Ifftshift(OpDef):
    """Ifftshift class."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def ifftshift(input: Tensor, *args: object, **kwargs: object) -> Tensor:
    """Evaluate ifftshift operation.

    Args:
        input (Tensor): The input parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Ifftshift", input.data, *args, **kwargs)
        return Tensor(data, TensorConfig(getattr(data, "shape", getattr(input, "shape", ())), getattr(input, "dtype", "float32"), getattr(input, "device", None)))
    return _emit_signal_node("Ifftshift", [input], kwargs, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


@register_op("Hfft")
class Hfft(OpDef):
    """Hfft class."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def hfft(input: Tensor, *args: object, **kwargs: object) -> Tensor:
    """Evaluate hfft operation.

    Args:
        input (Tensor): The input parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Hfft", input.data, *args, **kwargs)
        return Tensor(data, TensorConfig(getattr(data, "shape", getattr(input, "shape", ())), getattr(input, "dtype", "float32"), getattr(input, "device", None)))
    return _emit_signal_node("Hfft", [input], kwargs, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


@register_op("Rfftfreq")
class Rfftfreq(OpDef):
    """Rfftfreq class."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def rfftfreq(input: int, *args: object, **kwargs: object) -> Tensor:
    """Evaluate rfftfreq operation.

    Args:
        input (int): The input parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Rfftfreq", getattr(input, "data", input), *args, **kwargs)
        return Tensor(data, TensorConfig(getattr(data, "shape", getattr(input, "shape", ())), getattr(input, "dtype", "float32"), getattr(input, "device", None)))
    return _emit_signal_node("Rfftfreq", [input], kwargs, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))
