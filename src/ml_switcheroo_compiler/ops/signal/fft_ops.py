"""Fast Fourier Transform (FFT) and frequency-domain signal processing operations."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol, get_active_backend
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op

from .common_ops import _emit_signal_node


@register_op("Fftconvolve")
class Fftconvolve(OpDef):
    """FFT-based N-dimensional convolution operation definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer output shape for `Fftconvolve` from the first input tensor.

        Args:
            *args (EagerTensorProtocol): Input signal and kernel tensors.
            **kwargs (TensorOrScalar): Keyword mode and axis attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
        _ = kwargs
        return args[0].shape


def fftconvolve(in1: Tensor, in2: Tensor, mode: str = "full", axes: TensorOrScalar = None) -> TensorOrScalar:
    """Convolve two N-dimensional arrays using the Fast Fourier Transform.

    Args:
        in1 (Tensor): First input tensor.
        in2 (Tensor): Second input kernel tensor.
        mode (str): Convolution output size mode (`'full'`, `'valid'`, or `'same'`).
        axes (TensorOrScalar): Optional axis or axes over which to compute the convolution.

    Returns:
        TensorOrScalar: Convolved output `Tensor`.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Fftconvolve", in1.data, in2.data, mode=mode, axes=axes)
        return Tensor(data, TensorConfig(getattr(data, "shape"), in1.dtype, in1.device))

    return _emit_signal_node(
        "Fftconvolve",
        [in1, in2],
        {"mode": mode, "axes": axes},
        in1.shape,
        in1.dtype,
    )


@register_op("Fft")
class Fft(OpDef):
    """1D discrete Fourier Transform operation definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer output shape for `Fft`.

        Args:
            *args (EagerTensorProtocol): Input tensor positional arguments.
            **kwargs (TensorOrScalar): Keyword FFT attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
        _ = kwargs
        return args[0].shape


def fft(input: Tensor, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Compute the one-dimensional discrete Fourier Transform.

    Args:
        input (Tensor): Input signal tensor.
        *args (TensorOrScalar): Optional length and axis positional arguments.
        **kwargs (TensorOrScalar): Optional keyword arguments (`n`, `axis`, `norm`).

    Returns:
        TensorOrScalar: Complex-valued 1D FFT output `Tensor`.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Fft", input.data, *args, **kwargs)
        return Tensor(data, TensorConfig(getattr(data, "shape"), input.dtype, input.device))
    return _emit_signal_node("Fft", [input], kwargs, input.shape, input.dtype)


@register_op("Rfft")
class Rfft(OpDef):
    """1D real-input discrete Fourier Transform operation definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer output shape for `Rfft`.

        Args:
            *args (EagerTensorProtocol): Input tensor positional arguments.
            **kwargs (TensorOrScalar): Keyword FFT attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
        _ = kwargs
        return args[0].shape


@register_op("Fft2")
class Fft2(OpDef):
    """2D discrete Fourier Transform operation definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer output shape for `Fft2`.

        Args:
            *args (EagerTensorProtocol): Input tensor positional arguments.
            **kwargs (TensorOrScalar): Keyword FFT attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
        _ = kwargs
        return args[0].shape


@register_op("Fftfreq")
class Fftfreq(OpDef):
    """Discrete Fourier Transform sample frequency bin operation definition."""

    def infer_shape(
        self,
        n: EagerTensorProtocol,
        d: TensorOrScalar = 1.0,
        *args: EagerTensorProtocol,
        **kwargs: TensorOrScalar,
    ) -> TensorOrScalar:
        """Infer 1D `(n,)` output shape for `Fftfreq`.

        Args:
            n (EagerTensorProtocol): Window length integer.
            d (TensorOrScalar): Sample spacing scalar.
            *args (EagerTensorProtocol): Additional positional arguments.
            **kwargs (TensorOrScalar): Additional keyword attributes.

        Returns:
            TensorOrScalar: Inferred 1-tuple `(n,)` or empty tuple `()`.
        """
        _ = (d, args, kwargs)
        return (n,) if isinstance(n, int) else ()


@register_op("Irfft")
class Irfft(OpDef):
    """Inverse 1D real-output discrete Fourier Transform operation definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer output shape for `Irfft`.

        Args:
            *args (EagerTensorProtocol): Input tensor positional arguments.
            **kwargs (TensorOrScalar): Keyword FFT attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
        _ = kwargs
        return args[0].shape


@register_op("Ihfft")
class Ihfft(OpDef):
    """Inverse Hermitian discrete Fourier Transform operation definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer output shape for `Ihfft`.

        Args:
            *args (EagerTensorProtocol): Input tensor positional arguments.
            **kwargs (TensorOrScalar): Keyword FFT attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
        _ = kwargs
        return args[0].shape


@register_op("Ifft")
class Ifft(OpDef):
    """1D inverse discrete Fourier Transform operation definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer output shape for `Ifft`.

        Args:
            *args (EagerTensorProtocol): Input tensor positional arguments.
            **kwargs (TensorOrScalar): Keyword FFT attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
        _ = kwargs
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def ifft(input: Tensor, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Compute the one-dimensional inverse discrete Fourier Transform.

    Args:
        input (Tensor): Frequency-domain input tensor.
        *args (TensorOrScalar): Optional length and axis positional arguments.
        **kwargs (TensorOrScalar): Optional keyword arguments (`n`, `axis`, `norm`).

    Returns:
        TensorOrScalar: Reconstructed time-domain `Tensor`.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Ifft", input.data, *args, **kwargs)
        return Tensor(
            data,
            TensorConfig(
                getattr(data, "shape", getattr(input, "shape", ())),
                getattr(input, "dtype", "float32"),
                getattr(input, "device", None),
            ),
        )
    return _emit_signal_node("Ifft", [input], kwargs, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


@register_op("Fftn")
class Fftn(OpDef):
    """N-dimensional discrete Fourier Transform operation definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer output shape for `Fftn`.

        Args:
            *args (EagerTensorProtocol): Input tensor positional arguments.
            **kwargs (TensorOrScalar): Keyword FFT attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
        _ = kwargs
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def fftn(input: Tensor, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Compute the N-dimensional discrete Fourier Transform.

    Args:
        input (Tensor): Input signal tensor.
        *args (TensorOrScalar): Optional shape and axes positional arguments.
        **kwargs (TensorOrScalar): Optional keyword arguments (`s`, `axes`, `norm`).

    Returns:
        TensorOrScalar: Complex N-dimensional FFT `Tensor`.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Fftn", input.data, *args, **kwargs)
        return Tensor(
            data,
            TensorConfig(
                getattr(data, "shape", getattr(input, "shape", ())),
                getattr(input, "dtype", "float32"),
                getattr(input, "device", None),
            ),
        )
    return _emit_signal_node("Fftn", [input], kwargs, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


@register_op("Ifftn")
class Ifftn(OpDef):
    """N-dimensional inverse discrete Fourier Transform operation definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer output shape for `Ifftn`.

        Args:
            *args (EagerTensorProtocol): Input tensor positional arguments.
            **kwargs (TensorOrScalar): Keyword FFT attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
        _ = kwargs
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def ifftn(input: Tensor, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Compute the N-dimensional inverse discrete Fourier Transform.

    Args:
        input (Tensor): Frequency-domain input tensor.
        *args (TensorOrScalar): Optional shape and axes positional arguments.
        **kwargs (TensorOrScalar): Optional keyword arguments (`s`, `axes`, `norm`).

    Returns:
        TensorOrScalar: Reconstructed N-dimensional `Tensor`.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Ifftn", input.data, *args, **kwargs)
        return Tensor(
            data,
            TensorConfig(
                getattr(data, "shape", getattr(input, "shape", ())),
                getattr(input, "dtype", "float32"),
                getattr(input, "device", None),
            ),
        )
    return _emit_signal_node("Ifftn", [input], kwargs, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


@register_op("Rfftn")
class Rfftn(OpDef):
    """N-dimensional real-input discrete Fourier Transform operation definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer output shape for `Rfftn`.

        Args:
            *args (EagerTensorProtocol): Input tensor positional arguments.
            **kwargs (TensorOrScalar): Keyword FFT attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
        _ = kwargs
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def rfftn(input: Tensor, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Compute the N-dimensional discrete Fourier Transform for real input.

    Args:
        input (Tensor): Real-valued input tensor.
        *args (TensorOrScalar): Optional shape and axes positional arguments.
        **kwargs (TensorOrScalar): Optional keyword arguments (`s`, `axes`, `norm`).

    Returns:
        TensorOrScalar: Complex frequency-domain `Tensor`.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Rfftn", input.data, *args, **kwargs)
        return Tensor(
            data,
            TensorConfig(
                getattr(data, "shape", getattr(input, "shape", ())),
                getattr(input, "dtype", "float32"),
                getattr(input, "device", None),
            ),
        )
    return _emit_signal_node("Rfftn", [input], kwargs, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


@register_op("Irfftn")
class Irfftn(OpDef):
    """Inverse N-dimensional real-output discrete Fourier Transform operation definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer output shape for `Irfftn`.

        Args:
            *args (EagerTensorProtocol): Input tensor positional arguments.
            **kwargs (TensorOrScalar): Keyword FFT attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
        _ = kwargs
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def irfftn(input: Tensor, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Compute the inverse of `rfftn` to produce a real-valued N-dimensional tensor.

    Args:
        input (Tensor): Hermitian-symmetric frequency-domain input tensor.
        *args (TensorOrScalar): Optional shape and axes positional arguments.
        **kwargs (TensorOrScalar): Optional keyword arguments (`s`, `axes`, `norm`).

    Returns:
        TensorOrScalar: Real-valued time-domain `Tensor`.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Irfftn", input.data, *args, **kwargs)
        return Tensor(
            data,
            TensorConfig(
                getattr(data, "shape", getattr(input, "shape", ())),
                getattr(input, "dtype", "float32"),
                getattr(input, "device", None),
            ),
        )
    return _emit_signal_node("Irfftn", [input], kwargs, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


@register_op("Ifft2")
class Ifft2(OpDef):
    """2D inverse discrete Fourier Transform operation definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer output shape for `Ifft2`.

        Args:
            *args (EagerTensorProtocol): Input tensor positional arguments.
            **kwargs (TensorOrScalar): Keyword FFT attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
        _ = kwargs
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def ifft2(input: Tensor, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Compute the 2-dimensional inverse discrete Fourier Transform.

    Args:
        input (Tensor): 2D frequency-domain input tensor.
        *args (TensorOrScalar): Optional shape and axes positional arguments.
        **kwargs (TensorOrScalar): Optional keyword arguments (`s`, `axes`, `norm`).

    Returns:
        TensorOrScalar: Reconstructed 2D `Tensor`.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Ifft2", input.data, *args, **kwargs)
        return Tensor(
            data,
            TensorConfig(
                getattr(data, "shape", getattr(input, "shape", ())),
                getattr(input, "dtype", "float32"),
                getattr(input, "device", None),
            ),
        )
    return _emit_signal_node("Ifft2", [input], kwargs, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


@register_op("Rfft2")
class Rfft2(OpDef):
    """2D real-input discrete Fourier Transform operation definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer output shape for `Rfft2`.

        Args:
            *args (EagerTensorProtocol): Input tensor positional arguments.
            **kwargs (TensorOrScalar): Keyword FFT attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
        _ = kwargs
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def rfft2(input: Tensor, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Compute the 2-dimensional discrete Fourier Transform of a real array.

    Args:
        input (Tensor): Real-valued 2D input tensor.
        *args (TensorOrScalar): Optional shape and axes positional arguments.
        **kwargs (TensorOrScalar): Optional keyword arguments (`s`, `axes`, `norm`).

    Returns:
        TensorOrScalar: Complex 2D FFT `Tensor`.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Rfft2", input.data, *args, **kwargs)
        return Tensor(
            data,
            TensorConfig(
                getattr(data, "shape", getattr(input, "shape", ())),
                getattr(input, "dtype", "float32"),
                getattr(input, "device", None),
            ),
        )
    return _emit_signal_node("Rfft2", [input], kwargs, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


@register_op("Irfft2")
class Irfft2(OpDef):
    """2D inverse real-output discrete Fourier Transform operation definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer output shape for `Irfft2`.

        Args:
            *args (EagerTensorProtocol): Input tensor positional arguments.
            **kwargs (TensorOrScalar): Keyword FFT attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
        _ = kwargs
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def irfft2(input: Tensor, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Compute the 2-dimensional inverse of `rfft2`.

    Args:
        input (Tensor): Complex frequency-domain input tensor.
        *args (TensorOrScalar): Optional shape and axes positional arguments.
        **kwargs (TensorOrScalar): Optional keyword arguments (`s`, `axes`, `norm`).

    Returns:
        TensorOrScalar: Real-valued 2D output `Tensor`.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Irfft2", input.data, *args, **kwargs)
        return Tensor(
            data,
            TensorConfig(
                getattr(data, "shape", getattr(input, "shape", ())),
                getattr(input, "dtype", "float32"),
                getattr(input, "device", None),
            ),
        )
    return _emit_signal_node("Irfft2", [input], kwargs, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


@register_op("Fftnd")
class Fftnd(OpDef):
    """Multi-dimensional discrete Fourier Transform operation definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer output shape for `Fftnd`.

        Args:
            *args (EagerTensorProtocol): Input tensor positional arguments.
            **kwargs (TensorOrScalar): Keyword FFT attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
        _ = kwargs
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def fftnd(input: Tensor, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Compute the multi-dimensional discrete Fourier Transform.

    Args:
        input (Tensor): Input signal tensor.
        *args (TensorOrScalar): Optional shape and axes positional arguments.
        **kwargs (TensorOrScalar): Optional keyword arguments.

    Returns:
        TensorOrScalar: Complex multi-dimensional FFT `Tensor`.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Fftnd", input.data, *args, **kwargs)
        return Tensor(
            data,
            TensorConfig(
                getattr(data, "shape", getattr(input, "shape", ())),
                getattr(input, "dtype", "float32"),
                getattr(input, "device", None),
            ),
        )
    return _emit_signal_node("Fftnd", [input], kwargs, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


@register_op("Ifftnd")
class Ifftnd(OpDef):
    """Multi-dimensional inverse discrete Fourier Transform operation definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer output shape for `Ifftnd`.

        Args:
            *args (EagerTensorProtocol): Input tensor positional arguments.
            **kwargs (TensorOrScalar): Keyword FFT attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
        _ = kwargs
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def ifftnd(input: Tensor, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Compute the multi-dimensional inverse discrete Fourier Transform.

    Args:
        input (Tensor): Frequency-domain input tensor.
        *args (TensorOrScalar): Optional shape and axes positional arguments.
        **kwargs (TensorOrScalar): Optional keyword arguments.

    Returns:
        TensorOrScalar: Reconstructed multi-dimensional `Tensor`.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Ifftnd", input.data, *args, **kwargs)
        return Tensor(
            data,
            TensorConfig(
                getattr(data, "shape", getattr(input, "shape", ())),
                getattr(input, "dtype", "float32"),
                getattr(input, "device", None),
            ),
        )
    return _emit_signal_node("Ifftnd", [input], kwargs, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


@register_op("Rfftnd")
class Rfftnd(OpDef):
    """Multi-dimensional real-input discrete Fourier Transform operation definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer output shape for `Rfftnd`.

        Args:
            *args (EagerTensorProtocol): Input tensor positional arguments.
            **kwargs (TensorOrScalar): Keyword FFT attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
        _ = kwargs
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def rfftnd(input: Tensor, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Compute the multi-dimensional discrete Fourier Transform for real input.

    Args:
        input (Tensor): Real-valued input tensor.
        *args (TensorOrScalar): Optional shape and axes positional arguments.
        **kwargs (TensorOrScalar): Optional keyword arguments.

    Returns:
        TensorOrScalar: Complex frequency-domain `Tensor`.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Rfftnd", input.data, *args, **kwargs)
        return Tensor(
            data,
            TensorConfig(
                getattr(data, "shape", getattr(input, "shape", ())),
                getattr(input, "dtype", "float32"),
                getattr(input, "device", None),
            ),
        )
    return _emit_signal_node("Rfftnd", [input], kwargs, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


@register_op("Irfftnd")
class Irfftnd(OpDef):
    """Multi-dimensional inverse real-output discrete Fourier Transform operation definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer output shape for `Irfftnd`.

        Args:
            *args (EagerTensorProtocol): Input tensor positional arguments.
            **kwargs (TensorOrScalar): Keyword FFT attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
        _ = kwargs
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def irfftnd(input: Tensor, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Compute the inverse of `rfftnd` to produce a real multi-dimensional tensor.

    Args:
        input (Tensor): Complex frequency-domain input tensor.
        *args (TensorOrScalar): Optional shape and axes positional arguments.
        **kwargs (TensorOrScalar): Optional keyword arguments.

    Returns:
        TensorOrScalar: Real-valued output `Tensor`.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Irfftnd", input.data, *args, **kwargs)
        return Tensor(
            data,
            TensorConfig(
                getattr(data, "shape", getattr(input, "shape", ())),
                getattr(input, "dtype", "float32"),
                getattr(input, "device", None),
            ),
        )
    return _emit_signal_node("Irfftnd", [input], kwargs, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


@register_op("Fftshift")
class Fftshift(OpDef):
    """Shift zero-frequency component to the center of the spectrum."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer output shape for `Fftshift`.

        Args:
            *args (EagerTensorProtocol): Input tensor positional arguments.
            **kwargs (TensorOrScalar): Keyword axes attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
        _ = kwargs
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def fftshift(input: Tensor, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Shift the zero-frequency component to the center of the spectrum.

    Args:
        input (Tensor): Input spectrum tensor.
        *args (TensorOrScalar): Optional axes positional arguments.
        **kwargs (TensorOrScalar): Optional keyword arguments (`axes`).

    Returns:
        TensorOrScalar: Frequency-shifted `Tensor`.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Fftshift", input.data, *args, **kwargs)
        return Tensor(
            data,
            TensorConfig(
                getattr(data, "shape", getattr(input, "shape", ())),
                getattr(input, "dtype", "float32"),
                getattr(input, "device", None),
            ),
        )
    return _emit_signal_node("Fftshift", [input], kwargs, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


@register_op("Ifftshift")
class Ifftshift(OpDef):
    """Inverse of `Fftshift` shifting zero-frequency component back to origin."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer output shape for `Ifftshift`.

        Args:
            *args (EagerTensorProtocol): Input tensor positional arguments.
            **kwargs (TensorOrScalar): Keyword axes attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
        _ = kwargs
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def ifftshift(input: Tensor, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Inverse of `fftshift`, shifting the zero-frequency component back to the origin.

    Args:
        input (Tensor): Centered spectrum tensor.
        *args (TensorOrScalar): Optional axes positional arguments.
        **kwargs (TensorOrScalar): Optional keyword arguments (`axes`).

    Returns:
        TensorOrScalar: Unshifted spectrum `Tensor`.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Ifftshift", input.data, *args, **kwargs)
        return Tensor(
            data,
            TensorConfig(
                getattr(data, "shape", getattr(input, "shape", ())),
                getattr(input, "dtype", "float32"),
                getattr(input, "device", None),
            ),
        )
    return _emit_signal_node("Ifftshift", [input], kwargs, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


@register_op("Hfft")
class Hfft(OpDef):
    """Hermitian discrete Fourier Transform operation definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer output shape for `Hfft`.

        Args:
            *args (EagerTensorProtocol): Input tensor positional arguments.
            **kwargs (TensorOrScalar): Keyword FFT attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
        _ = kwargs
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def hfft(input: Tensor, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Compute the FFT of a signal that has Hermitian symmetry, yielding a real spectrum.

    Args:
        input (Tensor): Hermitian-symmetric input tensor.
        *args (TensorOrScalar): Optional length and axis positional arguments.
        **kwargs (TensorOrScalar): Optional keyword arguments (`n`, `axis`, `norm`).

    Returns:
        TensorOrScalar: Real-valued spectrum `Tensor`.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Hfft", input.data, *args, **kwargs)
        return Tensor(
            data,
            TensorConfig(
                getattr(data, "shape", getattr(input, "shape", ())),
                getattr(input, "dtype", "float32"),
                getattr(input, "device", None),
            ),
        )
    return _emit_signal_node("Hfft", [input], kwargs, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))


@register_op("Rfftfreq")
class Rfftfreq(OpDef):
    """Real-input discrete Fourier Transform sample frequency bin operation definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer output shape for `Rfftfreq`.

        Args:
            *args (EagerTensorProtocol): Input window length arguments.
            **kwargs (TensorOrScalar): Keyword sample spacing attributes.

        Returns:
            TensorOrScalar: Inferred output shape tuple.
        """
        _ = kwargs
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def rfftfreq(input: int | Tensor, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Return the Discrete Fourier Transform sample frequencies for usage with `rfft`.

    Args:
        input (int | Tensor): Window length integer or scalar tensor.
        *args (TensorOrScalar): Optional sample spacing positional arguments.
        **kwargs (TensorOrScalar): Optional keyword arguments (`d`).

    Returns:
        TensorOrScalar: 1D `Tensor` of sample frequency bin centers.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Rfftfreq", getattr(input, "data", input), *args, **kwargs)
        return Tensor(
            data,
            TensorConfig(
                getattr(data, "shape", getattr(input, "shape", ())),
                getattr(input, "dtype", "float32"),
                getattr(input, "device", None),
            ),
        )
    t_in = input if isinstance(input, Tensor) else Tensor(input, TensorConfig((), "int64", None))
    return _emit_signal_node("Rfftfreq", [t_in], kwargs, getattr(t_in, "shape", ()), getattr(t_in, "dtype", "float32"))
