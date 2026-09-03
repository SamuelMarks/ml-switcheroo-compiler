"""Module fft_ops.py."""

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


@register_op("Fftconvolve")
class Fftconvolve(OpDef):
    """Fftconvolve."""

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0].shape


def fftconvolve(in1: Tensor, in2: Tensor, mode: str = "full", axes=None):
    """Evaluate fftconvolve operation.

    Args:
        in1 (Tensor): The in1 parameter.
        in2 (Tensor): The in2 parameter.
        mode (str): The mode parameter.
        axes (Any): The axes parameter.

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


@register_op("Fft")
class Fft(OpDef):
    """Fft class."""

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0].shape


def fft(input: Tensor, *args, **kwargs):
    """Evaluate fft operation.

    Args:
        input (Tensor): The input parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Fft", input.data, *args, **kwargs)
        return Tensor(data, TensorConfig(data.shape, input.dtype, input.device))
    return _emit_signal_node("Fft", [input], kwargs, input.shape, input.dtype)


@register_op("Rfft")
class Rfft(OpDef):
    """Rfft class."""

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0].shape


@register_op("Fft2")
class Fft2(OpDef):
    """Fft2 class."""

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0].shape


@register_op("Fftfreq")
class Fftfreq(OpDef):
    """Fftfreq class."""

    def infer_shape(self, n, d=1.0, *args, **kwargs):
        """Infer shape.

        Args:
            n (Any): The n parameter.
            d (Any): The d parameter.
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return (n,) if isinstance(n, int) else ()


@register_op("Irfft")
class Irfft(OpDef):
    """Irfft class."""

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0].shape


@register_op("Ihfft")
class Ihfft(OpDef):
    """Ihfft class."""

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0].shape


@register_op("Ifft")
class Ifft(OpDef):
    """Ifft class."""

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def ifft(input: Tensor, *args, **kwargs):
    """Evaluate ifft operation.

    Args:
        input (Tensor): The input parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

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

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def fftn(input: Tensor, *args, **kwargs):
    """Evaluate fftn operation.

    Args:
        input (Tensor): The input parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

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

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def ifftn(input: Tensor, *args, **kwargs):
    """Evaluate ifftn operation.

    Args:
        input (Tensor): The input parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

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

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def rfftn(input: Tensor, *args, **kwargs):
    """Evaluate rfftn operation.

    Args:
        input (Tensor): The input parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

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

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def irfftn(input: Tensor, *args, **kwargs):
    """Evaluate irfftn operation.

    Args:
        input (Tensor): The input parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

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

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def ifft2(input: Tensor, *args, **kwargs):
    """Evaluate ifft2 operation.

    Args:
        input (Tensor): The input parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

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

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def rfft2(input: Tensor, *args, **kwargs):
    """Evaluate rfft2 operation.

    Args:
        input (Tensor): The input parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

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

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def irfft2(input: Tensor, *args, **kwargs):
    """Evaluate irfft2 operation.

    Args:
        input (Tensor): The input parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

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

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def fftnd(input: Tensor, *args, **kwargs):
    """Evaluate fftnd operation.

    Args:
        input (Tensor): The input parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

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

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def ifftnd(input: Tensor, *args, **kwargs):
    """Evaluate ifftnd operation.

    Args:
        input (Tensor): The input parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

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

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def rfftnd(input: Tensor, *args, **kwargs):
    """Evaluate rfftnd operation.

    Args:
        input (Tensor): The input parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

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

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def irfftnd(input: Tensor, *args, **kwargs):
    """Evaluate irfftnd operation.

    Args:
        input (Tensor): The input parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

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

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def fftshift(input: Tensor, *args, **kwargs):
    """Evaluate fftshift operation.

    Args:
        input (Tensor): The input parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

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

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def ifftshift(input: Tensor, *args, **kwargs):
    """Evaluate ifftshift operation.

    Args:
        input (Tensor): The input parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

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

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def hfft(input: Tensor, *args, **kwargs):
    """Evaluate hfft operation.

    Args:
        input (Tensor): The input parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

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

    def infer_shape(self, *args, **kwargs):
        """Infer shape.

        Args:
            *args (Any): Positional args.
            **kwargs (Any): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0].shape if args and hasattr(args[0], "shape") else ()


def rfftfreq(input: int, *args, **kwargs):
    """Evaluate rfftfreq operation.

    Args:
        input (int): The input parameter.
        *args (Any): Positional args.
        **kwargs (Any): Keyword args.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        backend = get_active_backend()
        data = backend.execute_op("Rfftfreq", getattr(input, "data", input), *args, **kwargs)
        return Tensor(data, TensorConfig(getattr(data, "shape", getattr(input, "shape", ())), getattr(input, "dtype", "float32"), getattr(input, "device", None)))
    return _emit_signal_node("Rfftfreq", [input], kwargs, getattr(input, "shape", ()), getattr(input, "dtype", "float32"))
