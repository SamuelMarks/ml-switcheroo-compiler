"""Linear algebra operations."""

from __future__ import annotations


from typing import TYPE_CHECKING
from collections.abc import Sequence

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.linalg.frontend import _emit_linalg_node

if TYPE_CHECKING:
    pass


def fft(a: Tensor, n: int | None = None, axis: int = -1) -> Tensor:
    """Computes the one-dimensional discrete Fourier Transform.

    Args:
        a (Tensor): The input tensor
        n (int | None): Length of the transformed axis of the output
        axis (int): Axis over which to compute the FFT

    Returns:
    Tensor: The transformed tensor

    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Fft", a.data, n=n, axis=axis)
        # Note: returning proper complex type is complex, using a mock DType mapping if possible
        # We will just return float32 here if complex not supported
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))

    from ml_switcheroo_compiler.ops.linalg.basic import Fft

    op = Fft()
    out_shape = op.infer_shape(a, n, axis)

    return _emit_linalg_node("Fft", [a], {"n": n, "axis": axis}, [out_shape], [a.dtype])


def rfft(a: Tensor, n: int | None = None, axis: int = -1) -> Tensor:
    """Computes the one-dimensional discrete Fourier Transform for real input.

    Args:
        a (Tensor): The input tensor
        n (int | None): Length of the transformed axis of the output
        axis (int): Axis over which to compute the FFT

    Returns:
    Tensor: The transformed tensor

    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Rfft", a.data, n=n, axis=axis)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))

    from ml_switcheroo_compiler.ops.linalg.basic import Rfft

    op = Rfft()
    out_shape = op.infer_shape(a, n, axis)

    return _emit_linalg_node("Rfft", [a], {"n": n, "axis": axis}, [out_shape], [a.dtype])


def ifft(a: Tensor, n: int | None = None, axis: int = -1) -> Tensor:
    """Computes the one-dimensional inverse discrete Fourier Transform.

    Args:
        a (Tensor): The input tensor
        n (int | None): Length of the transformed axis of the output
        axis (int): Axis over which to compute the IFFT

    Returns:
    Tensor: The transformed tensor

    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Ifft", a.data, n=n, axis=axis)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))

    from ml_switcheroo_compiler.ops.linalg.basic import Fft

    op = Fft()
    out_shape = op.infer_shape(a, n, axis)

    return _emit_linalg_node("Ifft", [a], {"n": n, "axis": axis}, [out_shape], [a.dtype])


def fft2d(a: Tensor, s: tuple[int, int] | None = None, axes: tuple[int, int] = (-2, -1)) -> Tensor:
    """Computes the 2-dimensional discrete Fourier Transform.

    Args:
        a (Tensor): The input tensor
        s (tuple[int, int] | None): Shape of the result
        axes (tuple[int, int]): Axes over which to compute the FFT

    Returns:
    Tensor: The transformed tensor

    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Fft2d", a.data, s=s, axes=axes)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))

    out_shape = list(a.shape)
    if s is not None:
        out_shape[axes[0]] = s[0]
        out_shape[axes[1]] = s[1]

    return _emit_linalg_node("Fft2d", [a], {"s": s, "axes": axes}, [tuple(out_shape)], [a.dtype])


def ifft2d(a: Tensor, s: tuple[int, int] | None = None, axes: tuple[int, int] = (-2, -1)) -> Tensor:
    """Computes the 2-dimensional inverse discrete Fourier Transform.

    Args:
        a (Tensor): The input tensor
        s (tuple[int, int] | None): Shape of the result
        axes (tuple[int, int]): Axes over which to compute the IFFT

    Returns:
    Tensor: The transformed tensor

    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Ifft2d", a.data, s=s, axes=axes)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))

    out_shape = list(a.shape)
    if s is not None:
        out_shape[axes[0]] = s[0]
        out_shape[axes[1]] = s[1]

    return _emit_linalg_node("Ifft2d", [a], {"s": s, "axes": axes}, [tuple(out_shape)], [a.dtype])


def irfft(a: Tensor, n: int | None = None, axis: int = -1) -> Tensor:
    """Computes the one-dimensional inverse discrete Fourier Transform for real input.

    Args:
        a (Tensor): The input tensor
        n (int | None): Length of the transformed axis of the output
        axis (int): Axis over which to compute the IFFT

    Returns:
    Tensor: The transformed tensor

    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Irfft", a.data, n=n, axis=axis)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))

    out_shape = list(a.shape)
    if n is not None:
        out_shape[axis] = n
    else:
        out_shape[axis] = 2 * (a.shape[axis] - 1)

    return _emit_linalg_node("Irfft", [a], {"n": n, "axis": axis}, [tuple(out_shape)], [a.dtype])


def fft3d(
    a: Tensor, s: tuple[int, int, int] | None = None, axes: tuple[int, int, int] = (-3, -2, -1)
) -> Tensor:
    """Computes the 3-dimensional discrete Fourier Transform.

    Args:
        a (Tensor): The input tensor
        s (tuple[int, int, int] | None): Shape of the result
        axes (tuple[int, int, int]): Axes over which to compute the FFT

    Returns:
    Tensor: The transformed tensor
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Fft3d", a.data, s=s, axes=axes)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))

    out_shape = list(a.shape)
    if s is not None:
        out_shape[axes[0]] = s[0]
        out_shape[axes[1]] = s[1]
        out_shape[axes[2]] = s[2]

    return _emit_linalg_node("Fft3d", [a], {"s": s, "axes": axes}, [tuple(out_shape)], [a.dtype])


def ifft3d(
    a: Tensor, s: tuple[int, int, int] | None = None, axes: tuple[int, int, int] = (-3, -2, -1)
) -> Tensor:
    """Computes the 3-dimensional inverse discrete Fourier Transform.

    Args:
        a (Tensor): The input tensor
        s (tuple[int, int, int] | None): Shape of the result
        axes (tuple[int, int, int]): Axes over which to compute the IFFT

    Returns:
    Tensor: The transformed tensor
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Ifft3d", a.data, s=s, axes=axes)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))

    out_shape = list(a.shape)
    if s is not None:
        out_shape[axes[0]] = s[0]
        out_shape[axes[1]] = s[1]
        out_shape[axes[2]] = s[2]

    return _emit_linalg_node("Ifft3d", [a], {"s": s, "axes": axes}, [tuple(out_shape)], [a.dtype])


def rfft2d(a: Tensor, s: tuple[int, int] | None = None, axes: tuple[int, int] = (-2, -1)) -> Tensor:
    """Computes the 2-dimensional discrete Fourier Transform for real input.

    Args:
        a (Tensor): The input tensor
        s (tuple[int, int] | None): Shape of the result
        axes (tuple[int, int]): Axes over which to compute the FFT

    Returns:
    Tensor: The transformed tensor
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Rfft2d", a.data, s=s, axes=axes)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))

    out_shape = list(a.shape)
    if s is not None:
        out_shape[axes[0]] = s[0]
        out_shape[axes[1]] = s[1] // 2 + 1
    else:
        out_shape[axes[1]] = out_shape[axes[1]] // 2 + 1

    return _emit_linalg_node("Rfft2d", [a], {"s": s, "axes": axes}, [tuple(out_shape)], [a.dtype])


def rfft3d(
    a: Tensor, s: tuple[int, int, int] | None = None, axes: tuple[int, int, int] = (-3, -2, -1)
) -> Tensor:
    """Computes the 3-dimensional discrete Fourier Transform for real input.

    Args:
        a (Tensor): The input tensor
        s (tuple[int, int, int] | None): Shape of the result
        axes (tuple[int, int, int]): Axes over which to compute the FFT

    Returns:
    Tensor: The transformed tensor
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Rfft3d", a.data, s=s, axes=axes)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))

    out_shape = list(a.shape)
    if s is not None:
        out_shape[axes[0]] = s[0]
        out_shape[axes[1]] = s[1]
        out_shape[axes[2]] = s[2] // 2 + 1
    else:
        out_shape[axes[2]] = out_shape[axes[2]] // 2 + 1

    return _emit_linalg_node("Rfft3d", [a], {"s": s, "axes": axes}, [tuple(out_shape)], [a.dtype])


def irfft2d(
    a: Tensor, s: tuple[int, int] | None = None, axes: tuple[int, int] = (-2, -1)
) -> Tensor:
    """Computes the 2-dimensional inverse discrete Fourier Transform for real input.

    Args:
        a (Tensor): The input tensor
        s (tuple[int, int] | None): Shape of the result
        axes (tuple[int, int]): Axes over which to compute the IFFT

    Returns:
    Tensor: The transformed tensor
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Irfft2d", a.data, s=s, axes=axes)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))

    out_shape = list(a.shape)
    if s is not None:
        out_shape[axes[0]] = s[0]
        out_shape[axes[1]] = s[1]
    else:
        out_shape[axes[1]] = 2 * (out_shape[axes[1]] - 1)

    return _emit_linalg_node("Irfft2d", [a], {"s": s, "axes": axes}, [tuple(out_shape)], [a.dtype])


def irfft3d(
    a: Tensor, s: tuple[int, int, int] | None = None, axes: tuple[int, int, int] = (-3, -2, -1)
) -> Tensor:
    """Computes the 3-dimensional inverse discrete Fourier Transform for real input.

    Args:
        a (Tensor): The input tensor
        s (tuple[int, int, int] | None): Shape of the result
        axes (tuple[int, int, int]): Axes over which to compute the IFFT

    Returns:
    Tensor: The transformed tensor
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Irfft3d", a.data, s=s, axes=axes)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))

    out_shape = list(a.shape)
    if s is not None:
        out_shape[axes[0]] = s[0]
        out_shape[axes[1]] = s[1]
        out_shape[axes[2]] = s[2]
    else:
        out_shape[axes[2]] = 2 * (out_shape[axes[2]] - 1)

    return _emit_linalg_node("Irfft3d", [a], {"s": s, "axes": axes}, [tuple(out_shape)], [a.dtype])


fft3 = fft3d
ifft3 = ifft3d

fft2 = fft2d
ifft2 = ifft2d
fft3 = fft3d
ifft3 = ifft3d


def fftnd(a: Tensor, s: Sequence[int] | None = None, axes: Sequence[int] | None = None) -> Tensor:
    """Computes the n-dimensional discrete Fourier Transform."""
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Fftnd", a.data, s=s, axes=axes)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))

    from ml_switcheroo_compiler.ops.linalg.basic import Fftnd

    op = Fftnd()
    out_shape = op.infer_shape(a, s=s, axes=axes)
    return _emit_linalg_node("Fftnd", [a], {"s": s, "axes": axes}, [tuple(out_shape)], [a.dtype])


def ifftnd(a: Tensor, s: Sequence[int] | None = None, axes: Sequence[int] | None = None) -> Tensor:
    """Computes the n-dimensional inverse discrete Fourier Transform."""
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Ifftnd", a.data, s=s, axes=axes)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))

    from ml_switcheroo_compiler.ops.linalg.basic import Ifftnd

    op = Ifftnd()
    out_shape = op.infer_shape(a, s=s, axes=axes)
    return _emit_linalg_node("Ifftnd", [a], {"s": s, "axes": axes}, [tuple(out_shape)], [a.dtype])


def rfftnd(a: Tensor, s: Sequence[int] | None = None, axes: Sequence[int] | None = None) -> Tensor:
    """Computes the n-dimensional discrete Fourier Transform of real input."""
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Rfftnd", a.data, s=s, axes=axes)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))

    from ml_switcheroo_compiler.ops.linalg.basic import Rfftnd

    op = Rfftnd()
    out_shape = op.infer_shape(a, s=s, axes=axes)
    return _emit_linalg_node("Rfftnd", [a], {"s": s, "axes": axes}, [tuple(out_shape)], [a.dtype])


def irfftnd(a: Tensor, s: Sequence[int] | None = None, axes: Sequence[int] | None = None) -> Tensor:
    """Computes the inverse n-dimensional discrete Fourier Transform of real input."""
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Irfftnd", a.data, s=s, axes=axes)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))

    from ml_switcheroo_compiler.ops.linalg.basic import Irfftnd

    op = Irfftnd()
    out_shape = op.infer_shape(a, s=s, axes=axes)
    return _emit_linalg_node("Irfftnd", [a], {"s": s, "axes": axes}, [tuple(out_shape)], [a.dtype])


def fftshift(a: Tensor, axes: Sequence[int] | None = None) -> Tensor:
    """Shift the zero-frequency component to the center of the spectrum."""
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Fftshift", a.data, axes=axes)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))

    from ml_switcheroo_compiler.ops.linalg.basic import Fftshift

    op = Fftshift()
    out_shape = op.infer_shape(a, axes=axes)
    return _emit_linalg_node("Fftshift", [a], {"axes": axes}, [tuple(out_shape)], [a.dtype])


def ifftshift(a: Tensor, axes: Sequence[int] | None = None) -> Tensor:
    """The inverse of fftshift."""
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Ifftshift", a.data, axes=axes)
        return Tensor(data, TensorConfig(data.shape, a.dtype, a.device))

    from ml_switcheroo_compiler.ops.linalg.basic import Ifftshift

    op = Ifftshift()
    out_shape = op.infer_shape(a, axes=axes)
    return _emit_linalg_node("Ifftshift", [a], {"axes": axes}, [tuple(out_shape)], [a.dtype])
