"""Linear algebra operations."""

from __future__ import annotations


from typing import TYPE_CHECKING

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
