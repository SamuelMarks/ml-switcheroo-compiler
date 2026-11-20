"""Operations and utilities for `ml_switcheroo_compiler.ops.linalg.decompositions.inv`."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.registry import unpack_pair
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.constants import MATRIX_RANK
from ml_switcheroo_compiler.core.shape import OpValue, ShapeProtocol, extract_shape
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node, _extract_linalg_shape

if typing.TYPE_CHECKING:
    pass


@register_op("Pinv")
class Pinv(OpDef):
    """Moore-Penrose pseudo-inverse operator definition."""

    def infer_shape(self, a: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer the transposed inner-matrix shape of the pseudo-inverse operation.

        Args:
            a (OpValue): Input matrix tensor.
            **kwargs (OpValue): Optional keyword attributes.

        Returns:
            OpValue: Inferred output shape `(..., N, M)`.
        """
        raw_a: OpValue = a
        if isinstance(raw_a, ShapeProtocol):
            s = list(extract_shape(raw_a))
            if len(s) >= MATRIX_RANK:
                s[-2], s[-1] = s[-1], s[-2]
            return tuple(s)
        return ()


@register_op("Inv")
class Inv(OpDef):
    """Inv Operation Definition."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> tuple[int, ...]:
        """Infer output shape for matrix inverse.

        Args:
            *args (OpValue): Input tensor or shape arguments.
            **kwargs (OpValue): Optional keyword arguments.

        Returns:
            tuple[int, ...]: Matrix inverse shape (..., N, N).
        """
        inp = args[0] if args else kwargs.get("input")
        if inp is None:
            return ()
        return _extract_linalg_shape(inp)


@register_op("InvEx")
class InvEx(OpDef):
    """InvEx Operation Definition."""

    def infer_shape(
        self, *args: OpValue, **kwargs: OpValue
    ) -> tuple[tuple[int, ...], tuple[int, ...]] | tuple[int, ...]:
        """Infer output shapes for InvEx matrix inverse with info.

        Args:
            *args (OpValue): Input tensor or shape arguments.
            **kwargs (OpValue): Optional keyword arguments.

        Returns:
            tuple: Tuple of (inv_shape (..., N, N), info_shape (...)).
        """
        inp = args[0] if args else kwargs.get("input")
        if inp is None:
            return ()
        shape = _extract_linalg_shape(inp)
        info_shape = shape[:-2] if len(shape) >= 2 else ()
        return (shape, info_shape)


@register_op("TriInv")
class TriInv(OpDef):
    """TriInv Operation Definition."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> tuple[int, ...]:
        """Infer output shape for triangular matrix inverse.

        Args:
            *args (OpValue): Input tensor or shape arguments.
            **kwargs (OpValue): Optional keyword arguments.

        Returns:
            tuple[int, ...]: Triangular inverse shape (..., N, N).
        """
        inp = args[0] if args else kwargs.get("a", kwargs.get("input"))
        if inp is None:
            return ()
        return _extract_linalg_shape(inp)


def inv(input: Tensor) -> OpValue:
    """Compute the multiplicative inverse of a square matrix.

    Args:
        input (Tensor): Input `input` argument (input).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Inv", input.data)
        arr = backend.array(data)
        return Tensor(arr, TensorConfig(extract_shape(arr), input.dtype, input.device))
    return _emit_linalg_node("Inv", [input], {}, [input.shape], [input.dtype])


def inv_ex(input: Tensor, check_errors: bool = False) -> OpValue:
    """Compute the multiplicative inverse of a square matrix with info tensor.

    Args:
        input (Tensor): The square matrix to invert
        check_errors (bool): If True, throws an error if the decomposition fails

    Returns:
        tuple[Tensor, Tensor]: The inverted matrix and the info tensor
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        inv_out, info = unpack_pair(backend.execute_op("InvEx", input.data, check_errors=check_errors))
        return (
            Tensor(inv_out, TensorConfig(extract_shape(inv_out), input.dtype, input.device)),
            Tensor(info, TensorConfig(extract_shape(info), "int32", input.device)),
        )
    return _emit_linalg_node(
        "InvEx", [input], {"check_errors": check_errors}, [input.shape, input.shape[:-2]], [input.dtype, "int32"]
    )


def pinv(input: Tensor, rcond: float = 1e-15) -> OpValue:
    """Compute the Moore-Penrose pseudo-inverse of a matrix.

    Args:
        input (Tensor): Input `input` argument (input).
        rcond (float): Input `rcond` argument (rcond).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Pinv", input.data, rcond=rcond)
        return Tensor(data, TensorConfig(extract_shape(data), input.dtype, input.device))
    return _emit_linalg_node("Pinv", [input], {"rcond": rcond}, [input.shape], [input.dtype])


def tri_inv(a: Tensor, lower: bool = False) -> OpValue:
    """Compute the inverse of a triangular matrix.

    Args:
        a (Tensor): Input `a` argument (a).
        lower (bool): Input `lower` argument (lower).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("TriInv", a.data, lower=lower)
        return Tensor(data, TensorConfig(a.shape, a.dtype, a.device))
    return _emit_linalg_node(
        "TriInv",
        [a],
        {"lower": lower},
        [a.shape],
        [a.dtype],
    )
