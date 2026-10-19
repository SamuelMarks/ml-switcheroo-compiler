"""Implementation of inv operations and utilities for the ops subsystem (`ml_switcheroo_compiler.ops.linalg.decompositions.inv`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node

if TYPE_CHECKING:
    from ml_switcheroo_compiler.interpreter.evaluator import TensorLike


@register_op("Inv")
class Inv(OpDef):
    """Inv Operation Definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> tuple[int, ...]:
        """Infer output shape for matrix inverse.

        Args:
            *args (EagerTensorProtocol): Input tensor or shape arguments.
            **kwargs (EagerTensorProtocol): Optional keyword arguments.

        Returns:
            tuple[int, ...]: Matrix inverse shape (..., N, N).
        """
        inp = args[0] if args else kwargs.get("input")
        if inp is None:
            return ()
        shape = tuple(int(d) for d in getattr(inp, "shape", getattr(inp, "shape_metadata", inp if isinstance(inp, (list, tuple)) else ())))
        return shape


@register_op("InvEx")
class InvEx(OpDef):
    """InvEx Operation Definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> tuple[tuple[int, ...], tuple[int, ...]] | tuple[int, ...]:
        """Infer output shapes for InvEx matrix inverse with info.

        Args:
            *args (EagerTensorProtocol): Input tensor or shape arguments.
            **kwargs (EagerTensorProtocol): Optional keyword arguments.

        Returns:
            tuple: Tuple of (inv_shape (..., N, N), info_shape (...)).
        """
        inp = args[0] if args else kwargs.get("input")
        if inp is None:
            return ()
        shape = tuple(int(d) for d in getattr(inp, "shape", getattr(inp, "shape_metadata", inp if isinstance(inp, (list, tuple)) else ())))
        info_shape = shape[:-2] if len(shape) >= 2 else ()
        return (shape, info_shape)


@register_op("TriInv")
class TriInv(OpDef):
    """TriInv Operation Definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> tuple[int, ...]:
        """Infer output shape for triangular matrix inverse.

        Args:
            *args (EagerTensorProtocol): Input tensor or shape arguments.
            **kwargs (EagerTensorProtocol): Optional keyword arguments.

        Returns:
            tuple[int, ...]: Triangular inverse shape (..., N, N).
        """
        inp = args[0] if args else kwargs.get("a", kwargs.get("input"))
        if inp is None:
            return ()
        shape = tuple(int(d) for d in getattr(inp, "shape", getattr(inp, "shape_metadata", inp if isinstance(inp, (list, tuple)) else ())))
        return shape


def inv(input: Tensor) -> TensorOrScalar:
    """Compute the multiplicative inverse of a square matrix.

    Args:
        input (Tensor): The input parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Inv", input.data)
        arr = cast("TensorLike", backend.array(data))
        return Tensor(arr, TensorConfig(arr.shape, input.dtype, input.device))
    return _emit_linalg_node("Inv", [input], {}, [input.shape], [input.dtype])


def inv_ex(input: Tensor, check_errors: bool = False) -> TensorOrScalar:
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
        inv_out, info = cast("tuple[TensorLike, TensorLike]", backend.execute_op("InvEx", input.data, check_errors=check_errors))
        return (
            Tensor(inv_out, TensorConfig(inv_out.shape, input.dtype, input.device)),
            Tensor(info, TensorConfig(info.shape, "int32", input.device)),
        )
    return _emit_linalg_node("InvEx", [input], {"check_errors": check_errors}, [input.shape, input.shape[:-2]], [input.dtype, "int32"])


def pinv(input: Tensor, rcond: float = 1e-15) -> TensorOrScalar:
    """Compute the Moore-Penrose pseudo-inverse of a matrix.

    Args:
        input (Tensor): The input parameter.
        rcond (float): The rcond parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = cast("TensorLike", backend.execute_op("Pinv", input.data, rcond=rcond))
        return Tensor(data, TensorConfig(data.shape, input.dtype, input.device))
    return _emit_linalg_node("Pinv", [input], {"rcond": rcond}, [input.shape], [input.dtype])


def tri_inv(a: Tensor, lower: bool = False) -> TensorOrScalar:
    """Compute the inverse of a triangular matrix.

    Args:
        a (Tensor): The a parameter.
        lower (bool): The lower parameter.

    Returns:
        Tensor: Result.
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
