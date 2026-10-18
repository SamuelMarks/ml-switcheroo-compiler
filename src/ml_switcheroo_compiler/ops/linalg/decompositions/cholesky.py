"""Core abstractions and logic definitions for cholesky.py."""

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


@register_op("Cholesky")
class Cholesky(OpDef):
    """Cholesky Operation Definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> tuple[int, ...]:
        """Infer output shape for Cholesky decomposition.

        Args:
            *args (EagerTensorProtocol): Input tensor or shape arguments.
            **kwargs (EagerTensorProtocol): Optional keyword arguments.

        Returns:
            tuple[int, ...]: Triangular factor output shape (..., N, N).
        """
        inp = args[0] if args else kwargs.get("input")
        if inp is None:
            return ()
        shape = tuple(int(d) for d in getattr(inp, "shape", getattr(inp, "shape_metadata", inp if isinstance(inp, (list, tuple)) else ())))
        return shape


def cholesky(input: Tensor) -> TensorOrScalar:
    """Compute the Cholesky decomposition of a symmetric/Hermitian positive-definite.

    Args:
        input (Tensor): The input parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Cholesky", input.data)
        arr = cast("TensorLike", backend.array(data))
        return Tensor(arr, TensorConfig(arr.shape, input.dtype, input.device))
    return _emit_linalg_node("Cholesky", [input], {}, [input.shape], [input.dtype])


@register_op("CholeskyEx")
class CholeskyEx(OpDef):
    """CholeskyEx Operation Definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> tuple[tuple[int, ...], tuple[int, ...]] | tuple[int, ...]:
        """Infer output shapes for CholeskyEx decomposition.

        Args:
            *args (EagerTensorProtocol): Input tensor or shape arguments.
            **kwargs (EagerTensorProtocol): Optional keyword arguments.

        Returns:
            tuple: Tuple of (factor_shape (..., N, N), info_shape (...)).
        """
        inp = args[0] if args else kwargs.get("input")
        if inp is None:
            return ()
        shape = tuple(int(d) for d in getattr(inp, "shape", getattr(inp, "shape_metadata", inp if isinstance(inp, (list, tuple)) else ())))
        info_shape = shape[:-2] if len(shape) >= 2 else ()
        return (shape, info_shape)


def cholesky_ex(input: Tensor, check_errors: bool = False) -> TensorOrScalar:
    """Compute the Cholesky decomposition with an info tensor.

    Args:
        input (Tensor): The input symmetric/Hermitian positive-definite matrix
        check_errors (bool): If True, throws an error if the decomposition fails

    Returns:
        tuple[Tensor, Tensor]: The Cholesky factor and the info tensor
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        L, info = cast("tuple[TensorLike, TensorLike]", backend.execute_op("CholeskyEx", input.data, check_errors=check_errors))
        return (
            Tensor(L, TensorConfig(L.shape, input.dtype, input.device)),
            Tensor(info, TensorConfig(info.shape, "int32", input.device)),
        )
    return _emit_linalg_node("CholeskyEx", [input], {"check_errors": check_errors}, [input.shape, input.shape[:-2]], [input.dtype, "int32"])
