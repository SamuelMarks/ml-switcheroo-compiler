"""Core abstractions and logic definitions for det.py."""

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


@register_op("Det")
class Det(OpDef):
    """Det Operation Definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> tuple[int, ...]:
        """Infer determinant output scalar/batch shape (...).

        Args:
            *args (EagerTensorProtocol): Input tensor or shape arguments.
            **kwargs (EagerTensorProtocol): Optional keyword arguments.

        Returns:
            tuple[int, ...]: Reduced batch shape (...).
        """
        inp = args[0] if args else kwargs.get("input")
        if inp is None:
            return ()
        shape = tuple(int(d) for d in getattr(inp, "shape", getattr(inp, "shape_metadata", inp if isinstance(inp, (list, tuple)) else ())))
        return shape[:-2] if len(shape) >= 2 else ()


@register_op("Slogdet")
class Slogdet(OpDef):
    """Slogdet Operation Definition."""

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> tuple[tuple[int, ...], tuple[int, ...]] | tuple[int, ...]:
        """Infer sign and log determinant output shapes.

        Args:
            *args (EagerTensorProtocol): Input tensor or shape arguments.
            **kwargs (EagerTensorProtocol): Optional keyword arguments.

        Returns:
            tuple: Tuple of (sign_shape (...), logdet_shape (...)).
        """
        inp = args[0] if args else kwargs.get("input")
        if inp is None:
            return ()
        shape = tuple(int(d) for d in getattr(inp, "shape", getattr(inp, "shape_metadata", inp if isinstance(inp, (list, tuple)) else ())))
        batch_shape = shape[:-2] if len(shape) >= 2 else ()
        return (batch_shape, batch_shape)


def det(input: Tensor) -> TensorOrScalar:
    """Compute the determinant of a square matrix.

    Args:
        input (Tensor): The input parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Det", (input.data if type(input).__name__ == "Tensor" else input))
        arr = cast("TensorLike", backend.array(data))
        return Tensor(
            arr,
            TensorConfig(arr.shape, getattr(input, "dtype", None), getattr(input, "device", None)),
        )
    return _emit_linalg_node("Det", [input], {}, [()], [getattr(input, "dtype", None)])


def slogdet(input: Tensor) -> TensorOrScalar:
    """Compute the sign and natural logarithm of the determinant of a square matrix.

    Args:
        input (Tensor): The input parameter.

    Returns:
        tuple: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        sign, logdet = cast("tuple[TensorOrScalar, TensorOrScalar]", backend.execute_op("Slogdet", (input.data if type(input).__name__ == "Tensor" else input)))
        sign_arr = cast("TensorLike", backend.array(sign))
        logdet_arr = cast("TensorLike", backend.array(logdet))
        return (
            Tensor(
                sign_arr,
                TensorConfig(sign_arr.shape, getattr(input, "dtype", None), getattr(input, "device", None)),
            ),
            Tensor(
                logdet_arr,
                TensorConfig(logdet_arr.shape, getattr(input, "dtype", None), getattr(input, "device", None)),
            ),
        )
    return _emit_linalg_node("Slogdet", [input], {}, [(), ()], [getattr(input, "dtype", None)] * 2)
