"""Operations and utilities for `ml_switcheroo_compiler.ops.linalg.decompositions.det`."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.registry import unpack_pair
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import OpValue, extract_device, extract_shape
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.utils import (
    _emit_linalg_node,
    _extract_linalg_dtype,
    _extract_linalg_shape,
)

if typing.TYPE_CHECKING:
    pass


@register_op("Det")
class Det(OpDef):
    """Det Operation Definition."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> tuple[int, ...]:
        """Infer determinant output scalar/batch shape (...).

        Args:
            *args (OpValue): Input tensor or shape arguments.
            **kwargs (OpValue): Optional keyword arguments.

        Returns:
            tuple[int, ...]: Reduced batch shape (...).
        """
        inp = args[0] if args else kwargs.get("input")
        if inp is None:
            return ()
        shape = _extract_linalg_shape(inp)
        return shape[:-2] if len(shape) >= 2 else ()


@register_op("Slogdet")
class Slogdet(OpDef):
    """Slogdet Operation Definition."""

    def infer_shape(
        self, *args: OpValue, **kwargs: OpValue
    ) -> tuple[tuple[int, ...], tuple[int, ...]] | tuple[int, ...]:
        """Infer sign and log determinant output shapes.

        Args:
            *args (OpValue): Input tensor or shape arguments.
            **kwargs (OpValue): Optional keyword arguments.

        Returns:
            tuple: Tuple of (sign_shape (...), logdet_shape (...)).
        """
        inp = args[0] if args else kwargs.get("input")
        if inp is None:
            return ()
        shape = _extract_linalg_shape(inp)
        batch_shape = shape[:-2] if len(shape) >= 2 else ()
        return (batch_shape, batch_shape)


def det(input: Tensor) -> OpValue:
    """Compute the determinant of a square matrix.

    Args:
        input (Tensor): Input `input` argument (input).

    Returns:
        Tensor: Computed `Tensor` output.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Det", (input.data if type(input).__name__ == "Tensor" else input))
        arr = backend.array(data)
        return Tensor(
            arr,
            TensorConfig(extract_shape(arr), _extract_linalg_dtype(input, None), extract_device(input, None)),
        )
    return _emit_linalg_node("Det", [input], {}, [()], [_extract_linalg_dtype(input, None)])


def slogdet(input: Tensor) -> OpValue:
    """Compute the sign and natural logarithm of the determinant of a square matrix.

    Args:
        input (Tensor): Input `input` argument (input).

    Returns:
        tuple: Computed `tuple` output.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        sign, logdet = unpack_pair(
            backend.execute_op("Slogdet", (input.data if type(input).__name__ == "Tensor" else input))
        )
        sign_arr = backend.array(sign)
        logdet_arr = backend.array(logdet)
        return (
            Tensor(
                sign_arr,
                TensorConfig(extract_shape(sign_arr), _extract_linalg_dtype(input, None), extract_device(input, None)),
            ),
            Tensor(
                logdet_arr,
                TensorConfig(
                    extract_shape(logdet_arr), _extract_linalg_dtype(input, None), extract_device(input, None)
                ),
            ),
        )
    return _emit_linalg_node("Slogdet", [input], {}, [(), ()], [_extract_linalg_dtype(input, None)] * 2)
