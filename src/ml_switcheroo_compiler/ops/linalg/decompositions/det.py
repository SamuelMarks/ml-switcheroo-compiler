"""Core abstractions and logic definitions for det.py."""

from __future__ import annotations

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node


@register_op("Det")
class Det(OpDef):
    """Det Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: The shape.
        """
        return ()


@register_op("Slogdet")
class Slogdet(OpDef):
    """Slogdet Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: The shape.
        """
        return ()


def det(input: Tensor) -> Tensor:
    """Computes the determinant of a square matrix.

    Args:
        input (Tensor): The square matrix

    Returns:
    Tensor: The determinant of the input matrix
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Det", (input.data if type(input).__name__ == "Tensor" else input))
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, getattr(input, "dtype", None), getattr(input, "device", None)),
        )
    return _emit_linalg_node("Det", [input], {}, [()], [getattr(input, "dtype", None)])


def slogdet(input: Tensor) -> tuple[Tensor, Tensor]:
    """Computes the sign and natural logarithm of the determinant of a square matrix.

    Args:
        input (Tensor): The square matrix

    Returns:
    tuple[Tensor, Tensor]: A tuple containing:
        - sign (Tensor): A number representing the sign of the determinant
    - logdet (Tensor): The natural logarithm of the absolute value of the
    determinant
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        sign, logdet = backend.execute_op("Slogdet", (input.data if type(input).__name__ == "Tensor" else input))
        return (
            Tensor(
                backend.array(sign),
                TensorConfig(backend.array(sign).shape, getattr(input, "dtype", None), getattr(input, "device", None)),
            ),
            Tensor(
                backend.array(logdet),
                TensorConfig(backend.array(logdet).shape, getattr(input, "dtype", None), getattr(input, "device", None)),
            ),
        )
    return _emit_linalg_node("Slogdet", [input], {}, [(), ()], [getattr(input, "dtype", None)] * 2)
