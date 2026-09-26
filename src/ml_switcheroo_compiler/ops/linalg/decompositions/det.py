"""Module det.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Core abstractions and logic definitions for det.py."""

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node


@register_op("Det")
class Det(OpDef):
    """Det Operation Definition."""

    def infer_shape(self, *args, **kwargs) -> tuple[int, ...]:
        """Infer determinant output scalar/batch shape (...).

        Args:
            *args (object): Input tensor or shape arguments.
            **kwargs (object): Optional keyword arguments.

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

    def infer_shape(self, *args, **kwargs) -> tuple[tuple[int, ...], tuple[int, ...]] | tuple[int, ...]:
        """Infer sign and log determinant output shapes.

        Args:
            *args (object): Input tensor or shape arguments.
            **kwargs (object): Optional keyword arguments.

        Returns:
            tuple: Tuple of (sign_shape (...), logdet_shape (...)).
        """
        inp = args[0] if args else kwargs.get("input")
        if inp is None:
            return ()
        shape = tuple(int(d) for d in getattr(inp, "shape", getattr(inp, "shape_metadata", inp if isinstance(inp, (list, tuple)) else ())))
        batch_shape = shape[:-2] if len(shape) >= 2 else ()
        return (batch_shape, batch_shape)


def det(input: Tensor):
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
        return Tensor(
            backend.array(data),
            TensorConfig(backend.array(data).shape, getattr(input, "dtype", None), getattr(input, "device", None)),
        )
    return _emit_linalg_node("Det", [input], {}, [()], [getattr(input, "dtype", None)])


def slogdet(input: Tensor):
    """Compute the sign and natural logarithm of the determinant of a square matrix.

    Args:
        input (Tensor): The input parameter.

    Returns:
        tuple: Result.
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
