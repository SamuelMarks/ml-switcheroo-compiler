"""Module docstring."""

from __future__ import annotations

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node


@register_op("Inv")
class Inv(OpDef):
    """Inv Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: The shape.
        """
        return ()


@register_op("TriInv")
class TriInv(OpDef):
    """TriInv Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: The shape.
        """
        return ()


def inv(input: Tensor) -> Tensor:
    """Computes the multiplicative inverse of a square matrix.

    Args:
        input (Tensor): The square matrix to invert

    Returns:
    Tensor: The multiplicative inverse of the input matrix
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Inv", input.data)
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device))
    return _emit_linalg_node("Inv", [input], {}, [input.shape], [input.dtype])


def pinv(input: Tensor, rcond: float = 1e-15) -> Tensor:
    """Computes the Moore-Penrose pseudo-inverse of a matrix.

    Args:
        input (Tensor): The matrix to invert
        rcond (float): Cutoff for small singular values. Defaults to 1e-15

    Returns:
    Tensor: The pseudo-inverse of the input matrix
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Pinv", input.data, rcond=rcond)
        return Tensor(data, TensorConfig(data.shape, input.dtype, input.device))
    return _emit_linalg_node("Pinv", [input], {"rcond": rcond}, [input.shape], [input.dtype])


def tri_inv(a: Tensor, lower: bool = False) -> Tensor:
    """Computes the inverse of a triangular matrix.

    Args:
        a (Tensor): Triangular matrix
        lower (bool): If True, a is assumed to be lower triangular. Otherwise, upper.

    Returns:
    Tensor: The inverse matrix
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
