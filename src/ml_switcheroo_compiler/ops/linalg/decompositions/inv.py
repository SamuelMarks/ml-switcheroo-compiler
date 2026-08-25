"""Module inv.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Core abstractions and logic definitions for inv.py."""

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
            tuple[int, ...]: Result.
        """
        return ()


@register_op("InvEx")
class InvEx(OpDef):
    """InvEx Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
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

        Returns: object: The shape.
        """
        return ()


def inv(input: Tensor) -> object:
    """Compute the multiplicative inverse of a square matrix.

    Args:
        input (Tensor): The input parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend: object = get_active_backend()
        data: object = backend.execute_op("Inv", input.data)
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device))
    return _emit_linalg_node("Inv", [input], {}, [input.shape], [input.dtype])


def inv_ex(input: Tensor, check_errors: bool = False) -> object:
    """Compute the multiplicative inverse of a square matrix with info tensor.

    Args:
        input (Tensor): The square matrix to invert
        check_errors (bool): If True, throws an error if the decomposition fails

    Returns:
        tuple[Tensor, Tensor]: The inverted matrix and the info tensor
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend: object = get_active_backend()
        inv_out, info = backend.execute_op("InvEx", input.data, check_errors=check_errors)
        return (
            Tensor(inv_out, TensorConfig(inv_out.shape, input.dtype, input.device)),
            Tensor(info, TensorConfig(info.shape, "int32", input.device)),
        )
    return _emit_linalg_node("InvEx", [input], {"check_errors": check_errors}, [input.shape, input.shape[:-2]], [input.dtype, "int32"])


def pinv(input: Tensor, rcond: float = 1e-15) -> object:
    """Compute the Moore-Penrose pseudo-inverse of a matrix.

    Args:
        input (Tensor): The input parameter.
        rcond (float): The rcond parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend: object = get_active_backend()
        data: object = backend.execute_op("Pinv", input.data, rcond=rcond)
        return Tensor(data, TensorConfig(data.shape, input.dtype, input.device))
    return _emit_linalg_node("Pinv", [input], {"rcond": rcond}, [input.shape], [input.dtype])


def tri_inv(a: Tensor, lower: bool = False) -> object:
    """Compute the inverse of a triangular matrix.

    Args:
        a (Tensor): The a parameter.
        lower (bool): The lower parameter.

    Returns:
        Tensor: Result.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend: object = get_active_backend()
        data: object = backend.execute_op("TriInv", a.data, lower=lower)
        return Tensor(data, TensorConfig(a.shape, a.dtype, a.device))
    return _emit_linalg_node(
        "TriInv",
        [a],
        {"lower": lower},
        [a.shape],
        [a.dtype],
    )
