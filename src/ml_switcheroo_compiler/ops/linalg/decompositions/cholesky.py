"""Core abstractions and logic definitions for cholesky.py."""

from __future__ import annotations

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node


@register_op("Cholesky")
class Cholesky(OpDef):
    """Cholesky Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
        object: Result.
        """
        return ()


def cholesky(input: Tensor) -> Tensor:
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
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device))
    return _emit_linalg_node("Cholesky", [input], {}, [input.shape], [input.dtype])


@register_op("CholeskyEx")
class CholeskyEx(OpDef):
    """CholeskyEx Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return ()


def cholesky_ex(input: Tensor, check_errors: bool = False) -> tuple[Tensor, Tensor]:
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
        L, info = backend.execute_op("CholeskyEx", input.data, check_errors=check_errors)
        return (
            Tensor(L, TensorConfig(L.shape, input.dtype, input.device)),
            Tensor(info, TensorConfig(info.shape, "int32", input.device)),
        )
    return _emit_linalg_node("CholeskyEx", [input], {"check_errors": check_errors}, [input.shape, input.shape[:-2]], [input.dtype, "int32"])
