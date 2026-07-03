"""Module docstring."""

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
            object: The shape.
        """
        return ()


def cholesky(input: Tensor) -> Tensor:
    """Computes the Cholesky decomposition of a symmetric/Hermitian positive-definite.

    matrix

    Args:
        input (Tensor): The input symmetric/Hermitian positive-definite matrix

    Returns:
    Tensor: The lower-triangular or upper-triangular Cholesky factor
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        data = backend.execute_op("Cholesky", input.data)
        return Tensor(backend.array(data), TensorConfig(backend.array(data).shape, input.dtype, input.device))
    return _emit_linalg_node("Cholesky", [input], {}, [input.shape], [input.dtype])
