"""Core abstractions and logic definitions for svd.py."""

from __future__ import annotations

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node


@register_op("Svd")
class Svd(OpDef):
    """Svd Operation Definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: The shape.
        """
        if not args:
            return ()
        a_shape = args[0].shape
        full_matrices = kwargs.get("full_matrices", True)
        m, n = a_shape[-2], a_shape[-1]
        k = min(m, n)
        s_shape = a_shape[:-2] + (k,)
        if full_matrices:
            u_shape = a_shape[:-2] + (m, m)
            vh_shape = a_shape[:-2] + (n, n)
        else:
            u_shape = a_shape[:-2] + (m, k)
            vh_shape = a_shape[:-2] + (k, n)
        if kwargs.get("compute_uv", True):
            return u_shape, s_shape, vh_shape
        return (s_shape,)


def svd(
    input: Tensor,
    full_matrices: bool = True,
    compute_uv: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Computes the Singular Value Decomposition (SVD) of a matrix.

    Args:
        input (Tensor): The input matrix of shape (..., M, N)
        full_matrices (bool): If True, matrices U and Vh have shapes (..., M, M)
        and (..., N, N). Otherwise, shapes are (..., M, K) and (..., K, N)
        where K = min(M, N). Defaults to True
        compute_uv (bool): Whether to compute U and Vh in addition to S. Defaults to
        True

    Returns:
    tuple[Tensor, Tensor, Tensor]: A tuple containing:
        - U (Tensor): Left singular vectors
        - S (Tensor): Singular values
        - Vh (Tensor): Right singular vectors (conjugate transposed)
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        res = backend.execute_op(
            "Svd",
            input.data,
            full_matrices=full_matrices,
            compute_uv=compute_uv,
        )
        if not compute_uv:
            return Tensor(res, TensorConfig(res.shape, input.dtype, input.device))
        u, s, vh = res
        return (
            Tensor(u, TensorConfig(u.shape, input.dtype, input.device)),
            Tensor(s, TensorConfig(s.shape, input.dtype, input.device)),
            Tensor(vh, TensorConfig(vh.shape, input.dtype, input.device)),
        )
    m, n = input.shape[-2], input.shape[-1]
    k = min(m, n)
    s_shape = input.shape[:-2] + (k,)
    if full_matrices:
        u_shape = input.shape[:-2] + (m, m)
        vh_shape = input.shape[:-2] + (n, n)
    else:
        u_shape = input.shape[:-2] + (m, k)
        vh_shape = input.shape[:-2] + (k, n)
    if not compute_uv:
        return _emit_linalg_node(
            "Svd",
            [input],
            {"full_matrices": full_matrices, "compute_uv": compute_uv},
            [s_shape],
            [input.dtype],
        )
    return _emit_linalg_node(
        "Svd",
        [input],
        {"full_matrices": full_matrices, "compute_uv": compute_uv},
        [u_shape, s_shape, vh_shape],
        [input.dtype] * 3,
    )
