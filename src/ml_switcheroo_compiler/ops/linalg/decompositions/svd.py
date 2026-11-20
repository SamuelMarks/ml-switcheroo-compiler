"""Operations and utilities for `ml_switcheroo_compiler.ops.linalg.decompositions.svd`."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.registry import (
    unpack_pair,
    unpack_triple,
)
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.shape import OpValue, extract_shape
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.ops.linalg.utils import _emit_linalg_node

if typing.TYPE_CHECKING:
    pass


@register_op("Svd")
class Svd(OpDef):
    """Svd Operation Definition."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer the output shape for the infer_shape operation.

        Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        if not args:
            return ()
        a_shape = extract_shape(args[0])
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
) -> OpValue:
    """Compute the Singular Value Decomposition (SVD) of a matrix.

    Args:
        input (Tensor): Input `input` argument (input).
        full_matrices (bool): Input `full_matrices` argument (full matrices).
        compute_uv (bool): Input `compute_uv` argument (compute uv).

    Returns:
        tuple: Computed `tuple` output.
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
            res_t = res
            return Tensor(res_t, TensorConfig(extract_shape(res_t), input.dtype, input.device))
        u, s, vh = unpack_triple(res)
        return (
            Tensor(u, TensorConfig(extract_shape(u), input.dtype, input.device)),
            Tensor(s, TensorConfig(extract_shape(s), input.dtype, input.device)),
            Tensor(vh, TensorConfig(extract_shape(vh), input.dtype, input.device)),
        )
    in_shape = extract_shape(input)
    m, n = in_shape[-2], in_shape[-1]
    k = min(m, n)
    s_shape = in_shape[:-2] + (k,)
    if full_matrices:
        u_shape = in_shape[:-2] + (m, m)
        vh_shape = in_shape[:-2] + (n, n)
    else:
        u_shape = in_shape[:-2] + (m, k)
        vh_shape = in_shape[:-2] + (k, n)
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


@register_op("Polar")
class Polar(OpDef):
    """Polar matrix decomposition operator definition."""

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer `(U_shape, P_shape)` for polar decomposition.

        Args:
            *args (OpValue): Input matrix tensor.
            **kwargs (OpValue): Optional keyword attributes.

        Returns:
            OpValue: Pair of `(U_shape, P_shape)`.
        """
        return extract_shape(args[0]), extract_shape(args[0])


def polar(a: Tensor, side: str = "right") -> OpValue:
    """Compute the polar decomposition `A = U P` (or `P U`) of matrix `a`.

    Args:
        a (Tensor): Input matrix tensor.
        side (str): Factor ordering (`'right'` for `U @ P` or `'left'` for `P @ U`).

    Returns:
        OpValue: Tuple `(U, P)` of unitary and positive-semidefinite factors.
    """
    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        u, p = unpack_pair(backend.execute_op("Polar", a.data, side=side))
        return (
            Tensor(u, TensorConfig(extract_shape(u), a.dtype, a.device)),
            Tensor(p, TensorConfig(extract_shape(p), a.dtype, a.device)),
        )
    return _emit_linalg_node("Polar", [a], {"side": side}, [a.shape, a.shape], [a.dtype] * 2)
