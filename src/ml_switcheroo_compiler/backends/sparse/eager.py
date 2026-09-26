"""Dedicated eager execution handlers and dispatch for the Sparse COO backend."""

from typing import Union

import numpy as np

from ml_switcheroo_compiler.backends import mapping_loader
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from ml_switcheroo_compiler.backends.sparse import kernels, types
from ml_switcheroo_compiler.backends.sparse.types import (
    COOTensor,
    CSCTensor,
    CSRTensor,
)
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

_DIRECT_OPS = {
    "coo_fromdense": kernels.coo_fromdense,
    "coo_todense": kernels.coo_todense,
    "coo_matmat": kernels.coo_matmat,
    "coo_matvec": kernels.coo_matvec,
    "Add": kernels.coo_add,
    "add": kernels.coo_add,
    "Sub": kernels.coo_sub,
    "sub": kernels.coo_sub,
    "Subtract": kernels.coo_sub,
    "subtract": kernels.coo_sub,
    "Mul": kernels.coo_mul,
    "mul": kernels.coo_mul,
    "Multiply": kernels.coo_mul,
    "multiply": kernels.coo_mul,
    "Neg": kernels.coo_neg,
    "neg": kernels.coo_neg,
    "Abs": kernels.coo_abs,
    "abs": kernels.coo_abs,
    "Transpose": kernels.coo_transpose,
    "transpose": kernels.coo_transpose,
    "Sum": kernels.coo_sum,
    "sum": kernels.coo_sum,
    "Mean": kernels.coo_mean,
    "mean": kernels.coo_mean,
    "Relu": kernels.coo_relu,
    "relu": kernels.coo_relu,
    "Reshape": kernels.coo_reshape,
    "reshape": kernels.coo_reshape,
    "Dot": kernels.coo_dot,
    "dot": kernels.coo_dot,
    "MatMul": kernels.coo_matmat,
    "matmul": kernels.coo_matmat,
    "spmm": kernels.spmm,
    "spgemm": kernels.spgemm,
    "dense_spmm": kernels.dense_spmm,
    "csr_add": kernels.csr_add,
    "csc_add": kernels.csc_add,
    "csr_sub": kernels.csr_sub,
    "csc_sub": kernels.csc_sub,
    "sparse_mask": kernels.sparse_mask,
    "sparse_conv2d_mask": kernels.sparse_conv2d_mask,
    "sparse_softmax": kernels.sparse_softmax,
    "graph_norm_adjacency": kernels.graph_norm_adjacency,
    "gnn_spmm_attention": kernels.gnn_spmm_attention,
    "sparse_dropout": kernels.sparse_dropout,
    "coo_to_csr": types.coo_to_csr,
    "coo_to_csc": types.coo_to_csc,
    "csr_to_coo": types.csr_to_coo,
    "csr_to_csc": types.csr_to_csc,
    "csc_to_coo": types.csc_to_coo,
    "csc_to_csr": types.csc_to_csr,
    "SparseCooTensor": lambda indices, values, shape: COOTensor(indices=indices, values=values, shape=shape),
}


def _dispatch_sparse_linalg(  # noqa: C901, PLR0911, PLR0912
    op_type: str,
    a: object,
    b: object,
) -> tuple[bool, object]:
    """Dispatch sparse linear algebra and arithmetic based on operand format types.

    Args:
        op_type (str): Name of the operation.
        a (object): First operand.
        b (object): Second operand.

    Returns:
        tuple[bool, object]: Handled flag and computed result.
    """
    if op_type in ("MatMul", "matmul", "Dot", "dot"):
        if isinstance(a, COOTensor) and isinstance(b, np.ndarray):
            return True, kernels.spmm(a, b)
        if isinstance(a, np.ndarray) and isinstance(b, COOTensor):
            return True, kernels.dense_spmm(a, b)
        if isinstance(a, CSRTensor) and isinstance(b, np.ndarray):
            return True, kernels.spmm(a.to_coo(), b)
        if isinstance(a, CSCTensor) and isinstance(b, np.ndarray):
            return True, kernels.spmm(a.to_coo(), b)
        if isinstance(a, np.ndarray) and isinstance(b, (CSRTensor, CSCTensor)):
            return True, kernels.dense_spmm(a, b.to_coo())
        if isinstance(a, COOTensor) and isinstance(b, COOTensor):
            return True, kernels.spgemm(a, b).to_coo()
        if isinstance(a, CSRTensor) and isinstance(b, CSRTensor):
            return True, kernels.spgemm(a.to_coo(), b.to_coo())
        if isinstance(a, CSCTensor) and isinstance(b, CSCTensor):
            return True, kernels.spgemm(a.to_coo(), b.to_coo()).to_coo().to_csc()

    if op_type in ("Add", "add"):
        if isinstance(a, CSRTensor) and isinstance(b, CSRTensor):
            return True, kernels.csr_add(a, b)
        if isinstance(a, CSCTensor) and isinstance(b, CSCTensor):
            return True, kernels.csc_add(a, b)

    if op_type in ("Sub", "sub"):
        if isinstance(a, CSRTensor) and isinstance(b, CSRTensor):
            return True, kernels.csr_sub(a, b)
        if isinstance(a, CSCTensor) and isinstance(b, CSCTensor):
            return True, kernels.csc_sub(a, b)

    return False, None


def execute_op(  # noqa: C901
    cls: type,
    op_type: str,
    *args: object,
    **kwargs: object,
) -> object:
    """Execute operation eagerly on the Sparse COO backend.

    Args:
        cls (type): The caller class or context.
        op_type (str): The name of the operation.
        *args (object): Positional arguments for the op.
        **kwargs (object): Keyword arguments for the op.

    Returns:
        object: Result of sparse COO kernel evaluation.

    Raises:
        BackendNotSupportedError: If op is unmapped and not in eager registry.
    """
    del cls
    if len(args) >= 2:
        handled, linalg_res = _dispatch_sparse_linalg(op_type, args[0], args[1])
        if handled:
            return linalg_res

    if op_type in _DIRECT_OPS:
        handler = _DIRECT_OPS[op_type]
        return handler(*args, **kwargs)

    schema = mapping_loader.load_backend_mappings("sparse")
    if op_type in schema.operations:
        try:
            return mapping_loader.dispatch_eager_op(
                "sparse",
                op_type,
                list(args),
                dict(kwargs),
                backend_module=kernels,
            )
        except BackendNotSupportedError:
            pass

    func = global_eager_registry.get(op_type)
    if func is not None:
        processed_args: list[Union[np.ndarray, object]] = []
        for a in args:
            if isinstance(a, (COOTensor, CSRTensor, CSCTensor)):
                processed_args.append(a.to_dense())
            else:
                processed_args.append(a)
        res = func(np, *processed_args, **kwargs)
        if isinstance(res, np.ndarray):
            return COOTensor.from_dense(res)
        return res

    msg = f"Operation '{op_type}' is not implemented for Sparse COO backend."
    raise BackendNotSupportedError(msg)
