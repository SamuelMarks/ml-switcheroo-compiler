"""Dedicated eager execution handlers and dispatch for the Sparse COO backend."""

from typing import Union

import numpy as np

from ml_switcheroo_compiler.backends import mapping_loader
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from ml_switcheroo_compiler.backends.sparse import kernels, types
from ml_switcheroo_compiler.backends.sparse.types import COOTensor
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

_DIRECT_OPS = {
    "coo_fromdense": kernels.coo_fromdense,
    "coo_todense": kernels.coo_todense,
    "coo_matmat": kernels.coo_matmat,
    "coo_matvec": kernels.coo_matvec,
    "Add": kernels.coo_add,
    "Sub": kernels.coo_sub,
    "Mul": kernels.coo_mul,
    "Neg": kernels.coo_neg,
    "Abs": kernels.coo_abs,
    "Transpose": kernels.coo_transpose,
    "Sum": kernels.coo_sum,
    "Mean": kernels.coo_mean,
    "Relu": kernels.coo_relu,
    "Reshape": kernels.coo_reshape,
    "Dot": kernels.coo_dot,
    "MatMul": kernels.coo_matmat,
    "spmm": kernels.spmm,
    "spgemm": kernels.spgemm,
    "dense_spmm": kernels.dense_spmm,
    "csr_add": kernels.csr_add,
    "csc_add": kernels.csc_add,
    "sparse_mask": kernels.sparse_mask,
    "sparse_conv2d_mask": kernels.sparse_conv2d_mask,
    "coo_to_csr": types.coo_to_csr,
    "coo_to_csc": types.coo_to_csc,
    "csr_to_coo": types.csr_to_coo,
    "csr_to_csc": types.csr_to_csc,
    "csc_to_coo": types.csc_to_coo,
    "csc_to_csr": types.csc_to_csr,
    "SparseCooTensor": lambda indices, values, shape: COOTensor(indices=indices, values=values, shape=shape),
}


def execute_op(
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
            if isinstance(a, COOTensor):
                processed_args.append(a.to_dense())
            else:
                processed_args.append(a)
        res = func(np, *processed_args, **kwargs)
        if isinstance(res, np.ndarray):
            return COOTensor.from_dense(res)
        return res

    msg = f"Operation '{op_type}' is not implemented for Sparse COO backend."
    raise BackendNotSupportedError(msg)
