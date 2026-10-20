"""Dedicated Sparse Coordinate (COO) Tensor execution kernels and operations."""

from __future__ import annotations

import typing
from collections.abc import Sequence

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue
from ml_switcheroo_compiler.backends.sparse.types import (
    _SP_BACKEND,
    COOTensor,
    CSCTensor,
    CSRTensor,
)


def coo_sum_duplicates(
    indices: EagerValue,
    values: EagerValue,
    shape: tuple[int, ...],
) -> COOTensor:
    """Sum duplicate coordinate entries in a sparse COO representation.

    Args:
        indices (_SP_BACKEND.ndarray): Coordinate array of shape (ndim, nnz).
        values (_SP_BACKEND.ndarray): Data array of shape (nnz,).
        shape (tuple[int, ...]): Shape of the tensor.

    Returns:
        COOTensor: Canonical COO tensor with unique sorted coordinates.
    """
    values_arr = typing.cast(EagerTensorProtocol, values)
    if values_arr.size == 0:
        return COOTensor(indices=indices, values=values, shape=shape)

    ndim: int = len(shape)
    indices_arr = _SP_BACKEND.asarray(indices, dtype=_SP_BACKEND.int64)
    if indices_arr.ndim == 1:
        indices_arr = indices_arr.reshape(ndim, -1)

    flat_indices = _SP_BACKEND.ravel_multi_index(
        tuple(indices_arr[i] for i in range(ndim)),
        shape,
    )
    sort_order = _SP_BACKEND.argsort(flat_indices)
    sorted_flat = flat_indices[sort_order]
    sorted_values = values_arr[sort_order]

    unique_flat, split_indices = _SP_BACKEND.unique(sorted_flat, return_index=True)
    summed_values = _SP_BACKEND.add.reduceat(sorted_values, split_indices)

    non_zeros = typing.cast(EagerTensorProtocol, typing.cast(EagerValue, summed_values) != 0)
    unique_flat = unique_flat[non_zeros]
    summed_values = summed_values[non_zeros]

    if unique_flat.size == 0:
        empty_indices = _SP_BACKEND.empty((ndim, 0), dtype=_SP_BACKEND.int64)
        empty_values = _SP_BACKEND.empty((0,), dtype=values_arr.dtype)
        return COOTensor(indices=empty_indices, values=empty_values, shape=shape)

    unraveled = _SP_BACKEND.unravel_index(unique_flat, shape)
    new_indices = _SP_BACKEND.vstack(unraveled)
    return COOTensor(indices=new_indices, values=summed_values, shape=shape)


def coo_fromdense(
    mat: EagerValue | Sequence[float] | COOTensor,
    nse: int | None = None,
    index_dtype: EagerValue | str | None = None,
) -> COOTensor:
    """Create a COO-format sparse matrix from a dense matrix.

    Args:
        mat (Union[_SP_BACKEND.ndarray, Sequence[float], COOTensor]): Input dense array or tensor.
        nse (Optional[int]): Optional maximum number of specified non-zero elements.
        index_dtype (Optional[Union[_SP_BACKEND.dtype, str]]): Data type for indices.

    Returns:
        COOTensor: Sparse COO representation.
    """
    del nse
    if isinstance(mat, COOTensor):
        if index_dtype is not None:
            return COOTensor(
                indices=mat.indices.astype(index_dtype),
                values=mat.values,
                shape=mat.shape,
            )
        return mat

    dense_arr = _SP_BACKEND.asarray(mat)
    nonzero_mask = typing.cast(EagerTensorProtocol, typing.cast(EagerValue, dense_arr) != 0)
    indices = _SP_BACKEND.vstack(_SP_BACKEND.nonzero(nonzero_mask))
    if index_dtype is not None:
        indices = indices.astype(index_dtype)
    values = dense_arr[nonzero_mask]
    return COOTensor(indices=indices, values=values, shape=dense_arr.shape)


def coo_todense(mat: COOTensor | EagerValue | Sequence[float]) -> EagerTensorProtocol:
    """Convert a COO-format sparse matrix to a dense matrix.

    Args:
        mat (Union[COOTensor, _SP_BACKEND.ndarray, Sequence[float]]): Input sparse or dense array.

    Returns:
        _SP_BACKEND.ndarray: Dense NumPy array.
    """
    if isinstance(mat, COOTensor):
        return mat.to_dense()
    return typing.cast(EagerTensorProtocol, _SP_BACKEND.asarray(mat))


def coo_matmat(
    mat: COOTensor | EagerValue,
    B: COOTensor | EagerValue,
    transpose: bool = False,
) -> COOTensor | EagerValue:
    """Product of COO sparse matrix and a dense or sparse matrix.

    Args:
        mat (Union[COOTensor, _SP_BACKEND.ndarray]): Left operand matrix (sparse or dense).
        B (Union[COOTensor, _SP_BACKEND.ndarray]): Right operand matrix (sparse or dense).
        transpose (bool): Whether to transpose the left matrix before multiplication.

    Returns:
        Union[COOTensor, _SP_BACKEND.ndarray]: Matrix product result.
    """
    a_coo: COOTensor = mat if isinstance(mat, COOTensor) else coo_fromdense(mat)
    if transpose:
        a_coo = coo_transpose(a_coo)

    m: int = a_coo.shape[0]

    if isinstance(B, COOTensor):
        n: int = B.shape[1]
        out_shape: tuple[int, int] = (m, n)
        if a_coo.nnz == 0 or B.nnz == 0:
            return COOTensor(
                indices=_SP_BACKEND.empty((2, 0), dtype=_SP_BACKEND.int64),
                values=_SP_BACKEND.empty((0,), dtype=_SP_BACKEND.promote_types(a_coo.dtype, B.dtype)),
                shape=out_shape,
            )

        match_mask = typing.cast(EagerTensorProtocol, typing.cast(EagerValue, a_coo.indices[1, :, None]) == typing.cast(EagerValue, B.indices[0, None, :]))
        a_matches, b_matches = _SP_BACKEND.nonzero(match_mask)

        if a_matches.size == 0:
            return COOTensor(
                indices=_SP_BACKEND.empty((2, 0), dtype=_SP_BACKEND.int64),
                values=_SP_BACKEND.empty((0,), dtype=_SP_BACKEND.promote_types(a_coo.dtype, B.dtype)),
                shape=out_shape,
            )

        res_rows = a_coo.indices[0, a_matches]
        res_cols = B.indices[1, b_matches]
        res_vals = a_coo.values[a_matches] * B.values[b_matches]
        stacked_indices = _SP_BACKEND.vstack([res_rows, res_cols])
        return coo_sum_duplicates(stacked_indices, res_vals, out_shape)

    b_dense = _SP_BACKEND.asarray(B)
    n = b_dense.shape[1]
    res_dense = _SP_BACKEND.zeros((m, n), dtype=_SP_BACKEND.promote_types(a_coo.dtype, b_dense.dtype))
    if a_coo.nnz > 0:
        row_indices = a_coo.indices[0]
        col_indices = a_coo.indices[1]
        contributions = a_coo.values[:, None] * b_dense[col_indices]
        _SP_BACKEND.add.at(res_dense, row_indices, contributions)
    return typing.cast(typing.Union[COOTensor, "EagerValue"], res_dense)


def coo_matvec(
    mat: COOTensor | EagerValue,
    v: COOTensor | EagerValue,
    transpose: bool = False,
) -> EagerValue:
    """Product of COO sparse matrix and a dense vector.

    Args:
        mat (Union[COOTensor, _SP_BACKEND.ndarray]): Matrix operand.
        v (Union[COOTensor, _SP_BACKEND.ndarray]): Vector operand.
        transpose (bool): Whether to transpose matrix before multiplication.

    Returns:
        _SP_BACKEND.ndarray: Matrix-vector product result.
    """
    a_coo: COOTensor = mat if isinstance(mat, COOTensor) else coo_fromdense(mat)
    if transpose:
        a_coo = coo_transpose(a_coo)

    v_dense = v.to_dense() if isinstance(v, COOTensor) else _SP_BACKEND.asarray(v)
    m: int = a_coo.shape[0]
    res = _SP_BACKEND.zeros(m, dtype=_SP_BACKEND.promote_types(a_coo.dtype, v_dense.dtype))
    if a_coo.nnz > 0:
        row_indices = a_coo.indices[0]
        col_indices = a_coo.indices[1]
        contributions = a_coo.values * v_dense[col_indices]
        _SP_BACKEND.add.at(res, row_indices, contributions)
    return typing.cast(EagerValue, res)


def coo_transpose(
    mat: COOTensor,
    axes: Sequence[int] | None = None,
) -> COOTensor:
    """Transpose axes of a COO tensor.

    Args:
        mat (COOTensor): Input sparse tensor.
        axes (Optional[Sequence[int]]): Permutation of axes. Defaults to reversing axes.

    Returns:
        COOTensor: Transposed COO tensor.
    """
    if axes is None:
        perm: list[int] = list(reversed(range(mat.ndim)))
    else:
        perm = [int(ax) for ax in axes]

    new_shape: tuple[int, ...] = tuple(mat.shape[p] for p in perm)
    if mat.nnz == 0:
        new_indices = _SP_BACKEND.empty((mat.ndim, 0), dtype=_SP_BACKEND.int64)
        return COOTensor(indices=new_indices, values=mat.values.copy(), shape=new_shape)

    perm_indices = mat.indices[perm]
    return coo_sum_duplicates(perm_indices, mat.values.copy(), new_shape)


def coo_add(
    a: COOTensor | EagerValue | float,
    b: COOTensor | EagerValue | float,
) -> COOTensor | EagerValue:
    """Element-wise addition for sparse COO tensors.

    Args:
        a (Union[COOTensor, _SP_BACKEND.ndarray, float]): First addend.
        b (Union[COOTensor, _SP_BACKEND.ndarray, float]): Second addend.

    Returns:
        Union[COOTensor, _SP_BACKEND.ndarray]: Sum result.
    """
    if isinstance(a, COOTensor) and isinstance(b, COOTensor):
        if a.shape != b.shape:
            msg = f"Shape mismatch for sparse addition: {a.shape} vs {b.shape}"
            raise ValueError(msg)
        if a.nnz == 0:
            return b
        if b.nnz == 0:
            return a
        stacked_indices = _SP_BACKEND.hstack([a.indices, b.indices])
        stacked_values = _SP_BACKEND.concatenate([a.values, b.values])
        return coo_sum_duplicates(stacked_indices, stacked_values, a.shape)

    if isinstance(a, COOTensor):
        return a.to_dense() + typing.cast(EagerTensorProtocol, b)
    if isinstance(b, COOTensor):
        return typing.cast(EagerTensorProtocol, a) + b.to_dense()
    return typing.cast(typing.Union[COOTensor, "EagerValue"], _SP_BACKEND.asarray(a) + _SP_BACKEND.asarray(b))


def coo_neg(a: COOTensor) -> COOTensor:
    """Element-wise negation of a sparse COO tensor.

    Args:
        a (COOTensor): Input sparse tensor.

    Returns:
        COOTensor: Negated sparse tensor.
    """
    return COOTensor(indices=a.indices.copy(), values=-a.values, shape=a.shape)


def coo_sub(
    a: COOTensor | EagerValue | float,
    b: COOTensor | EagerValue | float,
) -> COOTensor | EagerValue:
    """Element-wise subtraction for sparse COO tensors.

    Args:
        a (Union[COOTensor, _SP_BACKEND.ndarray, float]): Minuend.
        b (Union[COOTensor, _SP_BACKEND.ndarray, float]): Subtrahend.

    Returns:
        Union[COOTensor, _SP_BACKEND.ndarray]: Difference result.
    """
    if isinstance(b, COOTensor):
        return coo_add(a, coo_neg(b))
    return coo_add(a, -typing.cast(EagerTensorProtocol, b))


def _coo_mul_scalar(t: COOTensor, scalar: int | float | EagerValue) -> COOTensor:
    """Multiply sparse COO tensor by a scalar value.

    Args:
        t (COOTensor): Target sparse tensor.
        scalar (Union[int, float, _SP_BACKEND.number]): Scalar factor.

    Returns:
        COOTensor: Scaled sparse tensor.
    """
    if scalar == 0:
        return COOTensor(
            indices=_SP_BACKEND.empty((t.ndim, 0), dtype=_SP_BACKEND.int64),
            values=_SP_BACKEND.empty((0,), dtype=t.dtype),
            shape=t.shape,
        )
    return COOTensor(indices=t.indices.copy(), values=t.values * typing.cast(EagerTensorProtocol, scalar), shape=t.shape)


def _coo_mul_coo(a: COOTensor, b: COOTensor) -> COOTensor:
    """Multiply two sparse COO tensors element-wise.

    Args:
        a (COOTensor): First sparse factor.
        b (COOTensor): Second sparse factor.

    Returns:
        COOTensor: Sparse Hadamard product.
    """
    if a.shape != b.shape:
        msg = f"Shape mismatch for sparse multiply: {a.shape} vs {b.shape}"
        raise ValueError(msg)
    if a.nnz == 0 or b.nnz == 0:
        return COOTensor(
            indices=_SP_BACKEND.empty((a.ndim, 0), dtype=_SP_BACKEND.int64),
            values=_SP_BACKEND.empty((0,), dtype=_SP_BACKEND.promote_types(a.dtype, b.dtype)),
            shape=a.shape,
        )
    flat_a = _SP_BACKEND.ravel_multi_index(tuple(a.indices[i] for i in range(a.ndim)), a.shape)
    flat_b = _SP_BACKEND.ravel_multi_index(tuple(b.indices[i] for i in range(b.ndim)), b.shape)
    common, idx_a, idx_b = _SP_BACKEND.intersect1d(flat_a, flat_b, return_indices=True)
    if common.size == 0:
        return COOTensor(
            indices=_SP_BACKEND.empty((a.ndim, 0), dtype=_SP_BACKEND.int64),
            values=_SP_BACKEND.empty((0,), dtype=_SP_BACKEND.promote_types(a.dtype, b.dtype)),
            shape=a.shape,
        )
    new_values = a.values[idx_a] * b.values[idx_b]
    new_indices = a.indices[:, idx_a]
    return COOTensor(indices=new_indices, values=new_values, shape=a.shape)


def _coo_mul_dense(a: COOTensor, b_dense: EagerTensorProtocol) -> COOTensor:
    """Multiply a sparse COO tensor with a dense array element-wise.

    Args:
        a (COOTensor): Sparse tensor factor.
        b_dense (_SP_BACKEND.ndarray): Dense array factor.

    Returns:
        COOTensor: Filtered sparse product.
    """
    if a.nnz == 0:
        return COOTensor(indices=a.indices.copy(), values=a.values.copy(), shape=a.shape)
    coords_tuple: tuple[EagerValue, ...] = tuple(a.indices[i] for i in range(a.ndim))
    sampled_b = b_dense[coords_tuple]
    return COOTensor(indices=a.indices.copy(), values=a.values * sampled_b, shape=a.shape)


def coo_mul(
    a: COOTensor | EagerValue | float | int,
    b: COOTensor | EagerValue | float | int,
) -> COOTensor | EagerValue:
    """Element-wise multiplication (Hadamard product) for sparse COO tensors.

    Args:
        a (Union[COOTensor, _SP_BACKEND.ndarray, float, int]): First factor.
        b (Union[COOTensor, _SP_BACKEND.ndarray, float, int]): Second factor.

    Returns:
        Union[COOTensor, _SP_BACKEND.ndarray]: Product result.
    """
    if isinstance(a, (int, float, _SP_BACKEND.number)):
        return _coo_mul_scalar(typing.cast(COOTensor, b), a)
    if isinstance(b, (int, float, _SP_BACKEND.number)):
        return _coo_mul_scalar(typing.cast(COOTensor, a), b)
    if isinstance(a, COOTensor) and isinstance(b, COOTensor):
        return _coo_mul_coo(a, b)
    if isinstance(a, COOTensor):
        return _coo_mul_dense(a, _SP_BACKEND.asarray(b))
    if isinstance(b, COOTensor):
        return _coo_mul_dense(b, _SP_BACKEND.asarray(a))
    return typing.cast(typing.Union[COOTensor, "EagerValue"], _SP_BACKEND.asarray(a) * _SP_BACKEND.asarray(b))


def coo_abs(a: COOTensor) -> COOTensor:
    """Element-wise absolute value of a sparse COO tensor.

    Args:
        a (COOTensor): Input sparse tensor.

    Returns:
        COOTensor: Tensor with absolute values.
    """
    return COOTensor(indices=a.indices.copy(), values=_SP_BACKEND.abs(a.values), shape=a.shape)


def coo_relu(a: COOTensor) -> COOTensor:
    """Apply Rectified Linear Unit (ReLU) to sparse tensor values.

    Args:
        a (COOTensor): Input sparse tensor.

    Returns:
        COOTensor: Filtered tensor retaining only strictly positive elements.
    """
    mask = a.values > 0
    return COOTensor(
        indices=a.indices[:, mask],
        values=a.values[mask],
        shape=a.shape,
    )


def coo_reshape(a: COOTensor, shape: tuple[int, ...]) -> COOTensor:
    """Reshape a sparse COO tensor to a new shape.

    Args:
        a (COOTensor): Input sparse tensor.
        shape (tuple[int, ...]): New dense shape.

    Returns:
        COOTensor: Reshaped COO tensor.
    """
    target_shape: tuple[int, ...] = tuple(int(s) for s in shape)
    if a.nnz == 0:
        return COOTensor(
            indices=_SP_BACKEND.empty((len(target_shape), 0), dtype=_SP_BACKEND.int64),
            values=a.values.copy(),
            shape=target_shape,
        )

    flat = _SP_BACKEND.ravel_multi_index(tuple(a.indices[i] for i in range(a.ndim)), a.shape)
    unraveled = _SP_BACKEND.unravel_index(flat, target_shape)
    new_indices = _SP_BACKEND.vstack(unraveled)
    return COOTensor(indices=new_indices, values=a.values.copy(), shape=target_shape)


def _coo_sum_empty(
    a: COOTensor,
    axis: int | Sequence[int] | None,
    keepdims: bool,
) -> COOTensor | float:
    """Compute sum reduction for empty sparse tensor.

    Args:
        a (COOTensor): Empty sparse tensor.
        axis (Optional[Union[int, Sequence[int]]]): Reduction axes.
        keepdims (bool): Whether to preserve reduced dimensions.

    Returns:
        Union[COOTensor, float]: Empty sum result.
    """
    if axis is None:
        if not keepdims:
            return 0.0
        out_shape: tuple[int, ...] = tuple(1 for _ in a.shape)
        return COOTensor(_SP_BACKEND.empty((len(out_shape), 0), dtype=_SP_BACKEND.int64), _SP_BACKEND.empty((0,), dtype=a.dtype), out_shape)

    axes_list: list[int] = [axis] if isinstance(axis, int) else [int(ax) for ax in axis]
    reduced_shape: list[int] = list(a.shape)
    for ax in axes_list:
        reduced_shape[ax % a.ndim] = 1
    final_shape: tuple[int, ...] = tuple(reduced_shape) if keepdims else tuple(s for i, s in enumerate(reduced_shape) if (i % a.ndim) not in [x % a.ndim for x in axes_list])
    return COOTensor(_SP_BACKEND.empty((len(final_shape), 0), dtype=_SP_BACKEND.int64), _SP_BACKEND.empty((0,), dtype=a.dtype), final_shape)


def _coo_sum_axis(
    a: COOTensor,
    axis: int | Sequence[int],
    keepdims: bool,
) -> COOTensor | float | EagerValue:
    """Compute sum reduction along specified axis or axes.

    Args:
        a (COOTensor): Non-empty sparse tensor.
        axis (Union[int, Sequence[int]]): Axes to reduce.
        keepdims (bool): Whether to retain reduced dimensions.

    Returns:
        Union[COOTensor, float, _SP_BACKEND.ndarray]: Reduced sparse tensor or scalar.
    """
    axes_set: set[int] = {axis % a.ndim} if isinstance(axis, int) else {(ax % a.ndim) for ax in axis}
    kept_dims: list[int] = [i for i in range(a.ndim) if i not in axes_set]

    if not kept_dims:
        return coo_sum(a, axis=None, keepdims=keepdims)

    out_shape_kept: tuple[int, ...] = tuple(a.shape[i] for i in kept_dims)
    kept_indices = a.indices[kept_dims]
    collapsed: COOTensor = coo_sum_duplicates(kept_indices, a.values.copy(), out_shape_kept)

    if keepdims:
        final_shape: list[int] = list(a.shape)
        for ax in axes_set:
            final_shape[ax] = 1
        return coo_reshape(collapsed, tuple(final_shape))
    return collapsed


def coo_sum(
    a: COOTensor,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> COOTensor | float | EagerValue:
    """Sum elements of a sparse COO tensor over given axes.

    Args:
        a (COOTensor): Input sparse tensor.
        axis (Optional[Union[int, Sequence[int]]]): Axis or axes along which to sum.
        keepdims (bool): Whether to retain reduced dimensions.

    Returns:
        Union[COOTensor, float, _SP_BACKEND.ndarray]: Sum reduction result.
    """
    if a.nnz == 0:
        return _coo_sum_empty(a, axis, keepdims)

    if axis is None:
        total: float = float(_SP_BACKEND.sum(a.values))
        if keepdims:
            out_shape = tuple(1 for _ in a.shape)
            idx = _SP_BACKEND.zeros((len(out_shape), 1), dtype=_SP_BACKEND.int64)
            val = _SP_BACKEND.array([total], dtype=a.dtype)
            return COOTensor(indices=idx, values=val, shape=out_shape)
        return total

    return _coo_sum_axis(a, axis, keepdims)


def coo_mean(
    a: COOTensor,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> COOTensor | float | EagerValue:
    """Compute mean of elements in a sparse COO tensor over given axes.

    Args:
        a (COOTensor): Input sparse tensor.
        axis (Optional[Union[int, Sequence[int]]]): Reduction axis or axes.
        keepdims (bool): Whether to retain reduced dimensions.

    Returns:
        Union[COOTensor, float, _SP_BACKEND.ndarray]: Mean reduction result.
    """
    if axis is None:
        num_elements: int = a.size
        sum_val: COOTensor | float | EagerValue = coo_sum(a, axis=None, keepdims=keepdims)
        if isinstance(sum_val, COOTensor):
            return COOTensor(indices=sum_val.indices.copy(), values=sum_val.values / num_elements, shape=sum_val.shape)
        return float(typing.cast(float, sum_val)) / num_elements

    axes_list: list[int] = [axis] if isinstance(axis, int) else [int(ax) for ax in axis]
    num_reduced: int = 1
    for ax in axes_list:
        num_reduced *= a.shape[ax % a.ndim]

    sum_res: COOTensor | float | EagerValue = coo_sum(a, axis=axis, keepdims=keepdims)
    if isinstance(sum_res, COOTensor):
        return COOTensor(indices=sum_res.indices.copy(), values=sum_res.values / num_reduced, shape=sum_res.shape)
    return float(typing.cast(float, sum_res)) / num_reduced


def coo_dot(
    a: COOTensor | EagerValue,
    b: COOTensor | EagerValue,
) -> COOTensor | EagerValue | float:
    """Dot product for sparse COO tensors.

    Args:
        a (Union[COOTensor, _SP_BACKEND.ndarray]): Left operand.
        b (Union[COOTensor, _SP_BACKEND.ndarray]): Right operand.

    Returns:
        Union[COOTensor, _SP_BACKEND.ndarray, float]: Dot product result.
    """
    ndim_a: int = a.ndim if isinstance(a, COOTensor) else typing.cast(EagerTensorProtocol, a).ndim
    ndim_b: int = b.ndim if isinstance(b, COOTensor) else typing.cast(EagerTensorProtocol, b).ndim

    if ndim_a == 2 and ndim_b == 2:
        return coo_matmat(a, b)
    if ndim_a == 2 and ndim_b == 1:
        return coo_matvec(a, b)
    if ndim_a == 1 and ndim_b == 1:
        elem: COOTensor | EagerValue = coo_mul(a, b)
        if isinstance(elem, COOTensor):
            return float(_SP_BACKEND.sum(elem.values))
        return float(_SP_BACKEND.sum(elem))

    return coo_matmat(a, b)


def spmm(
    sparse_mat: CSRTensor | CSCTensor | COOTensor | EagerValue,
    dense_mat: EagerValue,
) -> EagerTensorProtocol:
    """Perform sparse-dense matrix multiplication (SpMM): out = sparse_mat @ dense_mat.

    Args:
        sparse_mat (Union[CSRTensor, CSCTensor, COOTensor, _SP_BACKEND.ndarray]): Left sparse matrix operand.
        dense_mat (_SP_BACKEND.ndarray): Right dense matrix or vector operand.

    Returns:
        _SP_BACKEND.ndarray: Evaluated dense matrix product.
    """
    b_arr = _SP_BACKEND.asarray(dense_mat)
    if isinstance(sparse_mat, CSRTensor):
        nrows, _ = sparse_mat.shape
        ncols_b = b_arr.shape[1] if b_arr.ndim > 1 else 1
        out_shape: tuple[int, ...] = (nrows, ncols_b) if b_arr.ndim > 1 else (nrows,)
        out = _SP_BACKEND.zeros(out_shape, dtype=b_arr.dtype)
        for r in range(nrows):
            start = int(sparse_mat.indptr[r])
            end = int(sparse_mat.indptr[r + 1])
            for idx in range(start, end):
                c = sparse_mat.indices[idx]
                val = sparse_mat.data[idx]
                out[r] += val * b_arr[c]
        return typing.cast(EagerTensorProtocol, out)

    if isinstance(sparse_mat, CSCTensor):
        return spmm(sparse_mat.to_csr(), b_arr)

    if isinstance(sparse_mat, COOTensor):
        return spmm(sparse_mat.to_csr(), b_arr)

    return typing.cast(EagerTensorProtocol, _SP_BACKEND.matmul(sparse_mat, b_arr))


def spgemm(
    a: CSRTensor | CSCTensor | COOTensor | EagerValue,
    b: CSRTensor | CSCTensor | COOTensor | EagerValue,
) -> CSRTensor:
    """Perform sparse-sparse matrix multiplication (SpGEMM): out = A @ B.

    Args:
        a (Union[CSRTensor, CSCTensor, COOTensor, _SP_BACKEND.ndarray]): Left sparse matrix operand.
        b (Union[CSRTensor, CSCTensor, COOTensor, _SP_BACKEND.ndarray]): Right sparse matrix operand.

    Returns:
        CSRTensor: Product sparse matrix in CSR format.
    """
    csr_a = a if isinstance(a, CSRTensor) else (a.to_csr() if isinstance(a, (CSCTensor, COOTensor)) else CSRTensor.from_dense(a))
    csr_b = b if isinstance(b, CSRTensor) else (b.to_csr() if isinstance(b, (CSCTensor, COOTensor)) else CSRTensor.from_dense(b))

    m, k1 = csr_a.shape
    k2, n = csr_b.shape
    if k1 != k2:
        raise ValueError(f"Shape mismatch for SpGEMM: {csr_a.shape} vs {csr_b.shape}")

    coo_res = coo_matmat(csr_a.to_coo(), csr_b.to_coo())
    if isinstance(coo_res, COOTensor):
        return coo_res.to_csr()
    return CSRTensor.from_dense(coo_res)


def spmm_grad(
    sparse_mat: CSRTensor | CSCTensor | COOTensor,
    dense_mat: EagerValue,
    cotangent: EagerValue,
) -> tuple[CSRTensor, EagerValue]:
    """Compute gradients for sparse-dense matrix multiplication (SpMM).

    Args:
        sparse_mat (Union[CSRTensor, CSCTensor, COOTensor]): Forward sparse matrix input.
        dense_mat (_SP_BACKEND.ndarray): Forward dense matrix input.
        cotangent (_SP_BACKEND.ndarray): Upstream gradient tensor.

    Returns:
        tuple[CSRTensor, _SP_BACKEND.ndarray]: Gradient with respect to sparse_mat (CSRTensor)
            and gradient with respect to dense_mat (_SP_BACKEND.ndarray).
    """
    csr_a = sparse_mat if isinstance(sparse_mat, CSRTensor) else sparse_mat.to_csr()
    b_dense = _SP_BACKEND.asarray(dense_mat)
    grad_out = _SP_BACKEND.asarray(cotangent)

    at_dense = csr_a.to_dense().T
    grad_dense = _SP_BACKEND.matmul(at_dense, grad_out)

    b_t = b_dense.T if b_dense.ndim > 1 else b_dense.reshape(1, -1)
    dense_grad_a = _SP_BACKEND.matmul(grad_out, b_t) if grad_out.ndim > 1 else _SP_BACKEND.outer(grad_out, b_t)

    nrows, ncols = csr_a.shape
    indptr = [0]
    indices = []
    data = []
    for r in range(nrows):
        start = int(csr_a.indptr[r])
        end = int(csr_a.indptr[r + 1])
        for idx in range(start, end):
            c = csr_a.indices[idx]
            indices.append(c)
            data.append(dense_grad_a[r, c])
        indptr.append(len(data))

    grad_sparse = CSRTensor(
        data=_SP_BACKEND.array(data, dtype=csr_a.dtype),
        indices=_SP_BACKEND.array(indices, dtype=_SP_BACKEND.int64),
        indptr=_SP_BACKEND.array(indptr, dtype=_SP_BACKEND.int64),
        shape=(nrows, ncols),
    )
    return grad_sparse, grad_dense


def spgemm_grad(
    a: CSRTensor | CSCTensor | COOTensor,
    b: CSRTensor | CSCTensor | COOTensor,
    cotangent: CSRTensor | CSCTensor | COOTensor | EagerValue,
) -> tuple[CSRTensor, CSRTensor]:
    """Compute gradients for sparse-sparse matrix multiplication (SpGEMM).

    Args:
        a (Union[CSRTensor, CSCTensor, COOTensor]): Forward sparse matrix A.
        b (Union[CSRTensor, CSCTensor, COOTensor]): Forward sparse matrix B.
        cotangent (Union[CSRTensor, CSCTensor, COOTensor, _SP_BACKEND.ndarray]): Upstream gradient tensor.

    Returns:
        tuple[CSRTensor, CSRTensor]: Gradient w.r.t A and gradient w.r.t B.
    """
    csr_a = a if isinstance(a, CSRTensor) else a.to_csr()
    csr_b = b if isinstance(b, CSRTensor) else b.to_csr()
    dense_cot = cotangent.to_dense() if isinstance(cotangent, (CSRTensor, CSCTensor, COOTensor)) else _SP_BACKEND.asarray(cotangent)

    dense_b = csr_b.to_dense()
    dense_grad_a = _SP_BACKEND.matmul(dense_cot, dense_b.T)
    grad_a_csr = CSRTensor.from_dense(dense_grad_a * typing.cast(EagerValue, typing.cast(EagerValue, csr_a.to_dense()) != 0))

    dense_a = csr_a.to_dense()
    dense_grad_b = _SP_BACKEND.matmul(dense_a.T, dense_cot)
    grad_b_csr = CSRTensor.from_dense(dense_grad_b * typing.cast(EagerValue, typing.cast(EagerValue, csr_b.to_dense()) != 0))

    return grad_a_csr, grad_b_csr


def dense_spmm(
    dense_mat: EagerValue,
    sparse_mat: CSRTensor | CSCTensor | COOTensor | EagerValue,
) -> EagerValue:
    """Perform dense-sparse matrix multiplication: out = dense_mat @ sparse_mat.

    Args:
        dense_mat (_SP_BACKEND.ndarray): Left dense matrix operand.
        sparse_mat (Union[CSRTensor, CSCTensor, COOTensor, _SP_BACKEND.ndarray]): Right sparse matrix operand.

    Returns:
        _SP_BACKEND.ndarray: Evaluated dense matrix product.
    """
    d_arr = _SP_BACKEND.asarray(dense_mat)
    if isinstance(sparse_mat, (CSRTensor, CSCTensor, COOTensor)):
        s_coo = sparse_mat if isinstance(sparse_mat, COOTensor) else sparse_mat.to_coo()
        s_t = coo_transpose(s_coo)
        d_t = d_arr.T
        res_t = spmm(s_t, d_t)
        return res_t.T
    return typing.cast(EagerValue, _SP_BACKEND.matmul(d_arr, sparse_mat))


def csr_add(a: CSRTensor, b: CSRTensor) -> CSRTensor:
    """Perform element-wise addition of two CSR matrices without dense materialization.

    Args:
        a (CSRTensor): First CSR operand.
        b (CSRTensor): Second CSR operand.

    Returns:
        CSRTensor: Resulting CSR matrix.

    Raises:
        ValueError: If matrix shapes do not match.
    """
    if a.shape != b.shape:
        msg = f"Shape mismatch for CSR addition: {a.shape} vs {b.shape}"
        raise ValueError(msg)
    if a.nnz == 0:
        return b
    if b.nnz == 0:
        return a
    coo_res = coo_add(a.to_coo(), b.to_coo())
    if isinstance(coo_res, COOTensor):
        return coo_res.to_csr()
    return CSRTensor.from_dense(coo_res)


def csc_add(a: CSCTensor, b: CSCTensor) -> CSCTensor:
    """Perform element-wise addition of two CSC matrices without dense materialization.

    Args:
        a (CSCTensor): First CSC operand.
        b (CSCTensor): Second CSC operand.

    Returns:
        CSCTensor: Resulting CSC matrix.

    Raises:
        ValueError: If matrix shapes do not match.
    """
    if a.shape != b.shape:
        msg = f"Shape mismatch for CSC addition: {a.shape} vs {b.shape}"
        raise ValueError(msg)
    if a.nnz == 0:
        return b
    if b.nnz == 0:
        return a
    coo_res = coo_add(a.to_coo(), b.to_coo())
    if isinstance(coo_res, COOTensor):
        return coo_res.to_csc()
    return CSCTensor.from_dense(coo_res)


def sparse_mask(
    tensor: COOTensor | EagerValue,
    mask: COOTensor | CSRTensor | CSCTensor | EagerValue,
) -> COOTensor:
    """Filter a dense or sparse tensor using a sparse boolean/coordinate mask.

    Args:
        tensor (Union[COOTensor, _SP_BACKEND.ndarray]): Input data tensor.
        mask (Union[COOTensor, CSRTensor, CSCTensor, _SP_BACKEND.ndarray]): Mask tensor specifying
            retained coordinates.

    Returns:
        COOTensor: Filtered sparse COO tensor with non-zero values only where mask is non-zero.
    """
    t_dense = tensor.to_dense() if isinstance(tensor, COOTensor) else _SP_BACKEND.asarray(tensor)
    m_dense = mask.to_dense() if isinstance(mask, (COOTensor, CSRTensor, CSCTensor)) else _SP_BACKEND.asarray(mask)
    filtered = _SP_BACKEND.where(typing.cast(EagerValue, m_dense) != 0, t_dense, 0)
    return COOTensor.from_dense(filtered)


def _prepare_conv2d_tensors(
    x: COOTensor | EagerValue,
    weight: EagerValue,
) -> tuple[EagerTensorProtocol, EagerTensorProtocol]:
    """Normalize input and weight dimensions to 4D tensors for convolution.

    Args:
        x (Union[COOTensor, _SP_BACKEND.ndarray]): Input tensor.
        weight (_SP_BACKEND.ndarray): Convolution filter weights.

    Returns:
        tuple[_SP_BACKEND.ndarray, _SP_BACKEND.ndarray]: 4D input array and 4D weight array.
    """
    x_dense = x.to_dense() if isinstance(x, COOTensor) else _SP_BACKEND.asarray(x)
    while x_dense.ndim < 4:
        x_dense = x_dense[None, :]

    w_arr = _SP_BACKEND.asarray(weight)
    while w_arr.ndim < 4:
        w_arr = w_arr[None, :] if w_arr.ndim == 3 else w_arr[None, None, :]
    return x_dense, w_arr


def _apply_sparse_mask(
    dense_out: EagerTensorProtocol,
    mask: COOTensor | CSRTensor | CSCTensor | EagerValue,
) -> COOTensor:
    """Filter evaluated dense convolution activations by an expanded sparse mask.

    Args:
        dense_out (_SP_BACKEND.ndarray): Dense convolution outputs.
        mask (Union[COOTensor, CSRTensor, CSCTensor, _SP_BACKEND.ndarray]): Target sparse mask.

    Returns:
        COOTensor: Filtered sparse COO tensor.
    """
    mask_coo = mask if isinstance(mask, COOTensor) else (mask.to_coo() if isinstance(mask, (CSRTensor, CSCTensor)) else COOTensor.from_dense(_SP_BACKEND.asarray(mask)))
    mask_dense = typing.cast(EagerTensorProtocol, typing.cast(EagerValue, mask_coo.to_dense()) != 0)
    while mask_dense.ndim < dense_out.ndim:
        mask_dense = mask_dense[None, :]
    mask_broadcast = _SP_BACKEND.broadcast_to(mask_dense, dense_out.shape)
    masked_out = _SP_BACKEND.where(mask_broadcast, dense_out, 0)
    return COOTensor.from_dense(masked_out)


def sparse_conv2d_mask(
    x: COOTensor | EagerValue,
    weight: EagerValue,
    mask: COOTensor | CSRTensor | CSCTensor | EagerValue,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
) -> COOTensor:
    """Perform 2D convolution and filter results using a sparse mask.

    Only locations where the spatial mask is non-zero are retained in the
    resulting sparse COO tensor.

    Args:
        x (Union[COOTensor, _SP_BACKEND.ndarray]): Input tensor of shape (N, C_in, H, W) or (H, W).
        weight (_SP_BACKEND.ndarray): Convolution kernel weights of shape (C_out, C_in, K_h, K_w).
        mask (Union[COOTensor, CSRTensor, CSCTensor, _SP_BACKEND.ndarray]): Binary or coordinate mask
            specifying active output locations.
        stride (tuple[int, int]): Stride along height and width. Defaults to (1, 1).
        padding (tuple[int, int]): Zero-padding added to height and width. Defaults to (0, 0).

    Returns:
        COOTensor: Output sparse COO tensor containing evaluated convolution activations
            strictly at masked locations.
    """
    x_dense, w_arr = _prepare_conv2d_tensors(x, weight)
    n_batch, c_in, h_in, w_in = x_dense.shape
    c_out, _, k_h, k_w = w_arr.shape
    s_h, s_w = stride
    p_h, p_w = padding

    h_out = (h_in + 2 * p_h - k_h) // s_h + 1
    w_out = (w_in + 2 * p_w - k_w) // s_w + 1

    if p_h > 0 or p_w > 0:
        x_padded = _SP_BACKEND.zeros((n_batch, c_in, h_in + 2 * p_h, w_in + 2 * p_w), dtype=x_dense.dtype)
        x_padded[:, :, p_h : p_h + h_in, p_w : p_w + w_in] = x_dense
    else:
        x_padded = x_dense

    out_dense = _SP_BACKEND.zeros((n_batch, c_out, h_out, w_out), dtype=_SP_BACKEND.promote_types(x_dense.dtype, w_arr.dtype))

    for b in range(n_batch):
        for o in range(c_out):
            for i in range(h_out):
                h_start = i * s_h
                h_end = h_start + k_h
                for j in range(w_out):
                    w_start = j * s_w
                    w_end = w_start + k_w
                    window = x_padded[b, :, h_start:h_end, w_start:w_end]
                    out_dense[b, o, i, j] = _SP_BACKEND.sum(window * w_arr[o])

    return _apply_sparse_mask(out_dense, mask)


def csr_sub(a: CSRTensor, b: CSRTensor) -> CSRTensor:
    """Subtract two Compressed Sparse Row (CSR) tensors.

    Args:
        a (CSRTensor): Left operand CSR tensor.
        b (CSRTensor): Right operand CSR tensor.

    Returns:
        CSRTensor: Resulting CSR tensor representing a - b.

    Raises:
        ValueError: If matrix shapes do not match.
    """
    if a.shape != b.shape:
        msg = f"Shape mismatch for CSR subtraction: {a.shape} vs {b.shape}"
        raise ValueError(msg)
    if a.nnz == 0:
        b_neg = coo_neg(b.to_coo())
        return b_neg.to_csr()
    if b.nnz == 0:
        return a
    return typing.cast(COOTensor, coo_sub(a.to_coo(), b.to_coo())).to_csr()


def csc_sub(a: CSCTensor, b: CSCTensor) -> CSCTensor:
    """Subtract two Compressed Sparse Column (CSC) tensors.

    Args:
        a (CSCTensor): Left operand CSC tensor.
        b (CSCTensor): Right operand CSC tensor.

    Returns:
        CSCTensor: Resulting CSC tensor representing a - b.

    Raises:
        ValueError: If matrix shapes do not match.
    """
    if a.shape != b.shape:
        msg = f"Shape mismatch for CSC subtraction: {a.shape} vs {b.shape}"
        raise ValueError(msg)
    if a.nnz == 0:
        b_neg = coo_neg(b.to_coo())
        return b_neg.to_csc()
    if b.nnz == 0:
        return a
    return typing.cast(COOTensor, coo_sub(a.to_coo(), b.to_coo())).to_csc()


def sparse_softmax(
    matrix: COOTensor | CSRTensor | CSCTensor | EagerValue,
    mask: COOTensor | EagerValue | None = None,
) -> COOTensor:
    """Compute row-wise softmax exclusively over non-zero elements for GNN attention.

    Args:
        matrix (Union[COOTensor, CSRTensor, CSCTensor, _SP_BACKEND.ndarray]): Input sparse or dense matrix.
        mask (Optional[Union[COOTensor, _SP_BACKEND.ndarray]]): Optional structural mask.

    Returns:
        COOTensor: Sparse attention matrix with normalized probabilities along rows.
    """
    coo: COOTensor = matrix if isinstance(matrix, COOTensor) else (matrix.to_coo() if isinstance(matrix, (CSRTensor, CSCTensor)) else COOTensor.from_dense(_SP_BACKEND.asarray(matrix)))
    if mask is not None:
        coo = sparse_mask(coo, mask)

    if coo.nnz == 0:
        return coo

    rows = coo.indices[0]
    vals = coo.values.astype(_SP_BACKEND.float64)
    n_rows = coo.shape[0]

    # Row-wise max for numerical stability
    row_max = _SP_BACKEND.full(n_rows, -_SP_BACKEND.inf, dtype=_SP_BACKEND.float64)
    for r, v in zip(rows, vals):
        if float(v) > float(row_max[r]):
            row_max[r] = v

    # Compute exp(v - max)
    exp_vals = _SP_BACKEND.zeros_like(vals)
    for idx, (r, v) in enumerate(zip(rows, vals)):
        exp_vals[idx] = _SP_BACKEND.exp(v - row_max[r])

    # Sum per row
    row_sum = _SP_BACKEND.zeros(n_rows, dtype=_SP_BACKEND.float64)
    for r, ev in zip(rows, exp_vals):
        row_sum[r] += ev

    # Normalize
    out_vals = _SP_BACKEND.zeros_like(exp_vals, dtype=coo.values.dtype)
    for idx, (r, ev) in enumerate(zip(rows, exp_vals)):
        out_vals[idx] = ev / row_sum[r] if float(row_sum[r]) > 0 else 0.0

    return COOTensor(indices=coo.indices, values=out_vals, shape=coo.shape)


def graph_norm_adjacency(
    adj: COOTensor | CSRTensor | CSCTensor | EagerValue,
    add_self_loops: bool = True,
) -> COOTensor:
    """Compute symmetric normalized graph adjacency matrix D^{-1/2} A D^{-1/2}.

    Args:
        adj (Union[COOTensor, CSRTensor, CSCTensor, _SP_BACKEND.ndarray]): Graph adjacency matrix.
        add_self_loops (bool): Whether to add self-loops (A + I) before normalizing.

    Returns:
        COOTensor: Symmetrically normalized sparse adjacency matrix.
    """
    coo: COOTensor = adj if isinstance(adj, COOTensor) else (adj.to_coo() if isinstance(adj, (CSRTensor, CSCTensor)) else COOTensor.from_dense(_SP_BACKEND.asarray(adj)))
    n = coo.shape[0]

    if add_self_loops:
        diag_indices = _SP_BACKEND.stack([_SP_BACKEND.arange(n), _SP_BACKEND.arange(n)], axis=0)
        diag_values = _SP_BACKEND.ones(n, dtype=coo.values.dtype)
        eye_coo = COOTensor(indices=diag_indices, values=diag_values, shape=(n, n))
        coo = typing.cast(COOTensor, coo_add(coo, eye_coo))

    # Compute row degrees
    deg = _SP_BACKEND.zeros(n, dtype=_SP_BACKEND.float64)
    for r, v in zip(coo.indices[0], coo.values):
        deg[r] += float(v)

    # D^{-1/2}
    deg_inv_sqrt = _SP_BACKEND.zeros(n, dtype=_SP_BACKEND.float64)
    nonzero_mask = typing.cast(EagerTensorProtocol, deg > 0)
    deg_inv_sqrt[nonzero_mask] = 1.0 / _SP_BACKEND.sqrt(deg[nonzero_mask])

    # Multiply entries: D^{-1/2}[i] * A[i, j] * D^{-1/2}[j]
    norm_vals = _SP_BACKEND.zeros_like(coo.values, dtype=_SP_BACKEND.float32)
    for idx, (i, j, v) in enumerate(zip(coo.indices[0], coo.indices[1], coo.values)):
        norm_vals[idx] = float(deg_inv_sqrt[i] * float(v) * deg_inv_sqrt[j])

    return COOTensor(indices=coo.indices, values=norm_vals, shape=coo.shape)


def gnn_spmm_attention(
    adj: COOTensor | CSRTensor | CSCTensor | EagerValue,
    features: EagerValue,
    attention_weights: COOTensor | EagerValue | None = None,
) -> EagerValue:
    """Compute Graph Neural Network message aggregation with optional attention weights.

    Args:
        adj (Union[COOTensor, CSRTensor, CSCTensor, _SP_BACKEND.ndarray]): Graph connectivity adjacency matrix.
        features (_SP_BACKEND.ndarray): Node feature representations of shape (N, F).
        attention_weights (Optional[Union[COOTensor, _SP_BACKEND.ndarray]]): Optional attention tensor.

    Returns:
        _SP_BACKEND.ndarray: Updated node feature matrix after message passing.
    """
    coo_adj: COOTensor = adj if isinstance(adj, COOTensor) else (adj.to_coo() if isinstance(adj, (CSRTensor, CSCTensor)) else COOTensor.from_dense(_SP_BACKEND.asarray(adj)))
    if attention_weights is not None:
        attn_coo = attention_weights if isinstance(attention_weights, COOTensor) else (attention_weights.to_coo() if isinstance(attention_weights, (CSRTensor, CSCTensor)) else COOTensor.from_dense(_SP_BACKEND.asarray(attention_weights)))
        norm_attn = sparse_softmax(attn_coo, mask=coo_adj)
        return spmm(norm_attn, features)
    return spmm(coo_adj, features)


def sparse_dropout(
    tensor: COOTensor | CSRTensor | CSCTensor,
    drop_rate: float = 0.5,
    training: bool = True,
    seed: int | None = None,
) -> COOTensor | CSRTensor | CSCTensor:
    """Apply edge dropout to non-zero values of a sparse matrix for DropEdge regularization.

    Args:
        tensor (Union[COOTensor, CSRTensor, CSCTensor]): Sparse tensor to regularize.
        drop_rate (float): Fraction of edges to drop. Defaults to 0.5.
        training (bool): Whether in training mode. If False, returns tensor unchanged.
        seed (Optional[int]): Random seed for reproducible dropout.

    Returns:
        Union[COOTensor, CSRTensor, CSCTensor]: Sparse tensor with dropped edges scaled by 1/(1-p).
    """
    if not training or drop_rate <= 0.0 or tensor.nnz == 0:
        return tensor

    coo = tensor if isinstance(tensor, COOTensor) else tensor.to_coo()

    if drop_rate >= 1.0:
        empty_indices = _SP_BACKEND.zeros((len(coo.shape), 0), dtype=coo.indices.dtype)
        empty_values = _SP_BACKEND.zeros(0, dtype=coo.values.dtype)
        coo_res = COOTensor(indices=empty_indices, values=empty_values, shape=coo.shape)
    else:
        keep_prob = 1.0 - drop_rate
        scale = 1.0 / keep_prob
        rng = _SP_BACKEND.random.default_rng(seed)
        keep_mask = typing.cast(EagerTensorProtocol, rng.uniform(0.0, 1.0, size=coo.nnz) < keep_prob)
        if not _SP_BACKEND.any(keep_mask):
            empty_indices = _SP_BACKEND.zeros((len(coo.shape), 0), dtype=coo.indices.dtype)
            empty_values = _SP_BACKEND.zeros(0, dtype=coo.values.dtype)
            coo_res = COOTensor(indices=empty_indices, values=empty_values, shape=coo.shape)
        else:
            new_indices = coo.indices[:, keep_mask]
            new_values = coo.values[keep_mask] * scale
            coo_res = COOTensor(indices=new_indices, values=new_values.astype(coo.values.dtype), shape=coo.shape)

    if isinstance(tensor, CSRTensor):
        return coo_res.to_csr()
    if isinstance(tensor, CSCTensor):
        return coo_res.to_csc()
    return coo_res


def __getattr__(name: str) -> EagerValue:
    """Fallback to numpy elementwise or array functions for sparse kernels.

    Args:
        name (str): Attribute or function name.

    Returns:
        EagerValue: Numpy function if available.

    Raises:
        AttributeError: If attribute is not found in numpy.
    """
    if hasattr(_SP_BACKEND, name):
        return typing.cast(EagerValue, getattr(_SP_BACKEND, name))
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
