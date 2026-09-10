"""Dedicated Sparse Coordinate (COO) Tensor execution kernels and operations."""

from collections.abc import Sequence
from typing import Optional, Union

import numpy as np

from ml_switcheroo_compiler.backends.sparse.types import COOTensor


def coo_sum_duplicates(
    indices: np.ndarray,
    values: np.ndarray,
    shape: tuple[int, ...],
) -> COOTensor:
    """Sum duplicate coordinate entries in a sparse COO representation.

    Args:
        indices (np.ndarray): Coordinate array of shape (ndim, nnz).
        values (np.ndarray): Data array of shape (nnz,).
        shape (tuple[int, ...]): Shape of the tensor.

    Returns:
        COOTensor: Canonical COO tensor with unique sorted coordinates.
    """
    if values.size == 0:
        return COOTensor(indices=indices, values=values, shape=shape)

    ndim: int = len(shape)
    indices_arr: np.ndarray = np.asarray(indices, dtype=np.int64)
    if indices_arr.ndim == 1:
        indices_arr = indices_arr.reshape(ndim, -1)

    flat_indices: np.ndarray = np.ravel_multi_index(
        tuple(indices_arr[i] for i in range(ndim)),
        shape,
    )
    sort_order: np.ndarray = np.argsort(flat_indices)
    sorted_flat: np.ndarray = flat_indices[sort_order]
    sorted_values: np.ndarray = values[sort_order]

    unique_flat, split_indices = np.unique(sorted_flat, return_index=True)
    summed_values: np.ndarray = np.add.reduceat(sorted_values, split_indices)

    non_zeros: np.ndarray = summed_values != 0
    unique_flat = unique_flat[non_zeros]
    summed_values = summed_values[non_zeros]

    if unique_flat.size == 0:
        empty_indices: np.ndarray = np.empty((ndim, 0), dtype=np.int64)
        empty_values: np.ndarray = np.empty((0,), dtype=values.dtype)
        return COOTensor(indices=empty_indices, values=empty_values, shape=shape)

    unraveled: tuple[np.ndarray, ...] = np.unravel_index(unique_flat, shape)
    new_indices: np.ndarray = np.vstack(unraveled)
    return COOTensor(indices=new_indices, values=summed_values, shape=shape)


def coo_fromdense(
    mat: Union[np.ndarray, Sequence[float], COOTensor],
    nse: Optional[int] = None,
    index_dtype: Optional[Union[np.dtype, str]] = None,
) -> COOTensor:
    """Create a COO-format sparse matrix from a dense matrix.

    Args:
        mat (Union[np.ndarray, Sequence[float], COOTensor]): Input dense array or tensor.
        nse (Optional[int]): Optional maximum number of specified non-zero elements.
        index_dtype (Optional[Union[np.dtype, str]]): Data type for indices.

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

    dense_arr: np.ndarray = np.asarray(mat)
    nonzero_mask: np.ndarray = dense_arr != 0
    indices: np.ndarray = np.vstack(np.nonzero(nonzero_mask))
    if index_dtype is not None:
        indices = indices.astype(index_dtype)
    values: np.ndarray = dense_arr[nonzero_mask]
    return COOTensor(indices=indices, values=values, shape=dense_arr.shape)


def coo_todense(mat: Union[COOTensor, np.ndarray, Sequence[float]]) -> np.ndarray:
    """Convert a COO-format sparse matrix to a dense matrix.

    Args:
        mat (Union[COOTensor, np.ndarray, Sequence[float]]): Input sparse or dense array.

    Returns:
        np.ndarray: Dense NumPy array.
    """
    if isinstance(mat, COOTensor):
        return mat.to_dense()
    return np.asarray(mat)


def coo_matmat(
    mat: Union[COOTensor, np.ndarray],
    B: Union[COOTensor, np.ndarray],
    transpose: bool = False,
) -> Union[COOTensor, np.ndarray]:
    """Product of COO sparse matrix and a dense or sparse matrix.

    Args:
        mat (Union[COOTensor, np.ndarray]): Left operand matrix (sparse or dense).
        B (Union[COOTensor, np.ndarray]): Right operand matrix (sparse or dense).
        transpose (bool): Whether to transpose the left matrix before multiplication.

    Returns:
        Union[COOTensor, np.ndarray]: Matrix product result.
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
                indices=np.empty((2, 0), dtype=np.int64),
                values=np.empty((0,), dtype=np.promote_types(a_coo.dtype, B.dtype)),
                shape=out_shape,
            )

        match_mask: np.ndarray = a_coo.indices[1, :, None] == B.indices[0, None, :]
        a_matches, b_matches = np.nonzero(match_mask)

        if a_matches.size == 0:
            return COOTensor(
                indices=np.empty((2, 0), dtype=np.int64),
                values=np.empty((0,), dtype=np.promote_types(a_coo.dtype, B.dtype)),
                shape=out_shape,
            )

        res_rows: np.ndarray = a_coo.indices[0, a_matches]
        res_cols: np.ndarray = B.indices[1, b_matches]
        res_vals: np.ndarray = a_coo.values[a_matches] * B.values[b_matches]
        stacked_indices: np.ndarray = np.vstack([res_rows, res_cols])
        return coo_sum_duplicates(stacked_indices, res_vals, out_shape)

    b_dense: np.ndarray = np.asarray(B)
    n = b_dense.shape[1]
    res_dense: np.ndarray = np.zeros((m, n), dtype=np.promote_types(a_coo.dtype, b_dense.dtype))
    if a_coo.nnz > 0:
        row_indices: np.ndarray = a_coo.indices[0]
        col_indices: np.ndarray = a_coo.indices[1]
        contributions: np.ndarray = a_coo.values[:, None] * b_dense[col_indices]
        np.add.at(res_dense, row_indices, contributions)
    return res_dense


def coo_matvec(
    mat: Union[COOTensor, np.ndarray],
    v: Union[COOTensor, np.ndarray],
    transpose: bool = False,
) -> np.ndarray:
    """Product of COO sparse matrix and a dense vector.

    Args:
        mat (Union[COOTensor, np.ndarray]): Matrix operand.
        v (Union[COOTensor, np.ndarray]): Vector operand.
        transpose (bool): Whether to transpose matrix before multiplication.

    Returns:
        np.ndarray: Matrix-vector product result.
    """
    a_coo: COOTensor = mat if isinstance(mat, COOTensor) else coo_fromdense(mat)
    if transpose:
        a_coo = coo_transpose(a_coo)

    v_dense: np.ndarray = v.to_dense() if isinstance(v, COOTensor) else np.asarray(v)
    m: int = a_coo.shape[0]
    res: np.ndarray = np.zeros(m, dtype=np.promote_types(a_coo.dtype, v_dense.dtype))
    if a_coo.nnz > 0:
        row_indices: np.ndarray = a_coo.indices[0]
        col_indices: np.ndarray = a_coo.indices[1]
        contributions: np.ndarray = a_coo.values * v_dense[col_indices]
        np.add.at(res, row_indices, contributions)
    return res


def coo_transpose(
    mat: COOTensor,
    axes: Optional[Sequence[int]] = None,
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
        new_indices: np.ndarray = np.empty((mat.ndim, 0), dtype=np.int64)
        return COOTensor(indices=new_indices, values=mat.values.copy(), shape=new_shape)

    perm_indices: np.ndarray = mat.indices[perm]
    return coo_sum_duplicates(perm_indices, mat.values.copy(), new_shape)


def coo_add(
    a: Union[COOTensor, np.ndarray, float],
    b: Union[COOTensor, np.ndarray, float],
) -> Union[COOTensor, np.ndarray]:
    """Element-wise addition for sparse COO tensors.

    Args:
        a (Union[COOTensor, np.ndarray, float]): First addend.
        b (Union[COOTensor, np.ndarray, float]): Second addend.

    Returns:
        Union[COOTensor, np.ndarray]: Sum result.
    """
    if isinstance(a, COOTensor) and isinstance(b, COOTensor):
        if a.shape != b.shape:
            msg = f"Shape mismatch for sparse addition: {a.shape} vs {b.shape}"
            raise ValueError(msg)
        if a.nnz == 0:
            return b
        if b.nnz == 0:
            return a
        stacked_indices: np.ndarray = np.hstack([a.indices, b.indices])
        stacked_values: np.ndarray = np.concatenate([a.values, b.values])
        return coo_sum_duplicates(stacked_indices, stacked_values, a.shape)

    if isinstance(a, COOTensor):
        return a.to_dense() + b
    if isinstance(b, COOTensor):
        return a + b.to_dense()
    return np.asarray(a) + np.asarray(b)


def coo_neg(a: COOTensor) -> COOTensor:
    """Element-wise negation of a sparse COO tensor.

    Args:
        a (COOTensor): Input sparse tensor.

    Returns:
        COOTensor: Negated sparse tensor.
    """
    return COOTensor(indices=a.indices.copy(), values=-a.values, shape=a.shape)


def coo_sub(
    a: Union[COOTensor, np.ndarray, float],
    b: Union[COOTensor, np.ndarray, float],
) -> Union[COOTensor, np.ndarray]:
    """Element-wise subtraction for sparse COO tensors.

    Args:
        a (Union[COOTensor, np.ndarray, float]): Minuend.
        b (Union[COOTensor, np.ndarray, float]): Subtrahend.

    Returns:
        Union[COOTensor, np.ndarray]: Difference result.
    """
    if isinstance(b, COOTensor):
        return coo_add(a, coo_neg(b))
    return coo_add(a, -b)


def _coo_mul_scalar(t: COOTensor, scalar: Union[int, float, np.number]) -> COOTensor:
    """Multiply sparse COO tensor by a scalar value.

    Args:
        t (COOTensor): Target sparse tensor.
        scalar (Union[int, float, np.number]): Scalar factor.

    Returns:
        COOTensor: Scaled sparse tensor.
    """
    if scalar == 0:
        return COOTensor(
            indices=np.empty((t.ndim, 0), dtype=np.int64),
            values=np.empty((0,), dtype=t.dtype),
            shape=t.shape,
        )
    return COOTensor(indices=t.indices.copy(), values=t.values * scalar, shape=t.shape)


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
            indices=np.empty((a.ndim, 0), dtype=np.int64),
            values=np.empty((0,), dtype=np.promote_types(a.dtype, b.dtype)),
            shape=a.shape,
        )
    flat_a: np.ndarray = np.ravel_multi_index(tuple(a.indices[i] for i in range(a.ndim)), a.shape)
    flat_b: np.ndarray = np.ravel_multi_index(tuple(b.indices[i] for i in range(b.ndim)), b.shape)
    common, idx_a, idx_b = np.intersect1d(flat_a, flat_b, return_indices=True)
    if common.size == 0:
        return COOTensor(
            indices=np.empty((a.ndim, 0), dtype=np.int64),
            values=np.empty((0,), dtype=np.promote_types(a.dtype, b.dtype)),
            shape=a.shape,
        )
    new_values: np.ndarray = a.values[idx_a] * b.values[idx_b]
    new_indices: np.ndarray = a.indices[:, idx_a]
    return COOTensor(indices=new_indices, values=new_values, shape=a.shape)


def _coo_mul_dense(a: COOTensor, b_dense: np.ndarray) -> COOTensor:
    """Multiply a sparse COO tensor with a dense array element-wise.

    Args:
        a (COOTensor): Sparse tensor factor.
        b_dense (np.ndarray): Dense array factor.

    Returns:
        COOTensor: Filtered sparse product.
    """
    if a.nnz == 0:
        return COOTensor(indices=a.indices.copy(), values=a.values.copy(), shape=a.shape)
    coords_tuple: tuple[np.ndarray, ...] = tuple(a.indices[i] for i in range(a.ndim))
    sampled_b: np.ndarray = b_dense[coords_tuple]
    return COOTensor(indices=a.indices.copy(), values=a.values * sampled_b, shape=a.shape)


def coo_mul(
    a: Union[COOTensor, np.ndarray, float, int],
    b: Union[COOTensor, np.ndarray, float, int],
) -> Union[COOTensor, np.ndarray]:
    """Element-wise multiplication (Hadamard product) for sparse COO tensors.

    Args:
        a (Union[COOTensor, np.ndarray, float, int]): First factor.
        b (Union[COOTensor, np.ndarray, float, int]): Second factor.

    Returns:
        Union[COOTensor, np.ndarray]: Product result.
    """
    if isinstance(a, (int, float, np.number)):
        return _coo_mul_scalar(b, a)
    if isinstance(b, (int, float, np.number)):
        return _coo_mul_scalar(a, b)
    if isinstance(a, COOTensor) and isinstance(b, COOTensor):
        return _coo_mul_coo(a, b)
    if isinstance(a, COOTensor):
        return _coo_mul_dense(a, np.asarray(b))
    if isinstance(b, COOTensor):
        return _coo_mul_dense(b, np.asarray(a))
    return np.asarray(a) * np.asarray(b)


def coo_abs(a: COOTensor) -> COOTensor:
    """Element-wise absolute value of a sparse COO tensor.

    Args:
        a (COOTensor): Input sparse tensor.

    Returns:
        COOTensor: Tensor with absolute values.
    """
    return COOTensor(indices=a.indices.copy(), values=np.abs(a.values), shape=a.shape)


def coo_relu(a: COOTensor) -> COOTensor:
    """Apply Rectified Linear Unit (ReLU) to sparse tensor values.

    Args:
        a (COOTensor): Input sparse tensor.

    Returns:
        COOTensor: Filtered tensor retaining only strictly positive elements.
    """
    mask: np.ndarray = a.values > 0
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
            indices=np.empty((len(target_shape), 0), dtype=np.int64),
            values=a.values.copy(),
            shape=target_shape,
        )

    flat: np.ndarray = np.ravel_multi_index(tuple(a.indices[i] for i in range(a.ndim)), a.shape)
    unraveled: tuple[np.ndarray, ...] = np.unravel_index(flat, target_shape)
    new_indices: np.ndarray = np.vstack(unraveled)
    return COOTensor(indices=new_indices, values=a.values.copy(), shape=target_shape)


def _coo_sum_empty(
    a: COOTensor,
    axis: Optional[Union[int, Sequence[int]]],
    keepdims: bool,
) -> Union[COOTensor, float]:
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
        return COOTensor(np.empty((len(out_shape), 0), dtype=np.int64), np.empty((0,), dtype=a.dtype), out_shape)

    axes_list: list[int] = [axis] if isinstance(axis, int) else [int(ax) for ax in axis]
    reduced_shape: list[int] = list(a.shape)
    for ax in axes_list:
        reduced_shape[ax % a.ndim] = 1
    final_shape: tuple[int, ...] = tuple(reduced_shape) if keepdims else tuple(s for i, s in enumerate(reduced_shape) if (i % a.ndim) not in [x % a.ndim for x in axes_list])
    return COOTensor(np.empty((len(final_shape), 0), dtype=np.int64), np.empty((0,), dtype=a.dtype), final_shape)


def _coo_sum_axis(
    a: COOTensor,
    axis: Union[int, Sequence[int]],
    keepdims: bool,
) -> Union[COOTensor, float, np.ndarray]:
    """Compute sum reduction along specified axis or axes.

    Args:
        a (COOTensor): Non-empty sparse tensor.
        axis (Union[int, Sequence[int]]): Axes to reduce.
        keepdims (bool): Whether to retain reduced dimensions.

    Returns:
        Union[COOTensor, float, np.ndarray]: Reduced sparse tensor or scalar.
    """
    axes_set: set[int] = {axis % a.ndim} if isinstance(axis, int) else {(ax % a.ndim) for ax in axis}
    kept_dims: list[int] = [i for i in range(a.ndim) if i not in axes_set]

    if not kept_dims:
        return coo_sum(a, axis=None, keepdims=keepdims)

    out_shape_kept: tuple[int, ...] = tuple(a.shape[i] for i in kept_dims)
    kept_indices: np.ndarray = a.indices[kept_dims]
    collapsed: COOTensor = coo_sum_duplicates(kept_indices, a.values.copy(), out_shape_kept)

    if keepdims:
        final_shape: list[int] = list(a.shape)
        for ax in axes_set:
            final_shape[ax] = 1
        return coo_reshape(collapsed, tuple(final_shape))
    return collapsed


def coo_sum(
    a: COOTensor,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
) -> Union[COOTensor, float, np.ndarray]:
    """Sum elements of a sparse COO tensor over given axes.

    Args:
        a (COOTensor): Input sparse tensor.
        axis (Optional[Union[int, Sequence[int]]]): Axis or axes along which to sum.
        keepdims (bool): Whether to retain reduced dimensions.

    Returns:
        Union[COOTensor, float, np.ndarray]: Sum reduction result.
    """
    if a.nnz == 0:
        return _coo_sum_empty(a, axis, keepdims)

    if axis is None:
        total: float = float(np.sum(a.values))
        if keepdims:
            out_shape = tuple(1 for _ in a.shape)
            idx = np.zeros((len(out_shape), 1), dtype=np.int64)
            val = np.array([total], dtype=a.dtype)
            return COOTensor(indices=idx, values=val, shape=out_shape)
        return total

    return _coo_sum_axis(a, axis, keepdims)


def coo_mean(
    a: COOTensor,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
) -> Union[COOTensor, float, np.ndarray]:
    """Compute mean of elements in a sparse COO tensor over given axes.

    Args:
        a (COOTensor): Input sparse tensor.
        axis (Optional[Union[int, Sequence[int]]]): Reduction axis or axes.
        keepdims (bool): Whether to retain reduced dimensions.

    Returns:
        Union[COOTensor, float, np.ndarray]: Mean reduction result.
    """
    if axis is None:
        num_elements: int = a.size
        sum_val: Union[COOTensor, float, np.ndarray] = coo_sum(a, axis=None, keepdims=keepdims)
        if isinstance(sum_val, COOTensor):
            return COOTensor(indices=sum_val.indices.copy(), values=sum_val.values / num_elements, shape=sum_val.shape)
        return float(sum_val) / num_elements

    axes_list: list[int] = [axis] if isinstance(axis, int) else [int(ax) for ax in axis]
    num_reduced: int = 1
    for ax in axes_list:
        num_reduced *= a.shape[ax % a.ndim]

    sum_res: Union[COOTensor, float, np.ndarray] = coo_sum(a, axis=axis, keepdims=keepdims)
    if isinstance(sum_res, COOTensor):
        return COOTensor(indices=sum_res.indices.copy(), values=sum_res.values / num_reduced, shape=sum_res.shape)
    return float(sum_res) / num_reduced


def coo_dot(
    a: Union[COOTensor, np.ndarray],
    b: Union[COOTensor, np.ndarray],
) -> Union[COOTensor, np.ndarray, float]:
    """Dot product for sparse COO tensors.

    Args:
        a (Union[COOTensor, np.ndarray]): Left operand.
        b (Union[COOTensor, np.ndarray]): Right operand.

    Returns:
        Union[COOTensor, np.ndarray, float]: Dot product result.
    """
    ndim_a: int = a.ndim
    ndim_b: int = b.ndim

    if ndim_a == 2 and ndim_b == 2:
        return coo_matmat(a, b)
    if ndim_a == 2 and ndim_b == 1:
        return coo_matvec(a, b)
    if ndim_a == 1 and ndim_b == 1:
        elem: Union[COOTensor, np.ndarray] = coo_mul(a, b)
        if isinstance(elem, COOTensor):
            return float(np.sum(elem.values))
        return float(np.sum(elem))

    return coo_matmat(a, b)
