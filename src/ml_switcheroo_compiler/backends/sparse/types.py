"""Sparse COO backend type definitions and array creation utilities."""

from collections.abc import Sequence
from typing import Optional, Union

import numpy as np


class COOTensor:
    """Dedicated multidimensional Sparse Coordinate (COO) Tensor implementation."""

    def __init__(
        self,
        indices: np.ndarray,
        values: np.ndarray,
        shape: tuple[int, ...],
    ) -> None:
        """Initialize a Sparse COO Tensor.

        Args:
            indices (np.ndarray): Integer coordinate array of shape (ndim, nnz).
            values (np.ndarray): Data array of shape (nnz,).
            shape (tuple[int, ...]): Shape of the full dense tensor.
        """
        self.shape: tuple[int, ...] = tuple(int(s) for s in shape)
        self.ndim: int = len(self.shape)

        indices_arr: np.ndarray = np.asarray(indices)
        if not np.issubdtype(indices_arr.dtype, np.integer):
            indices_arr = indices_arr.astype(np.int64)
        if indices_arr.ndim == 1:
            if self.ndim == 1:
                indices_arr = indices_arr.reshape(1, -1)
            else:
                indices_arr = indices_arr.reshape(self.ndim, -1)

        self.indices: np.ndarray = indices_arr
        self.values: np.ndarray = np.asarray(values)
        self.coords: np.ndarray = self.indices
        self.data: np.ndarray = self.values
        self.nnz: int = int(self.values.shape[0])

    @property
    def dtype(self) -> np.dtype:
        """Get the data type of the sparse tensor values.

        Returns:
            np.dtype: The underlying value dtype.
        """
        return self.values.dtype

    @property
    def size(self) -> int:
        """Get the total number of elements in the dense representation.

        Returns:
            int: The product of the dimension sizes.
        """
        prod: int = 1
        for s in self.shape:
            prod *= s
        return prod

    @classmethod
    def from_dense(cls, arr: np.ndarray) -> "COOTensor":
        """Construct a COOTensor from a dense NumPy array.

        Args:
            arr (np.ndarray): Input dense array.

        Returns:
            COOTensor: Sparse COO representation of non-zero elements.
        """
        arr_np: np.ndarray = np.asarray(arr)
        if arr_np.ndim == 0:
            arr_np_1d: np.ndarray = np.atleast_1d(arr_np)
            nonzero_mask_0d: np.ndarray = arr_np_1d != 0
            indices_0d: np.ndarray = np.empty((0, int(nonzero_mask_0d.sum())), dtype=np.int64)
            values_0d: np.ndarray = arr_np_1d[nonzero_mask_0d]
            return cls(indices=indices_0d, values=values_0d, shape=())

        nonzero_mask: np.ndarray = arr_np != 0
        indices: np.ndarray = np.vstack(np.nonzero(nonzero_mask))
        values: np.ndarray = arr_np[nonzero_mask]
        return cls(indices=indices, values=values, shape=arr_np.shape)

    def to_dense(self) -> np.ndarray:
        """Convert this sparse COO tensor into a dense NumPy array.

        Returns:
            np.ndarray: The dense tensor array.
        """
        if self.ndim == 0:
            val = self.values[0] if self.nnz > 0 else 0
            return np.array(val, dtype=self.dtype)

        dense: np.ndarray = np.zeros(self.shape, dtype=self.dtype)
        if self.nnz > 0:
            coords_tuple: tuple[np.ndarray, ...] = tuple(self.indices[i] for i in range(self.ndim))
            dense[coords_tuple] = self.values
        return dense

    def astype(self, dtype: Union[np.dtype, str]) -> "COOTensor":
        """Cast tensor values to a new data type.

        Args:
            dtype (Union[np.dtype, str]): Target data type.

        Returns:
            COOTensor: A new COOTensor with converted values.
        """
        return COOTensor(
            indices=self.indices.copy(),
            values=self.values.astype(dtype),
            shape=self.shape,
        )

    def to_csr(self) -> "CSRTensor":
        """Convert this COOTensor to a CSRTensor.

        Returns:
            CSRTensor: Equivalent CSR representation.
        """
        return CSRTensor.from_dense(self.to_dense())

    def to_csc(self) -> "CSCTensor":
        """Convert this COOTensor to a CSCTensor.

        Returns:
            CSCTensor: Equivalent CSC representation.
        """
        return CSCTensor.from_dense(self.to_dense())

    def __repr__(self) -> str:
        """Get string representation of the COOTensor.

        Returns:
            str: Representation string.
        """
        return f"<COOTensor shape={self.shape} nnz={self.nnz} dtype={self.dtype}>"


class CSRTensor:
    """Compressed Sparse Row (CSR) 2D matrix implementation."""

    def __init__(
        self,
        data: np.ndarray,
        indices: np.ndarray,
        indptr: np.ndarray,
        shape: tuple[int, int],
    ) -> None:
        """Initialize a CSRTensor.

        Args:
            data (np.ndarray): Non-zero element values of shape (nnz,).
            indices (np.ndarray): Column indices of shape (nnz,).
            indptr (np.ndarray): Row index pointers of shape (nrows + 1,).
            shape (tuple[int, int]): Matrix dimensions (nrows, ncols).
        """
        self.data: np.ndarray = np.asarray(data)
        self.indices: np.ndarray = np.asarray(indices, dtype=np.int64)
        self.indptr: np.ndarray = np.asarray(indptr, dtype=np.int64)
        self.shape: tuple[int, int] = (int(shape[0]), int(shape[1]))
        self.ndim: int = 2
        self.nnz: int = int(self.data.shape[0])

    @property
    def dtype(self) -> np.dtype:
        """Get the data type of the sparse tensor values.

        Returns:
            np.dtype: The underlying value dtype.
        """
        return self.data.dtype

    @classmethod
    def from_dense(cls, arr: np.ndarray) -> "CSRTensor":
        """Construct a CSRTensor from a dense 2D NumPy array.

        Args:
            arr (np.ndarray): Dense 2D array.

        Returns:
            CSRTensor: Compressed Sparse Row representation.
        """
        arr_2d = np.atleast_2d(arr)
        nrows, ncols = arr_2d.shape
        indptr = [0]
        indices = []
        data = []
        for r in range(nrows):
            for c in range(ncols):
                val = arr_2d[r, c]
                if val != 0:
                    indices.append(c)
                    data.append(val)
            indptr.append(len(data))
        return cls(
            data=np.array(data, dtype=arr_2d.dtype),
            indices=np.array(indices, dtype=np.int64),
            indptr=np.array(indptr, dtype=np.int64),
            shape=(nrows, ncols),
        )

    @classmethod
    def from_coo(cls, coo: COOTensor) -> "CSRTensor":
        """Construct a CSRTensor from a COOTensor.

        Args:
            coo (COOTensor): Source COO tensor.

        Returns:
            CSRTensor: Equivalent CSR representation.
        """
        return cls.from_dense(coo.to_dense())

    def to_dense(self) -> np.ndarray:
        """Convert this CSRTensor into a dense 2D NumPy array.

        Returns:
            np.ndarray: The dense matrix array.
        """
        nrows, ncols = self.shape
        dense = np.zeros((nrows, ncols), dtype=self.dtype)
        for r in range(nrows):
            start = self.indptr[r]
            end = self.indptr[r + 1]
            for idx in range(start, end):
                dense[r, self.indices[idx]] = self.data[idx]
        return dense

    def to_coo(self) -> COOTensor:
        """Convert this CSRTensor to a COOTensor.

        Returns:
            COOTensor: Equivalent COO representation.
        """
        nrows, _ = self.shape
        row_indices = []
        for r in range(nrows):
            count = self.indptr[r + 1] - self.indptr[r]
            row_indices.extend([r] * count)
        coords = np.vstack([np.array(row_indices, dtype=np.int64), self.indices]) if self.nnz > 0 else np.empty((2, 0), dtype=np.int64)
        return COOTensor(indices=coords, values=self.data, shape=self.shape)

    def to_csc(self) -> "CSCTensor":
        """Convert this CSRTensor to a CSCTensor.

        Returns:
            CSCTensor: Equivalent CSC representation.
        """
        return CSCTensor.from_dense(self.to_dense())

    def __repr__(self) -> str:
        """Get string representation of the CSRTensor.

        Returns:
            str: Representation string.
        """
        return f"<CSRTensor shape={self.shape} nnz={self.nnz} dtype={self.dtype}>"


class CSCTensor:
    """Compressed Sparse Column (CSC) 2D matrix implementation."""

    def __init__(
        self,
        data: np.ndarray,
        indices: np.ndarray,
        indptr: np.ndarray,
        shape: tuple[int, int],
    ) -> None:
        """Initialize a CSCTensor.

        Args:
            data (np.ndarray): Non-zero element values of shape (nnz,).
            indices (np.ndarray): Row indices of shape (nnz,).
            indptr (np.ndarray): Column index pointers of shape (ncols + 1,).
            shape (tuple[int, int]): Matrix dimensions (nrows, ncols).
        """
        self.data: np.ndarray = np.asarray(data)
        self.indices: np.ndarray = np.asarray(indices, dtype=np.int64)
        self.indptr: np.ndarray = np.asarray(indptr, dtype=np.int64)
        self.shape: tuple[int, int] = (int(shape[0]), int(shape[1]))
        self.ndim: int = 2
        self.nnz: int = int(self.data.shape[0])

    @property
    def dtype(self) -> np.dtype:
        """Get the data type of the sparse tensor values.

        Returns:
            np.dtype: The underlying value dtype.
        """
        return self.data.dtype

    @classmethod
    def from_dense(cls, arr: np.ndarray) -> "CSCTensor":
        """Construct a CSCTensor from a dense 2D NumPy array.

        Args:
            arr (np.ndarray): Dense 2D array.

        Returns:
            CSCTensor: Compressed Sparse Column representation.
        """
        arr_2d = np.atleast_2d(arr)
        nrows, ncols = arr_2d.shape
        indptr = [0]
        indices = []
        data = []
        for c in range(ncols):
            for r in range(nrows):
                val = arr_2d[r, c]
                if val != 0:
                    indices.append(r)
                    data.append(val)
            indptr.append(len(data))
        return cls(
            data=np.array(data, dtype=arr_2d.dtype),
            indices=np.array(indices, dtype=np.int64),
            indptr=np.array(indptr, dtype=np.int64),
            shape=(nrows, ncols),
        )

    @classmethod
    def from_coo(cls, coo: COOTensor) -> "CSCTensor":
        """Construct a CSCTensor from a COOTensor.

        Args:
            coo (COOTensor): Source COO tensor.

        Returns:
            CSCTensor: Equivalent CSC representation.
        """
        return cls.from_dense(coo.to_dense())

    def to_dense(self) -> np.ndarray:
        """Convert this CSCTensor into a dense 2D NumPy array.

        Returns:
            np.ndarray: The dense matrix array.
        """
        nrows, ncols = self.shape
        dense = np.zeros((nrows, ncols), dtype=self.dtype)
        for c in range(ncols):
            start = self.indptr[c]
            end = self.indptr[c + 1]
            for idx in range(start, end):
                dense[self.indices[idx], c] = self.data[idx]
        return dense

    def to_coo(self) -> COOTensor:
        """Convert this CSCTensor to a COOTensor.

        Returns:
            COOTensor: Equivalent COO representation.
        """
        _, ncols = self.shape
        col_indices = []
        for c in range(ncols):
            count = self.indptr[c + 1] - self.indptr[c]
            col_indices.extend([c] * count)
        coords = np.vstack([self.indices, np.array(col_indices, dtype=np.int64)]) if self.nnz > 0 else np.empty((2, 0), dtype=np.int64)
        return COOTensor(indices=coords, values=self.data, shape=self.shape)

    def to_csr(self) -> CSRTensor:
        """Convert this CSCTensor to a CSRTensor.

        Returns:
            CSRTensor: Equivalent CSR representation.
        """
        return CSRTensor.from_dense(self.to_dense())

    def __repr__(self) -> str:
        """Get string representation of the CSCTensor.

        Returns:
            str: Representation string.
        """
        return f"<CSCTensor shape={self.shape} nnz={self.nnz} dtype={self.dtype}>"


def zeros(cls: type, shape: tuple[int, ...]) -> COOTensor:
    """Create a Sparse COO tensor of all zeros.

    Args:
        cls (type): The backend class context.
        shape (tuple[int, ...]): The shape of the tensor.

    Returns:
        COOTensor: An empty sparse tensor with zero non-zero elements.
    """
    del cls
    ndim: int = len(shape)
    indices: np.ndarray = np.empty((ndim, 0), dtype=np.int64)
    values: np.ndarray = np.empty((0,), dtype=np.float32)
    return COOTensor(indices=indices, values=values, shape=shape)


def array(
    cls: type,
    data: Union[Sequence[float], np.ndarray, COOTensor],
    dtype: Optional[Union[np.dtype, str]] = None,
) -> COOTensor:
    """Create a Sparse COO tensor from input data.

    Args:
        cls (type): The backend class context.
        data (Union[Sequence[float], np.ndarray, COOTensor]): The input tensor data.
        dtype (Optional[Union[np.dtype, str]]): Target data type.

    Returns:
        COOTensor: The constructed Sparse COO tensor.
    """
    del cls
    if isinstance(data, COOTensor):
        if dtype is not None:
            return data.astype(dtype)
        return data

    dense: np.ndarray = np.asarray(data)
    if dtype is not None:
        dense = dense.astype(dtype)
    return COOTensor.from_dense(dense)


def asarray(cls: type, data: Union[Sequence[float], np.ndarray, COOTensor]) -> COOTensor:
    """Convert input data to a Sparse COO tensor.

    Args:
        cls (type): The backend class context.
        data (Union[Sequence[float], np.ndarray, COOTensor]): The input data.

    Returns:
        COOTensor: The Sparse COO tensor representation.
    """
    del cls
    if isinstance(data, COOTensor):
        return data
    return COOTensor.from_dense(np.asarray(data))


def item(cls: type, data: Union[COOTensor, np.ndarray]) -> float:
    """Extract a scalar item value from a single-element tensor.

    Args:
        cls (type): The backend class context.
        data (Union[COOTensor, np.ndarray]): The input tensor.

    Returns:
        float: The extracted scalar float value.
    """
    del cls
    if isinstance(data, COOTensor):
        dense = data.to_dense()
        return float(dense.item())
    return float(np.asarray(data).item())
