"""Sparse COO backend type definitions and array creation utilities."""

from __future__ import annotations

import typing
from collections.abc import Sequence
from types import ModuleType

from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class _SparseBackendProxy:
    """Lazy dense array backend proxy for sparse coordinate operations."""

    def __getattr__(self, name: str) -> EagerValue:
        """Resolve an attribute on the registered reference array backend.

        Args:
            name (str): Attribute name.

        Returns:
            EagerValue: Resolved attribute.
        """
        from ml_switcheroo_compiler.backends.registry import BackendRegistry

        return typing.cast(EagerValue, getattr(BackendRegistry.get("numpy").get_module(), name))


_SP_BACKEND: ModuleType = typing.cast(ModuleType, _SparseBackendProxy())


class COOTensor:
    """Dedicated multidimensional Sparse Coordinate (COO) Tensor implementation."""

    def __init__(
        self,
        indices: EagerValue,
        values: EagerValue,
        shape: tuple[int, ...],
    ) -> None:
        """Initialize a Sparse COO Tensor.

        Args:
            indices (_SP_BACKEND.ndarray): Integer coordinate array of shape (ndim, nnz).
            values (_SP_BACKEND.ndarray): Data array of shape (nnz,).
            shape (tuple[int, ...]): Shape of the full dense tensor.
        """
        self.shape: tuple[int, ...] = tuple(int(s) for s in shape)
        self.ndim: int = len(self.shape)

        indices_arr: EagerTensorProtocol = _SP_BACKEND.asarray(indices)
        if not _SP_BACKEND.issubdtype(indices_arr.dtype, _SP_BACKEND.integer):
            indices_arr = indices_arr.astype(_SP_BACKEND.int64)
        if indices_arr.ndim == 1:
            if self.ndim == 1:
                indices_arr = indices_arr.reshape(1, -1)
            else:
                indices_arr = indices_arr.reshape(self.ndim, -1)

        self.indices: EagerTensorProtocol = indices_arr
        self.values: EagerTensorProtocol = _SP_BACKEND.asarray(values)
        self.coords: EagerTensorProtocol = self.indices
        self.data: EagerTensorProtocol = self.values
        self.nnz: int = int(self.values.shape[0])

    @property
    def dtype(self) -> EagerValue:
        """Get the data type of the sparse tensor values.

        Returns:
            _SP_BACKEND.dtype: The underlying value dtype.
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
    def from_dense(cls, arr: EagerValue) -> COOTensor:
        """Construct a COOTensor from a dense NumPy array.

        Args:
            arr (_SP_BACKEND.ndarray): Input dense array.

        Returns:
            COOTensor: Sparse COO representation of non-zero elements.
        """
        arr_np: EagerTensorProtocol = _SP_BACKEND.asarray(arr)
        if arr_np.ndim == 0:
            arr_np_1d: EagerTensorProtocol = _SP_BACKEND.atleast_1d(arr_np)
            nonzero_mask_0d = typing.cast(EagerTensorProtocol, typing.cast(EagerValue, arr_np_1d) != 0)
            indices_0d: EagerTensorProtocol = _SP_BACKEND.empty((0, int(nonzero_mask_0d.sum())), dtype=_SP_BACKEND.int64)
            values_0d: EagerTensorProtocol = arr_np_1d[nonzero_mask_0d]
            return cls(indices=indices_0d, values=values_0d, shape=())

        nonzero_mask = typing.cast(EagerTensorProtocol, typing.cast(EagerValue, arr_np) != 0)
        indices: EagerTensorProtocol = _SP_BACKEND.vstack(_SP_BACKEND.nonzero(nonzero_mask))
        values: EagerTensorProtocol = arr_np[nonzero_mask]
        return cls(indices=indices, values=values, shape=arr_np.shape)

    def to_dense(self) -> EagerTensorProtocol:
        """Convert this sparse COO tensor into a dense NumPy array.

        Returns:
            _SP_BACKEND.ndarray: The dense tensor array.
        """
        if self.ndim == 0:
            val = self.values[0] if self.nnz > 0 else 0
            return typing.cast(EagerTensorProtocol, _SP_BACKEND.array(val, dtype=self.dtype))

        dense: EagerTensorProtocol = _SP_BACKEND.zeros(self.shape, dtype=self.dtype)
        if self.nnz > 0:
            coords_tuple: tuple[EagerValue, ...] = tuple(self.indices[i] for i in range(self.ndim))
            dense[coords_tuple] = self.values
        return dense

    def astype(self, dtype: EagerValue | str) -> COOTensor:
        """Cast tensor values to a new data type.

        Args:
            dtype (Union[_SP_BACKEND.dtype, str]): Target data type.

        Returns:
            COOTensor: A new COOTensor with converted values.
        """
        return COOTensor(
            indices=self.indices.copy(),
            values=self.values.astype(dtype),
            shape=self.shape,
        )

    def to_csr(self) -> CSRTensor:
        """Convert this COOTensor directly to a CSRTensor without dense materialization.

        Returns:
            CSRTensor: Equivalent CSR representation.
        """
        return coo_to_csr(self)

    def to_csc(self) -> CSCTensor:
        """Convert this COOTensor directly to a CSCTensor without dense materialization.

        Returns:
            CSCTensor: Equivalent CSC representation.
        """
        return coo_to_csc(self)

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
        data: EagerValue,
        indices: EagerValue,
        indptr: EagerValue,
        shape: tuple[int, int],
    ) -> None:
        """Initialize a CSRTensor.

        Args:
            data (_SP_BACKEND.ndarray): Non-zero element values of shape (nnz,).
            indices (_SP_BACKEND.ndarray): Column indices of shape (nnz,).
            indptr (_SP_BACKEND.ndarray): Row index pointers of shape (nrows + 1,).
            shape (tuple[int, int]): Matrix dimensions (nrows, ncols).
        """
        self.data: EagerTensorProtocol = _SP_BACKEND.asarray(data)
        self.indices: EagerTensorProtocol = _SP_BACKEND.asarray(indices, dtype=_SP_BACKEND.int64)
        self.indptr: EagerTensorProtocol = _SP_BACKEND.asarray(indptr, dtype=_SP_BACKEND.int64)
        self.shape: tuple[int, int] = (int(shape[0]), int(shape[1]))
        self.ndim: int = 2
        self.nnz: int = int(self.data.shape[0])

    @property
    def dtype(self) -> EagerValue:
        """Get the data type of the sparse tensor values.

        Returns:
            _SP_BACKEND.dtype: The underlying value dtype.
        """
        return self.data.dtype

    @classmethod
    def from_dense(cls, arr: EagerValue) -> CSRTensor:
        """Construct a CSRTensor from a dense 2D NumPy array.

        Args:
            arr (_SP_BACKEND.ndarray): Dense 2D array.

        Returns:
            CSRTensor: Compressed Sparse Row representation.
        """
        arr_2d = _SP_BACKEND.atleast_2d(arr)
        nrows, ncols = arr_2d.shape
        indptr = [0]
        indices = []
        data = []
        for r in range(nrows):
            for c in range(ncols):
                val: EagerValue = arr_2d[r, c]
                if val != 0:
                    indices.append(c)
                    data.append(val)
            indptr.append(len(data))
        return cls(
            data=_SP_BACKEND.array(data, dtype=arr_2d.dtype),
            indices=_SP_BACKEND.array(indices, dtype=_SP_BACKEND.int64),
            indptr=_SP_BACKEND.array(indptr, dtype=_SP_BACKEND.int64),
            shape=(nrows, ncols),
        )

    @classmethod
    def from_coo(cls, coo: COOTensor) -> CSRTensor:
        """Construct a CSRTensor directly from a COOTensor without dense materialization.

        Args:
            coo (COOTensor): Source COO tensor.

        Returns:
            CSRTensor: Equivalent CSR representation.
        """
        return coo_to_csr(coo)

    def to_dense(self) -> EagerTensorProtocol:
        """Convert this CSRTensor into a dense 2D NumPy array.

        Returns:
            _SP_BACKEND.ndarray: The dense matrix array.
        """
        nrows, ncols = self.shape
        dense = _SP_BACKEND.zeros((nrows, ncols), dtype=self.dtype)
        for r in range(nrows):
            start = int(self.indptr[r])
            end = int(self.indptr[r + 1])
            for idx in range(start, end):
                dense[r, self.indices[idx]] = self.data[idx]
        return typing.cast(EagerTensorProtocol, dense)

    def to_coo(self) -> COOTensor:
        """Convert this CSRTensor directly to a COOTensor without dense materialization.

        Returns:
            COOTensor: Equivalent COO representation.
        """
        return csr_to_coo(self)

    def to_csc(self) -> CSCTensor:
        """Convert this CSRTensor directly to a CSCTensor without dense materialization.

        Returns:
            CSCTensor: Equivalent CSC representation.
        """
        return csr_to_csc(self)

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
        data: EagerValue,
        indices: EagerValue,
        indptr: EagerValue,
        shape: tuple[int, int],
    ) -> None:
        """Initialize a CSCTensor.

        Args:
            data (_SP_BACKEND.ndarray): Non-zero element values of shape (nnz,).
            indices (_SP_BACKEND.ndarray): Row indices of shape (nnz,).
            indptr (_SP_BACKEND.ndarray): Column index pointers of shape (ncols + 1,).
            shape (tuple[int, int]): Matrix dimensions (nrows, ncols).
        """
        self.data: EagerTensorProtocol = _SP_BACKEND.asarray(data)
        self.indices: EagerTensorProtocol = _SP_BACKEND.asarray(indices, dtype=_SP_BACKEND.int64)
        self.indptr: EagerTensorProtocol = _SP_BACKEND.asarray(indptr, dtype=_SP_BACKEND.int64)
        self.shape: tuple[int, int] = (int(shape[0]), int(shape[1]))
        self.ndim: int = 2
        self.nnz: int = int(self.data.shape[0])

    @property
    def dtype(self) -> EagerValue:
        """Get the data type of the sparse tensor values.

        Returns:
            _SP_BACKEND.dtype: The underlying value dtype.
        """
        return self.data.dtype

    @classmethod
    def from_dense(cls, arr: EagerValue) -> CSCTensor:
        """Construct a CSCTensor from a dense 2D NumPy array.

        Args:
            arr (_SP_BACKEND.ndarray): Dense 2D array.

        Returns:
            CSCTensor: Compressed Sparse Column representation.
        """
        arr_2d = _SP_BACKEND.atleast_2d(arr)
        nrows, ncols = arr_2d.shape
        indptr = [0]
        indices = []
        data = []
        for c in range(ncols):
            for r in range(nrows):
                val: EagerValue = arr_2d[r, c]
                if val != 0:
                    indices.append(r)
                    data.append(val)
            indptr.append(len(data))
        return cls(
            data=_SP_BACKEND.array(data, dtype=arr_2d.dtype),
            indices=_SP_BACKEND.array(indices, dtype=_SP_BACKEND.int64),
            indptr=_SP_BACKEND.array(indptr, dtype=_SP_BACKEND.int64),
            shape=(nrows, ncols),
        )

    @classmethod
    def from_coo(cls, coo: COOTensor) -> CSCTensor:
        """Construct a CSCTensor directly from a COOTensor without dense materialization.

        Args:
            coo (COOTensor): Source COO tensor.

        Returns:
            CSCTensor: Equivalent CSC representation.
        """
        return coo_to_csc(coo)

    def to_dense(self) -> EagerTensorProtocol:
        """Convert this CSCTensor into a dense 2D NumPy array.

        Returns:
            _SP_BACKEND.ndarray: The dense matrix array.
        """
        nrows, ncols = self.shape
        dense = _SP_BACKEND.zeros((nrows, ncols), dtype=self.dtype)
        for c in range(ncols):
            start = int(self.indptr[c])
            end = int(self.indptr[c + 1])
            for idx in range(start, end):
                dense[self.indices[idx], c] = self.data[idx]
        return typing.cast(EagerTensorProtocol, dense)

    def to_coo(self) -> COOTensor:
        """Convert this CSCTensor directly to a COOTensor without dense materialization.

        Returns:
            COOTensor: Equivalent COO representation.
        """
        return csc_to_coo(self)

    def to_csr(self) -> CSRTensor:
        """Convert this CSCTensor directly to a CSRTensor without dense materialization.

        Returns:
            CSRTensor: Equivalent CSR representation.
        """
        return csc_to_csr(self)

    def __repr__(self) -> str:
        """Get string representation of the CSCTensor.

        Returns:
            str: Representation string.
        """
        return f"<CSCTensor shape={self.shape} nnz={self.nnz} dtype={self.dtype}>"


def coo_to_csr(coo: COOTensor) -> CSRTensor:
    """Convert a 2D COOTensor directly to a CSRTensor without dense materialization.

    Args:
        coo (COOTensor): Source sparse COO tensor.

    Returns:
        CSRTensor: Equivalent CSR representation.

    Raises:
        ValueError: If coo is not a 2D tensor.
    """
    if coo.ndim != 2:
        msg = f"CSR conversion requires 2D tensor, got shape {coo.shape}"
        raise ValueError(msg)
    nrows, ncols = coo.shape
    if coo.nnz == 0:
        return CSRTensor(
            data=_SP_BACKEND.empty((0,), dtype=coo.dtype),
            indices=_SP_BACKEND.empty((0,), dtype=_SP_BACKEND.int64),
            indptr=_SP_BACKEND.zeros(nrows + 1, dtype=_SP_BACKEND.int64),
            shape=(nrows, ncols),
        )
    rows = coo.indices[0]
    cols = coo.indices[1]
    order = _SP_BACKEND.lexsort((cols, rows))
    sorted_rows = rows[order]
    sorted_cols = cols[order]
    sorted_data = coo.values[order]

    counts = _SP_BACKEND.bincount(sorted_rows, minlength=nrows)
    indptr = _SP_BACKEND.zeros(nrows + 1, dtype=_SP_BACKEND.int64)
    indptr[1:] = _SP_BACKEND.cumsum(counts)

    return CSRTensor(
        data=sorted_data,
        indices=sorted_cols,
        indptr=indptr,
        shape=(nrows, ncols),
    )


def coo_to_csc(coo: COOTensor) -> CSCTensor:
    """Convert a 2D COOTensor directly to a CSCTensor without dense materialization.

    Args:
        coo (COOTensor): Source sparse COO tensor.

    Returns:
        CSCTensor: Equivalent CSC representation.

    Raises:
        ValueError: If coo is not a 2D tensor.
    """
    if coo.ndim != 2:
        msg = f"CSC conversion requires 2D tensor, got shape {coo.shape}"
        raise ValueError(msg)
    nrows, ncols = coo.shape
    if coo.nnz == 0:
        return CSCTensor(
            data=_SP_BACKEND.empty((0,), dtype=coo.dtype),
            indices=_SP_BACKEND.empty((0,), dtype=_SP_BACKEND.int64),
            indptr=_SP_BACKEND.zeros(ncols + 1, dtype=_SP_BACKEND.int64),
            shape=(nrows, ncols),
        )
    rows = coo.indices[0]
    cols = coo.indices[1]
    order = _SP_BACKEND.lexsort((rows, cols))
    sorted_rows = rows[order]
    sorted_cols = cols[order]
    sorted_data = coo.values[order]

    counts = _SP_BACKEND.bincount(sorted_cols, minlength=ncols)
    indptr = _SP_BACKEND.zeros(ncols + 1, dtype=_SP_BACKEND.int64)
    indptr[1:] = _SP_BACKEND.cumsum(counts)

    return CSCTensor(
        data=sorted_data,
        indices=sorted_rows,
        indptr=indptr,
        shape=(nrows, ncols),
    )


def csr_to_coo(csr: CSRTensor) -> COOTensor:
    """Convert a CSRTensor directly to a COOTensor without dense materialization.

    Args:
        csr (CSRTensor): Source sparse CSR tensor.

    Returns:
        COOTensor: Equivalent COO representation.
    """
    nrows, _ = csr.shape
    diffs = _SP_BACKEND.diff(csr.indptr)
    row_indices = _SP_BACKEND.repeat(_SP_BACKEND.arange(nrows, dtype=_SP_BACKEND.int64), diffs) if csr.nnz > 0 else _SP_BACKEND.empty((0,), dtype=_SP_BACKEND.int64)
    coords = _SP_BACKEND.vstack([row_indices, csr.indices]) if csr.nnz > 0 else _SP_BACKEND.empty((2, 0), dtype=_SP_BACKEND.int64)
    return COOTensor(indices=coords, values=csr.data.copy(), shape=csr.shape)


def csr_to_csc(csr: CSRTensor) -> CSCTensor:
    """Convert a CSRTensor directly to a CSCTensor without dense materialization.

    Args:
        csr (CSRTensor): Source sparse CSR tensor.

    Returns:
        CSCTensor: Equivalent CSC representation.
    """
    return coo_to_csc(csr_to_coo(csr))


def csc_to_coo(csc: CSCTensor) -> COOTensor:
    """Convert a CSCTensor directly to a COOTensor without dense materialization.

    Args:
        csc (CSCTensor): Source sparse CSC tensor.

    Returns:
        COOTensor: Equivalent COO representation.
    """
    _, ncols = csc.shape
    diffs = _SP_BACKEND.diff(csc.indptr)
    col_indices = _SP_BACKEND.repeat(_SP_BACKEND.arange(ncols, dtype=_SP_BACKEND.int64), diffs) if csc.nnz > 0 else _SP_BACKEND.empty((0,), dtype=_SP_BACKEND.int64)
    coords = _SP_BACKEND.vstack([csc.indices, col_indices]) if csc.nnz > 0 else _SP_BACKEND.empty((2, 0), dtype=_SP_BACKEND.int64)
    return COOTensor(indices=coords, values=csc.data.copy(), shape=csc.shape)


def csc_to_csr(csc: CSCTensor) -> CSRTensor:
    """Convert a CSCTensor directly to a CSRTensor without dense materialization.

    Args:
        csc (CSCTensor): Source sparse CSC tensor.

    Returns:
        CSRTensor: Equivalent CSR representation.
    """
    return coo_to_csr(csc_to_coo(csc))


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
    indices: EagerTensorProtocol = _SP_BACKEND.empty((ndim, 0), dtype=_SP_BACKEND.int64)
    values: EagerTensorProtocol = _SP_BACKEND.empty((0,), dtype=_SP_BACKEND.float32)
    return COOTensor(indices=indices, values=values, shape=shape)


def array(
    cls: type,
    data: Sequence[float] | EagerValue | COOTensor,
    dtype: EagerValue | str | None = None,
) -> COOTensor:
    """Create a Sparse COO tensor from input data.

    Args:
        cls (type): The backend class context.
        data (Union[Sequence[float], _SP_BACKEND.ndarray, COOTensor]): The input tensor data.
        dtype (Optional[Union[_SP_BACKEND.dtype, str]]): Target data type.

    Returns:
        COOTensor: The constructed Sparse COO tensor.
    """
    del cls
    if isinstance(data, COOTensor):
        if dtype is not None:
            return data.astype(dtype)
        return data

    dense: EagerTensorProtocol = _SP_BACKEND.asarray(data)
    if dtype is not None:
        dense = dense.astype(dtype)
    return COOTensor.from_dense(dense)


def asarray(cls: type, data: Sequence[float] | EagerValue | COOTensor) -> COOTensor:
    """Convert input data to a Sparse COO tensor.

    Args:
        cls (type): The backend class context.
        data (Union[Sequence[float], _SP_BACKEND.ndarray, COOTensor]): The input data.

    Returns:
        COOTensor: The Sparse COO tensor representation.
    """
    del cls
    if isinstance(data, COOTensor):
        return data
    return COOTensor.from_dense(_SP_BACKEND.asarray(data))


def item(cls: type, data: COOTensor | EagerValue) -> float:
    """Extract a scalar item value from a single-element tensor.

    Args:
        cls (type): The backend class context.
        data (Union[COOTensor, _SP_BACKEND.ndarray]): The input tensor.

    Returns:
        float: The extracted scalar float value.
    """
    del cls
    if isinstance(data, COOTensor):
        dense = data.to_dense()
        return float(typing.cast(float, dense.item()))
    return float(_SP_BACKEND.asarray(data).item())
