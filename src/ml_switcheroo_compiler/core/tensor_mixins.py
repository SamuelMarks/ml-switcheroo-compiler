"""Mixins for Tensor."""

from __future__ import annotations

import inspect
import types
import typing
import uuid
from collections.abc import Callable, Iterator, Sequence

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.registry import (
    EagerTensorProtocol,
    ShapeProtocol,
)
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.mixins import _as_tensor_self
from ml_switcheroo_compiler.core.shape import (
    BackendBufferProtocol as BackendBufferProtocol,
)
from ml_switcheroo_compiler.core.shape import (
    TensorOrScalar,
    is_backend_buffer,
    is_traced_data,
)
from ml_switcheroo_compiler.core.shape import (
    TracedDataProtocol as TracedDataProtocol,
)
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.ir.shape_system import SymNode

    from .tensor import ArrayAtIndexer, Tensor


def _is_mixin_eager_tensor(val: TensorOrScalar) -> typing.TypeGuard[EagerTensorProtocol]:
    """Type guard narrowing `val` to `EagerTensorProtocol`.

    Args:
        val (TensorOrScalar): Candidate array value.

    Returns:
        bool: True if non-None or None.
    """
    return not isinstance(val, type(Ellipsis))


def _ensure_mixin_tensor(val: TensorOrScalar) -> EagerTensorProtocol:
    """Narrow `val` to `EagerTensorProtocol`.

    Args:
        val (TensorOrScalar): Candidate array value.

    Returns:
        EagerTensorProtocol: Narrowed tensor.
    """
    assert _is_mixin_eager_tensor(val)
    return val


def _is_mixin_shape_tuple(seq: tuple[int | str | SymNode | None, ...]) -> typing.TypeGuard[tuple[int, ...]]:
    """Type guard narrowing a shape tuple to `tuple[int, ...]`.

    Args:
        seq (tuple[int | str | SymNode | None, ...]): Shape tuple.

    Returns:
        bool: True if seq is a tuple.
    """
    return isinstance(seq, tuple)


def _is_mixin_int(val: int | str | SymNode) -> typing.TypeGuard[int]:
    """Type guard narrowing a non-string dimension to `int`.

    Args:
        val (int | str | SymNode): Dimension value.

    Returns:
        bool: True if not a string.
    """
    return not isinstance(val, str)


def _extract_mixin_shape(seq: Sequence[int | str | SymNode | None]) -> tuple[int, ...]:
    """Convert a sequence of dimensions into a shape tuple.

    Args:
        seq (Sequence[int | str | SymNode | None]): Shape sequence.

    Returns:
        tuple[int, ...]: Shape tuple.
    """
    res = tuple(seq)
    assert _is_mixin_shape_tuple(res)
    return res


@typing.runtime_checkable
class _DataWrapperProtocol(typing.Protocol):
    """Protocol for objects wrapping a `.data` payload."""

    @property
    def data(self) -> TensorOrScalar:
        """Return underlying data payload.

        Returns:
            TensorOrScalar: Underlying data payload.
        """
        ...


class TensorPropertiesMixin:
    """Tensor properties mixin."""

    _shape: tuple[int | str | SymNode, ...]
    _dtype: DType
    _device: Device
    _requires_grad: bool
    _data: TensorOrScalar

    @property
    def ndim(self) -> int:
        """Get the number of dimensions of the tensor.

        Returns:
            int: The number of dimensions.
        """
        return len(self._shape)

    @property
    def size(self) -> int | None:
        """Get the number of elements in the tensor.

        Returns:
            int | None: The size of the tensor, or None if dimensions are symbolic.
        """
        # if there are strings in shape (unknown dims), return a ProxyTensor?
        # for eager evaluation, size should evaluate natively.
        prod = 1
        for s in self._shape:
            if isinstance(s, str):
                return None
            assert _is_mixin_int(s)
            prod *= s
        return prod

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the tensor.

        Returns:
            tuple[int, ...]: The shape tuple of the tensor.
        """
        return _extract_mixin_shape(self._shape)

    @property
    def dtype(self) -> DType:
        """Get the data type of the tensor.

        Returns:
            DType: The data type associated with the tensor.
        """
        return self._dtype

    @property
    def device(self) -> Device:
        """Get the device of the tensor.

        Returns:
            Device: The device associated with the tensor.
        """
        return self._device

    @property
    def requires_grad(self) -> bool:
        """Check if the tensor requires gradient computation.

        Returns:
            bool: A boolean indicating the result of the check.
        """
        return self._requires_grad

    @property
    def data(self) -> TensorOrScalar:
        """Get the underlying data payload.

        Returns:
            TensorOrScalar: The underlying array or proxy payload.
        """
        return self._data


class TensorConversionMixin(TensorPropertiesMixin):
    """Tensor conversion mixin."""

    if typing.TYPE_CHECKING:
        eval: Callable[[], Tensor]

    def numpy(self) -> TensorOrScalar:
        """Convert the tensor to a host array representation.

        Returns:
            TensorOrScalar: The host array representation.
        """
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        try:
            return get_active_backend().numpy(self._data)
        except Exception:
            return get_active_backend().asarray(self._data)

    def to_backend_array(self, dtype: DType | str | type | None = None) -> TensorOrScalar:
        """Convert the tensor to a backend array with optional dtype cast.

        Args:
            dtype (DType | str | type | None): Optional target data type.

        Returns:
            TensorOrScalar: The converted array.
        """
        for m_name, m_val in inspect.getmembers(self):
            if m_name.endswith("array__") and isinstance(m_val, types.FunctionType):
                res_hook: TensorOrScalar = m_val()
                return res_hook
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()

        if is_traced_data(self.data):
            data = backend.zeros(self.shape)
        else:
            data = self.data

        try:
            return backend.array(data, dtype=dtype) if dtype is not None else backend.asarray(data)
        except Exception:
            return backend.array(data.tolist() if is_backend_buffer(data) else data, dtype=dtype)

    def __array__(self, dtype: DType | str | type | None = None) -> TensorOrScalar:
        """Convert the tensor to a backend array with optional dtype cast.

        Args:
            dtype (DType | str | type | None): Optional target data type.

        Returns:
            TensorOrScalar: The converted array.
        """
        return self.to_backend_array(dtype=dtype)

    def item(self) -> float:
        """Return the value of this tensor as a standard Python number.

        Returns:
            float: The scalar numerical value.
        """
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()

        if type(self.eval()).__name__ == "Tensor":
            return backend.item(self.eval().data)
        return backend.item(self.eval())

    def __int__(self) -> int:
        """Convert the tensor scalar to an int.

        Returns:
            int: The integer value.
        """
        return int(self.item())

    def __index__(self) -> int:
        """Return the integer value of a scalar tensor.

        Returns:
            int: The scalar integer value.
        """
        return int(self.item())

    def __float__(self) -> float:
        """Convert the tensor scalar to a float.

        Returns:
            float: The float value.
        """
        return float(self.item())

    def __bool__(self) -> bool:
        """Convert the tensor scalar to a boolean.

        Returns:
            bool: The boolean value.

        Raises:
            ValueError: If the tensor has more than one element.
        """
        arr = self.to_backend_array()
        size = arr.size if is_backend_buffer(arr) else 1
        if size == 1:
            return bool(arr.item() if is_backend_buffer(arr) else arr)
        msg = "The truth value of an array with more than one element is ambiguous."
        raise ValueError(
            msg,
        )

    def __len__(self) -> int:
        """Return the size of the leading dimension.

        Returns:
            int: Leading dimension length, or 0 for scalar tensors.
        """
        return int(self.shape[0]) if self.shape else 0

    def __iter__(self) -> Iterator[Tensor]:
        """Iterate over the first dimension of the tensor.

        Yields:
            Tensor: A slice of the tensor along the first dimension.

        Raises:
            TypeError: If the tensor is 0-dimensional.
        """
        arr = self.to_backend_array()
        shape: Sequence[int | str] = arr.shape if isinstance(arr, ShapeProtocol) else [0]
        if not shape:
            raise TypeError("iteration over a 0-d tensor")
        for i in range(int(shape[0])):
            from .tensor import Tensor, TensorConfig

            item_arr: TensorOrScalar = _ensure_mixin_tensor(arr)[i]
            item_shape = _extract_mixin_shape(item_arr.shape) if isinstance(item_arr, ShapeProtocol) else ()
            yield Tensor(item_arr, TensorConfig(item_shape, self.dtype, self.device))


class TensorIndexingMixin(TensorConversionMixin):
    """Tensor indexing mixin."""

    def __getitem__(self, key: TensorOrScalar) -> Tensor:
        """Retrieve elements from the tensor.

        Args:
            key (TensorOrScalar): The index or slice key.

        Returns:
            Tensor: A new tensor containing the selected elements.

        Raises:
            IndexError: If the index is out of bounds or invalid.
            RuntimeError: If tracing but not in an active tracing context.
        """
        arr = self.to_backend_array()
        raw_key: TensorOrScalar = key
        if isinstance(raw_key, _DataWrapperProtocol):
            key = raw_key.data
        elif isinstance(key, tuple):
            key = tuple(k.data if isinstance(k, _DataWrapperProtocol) else k for k in key)

        res: TensorOrScalar = None
        try:
            raw_res = _ensure_mixin_tensor(arr)[tuple(key) if isinstance(key, list) else key]
            res = raw_res
            res_shape = _extract_mixin_shape(raw_res.shape)
        except IndexError as e:
            if config.eager_mode:
                raise
            res_shape = ()
            if "too many indices for array" in str(e):
                raise

        if config.eager_mode:
            from .tensor import Tensor, TensorConfig

            return Tensor(res, TensorConfig(res_shape, self.dtype, self.device))

        nid = f"getitem_{uuid.uuid4().hex[:6]}"
        input_id = self.data.id if is_traced_data(self.data) else "const"

        node = LogicalNode(
            id=nid,
            op_type="GetItem",
            inputs=[input_id],
            attributes={"key": str(key)},
            shape_metadata=(),
        )
        if global_tracing_state.is_tracing:
            global_tracing_state.add_node(node)
        else:
            raise RuntimeError("Cannot add node: not currently tracing.")

        from .tensor import Tensor, TensorConfig

        return Tensor(ProxyTensor(nid, (), self.dtype.value), TensorConfig((), self.dtype, self.device))

    def __setitem__(self, key: TensorOrScalar, value: TensorOrScalar) -> None:
        """Set elements in the tensor (only supported in eager mode).

        Args:
            key (TensorOrScalar): The index or slice key.
            value (TensorOrScalar): The value to set.

        Raises:
            TypeError: If attempted during tracing.
        """
        if config.eager_mode:
            val = value.data if isinstance(value, _DataWrapperProtocol) else value
            _ensure_mixin_tensor(self.data)[key] = val
        else:
            msg = "Tensor object does not support item assignment in tracing mode. Use .at[...].set(...) instead."
            raise TypeError(msg)

    @property
    def at(self) -> ArrayAtIndexer:
        """Get ArrayAtIndexer for the tensor.

        Returns:
            ArrayAtIndexer: The indexer.
        """
        from .tensor import ArrayAtIndexer

        return ArrayAtIndexer(_as_tensor_self(self))
