"""Mixins for Tensor."""

from __future__ import annotations

import typing
import uuid
from collections.abc import Callable, Iterator, Sequence

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.registry import (
    ShapeProtocol,
)
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import (
    BackendBufferProtocol as BackendBufferProtocol,
)
from ml_switcheroo_compiler.core.shape import (
    OpValue,
    ShapeDim,
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


@typing.runtime_checkable
class _MixinSubscriptableProtocol(typing.Protocol):
    """Protocol for backend arrays supporting element and slice lookup."""

    def __getitem__(self, key: OpValue) -> OpValue:
        """Retrieve element or slice at ``key``.

        Args:
            key (OpValue): Index or slice specification.

        Returns:
            OpValue: Indexed value or subarray.
        """
        ...


@typing.runtime_checkable
class _MixinMutableSubscriptProtocol(typing.Protocol):
    """Protocol for backend arrays supporting in-place item assignment."""

    def __setitem__(self, key: OpValue, val: OpValue) -> None:
        """Assign ``val`` into ``key``.

        Args:
            key (OpValue): Index or slice specification.
            val (OpValue): Replacement value.
        """
        ...


def _as_mixin_subscriptable(val: OpValue | _MixinSubscriptableProtocol) -> _MixinSubscriptableProtocol:
    """Verify and narrow ``val`` to ``_MixinSubscriptableProtocol``.

    Args:
        val (OpValue | _MixinSubscriptableProtocol): Candidate array value.

    Returns:
        _MixinSubscriptableProtocol: Verified subscriptable array.
    """
    assert isinstance(val, _MixinSubscriptableProtocol)
    return val


def _as_mixin_mutable_subscriptable(
    val: OpValue | _MixinMutableSubscriptProtocol,
) -> _MixinMutableSubscriptProtocol:
    """Verify and narrow ``val`` to ``_MixinMutableSubscriptProtocol``.

    Args:
        val (OpValue | _MixinMutableSubscriptProtocol): Candidate mutable array value.

    Returns:
        _MixinMutableSubscriptProtocol: Verified mutable array.
    """
    assert isinstance(val, _MixinMutableSubscriptProtocol)
    return val


@typing.runtime_checkable
class _DataWrapperProtocol(typing.Protocol):
    """Protocol for objects wrapping a `.data` payload."""

    @property
    def data(self) -> OpValue:
        """Return underlying data payload.

        Returns:
            OpValue: Underlying data payload.
        """
        ...


class TensorPropertiesMixin:
    """Tensor properties mixin."""

    _shape: tuple[int | str | SymNode, ...]
    _dtype: DType
    _device: Device
    _requires_grad: bool
    _data: OpValue

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
            if not isinstance(s, int):
                return None
            prod *= s
        return prod

    @property
    def shape(self) -> tuple[int | str | SymNode, ...]:
        """Get the shape tuple of the tensor (including symbolic dimensions when present).

        Returns:
            tuple[int | str | SymNode, ...]: The shape tuple of the tensor.
        """
        return self._shape

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
    def data(self) -> OpValue:
        """Get the underlying data payload.

        Returns:
            OpValue: The underlying array or proxy payload.
        """
        return self._data


class TensorConversionMixin(TensorPropertiesMixin):
    """Tensor conversion mixin."""

    if typing.TYPE_CHECKING:
        eval: Callable[[], Tensor]

    def numpy(self) -> OpValue:
        """Convert the tensor to a host array representation.

        Returns:
            OpValue: The host array representation.
        """
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()
        try:
            return backend.numpy(self._data)
        except (
            AttributeError,
            ImportError,
            LookupError,
            OSError,
            ArithmeticError,
            RuntimeError,
            SyntaxError,
            TypeError,
            ValueError,
        ):
            try:
                return backend.asarray(self._data)
            except AttributeError:
                try:
                    return backend.array(self._data)
                except AttributeError:
                    return self._data

    def to_backend_array(self, dtype: DType | str | type | None = None) -> OpValue:
        """Convert the tensor to a backend array with optional dtype cast.

        Args:
            dtype (DType | str | type | None): Optional target data type.

        Returns:
            OpValue: The converted array.
        """
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()

        if is_traced_data(self.data):
            data = backend.zeros(tuple(int(d) if isinstance(d, int) else 1 for d in self.shape))
        else:
            data = self.data

        try:
            return backend.array(data, dtype=dtype) if dtype is not None else backend.asarray(data)
        except (
            AttributeError,
            ImportError,
            LookupError,
            OSError,
            ArithmeticError,
            RuntimeError,
            SyntaxError,
            TypeError,
            ValueError,
        ):
            return backend.array(data.tolist() if is_backend_buffer(data) else data, dtype=dtype)

    def __array__(self, dtype: DType | str | type | None = None) -> OpValue:
        """Convert the tensor to a backend array with optional dtype cast.

        Args:
            dtype (DType | str | type | None): Optional target data type.

        Returns:
            OpValue: The converted array.
        """
        return self.to_backend_array(dtype=dtype)

    def item(self) -> float:
        """Return the value of this tensor as a standard Python number.

        Returns:
            float: The scalar numerical value.
        """
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()

        eval_val = self.eval()
        target_val: OpValue = eval_val.data if type(eval_val).__name__ == "Tensor" else eval_val
        try:
            return backend.item(target_val)
        except AttributeError:
            from ml_switcheroo_compiler.core._shape_protocols import _ItemMemberProtocol

            if isinstance(target_val, _ItemMemberProtocol):
                return float(target_val.item())
            return float(str(target_val))

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
        return int(str(self.shape[0])) if self.shape else 0

    def __iter__(self) -> Iterator[Tensor]:
        """Iterate over the first dimension of the tensor.

        Yields:
            Tensor: A slice of the tensor along the first dimension.

        Raises:
            TypeError: If the tensor is 0-dimensional.
        """
        arr = self.to_backend_array()
        shape: Sequence[ShapeDim] = arr.shape if isinstance(arr, ShapeProtocol) else [0]
        if not shape:
            raise TypeError("iteration over a 0-d tensor")
        for i in range(int(shape[0]) if isinstance(shape[0], int) else 0):
            from .tensor import Tensor, TensorConfig

            item_arr: OpValue = _as_mixin_subscriptable(arr)[i]
            item_shape = tuple(item_arr.shape) if isinstance(item_arr, ShapeProtocol) else ()
            yield Tensor(item_arr, TensorConfig(item_shape, self.dtype, self.device))


_IndexedSelfT = typing.TypeVar("_IndexedSelfT", bound="TensorIndexingMixin")


class TensorIndexingMixin(TensorConversionMixin):
    """Tensor indexing mixin."""

    def __getitem__(self, key: OpValue) -> Tensor:
        """Retrieve elements from the tensor.

        Args:
            key (OpValue): The index or slice key.

        Returns:
            Tensor: A new tensor containing the selected elements.

        Raises:
            IndexError: If the index is out of bounds or invalid.
            RuntimeError: If tracing but not in an active tracing context.
        """
        arr = self.to_backend_array()
        raw_key: OpValue = key
        if isinstance(raw_key, _DataWrapperProtocol):
            key = raw_key.data
        elif isinstance(key, tuple):
            key = tuple(k.data if isinstance(k, _DataWrapperProtocol) else k for k in key)

        res: OpValue = None
        try:
            raw_res = _as_mixin_subscriptable(arr)[tuple(key) if isinstance(key, list) else key]
            res = raw_res
            res_shape = tuple(raw_res.shape) if isinstance(raw_res, ShapeProtocol) else ()
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

    def __setitem__(self, key: OpValue, value: OpValue) -> None:
        """Set elements in the tensor (only supported in eager mode).

        Args:
            key (OpValue): The index or slice key.
            value (OpValue): The value to set.

        Raises:
            TypeError: If attempted during tracing.
        """
        if config.eager_mode:
            val = value.data if isinstance(value, _DataWrapperProtocol) else value
            _as_mixin_mutable_subscriptable(self.data)[key] = val
        else:
            msg = "Tensor object does not support item assignment in tracing mode. Use .at[...].set(...) instead."
            raise TypeError(msg)

    @property
    def at(self: _IndexedSelfT) -> ArrayAtIndexer[_IndexedSelfT]:
        """Get ArrayAtIndexer for the tensor.

        Returns:
            ArrayAtIndexer[_IndexedSelfT]: The indexer.
        """
        from .tensor import ArrayAtIndexer

        return ArrayAtIndexer(self)
