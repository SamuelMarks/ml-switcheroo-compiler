"""Define the unified backend array base class for ml-switcheroo."""

from __future__ import annotations

import typing
from collections.abc import Iterable, Sequence

import pydantic

from ml_switcheroo_compiler.core.shape import OpValue

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.ir.shape_system import SymNode

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device as Device
from ml_switcheroo_compiler.core.device import DeviceType as DeviceType
from ml_switcheroo_compiler.core.dtype import DType as DType
from ml_switcheroo_compiler.core.mixins import (
    TensorArithmeticMixin,
    TensorBitwiseMixin,
    TensorLogicalMixin,
)
from ml_switcheroo_compiler.core.shape import (
    IRAttrValue,
    ShapeLike,
    extract_device,
    extract_dtype,
    extract_shape,
    is_traced_data,
)
from ml_switcheroo_compiler.tracing.state import global_tracing_state, register_live_tensor

from .tensor_mixins import (
    TensorIndexingMixin,
    _DataWrapperProtocol,
)


@typing.runtime_checkable
class _IndexedBufferElementProtocol(typing.Protocol):
    """Protocol for scalar/subarray elements supporting arithmetic and ordering in ``ArrayAt``."""

    def __add__(self, other: OpValue) -> OpValue:
        """Add ``other`` to ``self``.

        Args:
            other (OpValue): Right-hand operand.

        Returns:
            OpValue: Sum value.
        """
        ...

    def __mul__(self, other: OpValue) -> OpValue:
        """Multiply ``self`` by ``other``.

        Args:
            other (OpValue): Right-hand operand.

        Returns:
            OpValue: Product value.
        """
        ...

    def __ge__(self, other: OpValue) -> bool:
        """Compare ``self >= other``.

        Args:
            other (OpValue): Right-hand operand.

        Returns:
            bool: Comparison result.
        """
        ...

    def __le__(self, other: OpValue) -> bool:
        """Compare ``self <= other``.

        Args:
            other (OpValue): Right-hand operand.

        Returns:
            bool: Comparison result.
        """
        ...


@typing.runtime_checkable
class _MutableIndexedBufferProtocol(typing.Protocol):
    """Protocol for copied buffers supporting indexed read and write in ``ArrayAt``."""

    def __getitem__(self, key: OpValue) -> OpValue:
        """Read element or slice at ``key``.

        Args:
            key (OpValue): Index specification.

        Returns:
            OpValue: Current element or slice.
        """
        ...

    def __setitem__(self, key: OpValue, val: OpValue) -> None:
        """Write ``val`` at ``key``.

        Args:
            key (OpValue): Index specification.
            val (OpValue): Replacement value.
        """
        ...


def _as_mutable_indexed_buffer(val: OpValue | _MutableIndexedBufferProtocol) -> _MutableIndexedBufferProtocol:
    """Verify and narrow ``val`` to ``_MutableIndexedBufferProtocol``.

    Args:
        val (OpValue | _MutableIndexedBufferProtocol): Candidate copied buffer.

    Returns:
        _MutableIndexedBufferProtocol: Verified mutable buffer.
    """
    assert isinstance(val, _MutableIndexedBufferProtocol)
    return val


def _as_indexed_element(val: OpValue | _IndexedBufferElementProtocol) -> _IndexedBufferElementProtocol:
    """Verify and narrow ``val`` to ``_IndexedBufferElementProtocol``.

    Args:
        val (OpValue | _IndexedBufferElementProtocol): Candidate element or slice.

    Returns:
        _IndexedBufferElementProtocol: Verified element supporting arithmetic/comparison.
    """
    assert isinstance(val, _IndexedBufferElementProtocol)
    return val


@typing.runtime_checkable
class _CopyableArrayProtocol(typing.Protocol):
    """Protocol for multi-dimensional array buffers supporting `.copy()` and `.ndim`."""

    @property
    def ndim(self) -> int | OpValue:
        """Return number of dimensions.

        Returns:
            int | OpValue: Number of dimensions.
        """
        ...

    def copy(self) -> OpValue:
        """Return a shallow copy of the buffer.

        Returns:
            OpValue: Copied buffer.
        """
        ...


@typing.runtime_checkable
class _DTypeValueProtocol(typing.Protocol):
    """Protocol for dtype objects exposing a `.value` attribute."""

    @property
    def value(self) -> str:
        """Return dtype string value.

        Returns:
            str: String value.
        """
        ...


@typing.runtime_checkable
class _DTypeNameProtocol(typing.Protocol):
    """Protocol for dtype objects exposing a `.name` attribute."""

    @property
    def name(self) -> str:
        """Return dtype string name.

        Returns:
            str: String name.
        """
        ...


@typing.runtime_checkable
class _DeviceLikeProtocol(typing.Protocol):
    """Protocol for device-like objects exposing a `.device_type` attribute."""

    @property
    def device_type(self) -> DeviceType | str:
        """Return device type.

        Returns:
            DeviceType | str: Device type.
        """
        ...


def ensure_tensor(val: OpValue) -> Tensor:
    """Return ``val`` directly if already a ``Tensor``, otherwise wrap it in a ``Tensor``.

    Args:
        val (OpValue): Input tensor, backend array, or scalar operand.

    Returns:
        Tensor: Canonical ``Tensor`` instance.
    """
    if isinstance(val, Tensor):
        return val
    return Tensor(val, TensorConfig(extract_shape(val), extract_dtype(val), extract_device(val)))


def as_opt_tensor_obj(val: OpValue) -> Tensor | None:
    """Convert an optional ``OpValue`` operand into ``Tensor | None``.

    Args:
        val (OpValue): Input optional tensor operand.

    Returns:
        Tensor | None: ``None`` if ``val`` is ``None``, otherwise a ``Tensor``.
    """
    if val is None:
        return None
    return ensure_tensor(val)


_AtTensorT = typing.TypeVar("_AtTensorT", bound=TensorIndexingMixin)


class ArrayAt(typing.Generic[_AtTensorT]):
    """Provide a helper object to apply updates at specific indices."""

    def __init__(self, tensor: _AtTensorT, indices: OpValue) -> None:
        """Initialize ArrayAt.

        Args:
            tensor (_AtTensorT): The tensor to update.
            indices (OpValue): The indices to update at.
        """
        self.tensor: _AtTensorT = tensor
        self.indices = indices

    def _apply_update(self, op_name: str, value: OpValue) -> _AtTensorT:
        """Apply indexed update on multi-dimensional tensor and return the tensor.

        Args:
            op_name (str): Update operation kind ('add', 'mul', 'set', 'max', 'min').
            value (OpValue): Value to apply at indices.

        Returns:
            _AtTensorT: Updated tensor.
        """
        raw_tensor: OpValue = self.tensor
        data: OpValue | _CopyableArrayProtocol = (
            raw_tensor.data if isinstance(raw_tensor, _DataWrapperProtocol) else None
        )
        idx = self.indices.data if isinstance(self.indices, _DataWrapperProtocol) else self.indices
        val = value.data if isinstance(value, _DataWrapperProtocol) else value
        if isinstance(data, _CopyableArrayProtocol) and isinstance(data.ndim, int) and data.ndim > 0:
            copied_raw = data.copy()
            copied = _as_mutable_indexed_buffer(copied_raw)
            cur = _as_indexed_element(copied[idx])
            if op_name == "add":
                new_val = cur + val
            elif op_name == "mul":
                new_val = cur * val
            elif op_name == "set":
                new_val = val
            elif op_name == "max":
                new_val = copied[idx] if bool(cur >= val) else val
            else:
                new_val = copied[idx] if bool(cur <= val) else val
            copied[idx] = new_val
            self.tensor._data = copied_raw
        return self.tensor

    def add(self, value: OpValue) -> _AtTensorT:
        """Add value at indices.

        Args:
            value (OpValue): Input `value` argument (value).

        Returns:
            _AtTensorT: Updated tensor.
        """
        return self._apply_update("add", value)

    def multiply(self, value: OpValue) -> _AtTensorT:
        """Multiply value at indices.

        Args:
            value (OpValue): Input `value` argument (value).

        Returns:
            _AtTensorT: Updated tensor.
        """
        return self._apply_update("mul", value)

    def set(self, value: OpValue) -> _AtTensorT:
        """Set value at indices.

        Args:
            value (OpValue): Input `value` argument (value).

        Returns:
            _AtTensorT: Updated tensor.
        """
        return self._apply_update("set", value)

    def maximum(self, value: OpValue) -> _AtTensorT:
        """Maximum value at indices.

        Args:
            value (OpValue): Input `value` argument (value).

        Returns:
            _AtTensorT: Updated tensor.
        """
        return self._apply_update("max", value)

    def minimum(self, value: OpValue) -> _AtTensorT:
        """Minimum value at indices.

        Args:
            value (OpValue): Input `value` argument (value).

        Returns:
            _AtTensorT: Updated tensor.
        """
        return self._apply_update("min", value)


class ArrayAtIndexer(typing.Generic[_AtTensorT]):
    """Provide a helper object to index a tensor for ArrayAt."""

    def __init__(self, tensor: _AtTensorT) -> None:
        """Initialize ArrayAtIndexer.

        Args:
            tensor (_AtTensorT): The tensor to index.
        """
        self.tensor: _AtTensorT = tensor

    def __getitem__(self, indices: OpValue) -> ArrayAt[_AtTensorT]:
        """Get ArrayAt for indices.

        Args:
            indices (OpValue): Input `indices` argument (indices).

        Returns:
            ArrayAt[_AtTensorT]: Indexed updater helper.
        """
        return ArrayAt(self.tensor, indices)


def _extract_dtype_str(dt: DType | str | _DTypeNameProtocol | OpValue | None) -> str:
    """Extract string representation from a dtype specification.

    Args:
        dt (DType | str | _DTypeNameProtocol | OpValue | None): Dtype specification.

    Returns:
        str: Extracted dtype string.
    """
    if dt is None:
        return "float32"
    if isinstance(dt, str):
        return dt
    raw_dt: OpValue | _DTypeNameProtocol = dt
    if isinstance(raw_dt, _DTypeValueProtocol):
        return str(raw_dt.value)
    return str(raw_dt.name) if isinstance(raw_dt, _DTypeNameProtocol) else str(dt)


def _normalize_config_dtype(
    dt: DType | str | _DTypeNameProtocol | OpValue | None,
) -> DType | str | _DTypeNameProtocol | OpValue:
    """Normalize a raw dtype specification into a `DType` enum when valid.

    Args:
        dt (DType | str | _DTypeNameProtocol | OpValue | None): Raw dtype specification.

    Returns:
        DType | str | _DTypeNameProtocol | OpValue: Normalized `DType` or original dtype object.
    """
    raw_val = _extract_dtype_str(dt)
    try:
        return DType(raw_val)
    except ValueError:
        return dt


class TensorConfig(pydantic.BaseModel):
    """Configuration for a Tensor."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    shape: ShapeLike = ()
    dtype: OpValue = DType.Float32
    device: OpValue = "cpu"
    requires_grad: bool = False
    trainable: bool = False

    def __init__(
        self,
        shape: ShapeLike = (),
        dtype: OpValue = DType.Float32,
        device: OpValue = "cpu",
        requires_grad: bool = False,
        trainable: bool = False,
    ) -> None:
        """Initialize TensorConfig.

        Args:
            shape (ShapeLike): Tensor shape specification.
            dtype (OpValue): Data type specification.
            device (OpValue): Execution device specification.
            requires_grad (bool): Whether gradient tracking is enabled.
            trainable (bool): Whether tensor is a trainable parameter.
        """
        norm_shape: ShapeLike
        if shape is None:
            norm_shape = ()
        elif not isinstance(shape, tuple):
            if isinstance(shape, Iterable) and not isinstance(shape, (str, bytes)):
                from ml_switcheroo_compiler.ir.shape_system import SymNode

                norm_shape = tuple(
                    d if isinstance(d, SymNode) else (int(d) if isinstance(d, (int, float)) else str(d)) for d in shape
                )
            else:
                norm_shape = ()
        else:
            norm_shape = shape
        norm_dtype = _normalize_config_dtype(dtype)
        norm_device: OpValue
        if device is None:
            norm_device = Device("cpu")
        elif isinstance(device, str):
            norm_device = Device(device)
        elif not isinstance(device, (Device, _DeviceLikeProtocol)):
            norm_device = Device(str(device))
        else:
            norm_device = device
        super().__init__(
            shape=norm_shape,
            dtype=norm_dtype,
            device=norm_device,
            requires_grad=requires_grad,
            trainable=trainable,
        )


class Tensor(
    TensorIndexingMixin,
    TensorArithmeticMixin,
    TensorBitwiseMixin,
    TensorLogicalMixin,
):
    """Return the unified backend array base class for ml-switcheroo.

    Represents a multi-dimensional array that wraps underlying backend-specific
    data payloads, supporting both eager execution and lazy tracing
    """

    grad: OpValue = None

    def __init__(self, data: OpValue, config: TensorConfig) -> None:
        """Initialize the Tensor.

        Args:
            data (OpValue): The actual data payload.
            config (TensorConfig): The tensor configuration.
        """
        self._data = data

        def _parse_dim(s: int | str | SymNode | OpValue) -> int | str | SymNode:
            """Parse dimension.

            Args:
                s (int | str | SymNode | OpValue): Input `s` argument (s).

            Returns:
                Union[int, str, SymNode]: Result dimension.
            """
            from ml_switcheroo_compiler.ir.shape_system import SymNode

            if isinstance(s, SymNode):
                return s
            try:
                return int(str(s))
            except (ValueError, TypeError):
                return str(s)

        shape_tuple = config.shape if isinstance(config.shape, tuple) else ()
        self._shape: tuple[int | str | SymNode, ...] = tuple(_parse_dim(s) for s in shape_tuple)
        norm_dt = _normalize_config_dtype(config.dtype)
        self._dtype: DType = norm_dt if isinstance(norm_dt, DType) else DType.Float32
        self._device: Device = config.device if isinstance(config.device, Device) else Device("cpu")
        self._requires_grad = config.requires_grad
        self.config = config
        register_live_tensor(self)

    def eval(self) -> Tensor:
        """Trigger evaluation of a lazy tensor.

        Returns:
            Tensor: The evaluated tensor
        """
        if config.eager_mode or not is_traced_data(self.data):
            return self

        if global_tracing_state.is_tracing and global_tracing_state.active_graph:
            graph = global_tracing_state.active_graph
            node_id = self.data.id
            if node_id not in graph.outputs:
                graph.outputs = [*list(graph.outputs), node_id]
        return self

    def backward(self, *args: OpValue, **kwargs: IRAttrValue) -> None:
        """Trigger the reverse-mode auto-differentiation.

        Args:
            *args (OpValue): Additional arguments.
            **kwargs (IRAttrValue): Additional keyword arguments.
        """
        import ml_switcheroo_compiler.grad.api  # noqa: F401
        from ml_switcheroo_compiler.core.errors import TracingError
        from ml_switcheroo_compiler.ops.registry import get_util

        try:
            get_util("backward")(self, *args, **kwargs)
        except TracingError:
            import logging

            logging.getLogger(__name__).debug("Ignored TracingError in Tensor.backward")

    def view(self, *shape: int | Sequence[int]) -> OpValue:
        """Return a new tensor with the same data but different size.

        Args:
            *shape (int | Sequence[int]): Target shape dimensions.

        Returns:
            OpValue: A tensor containing the reshaped view.
        """
        flat_shape: list[int] = []
        for s in shape:
            if isinstance(s, (list, tuple)):
                flat_shape.extend(s)
            elif isinstance(s, int):
                flat_shape.append(s)
            else:
                flat_shape.extend(s)
        from ml_switcheroo_compiler.core.shape import VariadicCallableProtocol
        from ml_switcheroo_compiler.ops.base import OpDef
        from ml_switcheroo_compiler.ops.registry import get_frontend, get_op

        reshape_fn: type | OpDef | None
        try:
            reshape_fn = get_frontend("reshape")
        except KeyError:
            reshape_fn = get_op("Reshape")()

        assert isinstance(reshape_fn, VariadicCallableProtocol)
        return reshape_fn(self, tuple(flat_shape))

    def contiguous(self) -> Tensor:
        """Return a contiguous in memory tensor.

        Returns:
            Tensor: A contiguous tensor.
        """
        return self

    def detach(self) -> Tensor:
        """Return a new Tensor, detached from the current graph.

        Returns:
            Tensor: A detached tensor.
        """
        return Tensor(self.eval().data, TensorConfig(self.shape, self.dtype, Device("cpu")))


class Variable(Tensor):
    """Provide a mutable variable tensor for tracking state."""

    def __init__(self, data: OpValue, config: TensorConfig) -> None:
        """Initialize Variable.

        Args:
            data (OpValue): The underlying data array or proxy object.
            config (TensorConfig): The tensor configuration.
        """
        super().__init__(data, config)
        self.trainable = config.trainable

    def assign(self, value: OpValue) -> Variable:
        """Assign a new value to the variable.

        Args:
            value (OpValue): The new value.

        Returns:
            Variable: The updated variable.
        """
        if config.eager_mode:
            from ml_switcheroo_compiler.backends.registry import get_active_backend

            backend = get_active_backend()
            val_data = value.data if isinstance(value, _DataWrapperProtocol) else value
            self._data = backend.execute_op("Assign", self._data, val_data)
        else:
            from ml_switcheroo_compiler.ops.registry import get_util

            _emit_shape_node = get_util("_emit_shape_node")
            _emit_shape_node("Assign", [self, value], {}, self.shape, self.dtype)
        return self

    def assign_add(self, value: OpValue) -> Variable:
        """Add a value to the variable in-place.

        Args:
            value (OpValue): The value to add.

        Returns:
            Variable: The updated variable.
        """
        if config.eager_mode:
            from ml_switcheroo_compiler.backends.registry import get_active_backend

            backend = get_active_backend()
            val_data = value.data if isinstance(value, _DataWrapperProtocol) else value
            self._data = backend.execute_op("AssignAdd", self._data, val_data)
        else:
            from ml_switcheroo_compiler.ops.registry import get_util

            _emit_shape_node = get_util("_emit_shape_node")
            _emit_shape_node("AssignAdd", [self, value], {}, self.shape, self.dtype)
        return self

    def assign_sub(self, value: OpValue) -> Variable:
        """Subtract a value from the variable in-place.

        Args:
            value (OpValue): The value to subtract.

        Returns:
            Variable: The updated variable.
        """
        if config.eager_mode:
            from ml_switcheroo_compiler.backends.registry import get_active_backend

            backend = get_active_backend()
            val_data = value.data if isinstance(value, _DataWrapperProtocol) else value
            self._data = backend.execute_op("AssignSub", self._data, val_data)
        else:
            from ml_switcheroo_compiler.ops.registry import get_util

            _emit_shape_node = get_util("_emit_shape_node")
            _emit_shape_node("AssignSub", [self, value], {}, self.shape, self.dtype)
        return self


class Parameter(Variable):
    """Provide a trainable parameter tensor."""

    def __init__(self, data: OpValue, config: TensorConfig) -> None:
        """Initialize Parameter.

        Args:
            data (OpValue): The underlying data array or proxy object.
            config (TensorConfig): The tensor configuration.
        """
        # Override config.trainable to True for Parameter
        config = TensorConfig(config.shape, config.dtype, config.device, config.requires_grad, True)
        super().__init__(data, config)

    def __index__(self) -> int:
        """Return the integer representation of the tensor.

        Returns:
            int: The integer value.
        """
        np_val = self.numpy()
        assert isinstance(np_val, int)
        return np_val
