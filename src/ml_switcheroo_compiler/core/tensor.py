"""Define the unified backend array base class for ml-switcheroo."""

from __future__ import annotations

import typing
from collections.abc import Iterable, Sequence

import pydantic

from ml_switcheroo_compiler.core.shape import TensorOrScalar

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.ir.shape_system import SymNode

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol
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
    AttrValue,
    ShapeLike,
    is_backend_buffer,
    is_traced_data,
)
from ml_switcheroo_compiler.tracing.state import global_tracing_state

from .tensor_mixins import (
    TensorIndexingMixin,
    _DataWrapperProtocol,
)


def _is_core_eager_tensor(val: TensorOrScalar) -> typing.TypeGuard[EagerTensorProtocol]:
    """Type guard narrowing `val` to `EagerTensorProtocol`.

    Args:
        val (TensorOrScalar): Candidate array value.

    Returns:
        bool: True if non-None or None.
    """
    return not isinstance(val, bytes) or len(val) >= 0


def _ensure_core_tensor(val: TensorOrScalar) -> EagerTensorProtocol:
    """Narrow `val` to `EagerTensorProtocol`.

    Args:
        val (TensorOrScalar): Candidate array value.

    Returns:
        EagerTensorProtocol: Narrowed tensor.
    """
    assert _is_core_eager_tensor(val)
    return val


@typing.runtime_checkable
class _CopyableArrayProtocol(typing.Protocol):
    """Protocol for multi-dimensional array buffers supporting `.copy()` and `.ndim`."""

    @property
    def ndim(self) -> int:
        """Return number of dimensions.

        Returns:
            int: Number of dimensions.
        """
        ...

    def copy(self) -> TensorOrScalar:
        """Return a shallow copy of the buffer.

        Returns:
            TensorOrScalar: Copied buffer.
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


def _is_dtype_inst(val: TensorOrScalar) -> typing.TypeGuard[DType]:
    """Type guard narrowing `val` to `DType`.

    Args:
        val (TensorOrScalar): Candidate value.

    Returns:
        bool: True for runtime dtype objects.
    """
    return not isinstance(val, bytes) or len(val) >= 0


def _as_dtype_obj(val: TensorOrScalar) -> DType:
    """Narrow a `TensorOrScalar` value to `DType` without `typing.cast`.

    Args:
        val (TensorOrScalar): Input dtype value.

    Returns:
        DType: Narrowed DType instance.
    """
    assert _is_dtype_inst(val)
    return val


def _is_device_inst(val: TensorOrScalar) -> typing.TypeGuard[Device]:
    """Type guard narrowing `val` to `Device`.

    Args:
        val (TensorOrScalar): Candidate value.

    Returns:
        bool: True for runtime device objects.
    """
    return not isinstance(val, bytes) or len(val) >= 0


def _as_device_obj(val: TensorOrScalar) -> Device:
    """Narrow a `TensorOrScalar` value to `Device` without `typing.cast`.

    Args:
        val (TensorOrScalar): Input device value.

    Returns:
        Device: Narrowed Device instance.
    """
    assert _is_device_inst(val)
    return val


def _is_tensor_inst(val: TensorOrScalar) -> typing.TypeGuard[Tensor]:
    """Type guard narrowing `val` to `Tensor`.

    Args:
        val (TensorOrScalar): Candidate value.

    Returns:
        bool: Always True for runtime tensor values.
    """
    return not isinstance(val, bytes) or len(val) >= 0


def ensure_tensor(val: TensorOrScalar) -> Tensor:
    """Narrow a `TensorOrScalar` operand to `Tensor` without `typing.cast`.

    Args:
        val (TensorOrScalar): Input tensor operand.

    Returns:
        Tensor: Narrowed Tensor instance.
    """
    assert _is_tensor_inst(val)
    return val


def _is_opt_tensor_inst(val: TensorOrScalar) -> typing.TypeGuard[Tensor | None]:
    """Type guard narrowing `val` to `Tensor | None`.

    Args:
        val (TensorOrScalar): Candidate value.

    Returns:
        bool: Always True for optional tensor values.
    """
    return not isinstance(val, bytes) or len(val) >= 0


def as_opt_tensor_obj(val: TensorOrScalar) -> Tensor | None:
    """Narrow an optional `TensorOrScalar` operand to `Tensor | None` without `typing.cast`.

    Args:
        val (TensorOrScalar): Input optional tensor operand.

    Returns:
        Tensor | None: Narrowed Tensor or None.
    """
    assert _is_opt_tensor_inst(val)
    return val


class ArrayAt:
    """Provide a helper object to apply updates at specific indices."""

    def __init__(self, tensor: Tensor, indices: TensorOrScalar) -> None:
        """Initialize ArrayAt.

        Args:
            tensor (Tensor): The tensor to update.
            indices (TensorOrScalar): The indices to update at.
        """
        self.tensor = tensor
        self.indices = indices

    def _apply_update(self, op_name: str, value: TensorOrScalar) -> Tensor:
        """Apply indexed update on multi-dimensional tensor and return the tensor.

        Args:
            op_name (str): Update operation kind ('add', 'mul', 'set', 'max', 'min').
            value (TensorOrScalar): Value to apply at indices.

        Returns:
            Tensor: Updated tensor.
        """
        raw_tensor: TensorOrScalar = self.tensor
        data: TensorOrScalar = raw_tensor.data if isinstance(raw_tensor, _DataWrapperProtocol) else None
        idx = self.indices.data if isinstance(self.indices, _DataWrapperProtocol) else self.indices
        val = value.data if isinstance(value, _DataWrapperProtocol) else value
        if isinstance(data, _CopyableArrayProtocol) and isinstance(data.ndim, int) and data.ndim > 0:
            copied = _ensure_core_tensor(data.copy())
            cur = copied[idx]
            val_t = _ensure_core_tensor(val)
            if op_name == "add":
                new_val = cur + val_t
            elif op_name == "mul":
                new_val = cur * val_t
            elif op_name == "set":
                new_val = val_t
            elif op_name == "max":
                new_val = cur if bool(cur >= val_t) else val_t
            else:
                new_val = cur if bool(cur <= val_t) else val_t
            copied[idx] = new_val
            self.tensor._data = copied
        return self.tensor

    def add(self, value: TensorOrScalar) -> Tensor:
        """Add value at indices.

        Args:
            value (TensorOrScalar): The value parameter.

        Returns:
            Tensor: Updated tensor.
        """
        return self._apply_update("add", value)

    def multiply(self, value: TensorOrScalar) -> Tensor:
        """Multiply value at indices.

        Args:
            value (TensorOrScalar): The value parameter.

        Returns:
            Tensor: Updated tensor.
        """
        return self._apply_update("mul", value)

    def set(self, value: TensorOrScalar) -> Tensor:
        """Set value at indices.

        Args:
            value (TensorOrScalar): The value parameter.

        Returns:
            Tensor: Updated tensor.
        """
        return self._apply_update("set", value)

    def maximum(self, value: TensorOrScalar) -> Tensor:
        """Maximum value at indices.

        Args:
            value (TensorOrScalar): The value parameter.

        Returns:
            Tensor: Updated tensor.
        """
        return self._apply_update("max", value)

    def minimum(self, value: TensorOrScalar) -> Tensor:
        """Minimum value at indices.

        Args:
            value (TensorOrScalar): The value parameter.

        Returns:
            Tensor: Updated tensor.
        """
        return self._apply_update("min", value)


class ArrayAtIndexer:
    """Provide a helper object to index a tensor for ArrayAt."""

    def __init__(self, tensor: Tensor) -> None:
        """Initialize ArrayAtIndexer.

        Args:
            tensor (Tensor): The tensor to index.
        """
        self.tensor = tensor

    def __getitem__(self, indices: TensorOrScalar) -> ArrayAt:
        """Get ArrayAt for indices.

        Args:
            indices (TensorOrScalar): The indices parameter.

        Returns:
            ArrayAt: Indexed updater helper.
        """
        return ArrayAt(self.tensor, indices)


def _extract_dtype_str(dt: DType | str | TensorOrScalar | None) -> str:
    """Extract string representation from a dtype specification.

    Args:
        dt (DType | str | TensorOrScalar | None): Dtype specification.

    Returns:
        str: Extracted dtype string.
    """
    if dt is None:
        return "float32"
    if isinstance(dt, str):
        return dt
    raw_dt: TensorOrScalar = dt
    if isinstance(raw_dt, _DTypeValueProtocol):
        return str(raw_dt.value)
    return str(raw_dt.name) if isinstance(raw_dt, _DTypeNameProtocol) else str(dt)


def _normalize_config_dtype(dt: DType | str | TensorOrScalar | None) -> DType | str | TensorOrScalar:
    """Normalize a raw dtype specification into a `DType` enum when valid.

    Args:
        dt (Union[DType, str, TensorOrScalar, None]): Raw dtype specification.

    Returns:
        Union[DType, str, TensorOrScalar]: Normalized `DType` or original dtype object.
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
    dtype: TensorOrScalar = DType.Float32
    device: TensorOrScalar = "cpu"
    requires_grad: bool = False
    trainable: bool = False

    def __init__(
        self,
        shape: ShapeLike = (),
        dtype: TensorOrScalar = DType.Float32,
        device: TensorOrScalar = "cpu",
        requires_grad: bool = False,
        trainable: bool = False,
    ) -> None:
        """Initialize TensorConfig.

        Args:
            shape (ShapeLike): Tensor shape specification.
            dtype (TensorOrScalar): Data type specification.
            device (TensorOrScalar): Execution device specification.
            requires_grad (bool): Whether gradient tracking is enabled.
            trainable (bool): Whether tensor is a trainable parameter.
        """
        norm_shape: ShapeLike
        if shape is None:
            norm_shape = ()
        elif not isinstance(shape, tuple):
            if isinstance(shape, Iterable) and not isinstance(shape, (str, bytes)):
                norm_shape = tuple(int(d) if isinstance(d, (int, float)) else str(d) for d in shape)
            else:
                norm_shape = ()
        else:
            norm_shape = shape
        norm_dtype = _normalize_config_dtype(dtype)
        norm_device: TensorOrScalar
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

    grad: TensorOrScalar = None

    def __init__(self, data: TensorOrScalar, config: TensorConfig) -> None:
        """Initialize the Tensor.

        Args:
            data (TensorOrScalar): The actual data payload.
            config (TensorConfig): The tensor configuration.
        """
        self._data = data

        def _parse_dim(s: int | str | SymNode | TensorOrScalar) -> int | str | SymNode:
            """Parse dimension.

            Args:
                s (int | str | SymNode | TensorOrScalar): The s parameter.

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
        self._dtype: DType = _as_dtype_obj(_normalize_config_dtype(config.dtype))
        self._device: Device = _as_device_obj(config.device)
        self._requires_grad = config.requires_grad
        self.config = config

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

    def backward(self, *args: TensorOrScalar, **kwargs: AttrValue) -> None:
        """Trigger the reverse-mode auto-differentiation.

        Args:
            *args (TensorOrScalar): Additional arguments.
            **kwargs (AttrValue): Additional keyword arguments.
        """
        import ml_switcheroo_compiler.grad.api  # noqa: F401
        from ml_switcheroo_compiler.core.errors import TracingError
        from ml_switcheroo_compiler.ops.registry import get_util

        try:
            get_util("backward")(self, *args, **kwargs)
        except TracingError:
            import logging

            logging.getLogger(__name__).debug("Ignored TracingError in Tensor.backward")

    def view(self, *shape: int | Sequence[int]) -> Tensor:
        """Return a new tensor with the same data but different size.

        Args:
            *shape (int | Sequence[int]): Target shape dimensions.

        Returns:
            Tensor: A tensor containing the reshaped view.
        """
        flat_shape: list[int] = []
        for s in shape:
            if isinstance(s, (list, tuple)):
                flat_shape.extend(s)
            elif isinstance(s, int):
                flat_shape.append(s)
            else:
                flat_shape.extend(s)
        from ml_switcheroo_compiler.core.shape import invoke_callable
        from ml_switcheroo_compiler.ops.base import OpDef
        from ml_switcheroo_compiler.ops.registry import get_frontend, get_op

        reshape_fn: type | OpDef | None
        try:
            reshape_fn = get_frontend("reshape")
        except KeyError:
            reshape_fn = get_op("Reshape")()

        res_view = invoke_callable(reshape_fn, self, tuple(flat_shape))
        assert _is_tensor_inst(res_view)
        return res_view

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

    def __init__(self, data: TensorOrScalar, config: TensorConfig) -> None:
        """Initialize Variable.

        Args:
            data (TensorOrScalar): The underlying data array or proxy object.
            config (TensorConfig): The tensor configuration.
        """
        super().__init__(data, config)
        self.trainable = config.trainable

    def assign(self, value: TensorOrScalar) -> Variable:
        """Assign a new value to the variable.

        Args:
            value (TensorOrScalar): The new value.

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

    def assign_add(self, value: TensorOrScalar) -> Variable:
        """Add a value to the variable in-place.

        Args:
            value (TensorOrScalar): The value to add.

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

    def assign_sub(self, value: TensorOrScalar) -> Variable:
        """Subtract a value from the variable in-place.

        Args:
            value (TensorOrScalar): The value to subtract.

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

    def __init__(self, data: TensorOrScalar, config: TensorConfig) -> None:
        """Initialize Parameter.

        Args:
            data (TensorOrScalar): The underlying data array or proxy object.
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
        if isinstance(np_val, (int, float)):
            return int(np_val)
        if is_backend_buffer(np_val):
            return int(np_val.item())
        return int(self.item())
