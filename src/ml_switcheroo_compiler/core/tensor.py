"""Define the unified backend array base class for ml-switcheroo."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union, cast

if TYPE_CHECKING:
    from ml_switcheroo_ir import TensorShape

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
from ml_switcheroo_compiler.core.shape import AttrValue, ShapeLike, TensorOrScalar
from ml_switcheroo_compiler.tracing.state import global_tracing_state

from .tensor_mixins import (
    TensorConversionMixin,
    TensorIndexingMixin,
    TensorPropertiesMixin,
)


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
        data = getattr(self.tensor, "data", None)
        idx = getattr(self.indices, "data", self.indices)
        val = getattr(value, "data", value)
        ndim = getattr(data, "ndim", 0)
        if data is not None and hasattr(data, "copy") and isinstance(ndim, int) and ndim > 0:
            copied: TensorOrScalar = cast(TensorOrScalar, getattr(data, "copy")())
            cur = getattr(copied, "__getitem__")(idx)
            if op_name == "add":
                new_val = getattr(cur, "__add__")(val)
            elif op_name == "mul":
                new_val = getattr(cur, "__mul__")(val)
            elif op_name == "set":
                new_val = val
            elif op_name == "max":
                new_val = cur if bool(getattr(cur, "__ge__")(val)) else val
            else:
                new_val = cur if bool(getattr(cur, "__le__")(val)) else val
            getattr(copied, "__setitem__")(idx, new_val)
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


def _normalize_config_dtype(dt: Union[DType, str, TensorOrScalar, None]) -> Union[DType, str, TensorOrScalar]:
    """Normalize a raw dtype specification into a `DType` enum when valid.

    Args:
        dt (Union[DType, str, TensorOrScalar, None]): Raw dtype specification.

    Returns:
        Union[DType, str, TensorOrScalar]: Normalized `DType` or original dtype object.
    """
    if dt is None:
        return DType.Float32
    if isinstance(dt, DType):
        return dt
    raw_val = dt if isinstance(dt, str) else getattr(dt, "value", getattr(dt, "name", str(dt)))
    try:
        return DType(str(raw_val))
    except ValueError:
        return dt


@dataclass(frozen=True)
class TensorConfig:
    """Configuration for a Tensor."""

    shape: Union[tuple[Union[int, str, SymNode, None], ...], Sequence[Union[int, str, SymNode, None]], TensorShape, ShapeLike, TensorOrScalar, None] = ()
    dtype: Union[DType, str, TensorOrScalar, None] = DType.Float32
    device: Union[Device, str, TensorOrScalar, None] = "cpu"
    requires_grad: bool = False
    trainable: bool = False

    def __post_init__(self) -> None:
        """Post initialization."""
        if self.shape is None:
            object.__setattr__(self, "shape", ())
        elif not isinstance(self.shape, tuple):
            try:
                object.__setattr__(self, "shape", tuple(cast("Sequence[Union[int, str, SymNode, None]]", self.shape)))
            except TypeError:
                object.__setattr__(self, "shape", ())
        object.__setattr__(self, "dtype", _normalize_config_dtype(self.dtype))
        if self.device is None:
            object.__setattr__(self, "device", Device("cpu"))
        elif isinstance(self.device, str):
            object.__setattr__(self, "device", Device(self.device))
        elif not isinstance(self.device, Device) and not hasattr(self.device, "device_type"):
            object.__setattr__(self, "device", Device(str(self.device)))


if TYPE_CHECKING:
    _TensorMixinBase = TensorIndexingMixin
else:

    class _TensorMixinBase(TensorPropertiesMixin, TensorConversionMixin, TensorIndexingMixin):
        """Runtime combined base class for Tensor mixins."""


class Tensor(
    _TensorMixinBase,
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

        def _parse_dim(s: int | str | SymNode) -> Union[int, str, SymNode]:
            """Parse dimension.

            Args:
                s (int | str | SymNode): The s parameter.

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
        self._shape: tuple[Union[int, str, SymNode], ...] = tuple(cast("Callable[..., Union[int, str, SymNode]]", _parse_dim)(s) for s in shape_tuple)
        self._dtype = cast(DType, config.dtype)
        self._device = cast(Device, config.device)
        self._requires_grad = config.requires_grad
        self.config = config

    def eval(self) -> Tensor:
        """Trigger evaluation of a lazy tensor.

        Returns:
            Tensor: The evaluated tensor
        """
        if config.eager_mode or not hasattr(self.data, "id"):
            return self

        if global_tracing_state.is_tracing and global_tracing_state.active_graph:
            graph = global_tracing_state.active_graph
            node_id = getattr(self.data, "id")
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
        from ml_switcheroo_compiler.ops.base import OpDef
        from ml_switcheroo_compiler.ops.registry import get_frontend, get_op

        reshape_fn: type | OpDef | None
        try:
            reshape_fn = get_frontend("reshape")
        except KeyError:
            reshape_fn = get_op("Reshape")()

        res = cast(Callable[..., Tensor], reshape_fn)(self, tuple(flat_shape))
        return res

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
            self._data = backend.execute_op("Assign", self._data, getattr(value, "data"))
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
            self._data = backend.execute_op("AssignAdd", self._data, getattr(value, "data"))
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
            self._data = backend.execute_op("AssignSub", self._data, getattr(value, "data"))
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
        return int(cast(int, self.numpy()))
