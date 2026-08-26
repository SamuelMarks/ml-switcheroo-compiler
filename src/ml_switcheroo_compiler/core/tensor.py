# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Define the unified backend array base class for ml-switcheroo."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Union

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.mixins import (
    TensorArithmeticMixin,
    TensorBitwiseMixin,
    TensorLogicalMixin,
)
from ml_switcheroo_compiler.tracing.state import global_tracing_state

from .tensor_mixins import TensorConversionMixin, TensorIndexingMixin, TensorPropertiesMixin


class ArrayAt:
    """Provide a helper object to apply updates at specific indices."""

    def __init__(self, tensor, indices) -> None:
        """Initialize ArrayAt.

        Args:
            tensor (Tensor[object]): The tensor to update
            indices (object): The indices to update at
        """
        self.tensor = tensor
        self.indices = indices

    def add(self, value):
        """Add value at indices.

        Args:
        value (object): The value parameter.

        Returns:
        str: Result.
        """
        return self.tensor

    def multiply(self, value):
        """Multiply value at indices.

        Args:
        value (object): The value parameter.

        Returns:
        str: Result.
        """
        return self.tensor

    def set(self, value):
        """Set value at indices.

        Args:
        value (object): The value parameter.

        Returns:
        str: Result.
        """
        return self.tensor

    def maximum(self, value):
        """Maximum value at indices.

        Args:
        value (object): The value parameter.

        Returns:
        str: Result.
        """
        return self.tensor

    def minimum(self, value):
        """Minimum value at indices.

        Args:
        value (object): The value parameter.

        Returns:
        str: Result.
        """
        return self.tensor


class ArrayAtIndexer:
    """Provide a helper object to index a tensor for ArrayAt."""

    def __init__(self, tensor) -> None:
        """Initialize ArrayAtIndexer.

        Args:
            tensor (Tensor[object]): The tensor to index
        """
        self.tensor = tensor

    def __getitem__(self, indices) -> ArrayAt:
        """Get ArrayAt for indices.

        Args:
        indices (object): The indices parameter.

        Returns:
        ArrayAt: Result.
        """
        return ArrayAt(self.tensor, indices)


@dataclass(frozen=True)
class TensorConfig:
    """Configuration for a Tensor."""

    shape: tuple[Union[int, str], ...]
    dtype: Union["DType", str]
    device: Union["Device", str]
    requires_grad: bool = False
    trainable: bool = False

    def __post_init__(self) -> None:
        """Post initialization."""
        if not isinstance(self.shape, tuple):
            object.__setattr__(self, "shape", tuple(self.shape))
        if isinstance(self.dtype, str):
            object.__setattr__(self, "dtype", DType(self.dtype))
        if isinstance(self.device, str):
            object.__setattr__(self, "device", Device(self.device))


from typing import Generic, TypeVar

T_Payload = TypeVar("T_Payload")


class Tensor(
    Generic[T_Payload],
    TensorPropertiesMixin,
    TensorConversionMixin,
    TensorIndexingMixin,
    TensorArithmeticMixin,
    TensorBitwiseMixin,
    TensorLogicalMixin,
):
    """Return the unified backend array base class for ml-switcheroo.

    Represents a multi-dimensional array that wraps underlying backend-specific
    data payloads, supporting both eager execution and lazy tracing
    """

    def __init__(self, data: T_Payload, config: TensorConfig) -> None:
        """Initialize the Tensor.

        Args:
            data (object): The actual data payload.
            config (TensorConfig): The tensor configuration.
        """
        self._data = data

        def _parse_dim(s) -> Union[int, str]:
            """Parse dimension.

            Args:
            s (object): The s parameter.

            Returns: Tensor: Result.
            """
            try:
                return int(s)
            except (ValueError, TypeError):
                return str(s)

        self._shape: tuple[Union[int, str], ...] = tuple(_parse_dim(s) for s in config.shape)
        self._dtype = config.dtype
        self._device = config.device
        self._requires_grad = config.requires_grad
        self.config = config

    def eval(self):
        """Trigger evaluation of a lazy tensor.

        Returns:
            Tensor: The evaluated tensor
        """
        if config.eager_mode or not hasattr(self.data, "id"):
            return self

        if global_tracing_state.is_tracing and global_tracing_state.active_graph:
            graph = global_tracing_state.active_graph
            if self.data.id not in graph.outputs:
                graph.outputs.append(self.data.id)
        return self

    def backward(self, *args, **kwargs) -> None:
        """Triggers the reverse-mode auto-differentiation.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        from ml_switcheroo_compiler.ops.registry import get_util

        get_util("backward")(self, *args, **kwargs)

    def view(self, *shape: int):
        """Return a new tensor with the same data but different size.

        Args:
            *shape: Additional arguments.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        flat_shape = []
        for s in shape:
            if isinstance(s, (list, tuple)):
                flat_shape.extend(s)
            else:
                flat_shape.append(s)
        from ml_switcheroo_compiler.ops.registry import get_frontend

        res = get_frontend("reshape")(self, tuple(flat_shape))
        return res

    def contiguous(self):
        """Return a contiguous in memory tensor.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self

    def detach(self):
        """Return a new Tensor, detached from the current graph.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return Tensor[object](self.eval().data, TensorConfig(self.shape, self.dtype, Device("cpu")))


class Variable(Tensor[T_Payload]):
    """Provide a mutable variable tensor for tracking state."""

    def __init__(self, data: T_Payload, config: TensorConfig) -> None:
        """Init.

        Args:
            data (object): The underlying data array or proxy object.
            config (TensorConfig): The tensor configuration.
        """
        super().__init__(data, config)
        self.trainable = config.trainable

    def assign(self, value):
        """Assign a new value to the variable.

        Args:
            value (Tensor[object]): The new value.

        Returns:
            Variable: The updated variable.
        """
        if config.eager_mode:
            from ml_switcheroo_compiler.backends.registry import get_active_backend

            backend = get_active_backend()
            self._data = backend.execute_op("Assign", self._data, value.data)
        else:
            from ml_switcheroo_compiler.ops.registry import get_util

            _emit_shape_node = get_util("_emit_shape_node")
            _emit_shape_node("Assign", [self, value], {}, self.shape, self.dtype)
        return self

    def assign_add(self, value):
        """Add a value to the variable in-place.

        Args:
            value (Tensor[object]): The value to add.

        Returns:
            Variable: The updated variable.
        """
        if config.eager_mode:
            from ml_switcheroo_compiler.backends.registry import get_active_backend

            backend = get_active_backend()
            self._data = backend.execute_op("AssignAdd", self._data, value.data)
        else:
            from ml_switcheroo_compiler.ops.registry import get_util

            _emit_shape_node = get_util("_emit_shape_node")
            _emit_shape_node("AssignAdd", [self, value], {}, self.shape, self.dtype)
        return self

    def assign_sub(self, value):
        """Subtract a value from the variable in-place.

        Args:
            value (Tensor[object]): The value to subtract.

        Returns:
            Variable: The updated variable.
        """
        if config.eager_mode:
            from ml_switcheroo_compiler.backends.registry import get_active_backend

            backend = get_active_backend()
            self._data = backend.execute_op("AssignSub", self._data, value.data)
        else:
            from ml_switcheroo_compiler.ops.registry import get_util

            _emit_shape_node = get_util("_emit_shape_node")
            _emit_shape_node("AssignSub", [self, value], {}, self.shape, self.dtype)
        return self


class Parameter(Variable[T_Payload]):
    """Provide a trainable parameter tensor."""

    def __init__(self, data: T_Payload, config: TensorConfig) -> None:
        """Init.

        Args:
            data (object): The underlying data array or proxy object.
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
        return int(self.numpy())
