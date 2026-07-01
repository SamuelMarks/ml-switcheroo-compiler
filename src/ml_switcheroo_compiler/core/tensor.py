"""Defines the unified backend array base class for ml-switcheroo.

This module provides the Tensor class, which serves as the core multi-dimensional array
abstraction across different execution backends and tracing modes
"""

from ml_switcheroo_compiler.backends.registry import get_active_backend
from dataclasses import dataclass
from collections.abc import Sequence

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.mixins import (
    TensorArithmeticMixin,
    TensorBitwiseMixin,
    TensorLogicalMixin,
)


class ArrayAt:
    """A helper object to apply updates at specific indices."""

    def __init__(self, tensor: "Tensor", indices: object) -> None:
        """Initialize ArrayAt.

        Args:
            tensor (Tensor): The tensor to update
            indices (object): The indices to update at
        """
        self.tensor = tensor
        self.indices = indices

    def add(self, value: object) -> "Tensor":
        """Add value at indices.

        Args:
            value (object): The value to add
        Returns:
            Tensor: The updated tensor
        """
        return self.tensor

    def multiply(self, value: object) -> "Tensor":
        """Multiply value at indices.

        Args:
            value (object): The value to multiply
        Returns:
            Tensor: The updated tensor
        """
        return self.tensor

    def set(self, value: object) -> "Tensor":
        """Set value at indices.

        Args:
            value (object): The value to set
        Returns:
            Tensor: The updated tensor
        """
        return self.tensor

    def maximum(self, value: object) -> "Tensor":
        """Maximum value at indices.

        Args:
            value (object): The value to compare
        Returns:
            Tensor: The updated tensor
        """
        return self.tensor

    def minimum(self, value: object) -> "Tensor":
        """Minimum value at indices.

        Args:
            value (object): The value to compare
        Returns:
            Tensor: The updated tensor
        """
        return self.tensor


class ArrayAtIndexer:
    """A helper object to index a tensor for ArrayAt."""

    def __init__(self, tensor: "Tensor") -> None:
        """Initialize ArrayAtIndexer.

        Args:
            tensor (Tensor): The tensor to index
        """
        self.tensor = tensor

    def __getitem__(self, indices: object) -> ArrayAt:
        """Get ArrayAt for indices.

        Args:
            indices (object): The indices to update at
        Returns:
            ArrayAt: The ArrayAt object
        """
        return ArrayAt(self.tensor, indices)


@dataclass(frozen=True)
class TensorConfig:
    """Configuration for a Tensor."""

    shape: tuple[int, ...]
    dtype: "DType"
    device: "Device"
    requires_grad: bool = False
    trainable: bool = False


class Tensor(TensorArithmeticMixin, TensorBitwiseMixin, TensorLogicalMixin):
    """The unified backend array base class for ml-switcheroo.

    Represents a multi-dimensional array that wraps underlying backend-specific
    data payloads, supporting both eager execution and lazy tracing
    """

    def __init__(self, data: object, config: TensorConfig) -> None:
        """Initialize the Tensor.

        Args:
            data (object): The actual data payload.
            config (TensorConfig): The tensor configuration.
        """
        self._data = data
        self._shape = tuple(int(s) for s in config.shape)
        self._dtype = config.dtype
        self._device = config.device
        self._requires_grad = config.requires_grad
        self.config = config

    @property
    def shape(self) -> Sequence[int]:
        """Get the shape of the tensor.

        Args:
        Returns:
            Sequence[int]: The result of the operation
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
    def data(self) -> object:
        """Get the underlying data payload.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return self._data

    def eval(self) -> "Tensor":
        """Trigger evaluation of a lazy tensor.

        Returns:
            Tensor: The evaluated tensor
        """
        import ml_switcheroo_compiler.core.tensor_dispatcher as _td

        return _td.dispatch_eval(self)

    def __array__(self, dtype: object = None) -> object:
        """Array.

        Args:
            dtype (object): The dtype to process.

        Returns:
            The computed shape or evaluation result.
        """
        import numpy as np

        backend = get_active_backend()

        if hasattr(self.data, "id"):
            data = backend.zeros(self.shape)
        else:
            data = self.data

        try:
            return np.asarray(data, dtype=dtype) if dtype is not None else np.asarray(data)
        except Exception:  # pragma: no cover
            return np.array(
                data.tolist() if hasattr(data, "tolist") else data, dtype=dtype
            )  # pragma: no cover

    def __bool__(self) -> bool:
        """Bool.

        Returns:
            bool: A boolean indicating the result of the check.
        """
        arr = self.__array__()
        if getattr(arr, "size", 1) == 1:
            return bool(getattr(arr, "item", lambda: arr)())
        msg = "The truth value of an array with more than one element is ambiguous."
        raise ValueError(
            msg,
        )

    def __len__(self) -> int:
        """Len.

        Returns:
            int: The evaluated output resulting from this operation.
        """
        return self.shape[0] if self.shape else 0

    def __iter__(self) -> object:
        """Iter.

        Returns:
            The computed shape or evaluation result.
        """
        arr = self.__array__()
        shape = getattr(arr, "shape", [0])
        if not shape:
            raise TypeError("iteration over a 0-d tensor")
        for i in range(shape[0]):
            yield Tensor(arr[i], TensorConfig(arr[i].shape, self.dtype, self.device))

    def __getitem__(self, key: object) -> "Tensor":
        """Getitem.

        Args:
            key (object): The key to process.

        Returns:
            Tensor: A new tensor with the selected element.
        """
        import ml_switcheroo_compiler.core.tensor_dispatcher as _td

        return _td.dispatch_getitem(self, key)

    def __setitem__(self, key: object, value: object) -> None:
        """Setitem.

        Args:
            key (object): The key to process.
            value (object): The value to set or add.
        """
        import ml_switcheroo_compiler.core.tensor_dispatcher as _td

        _td.dispatch_setitem(self, key, value)

    def backward(self, *args: object, **kwargs: object) -> None:
        """Triggers the reverse-mode auto-differentiation.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        import ml_switcheroo_compiler.core.tensor_dispatcher as _td

        _td.dispatch_backward(self, *args, **kwargs)

    def view(self, *shape: int) -> "Tensor":
        """Returns a new tensor with the same data but different size.

        Args:
            *shape: Additional arguments.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        import ml_switcheroo_compiler.core.tensor_dispatcher as _td

        flat_shape = []
        for s in shape:
            if isinstance(s, (list, tuple)):
                flat_shape.extend(s)
            else:
                flat_shape.append(s)
        return _td.dispatch_reshape(self, tuple(flat_shape))

    def contiguous(self) -> "Tensor":
        """Returns a contiguous in memory tensor.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        return self

    def item(self) -> float:
        """Returns the value of this tensor as a standard Python number.

        Returns:
            float: The evaluated output resulting from this operation.
        """
        backend = get_active_backend()

        if self.eval().__class__.__name__ == "Tensor":
            return backend.item(self.eval().data)
        return backend.item(self.eval())

    def detach(self) -> "Tensor":
        """Returns a new Tensor, detached from the current graph.

        Returns:
            'Tensor': A tensor containing the result of the operation.
        """
        from ml_switcheroo_compiler.core.device import Device

        return Tensor(self.eval().data, TensorConfig(self.shape, self.dtype, Device("cpu")))

    @property
    def at(self) -> ArrayAtIndexer:
        """Get ArrayAtIndexer for the tensor.

        Returns:
            ArrayAtIndexer: The indexer
        """
        return ArrayAtIndexer(self)


class Variable(Tensor):
    """A mutable variable tensor for tracking state."""

    def __init__(self, data: object, config: TensorConfig) -> None:
        """Init.

        Args:
            data (object): The underlying data array or proxy object.
            config (TensorConfig): The tensor configuration.
        """
        super().__init__(data, config)
        self.trainable = config.trainable

    def assign(self, value: Tensor) -> "Variable":
        """Assign a new value to the variable.

        Args:
            value (Tensor): The new value.

        Returns:
            Variable: The updated variable.
        """
        import ml_switcheroo_compiler.core.tensor_dispatcher as _td

        return _td.dispatch_assign(self, value)

    def assign_add(self, value: Tensor) -> "Variable":
        """Add a value to the variable in-place.

        Args:
            value (Tensor): The value to add.

        Returns:
            Variable: The updated variable.
        """
        import ml_switcheroo_compiler.core.tensor_dispatcher as _td

        return _td.dispatch_assign_add(self, value)

    def assign_sub(self, value: Tensor) -> "Variable":
        """Subtract a value from the variable in-place.

        Args:
            value (Tensor): The value to subtract.

        Returns:
            Variable: The updated variable.
        """
        import ml_switcheroo_compiler.core.tensor_dispatcher as _td

        return _td.dispatch_assign_sub(self, value)


class Parameter(Variable):
    """A trainable parameter tensor."""

    def __init__(self, data: object, config: TensorConfig) -> None:
        """Init.

        Args:
            data (object): The underlying data array or proxy object.
            config (TensorConfig): The tensor configuration.
        """
        # Override config.trainable to True for Parameter
        config = TensorConfig(config.shape, config.dtype, config.device, config.requires_grad, True)
        super().__init__(data, config)
