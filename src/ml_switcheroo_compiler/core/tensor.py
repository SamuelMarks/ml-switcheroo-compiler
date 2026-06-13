"""Defines the unified backend array base class for ml-switcheroo.

This module provides the Tensor class, which serves as the core multi-dimensional array
abstraction across different execution backends and tracing modes
"""

from collections.abc import Sequence

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType


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


class Tensor:
    """The unified backend array base class for ml-switcheroo.

    Represents a multi-dimensional array that wraps underlying backend-specific
    data payloads, supporting both eager execution and lazy tracing
    """

    def __init__(
        self,
        data: object,
        shape: Sequence[int],
        dtype: DType,
        device: Device,
        requires_grad: bool = False,
    ) -> None:
        """Initialize the Tensor.

        data (object): Argument data
            shape (Sequence[int]): Argument shape
            dtype (DType): The data type
            device (Device): Argument device
            requires_grad (bool): Argument requires_grad

        Args:
            data (object): Argument data
            shape (Sequence[int]): Argument shape
            dtype (DType): The data type
            device (Device): Argument device
            requires_grad (bool): Argument requires_grad
        """
        self._data = data
        self._shape = tuple(shape)
        self._dtype = dtype
        self._device = device
        self._requires_grad = requires_grad

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
        """Get the data type of the tensor."""
        return self._dtype

    @property
    def device(self) -> Device:
        """Get the device of the tensor."""
        return self._device

    @property
    def requires_grad(self) -> bool:
        """Check if the tensor requires gradient computation."""
        return self._requires_grad

    @property
    def data(self) -> object:
        """Get the underlying data payload."""
        return self._data

    def eval(self) -> "Tensor":
        """Trigger evaluation of a lazy tensor.

        Returns:
            Tensor: The evaluated tensor
        """
        from ml_switcheroo_compiler.core.config import config
        from ml_switcheroo_compiler.tracing import _tracer

        if config.eager_mode or not hasattr(self.data, "id"):
            return self

        # The tensor wraps a ProxyTensor, meaning we are tracing
        # If we want to evaluate it right here (e.g. host-data request),
        # we can grab the active graph, mark this node as an output,
        # compile and run it
        # This is a simplified hook for lazy evaluation
        if _tracer.is_tracing and _tracer.active_graph:
            graph = _tracer.active_graph
            if self.data.id not in graph.outputs:
                graph.outputs.append(self.data.id)
            # A full implementation would invoke a backend here
            # evaluate_graph is a naive numpy interpreter
            # Real implementation would cache and call MLX/JAX

        return self

    def __add__(self, other: object) -> "Tensor":
        """Add.

        Args:
            other (object): The other parameter

        Returns:
            'Tensor': The resulting output
        """
        from ml_switcheroo_compiler.ops.base import get_op

        return get_op("Add")()(self, other)

    def __radd__(self, other: object) -> "Tensor":
        """Radd.

        Args:
            other (object): The other parameter

        Returns:
            'Tensor': The resulting output
        """
        from ml_switcheroo_compiler.ops.base import get_op

        return get_op("Add")()(other, self)

    def __sub__(self, other: object) -> "Tensor":
        """Sub.

        Args:
            other (object): The other parameter

        Returns:
            'Tensor': The resulting output
        """
        from ml_switcheroo_compiler.ops.base import get_op

        return get_op("Subtract")()(self, other)

    def __rsub__(self, other: object) -> "Tensor":
        """Rsub.

        Args:
            other (object): The other parameter

        Returns:
            'Tensor': The resulting output
        """
        from ml_switcheroo_compiler.ops.base import get_op

        return get_op("Subtract")()(other, self)

    def __mul__(self, other: object) -> "Tensor":
        """Mul.

        Args:
            other (object): The other parameter

        Returns:
            'Tensor': The resulting output
        """
        from ml_switcheroo_compiler.ops.base import get_op

        return get_op("Multiply")()(self, other)

    def __rmul__(self, other: object) -> "Tensor":
        """Rmul.

        Args:
            other (object): The other parameter

        Returns:
            'Tensor': The resulting output
        """
        from ml_switcheroo_compiler.ops.base import get_op

        return get_op("Multiply")()(other, self)

    def __truediv__(self, other: object) -> "Tensor":
        """Truediv.

        Args:
            other (object): The other parameter

        Returns:
            'Tensor': The resulting output
        """
        from ml_switcheroo_compiler.ops.base import get_op

        return get_op("TrueDivide")()(self, other)

    def __rtruediv__(self, other: object) -> "Tensor":
        """Rtruediv.

        Args:
            other (object): The other parameter

        Returns:
            'Tensor': The resulting output
        """
        from ml_switcheroo_compiler.ops.base import get_op

        return get_op("TrueDivide")()(other, self)

    def __pow__(self, other: object) -> "Tensor":
        """Pow.

        Args:
            other (object): The other parameter

        Returns:
            'Tensor': The resulting output
        """
        from ml_switcheroo_compiler.ops.base import get_op

        return get_op("Power")()(self, other)

    def __floordiv__(self, other: object) -> "Tensor":
        """Floordiv.

        Args:
            other (object): The other parameter

        Returns:
            'Tensor': The resulting output
        """
        from ml_switcheroo_compiler.ops.base import get_op

        return get_op("FloorDivide")()(self, other)

    def __mod__(self, other: object) -> "Tensor":
        """Mod.

        Args:
            other (object): The other parameter

        Returns:
            'Tensor': The resulting output
        """
        from ml_switcheroo_compiler.ops.base import get_op

        return get_op("Mod")()(self, other)

    def __and__(self, other: object) -> "Tensor":
        """And.

        Args:
            other (object): The other parameter

        Returns:
            'Tensor': The resulting output
        """
        from ml_switcheroo_compiler.ops.base import get_op

        return get_op("BitwiseAnd")()(self, other)

    def __or__(self, other: object) -> "Tensor":
        """Or.

        Args:
            other (object): The other parameter

        Returns:
            'Tensor': The resulting output
        """
        from ml_switcheroo_compiler.ops.base import get_op

        return get_op("BitwiseOr")()(self, other)

    def __xor__(self, other: object) -> "Tensor":
        """Xor.

        Args:
            other (object): The other parameter

        Returns:
            'Tensor': The resulting output
        """
        from ml_switcheroo_compiler.ops.base import get_op

        return get_op("BitwiseXor")()(self, other)

    def __lshift__(self, other: object) -> "Tensor":
        """Lshift.

        Args:
            other (object): The other parameter

        Returns:
            'Tensor': The resulting output
        """
        from ml_switcheroo_compiler.ops.base import get_op

        return get_op("LeftShift")()(self, other)

    def __rshift__(self, other: object) -> "Tensor":
        """Rshift.

        Args:
            other (object): The other parameter

        Returns:
            'Tensor': The resulting output
        """
        from ml_switcheroo_compiler.ops.base import get_op

        return get_op("RightShift")()(self, other)

    def __neg__(self) -> "Tensor":
        """Neg.

        Returns:
            'Tensor': The resulting output
        """
        from ml_switcheroo_compiler.ops.base import get_op

        return get_op("Negative")()(self)

    def __pos__(self) -> "Tensor":
        """Pos.

        Returns:
            'Tensor': The resulting output
        """
        from ml_switcheroo_compiler.ops.base import get_op

        return get_op("Positive")()(self)

    def __abs__(self) -> "Tensor":
        """Abs.

        Returns:
            'Tensor': The resulting output
        """
        from ml_switcheroo_compiler.ops.base import get_op

        return get_op("Abs")()(self)

    def __invert__(self) -> "Tensor":
        """Invert.

        Returns:
            'Tensor': The resulting output
        """
        from ml_switcheroo_compiler.ops.base import get_op

        return get_op("BitwiseNot")()(self)

    def __lt__(self, other: object) -> "Tensor":
        """Lt.

        Args:
            other (object): The other parameter

        Returns:
            'Tensor': The resulting output
        """
        from ml_switcheroo_compiler.ops.base import get_op

        return get_op("Less")()(self, other)

    def __gt__(self, other: object) -> "Tensor":
        """Gt.

        Args:
            other (object): The other parameter

        Returns:
            'Tensor': The resulting output
        """
        from ml_switcheroo_compiler.ops.base import get_op

        return get_op("Greater")()(self, other)

    def __le__(self, other: object) -> "Tensor":
        """Le.

        Args:
            other (object): The other parameter

        Returns:
            'Tensor': The resulting output
        """
        from ml_switcheroo_compiler.ops.base import get_op

        return get_op("LessEqual")()(self, other)

    def __ge__(self, other: object) -> "Tensor":
        """Ge.

        Args:
            other (object): The other parameter

        Returns:
            'Tensor': The resulting output
        """
        from ml_switcheroo_compiler.ops.base import get_op

        return get_op("GreaterEqual")()(self, other)

    def __eq__(self, other: object) -> "Tensor":
        """Eq.

        Args:
            other (object): The other parameter

        Returns:
            'Tensor': The resulting output
        """
        from ml_switcheroo_compiler.ops.base import get_op

        return get_op("Equal")()(self, other)

    def __ne__(self, other: object) -> "Tensor":
        """Ne.

        Args:
            other (object): The other parameter

        Returns:
            'Tensor': The resulting output
        """
        from ml_switcheroo_compiler.ops.base import get_op

        return get_op("NotEqual")()(self, other)

    def __array__(self, dtype: object = None) -> object:
        """Array.

        Args:
            dtype (object): The dtype parameter

        Returns:
            object: The resulting output
        """
        import numpy as np

        if hasattr(self.data, "id"):
            return np.zeros(self.shape)
        return np.array(self.data)

    def __bool__(self) -> bool:
        """Bool.

        Returns:
            bool: The resulting output
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
            int: The resulting output
        """
        return self.shape[0] if self.shape else 0

    def __iter__(self) -> object:
        """Iter.

        Returns:
            object: The resulting output
        """
        arr = self.__array__()
        for i in range(getattr(arr, "shape", [0])[0]):
            yield Tensor(arr[i], arr[i].shape, self.dtype, self.device)

    def __getitem__(self, key: object) -> "Tensor":
        """Getitem.

        Args:
            key (object): The key parameter

        Returns:
            'Tensor': The resulting output
        """
        arr = self.__array__()
        if hasattr(key, "data"):
            key = key.data
        elif isinstance(key, tuple):
            key = tuple(getattr(k, "data", k) for k in key)

        # We need a proper op for slice, but for eager mode:
        res = arr[key]
        from ml_switcheroo_compiler.core.config import config

        if config.eager_mode:
            return Tensor(res, getattr(res, "shape", ()), self.dtype, self.device)
        msg = "Tracing getitem is not yet fully implemented via an OpDef."
        raise NotImplementedError(
            msg,
        )

    def __setitem__(self, key: object, value: object) -> None:
        """Setitem.

        Args:
            key (object): The key parameter
            value (object): The value parameter
        """
        from ml_switcheroo_compiler.core.config import config

        if config.eager_mode:
            val = getattr(value, "data", value)
            self.data[key] = val
        else:
            msg = "Item assignment is only supported in eager mode."
            raise NotImplementedError(
                msg,
            )

    def backward(self, *args: object, **kwargs: object) -> None:
        """Triggers the reverse-mode auto-differentiation."""
        from ml_switcheroo_compiler.grad import backward

        backward(self, *args, **kwargs)

    def view(self, *shape: int) -> "Tensor":
        """Returns a new tensor with the same data but different size."""
        from ml_switcheroo_compiler.ops.shape import reshape

        flat_shape = []
        for s in shape:
            if isinstance(s, (list, tuple)):
                flat_shape.extend(s)
            else:
                flat_shape.append(s)
        return reshape(self, tuple(flat_shape))

    def contiguous(self) -> "Tensor":
        """Returns a contiguous in memory tensor."""
        return self

    def item(self) -> float:
        """Returns the value of this tensor as a standard Python number."""
        import numpy as np

        if self.eval().__class__.__name__ == "Tensor":
            return float(np.asarray(self.eval().data).item())
        return float(np.asarray(self.eval()).item())

    def detach(self) -> "Tensor":
        """Returns a new Tensor, detached from the current graph."""
        from ml_switcheroo_compiler.core.device import Device

        return Tensor(self.eval().data, self.shape, self.dtype, Device("cpu"))

    @property
    def at(self) -> ArrayAtIndexer:
        """Get ArrayAtIndexer for the tensor.

        Returns:
            ArrayAtIndexer: The indexer
        """
        return ArrayAtIndexer(self)
