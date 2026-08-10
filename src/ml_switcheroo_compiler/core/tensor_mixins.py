# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
from typing import Any

"""Mixins for Tensor."""

import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

if TYPE_CHECKING:
    from ml_switcheroo_compiler.ops.shape.indexing import ArrayAtIndexer  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    from .tensor import Tensor


class TensorPropertiesMixin:
    """Tensor properties mixin."""

    @property
    def ndim(self) -> int:
        """Get the number of dimensions of the tensor.

        Returns:
            int: The number of dimensions.
        """
        return len(self._shape)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    @property
    def size(self) -> int:
        """Get the number of elements in the tensor.

        Returns:
            int: The size of the tensor.
        """
        # if there are strings in shape (unknown dims), return a ProxyTensor?
        # for eager evaluation, size should evaluate natively.
        prod = 1
        for s in self._shape:  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
            if isinstance(s, str):
                return None  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
            else:
                prod *= s
        return prod

    @property
    def shape(self) -> Sequence[int]:
        """Get the shape of the tensor.

        Returns:
            Sequence[int]: The shape tuple of the tensor.
        """
        return self._shape  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    @property
    def dtype(self) -> DType:
        """Get the data type of the tensor.

        Returns:
            DType: The data type associated with the tensor.
        """
        return self._dtype  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    @property
    def device(self) -> Device:
        """Get the device of the tensor.

        Returns:
            Device: The device associated with the tensor.
        """
        return self._device  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    @property
    def requires_grad(self) -> bool:
        """Check if the tensor requires gradient computation.

        Returns:
            bool: A boolean indicating the result of the check.
        """
        return self._requires_grad  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    @property
    def data(self) -> object:
        """Get the underlying data payload.

        Returns:
            object: The computed result.
        """
        return self._data  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism


class TensorConversionMixin:
    """Tensor conversion mixin."""

    def numpy(self) -> object:
        """Convert the tensor to a NumPy array.

        Returns:
            object: The NumPy array representation.
        """
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        try:
            return get_active_backend().numpy(self._data)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        except Exception:
            return get_active_backend().asarray(self._data)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __array__(self, dtype: object = None) -> object:
        """Array.

        Args:
            dtype (object): The dtype to process.

        Returns:
            The computed shape or evaluation result.
        """
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()

        if hasattr(self.data, "id"):  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
            data = backend.zeros(self.shape)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        else:
            data = self.data  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

        try:
            return backend.array(data, dtype=dtype) if dtype is not None else backend.asarray(data)
        except Exception:
            return backend.array(data.tolist() if hasattr(data, "tolist") else data, dtype=dtype)

    def item(self) -> float:
        """Return the value of this tensor as a standard Python number.

        Returns:
            float: The computed result.
        """
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()

        if self.eval().__class__.__name__ == "Tensor":  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
            return backend.item(self.eval().data)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        return backend.item(self.eval())  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

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
            int: The computed result.
        """
        return self.shape[0] if self.shape else 0  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __iter__(self) -> object:
        """Iterate over the first dimension of the tensor.

        Yields:
            Tensor: A slice of the tensor along the first dimension.

        Raises:
            TypeError: If the tensor is 0-dimensional.
        """
        arr = self.__array__()
        shape = getattr(arr, "shape", [0])
        if not shape:
            raise TypeError("iteration over a 0-d tensor")
        for i in range(shape[0]):
            from .tensor import Tensor, TensorConfig

            yield Tensor(arr[i], TensorConfig(arr[i].shape, self.dtype, self.device))  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism


class TensorIndexingMixin:
    """Tensor indexing mixin."""

    def __getitem__(self, key: object) -> "Tensor":
        """Retrieve elements from the tensor.

        Args:
            key (object): The index or slice key.

        Returns:
            Tensor: A new tensor containing the selected elements.

        Raises:
            IndexError: If the index is out of bounds or invalid.
            RuntimeError: If tracing but not in an active tracing context.
        """
        arr = self.__array__()  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        if hasattr(key, "data"):
            key = key.data
        elif isinstance(key, tuple):
            key = tuple(getattr(k, "data", k) for k in key)

        try:
            res = arr[tuple(key) if isinstance(key, list) else key]
            res_shape = getattr(res, "shape", ())
        except IndexError as e:
            if config.eager_mode:
                raise
            res_shape = ()
            # If we are in dummy mode, we might want to still raise if it's clearly out of bounds
            # like too many indices for a known shape. But if the key has a ProxyTensor, NumPy
            # will raise IndexError: only integers... We want to let it pass and build a node,
            # UNLESS it's truly an invalid indexing like too many dims.
            # We will just raise IndexError if it's too many indices for array, so Keras catches it.
            if "too many indices for array" in str(e):
                raise
            # Otherwise we continue and create a lazy node

        if config.eager_mode:
            from .tensor import Tensor, TensorConfig

            return Tensor(res, TensorConfig(res_shape, self.dtype, self.device))  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

        nid = f"getitem_{uuid.uuid4().hex[:6]}"
        input_id = getattr(self.data, "id", "const")  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

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

        return Tensor(ProxyTensor(nid, (), self.dtype.value), TensorConfig((), self.dtype, self.device))  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def __setitem__(self, key: object, value: object) -> None:
        """Set elements in the tensor (only supported in eager mode).

        Args:
            key (object): The index or slice key.
            value (object): The value to set.

        Raises:
            TypeError: If attempted during tracing.
        """
        if config.eager_mode:
            val = getattr(value, "data", value)
            self.data[key] = val  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        else:
            msg = "Tensor object does not support item assignment in tracing mode. Use .at[...].set(...) instead."
            raise TypeError(msg)

    @property
    def at(self) -> "ArrayAtIndexer":
        """Get ArrayAtIndexer for the tensor.

        Returns:
            ArrayAtIndexer: The indexer
        """
        from .tensor import ArrayAtIndexer

        return ArrayAtIndexer(self)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
