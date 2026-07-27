"""Mixins for Tensor."""

import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

if TYPE_CHECKING:
    from ml_switcheroo_compiler.ops.shape.indexing import ArrayAtIndexer

    from .tensor import Tensor


class TensorPropertiesMixin:
    """Tensor properties mixin."""

    @property
    def ndim(self) -> int:
        """Get the number of dimensions of the tensor."""
        return len(self._shape)

    @property
    def size(self) -> int:
        """Get the number of elements in the tensor."""
        # if there are strings in shape (unknown dims), return a ProxyTensor?
        # for eager evaluation, size should evaluate natively.
        prod = 1
        for s in self._shape:
            if isinstance(s, str):
                pass  # Ignore symbolic dims or handle differently? Keras usually expects an int or None.
            else:
                prod *= s
        return prod

    @property
    def shape(self) -> Sequence[int]:
        """Get the shape of the tensor.

        Args:
        Returns:
            Sequence[int]: The inferred shape or computed result
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


class TensorConversionMixin:
    """Tensor conversion mixin."""

    def numpy(self) -> object:
        """Numpy."""
        try:
            import importlib.util

            import numpy as np

            if importlib.util.find_spec("ml_switcheroo_compiler.backends.numpy.utils") is None:
                pass
        except ImportError as e:
            raise ImportError("numpy is required") from e
        return np.asarray(self._data)

    def __array__(self, dtype: object = None) -> object:
        """Array.

        Args:
            dtype (object): The dtype to process.

        Returns:
            The computed shape or evaluation result.
        """
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()

        if hasattr(self.data, "id"):
            data = backend.zeros(self.shape)
        else:
            data = self.data

        try:
            return np.asarray(data, dtype=dtype) if dtype is not None else np.asarray(data)
        except Exception:
            return np.array(data.tolist() if hasattr(data, "tolist") else data, dtype=dtype)

    def item(self) -> float:
        """Returns the value of this tensor as a standard Python number.

        Returns:
            float: The evaluated output resulting from this operation.
        """
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        backend = get_active_backend()

        if self.eval().__class__.__name__ == "Tensor":
            return backend.item(self.eval().data)
        return backend.item(self.eval())

    def __int__(self) -> int:
        """Int."""
        return int(self.item())

    def __index__(self) -> int:
        """Return the integer value of a scalar tensor.

        Returns:
            int: The scalar integer value.
        """
        return int(self.item())

    def __float__(self) -> float:
        """Float."""
        return float(self.item())

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
            from .tensor import Tensor, TensorConfig

            yield Tensor(arr[i], TensorConfig(arr[i].shape, self.dtype, self.device))


class TensorIndexingMixin:
    """Tensor indexing mixin."""

    def __getitem__(self, key: object) -> "Tensor":
        """Getitem.

        Args:
            key (object): The key to process.

        Returns:
            Tensor: A new tensor with the selected element.
        """
        arr = self.__array__()
        if hasattr(key, "data"):
            key = key.data
        elif isinstance(key, tuple):
            key = tuple(getattr(k, "data", k) for k in key)

        try:
            res = arr[tuple(key) if isinstance(key, list) else key]
            res_shape = getattr(res, "shape", ())
        except IndexError:
            if config.eager_mode:
                raise
            res_shape = ()
            # If we are in dummy mode, we might want to still raise if it's clearly out of bounds
            # like too many indices for a known shape. But if the key has a ProxyTensor, NumPy
            # will raise IndexError: only integers... We want to let it pass and build a node,
            # UNLESS it's truly an invalid indexing like too many dims.
            # We will just raise IndexError if it's too many indices for array, so Keras catches it.
            import traceback

            err_msg = traceback.format_exc()
            if "too many indices for array" in err_msg:
                raise
            # Otherwise we continue and create a lazy node

        if config.eager_mode:
            from .tensor import Tensor, TensorConfig

            return Tensor(res, TensorConfig(res_shape, self.dtype, self.device))

        nid = f"getitem_{uuid.uuid4().hex[:6]}"
        input_id = getattr(self.data, "id", "const")

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

    def __setitem__(self, key: object, value: object) -> None:
        """Setitem.

        Args:
            key (object): The key to process.
            value (object): The value to set or add.
        """
        if config.eager_mode:
            val = getattr(value, "data", value)
            self.data[key] = val
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

        return ArrayAtIndexer(self)
