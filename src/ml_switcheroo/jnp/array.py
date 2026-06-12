"""Docstring."""

import ml_switcheroo.ops as ops
from ml_switcheroo import Tensor
import ml_switcheroo


class ndarray:
    """A multi-dimensional array object backed by an ml-switcheroo tensor.

    This class provides a NumPy-like interface for tensor operations, supporting
    standard arithmetic, comparison, and array-manipulation magic methods.
    """

    def __init__(self, tensor: object) -> None:
        """Initialize the object.

        Args:
            tensor (Any): The underlying tensor data.

        Returns:
            None
        """
        self._tensor = tensor

    @property
    def shape(self) -> object:
        """Get the shape of the array.

        Returns:
            Any: The shape property of the underlying tensor.
        """
        return self._tensor.shape

    @property
    def dtype(self) -> object:
        """Get the dtype of the array.

        Returns:
            Any: The dtype property of the underlying tensor.
        """
        return self._tensor.dtype

    def __array__(self) -> object:
        """Perform the array operation.

        Returns:
            Any: The result of the array operation.
        """
        import numpy as np

        if hasattr(self._tensor.data, "id"):  # ProxyTensor check
            return np.zeros(self._tensor.shape)
        return np.array(
            self._tensor.data
        )  # Return dummy shape for tracing asserts if needed

    def __repr__(self) -> object:
        """Perform the repr operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the repr operation.
        """
        return repr(self.__array__())

    def __add__(self, other: object) -> object:
        """Perform the add operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the add operation.
        """
        from .math_ops import add

        return add(self, other)

    def __radd__(self, other: object) -> object:
        """Perform the radd operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the radd operation.
        """
        from .math_ops import add

        return add(other, self)

    def __sub__(self, other: object) -> object:
        """Perform the sub operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the sub operation.
        """
        from .math_ops import add, multiply

        return add(self, multiply(other, -1))

    def __rsub__(self, other: object) -> object:
        """Perform the rsub operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the rsub operation.
        """
        from .math_ops import add, multiply

        return add(other, multiply(self, -1))

    def __mul__(self, other: object) -> object:
        """Perform the mul operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the mul operation.
        """
        from .math_ops import multiply

        return multiply(self, other)

    def __rmul__(self, other: object) -> object:
        """Perform the rmul operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the rmul operation.
        """
        from .math_ops import multiply

        return multiply(other, self)

    def __pow__(self, other: object) -> object:
        """Perform the pow operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the pow operation.
        """
        from .math_ops import power

        return power(self, other)

    def __rpow__(self, other: object) -> object:
        """Perform the rpow operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the rpow operation.
        """
        from .math_ops import power

        return power(other, self)

    def __truediv__(self, other: object) -> object:
        """Perform the truediv operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the truediv operation.
        """
        from .math_ops import true_divide

        return true_divide(self, other)

    def __rtruediv__(self, other: object) -> object:
        """Perform the rtruediv operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the rtruediv operation.
        """
        from .math_ops import true_divide

        return true_divide(other, self)

    def __floordiv__(self, other: object) -> object:
        """Perform the floordiv operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the floordiv operation.
        """
        from .math_ops import floor_divide

        return floor_divide(self, other)

    def __rfloordiv__(self, other: object) -> object:
        """Perform the rfloordiv operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the rfloordiv operation.
        """
        from .math_ops import floor_divide

        return floor_divide(other, self)

    def __neg__(self) -> object:
        """Perform the neg operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the neg operation.
        """
        from ml_switcheroo.jnp.math_ops import negative

        return negative(self)

    def __lt__(self, other: object) -> object:
        """Perform the lt operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the lt operation.
        """
        return _wrap(ops.less(self._tensor, _to_tensor(other)))

    def __gt__(self, other: object) -> object:
        """Perform the gt operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the gt operation.
        """
        return _wrap(ops.greater(self._tensor, _to_tensor(other)))

    def __le__(self, other: object) -> object:
        """Perform the le operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the le operation.
        """
        return _wrap(ops.less_equal(self._tensor, _to_tensor(other)))

    def __ge__(self, other: object) -> object:
        """Perform the ge operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the ge operation.
        """
        return _wrap(ops.greater_equal(self._tensor, _to_tensor(other)))

    def __setitem__(self, key: object, value: object) -> None:
        """Perform the setitem operation.

        Args:
            key: The key.
            value: The value.

        Returns:
            Any: The result of the setitem operation.
        """
        from ml_switcheroo.core.config import config

        if config.eager_mode:
            val = getattr(value, "_tensor", value)
            val = getattr(val, "data", val)
            self._tensor.data[key] = val
        else:
            raise NotImplementedError(
                "Item assignment is only supported in eager mode."
            )

    def __getitem__(self, key: object) -> object:
        """Perform the getitem operation.

        Args:
            key: The key.

        Returns:
            Any: The result of the getitem operation.
        """
        arr = self.__array__()
        if hasattr(key, "_tensor"):
            key = key._tensor.data
        elif isinstance(key, tuple):
            key = tuple(
                getattr(getattr(k, "_tensor", k), "data", getattr(k, "_tensor", k))
                for k in key
            )
        return _wrap(_to_tensor(arr[key]))

    def __eq__(self, other: object) -> object:
        """Perform the eq operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the eq operation.
        """
        return _wrap(ops.equal(self._tensor, _to_tensor(other)))

    def __bool__(self) -> object:
        """Perform the bool operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the bool operation.
        """
        arr = self.__array__()
        if arr.size == 1:
            return bool(arr.item())
        raise ValueError(
            "The truth value of an array with more than one element is ambiguous."
        )

    def __len__(self) -> object:
        """Perform the len operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the len operation.
        """
        return self.shape[0] if self.shape else 0

    def __iter__(self) -> object:
        """Perform the iter operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the iter operation.
        """
        arr = self.__array__()
        for i in range(arr.shape[0]):
            from .creation import array

            yield array(arr[i])


def _to_tensor(x: object) -> object:
    """Convert the input to an ml-switcheroo Tensor.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    if isinstance(x, ndarray):
        x = x._tensor
    from ml_switcheroo.core.config import config
    from ml_switcheroo.tracing import _tracer, ProxyTensor
    from ml_switcheroo_ir import LogicalNode
    import uuid

    if isinstance(x, ml_switcheroo.Tensor):
        if _tracer.is_tracing and not hasattr(x.data, "id"):
            # lift eager tensor as constant
            out_id = str(uuid.uuid4())
            node = LogicalNode(
                id=out_id,
                op_type="Constant",
                attributes={"value": getattr(x.data, "tolist", lambda: x.data)()},
                shape_metadata=x.shape,
            )
            _tracer.add_node(node)
            pt = ProxyTensor(id=out_id, shape=x.shape, dtype=x.dtype.value)
            return ml_switcheroo.Tensor(
                data=pt, shape=x.shape, dtype=x.dtype, device=x.device
            )
        return x
    if isinstance(x, ProxyTensor):
        # We need a dtype. ProxyTensor has dtype as string.
        # But we'll just mock it or use default.
        return ml_switcheroo.Tensor(
            data=x,
            shape=x.shape,
            dtype=config.default_float_dtype,
            device=config.default_device,
        )

    import numpy as np

    arr = np.array(x)
    if config.eager_mode and not _tracer.is_tracing:
        return ml_switcheroo.Tensor(
            arr, arr.shape, config.default_float_dtype, config.default_device
        )
    else:
        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id, op_type="Constant", attributes={"value": arr.tolist()}
        )
        _tracer.add_node(node)
        pt = ProxyTensor(id=out_id, shape=arr.shape)
        return ml_switcheroo.Tensor(
            data=pt,
            shape=arr.shape,
            dtype=config.default_float_dtype,
            device=config.default_device,
        )


def _wrap(t: object) -> object:
    """Wrap an ml-switcheroo Tensor in an ndarray.

    Args:
        t (Any): Argument t.

    Returns:
        Any: The result of the operation.
    """
    if isinstance(t, Tensor):
        return ndarray(t)
    elif isinstance(t, tuple):
        return tuple(_wrap(x) for x in t)
    elif isinstance(t, list):
        return list(_wrap(x) for x in t)
    return t
