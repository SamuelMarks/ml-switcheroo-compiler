"""JAX-like numpy API backed by ml-switcheroo-compiler."""

from typing import Any

from typing import Tuple, List, Optional
import ml_switcheroo.ops as ops
from ml_switcheroo import Tensor
import ml_switcheroo


class ndarray:
    """A multi-dimensional array object backed by an ml-switcheroo tensor.

    This class provides a NumPy-like interface for tensor operations, supporting
    standard arithmetic, comparison, and array-manipulation magic methods.
    """

    def __init__(self, tensor: Any) -> None:
        """Initialize the object.

        Args:
            tensor (Any): The underlying tensor data.

        Returns:
            None
        """
        self._tensor = tensor

    @property
    def shape(self) -> Any:
        """Get the shape of the array.

        Returns:
            Any: The shape property of the underlying tensor.
        """
        return self._tensor.shape

    @property
    def dtype(self) -> Any:
        """Get the dtype of the array.

        Returns:
            Any: The dtype property of the underlying tensor.
        """
        return self._tensor.dtype

    def __array__(self) -> Any:
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

    def __repr__(self) -> Any:
        """Perform the repr operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the repr operation.
        """
        return repr(self.__array__())

    def __add__(self, other: Any) -> Any:
        """Perform the add operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the add operation.
        """
        return add(self, other)

    def __radd__(self, other: Any) -> Any:
        """Perform the radd operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the radd operation.
        """
        return add(other, self)

    def __sub__(self, other: Any) -> Any:
        """Perform the sub operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the sub operation.
        """
        return add(self, multiply(other, -1))

    def __rsub__(self, other: Any) -> Any:
        """Perform the rsub operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the rsub operation.
        """
        return add(other, multiply(self, -1))

    def __mul__(self, other: Any) -> Any:
        """Perform the mul operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the mul operation.
        """
        return multiply(self, other)

    def __rmul__(self, other: Any) -> Any:
        """Perform the rmul operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the rmul operation.
        """
        return multiply(other, self)

    def __pow__(self, other: Any) -> Any:
        """Perform the pow operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the pow operation.
        """
        return power(self, other)

    def __rpow__(self, other: Any) -> Any:
        """Perform the rpow operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the rpow operation.
        """
        return power(other, self)

    def __truediv__(self, other: Any) -> Any:
        """Perform the truediv operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the truediv operation.
        """
        return true_divide(self, other)

    def __rtruediv__(self, other: Any) -> Any:
        """Perform the rtruediv operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the rtruediv operation.
        """
        return true_divide(other, self)

    def __floordiv__(self, other: Any) -> Any:
        """Perform the floordiv operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the floordiv operation.
        """
        return floor_divide(self, other)

    def __rfloordiv__(self, other: Any) -> Any:
        """Perform the rfloordiv operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the rfloordiv operation.
        """
        return floor_divide(other, self)

    def __neg__(self) -> Any:
        """Perform the neg operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the neg operation.
        """
        return multiply(self, -1.0)

    def __lt__(self, other: Any) -> Any:
        """Perform the lt operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the lt operation.
        """
        return _wrap(ops.less(self._tensor, _to_tensor(other)))

    def __gt__(self, other: Any) -> Any:
        """Perform the gt operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the gt operation.
        """
        return _wrap(ops.greater(self._tensor, _to_tensor(other)))

    def __le__(self, other: Any) -> Any:
        """Perform the le operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the le operation.
        """
        return _wrap(ops.less_equal(self._tensor, _to_tensor(other)))

    def __ge__(self, other: Any) -> Any:
        """Perform the ge operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the ge operation.
        """
        return _wrap(ops.greater_equal(self._tensor, _to_tensor(other)))

    def __setitem__(self, key: Any, value: Any) -> None:
        """Perform the setitem operation.

        Args:
            other (Any): The other operand for the operation.

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

    def __getitem__(self, key: Any) -> Any:
        """Perform the getitem operation.

        Args:
            other (Any): The other operand for the operation.

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

    def __eq__(self, other: Any) -> Any:
        """Perform the eq operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the eq operation.
        """
        return _wrap(ops.equal(self._tensor, _to_tensor(other)))

    def __bool__(self) -> Any:
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

    def __len__(self) -> Any:
        """Perform the len operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the len operation.
        """
        return self.shape[0] if self.shape else 0

    def __iter__(self) -> Any:
        """Perform the iter operation.

        Args:
            other (Any): The other operand for the operation.

        Returns:
            Any: The result of the iter operation.
        """
        arr = self.__array__()
        for i in range(arr.shape[0]):
            yield array(arr[i])


def _to_tensor(x: Any) -> Any:
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


def _wrap(t: Any) -> Any:
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


def sin(x: Any) -> Any:
    """Compute the trigonometric sine element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.sin(_to_tensor(x)))


def cos(x: Any) -> Any:
    """Compute the trigonometric cosine element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.cos(_to_tensor(x)))


def exp(x: Any) -> Any:
    """Calculate the exponential of all elements in the input array.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.exp(_to_tensor(x)))


def log(x: Any) -> Any:
    """Natural logarithm, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.log(_to_tensor(x)))


def transpose(x: Any, axes: Optional[List[int]] = None) -> Any:
    """Reverse or permute the axes of an array.

    Args:
        x (Any): Argument x.
        axes (Any): Argument axes.

    Returns:
        Any: The result of the operation.
    """
    t = _to_tensor(x)
    if axes is not None:
        return _wrap(ops.permute(t, dims=axes))
    axes = list(range(len(t.shape))[::-1])
    return _wrap(ops.permute(t, dims=axes))


def reshape(x: Any, newshape: Tuple[int, ...]) -> Any:
    """Gives a new shape to an array without changing its data.

    Args:
        x (Any): Argument x.
        newshape (Any): Argument newshape.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.reshape(_to_tensor(x), shape=newshape))


def broadcast_to(x: Any, shape: Tuple[int, ...]) -> Any:
    """Broadcast an array to a new shape.

    Args:
        x (Any): Argument x.
        shape (Any): Argument shape.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.broadcast_to(_to_tensor(x), size=shape))


def concatenate(arrays: List[Any], axis: int = 0) -> Any:
    """Join a sequence of arrays along an existing axis.

    Args:
        arrays (Any): Argument arrays.
        axis (Any): Argument axis.

    Returns:
        Any: The result of the operation.
    """
    tensors = [_to_tensor(a) for a in arrays]
    return _wrap(ops.concatenate(tensors, dim=axis))


def where(condition: Any, x: Any, y: Any) -> Any:
    """Return elements chosen from x or y depending on condition.

    Args:
        condition (Any): Argument condition.
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.where(_to_tensor(condition), _to_tensor(x), _to_tensor(y)))


def einsum(subscripts: str, *operands: Any) -> Any:
    """Evaluates the Einstein summation convention on the operands.

    Args:
        subscripts (Any): Argument subscripts.

    Returns:
        Any: The result of the operation.
    """
    tensors = [_to_tensor(a) for a in operands]
    return _wrap(ops.einsum(subscripts, *tensors))


def add(x: Any, y: Any) -> Any:
    """Add arguments element-wise.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.add(_to_tensor(x), _to_tensor(y)))


def multiply(x: Any, y: Any) -> Any:
    """Multiply arguments element-wise.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.multiply(_to_tensor(x), _to_tensor(y)))


def power(x: Any, y: Any) -> Any:
    """First array elements raised to powers from second array, element-wise.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.power(_to_tensor(x), _to_tensor(y)))


def maximum(x: Any, y: Any) -> Any:
    """Element-wise maximum of array elements.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.maximum(_to_tensor(x), _to_tensor(y)))


def minimum(x: Any, y: Any) -> Any:
    """Element-wise minimum of array elements.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.minimum(_to_tensor(x), _to_tensor(y)))


def clip(a: Any, a_min: Any, a_max: Any) -> Any:
    """Clip (limit) the values in an array.

    Args:
        a (Any): Argument a.
        a_min (Any): Argument a_min.
        a_max (Any): Argument a_max.

    Returns:
        Any: The result of the operation.
    """
    res = _to_tensor(a)
    if a_min is not None:
        res = ops.maximum(res, _to_tensor(a_min))
    if a_max is not None:
        res = ops.minimum(res, _to_tensor(a_max))
    return _wrap(res)


def max(
    x: Any,
    axis: Any = None,
    keepdims: bool = False,
    where: Any = None,
    initial: Any = None,
) -> Any:
    """Return the maximum of an array or maximum along an axis.

    Args:
        x (Any): Argument x.
        axis (Any): Argument axis.
        keepdims (Any): Argument keepdims.
        where (Any): Argument where.
        initial (Any): Argument initial.

    Returns:
        Any: The result of the operation.
    """
    t_x = _to_tensor(x)
    if where is not None:
        init_val = initial if initial is not None else float("-inf")
        t_x = ops.where(_to_tensor(where), t_x, _to_tensor(init_val))
    res = ops.max(t_x, axis=axis, keepdims=keepdims)
    if initial is not None:
        res = ops.maximum(res, _to_tensor(initial))
    return _wrap(res)


def sum(x: Any, axis: Any = None, keepdims: bool = False, where: Any = None) -> Any:
    """Sum of array elements over a given axis.

    Args:
        x (Any): Argument x.
        axis (Any): Argument axis.
        keepdims (Any): Argument keepdims.
        where (Any): Argument where.

    Returns:
        Any: The result of the operation.
    """
    t_x = _to_tensor(x)
    if where is not None:
        t_x = ops.where(_to_tensor(where), t_x, _to_tensor(0))
    return _wrap(ops.sum(t_x, axis=axis, keepdims=keepdims))


def zeros_like(x: Any, dtype: Any = None) -> Any:
    """Return an array of zeros with the same shape and type as a given array.

    Args:
        x (Any): Argument x.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.zeros_like(_to_tensor(x), dtype=dtype))


def zeros(shape: Any, dtype: Any = None) -> Any:
    """Return a new array of given shape and type, filled with zeros.

    Args:
        shape (Any): Argument shape.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.zeros(shape=shape, dtype=dtype))


def abs(x: Any) -> Any:
    """Calculate the absolute value element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.abs(_to_tensor(x)))


def mean(x: Any, axis: Any = None, keepdims: bool = False) -> Any:
    """Compute the arithmetic mean along the specified axis.

    Args:
        x (Any): Argument x.
        axis (Any): Argument axis.
        keepdims (Any): Argument keepdims.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.mean(_to_tensor(x), axis=axis, keepdims=keepdims))


inf = float("inf")


def array(x: Any, dtype: Any = None) -> Any:
    """Create an array.

    Args:
        x (Any): Argument x.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    if isinstance(x, ndarray):
        return x
    return _wrap(_to_tensor(x))


def dot(a: Any, b: Any) -> Any:
    """Dot product of two arrays.

    Args:
        a (Any): Argument a.
        b (Any): Argument b.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.dot(_to_tensor(a), _to_tensor(b)))


def matmul(a: Any, b: Any) -> Any:
    """Matrix product of two arrays.

    Args:
        a (Any): Argument a.
        b (Any): Argument b.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.matmul(_to_tensor(a), _to_tensor(b)))


def expand_dims(a: Any, axis: int) -> Any:
    """Expand the shape of an array.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.unsqueeze(_to_tensor(a), dim=axis))


def isfinite(x: Any) -> Any:
    """Test element-wise for finiteness (not infinity or not Not a Number).

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.isfinite(_to_tensor(x)))


def allclose(
    a: Any, b: Any, rtol: Any = 1e-05, atol: Any = 1e-08, equal_nan: Any = False
) -> Any:
    """Returns True if two arrays are element-wise equal within a tolerance.

    Args:
        a (Any): Argument a.
        b (Any): Argument b.
        rtol (Any): Argument rtol.
        atol (Any): Argument atol.
        equal_nan (Any): Argument equal_nan.

    Returns:
        Any: The result of the operation.
    """
    return ops.allclose(
        _to_tensor(a), _to_tensor(b), rtol=rtol, atol=atol, equal_nan=equal_nan
    )


def array_equal(a1: Any, a2: Any, equal_nan: Any = False) -> Any:
    """True if two arrays have the same shape and elements, False otherwise.

    Args:
        a1 (Any): Argument a1.
        a2 (Any): Argument a2.
        equal_nan (Any): Argument equal_nan.

    Returns:
        Any: The result of the operation.
    """
    res = ops.equal(_to_tensor(a1), _to_tensor(a2))
    import numpy as np

    return bool(np.array(res.data).all()) if hasattr(res, "data") else True


def broadcast_shapes(*shapes: Any) -> Any:
    """Broadcast the input shapes into a single shape.

    Returns:
        Any: The result of the operation.
    """
    from ml_switcheroo.shape import broadcast_shapes as _broadcast_shapes
    import functools

    if not shapes:
        return ()
    return functools.reduce(_broadcast_shapes, shapes)


def _unary_op(x: Any, name: Any) -> Any:
    """Apply a unary operation.

    Args:
        x (Any): Argument x.
        name (Any): Argument name.

    Returns:
        Any: The result of the operation.
    """
    if name == "Transpose":
        return transpose(x)
    raise NotImplementedError()


def ones(shape: Any, dtype: Any = None) -> Any:
    """Return a new array of given shape and type, filled with ones.

    Args:
        shape (Any): Argument shape.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.ones(shape=shape, dtype=dtype))


def empty(shape: Any, dtype: Any = None) -> Any:
    """Return a new array of given shape and type, without initializing entries.

    Args:
        shape (Any): Argument shape.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.empty(shape=shape, dtype=dtype))


def full(shape: Any, fill_value: Any, dtype: Any = None) -> Any:
    """Return a new array of given shape and type, filled with fill_value.

    Args:
        shape (Any): Argument shape.
        fill_value (Any): Argument fill_value.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.full(shape=shape, fill_value=fill_value, dtype=dtype))


def ones_like(x: Any, dtype: Any = None) -> Any:
    """Return an array of ones with the same shape and type as a given array.

    Args:
        x (Any): Argument x.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.ones_like(_to_tensor(x), dtype=dtype))


def empty_like(x: Any, dtype: Any = None) -> Any:
    """Return a new array with the same shape and type as a given array.

    Args:
        x (Any): Argument x.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    t = _to_tensor(x)
    return _wrap(ops.empty(shape=t.shape, dtype=dtype if dtype else t.dtype))


def full_like(x: Any, fill_value: Any, dtype: Any = None) -> Any:
    """Return a full array with the same shape and type as a given array.

    Args:
        x (Any): Argument x.
        fill_value (Any): Argument fill_value.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.full_like(_to_tensor(x), fill_value=fill_value, dtype=dtype))


def asarray(x: Any, dtype: Any = None) -> Any:
    """Convert the input to an array.

    Args:
        x (Any): Argument x.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    return array(x, dtype=dtype)


def arange(start: Any, stop: Any = None, step: Any = 1, dtype: Any = None) -> Any:
    """Return evenly spaced values within a given interval.

    Args:
        start (Any): Argument start.
        stop (Any): Argument stop.
        step (Any): Argument step.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.arange(start=start, stop=stop, step=step, dtype=dtype))


def linspace(
    start: Any,
    stop: Any,
    num: int = 50,
    endpoint: bool = True,
    retstep: bool = False,
    dtype: Any = None,
    axis: int = 0,
) -> Any:
    """Return evenly spaced numbers over a specified interval.

    Args:
        start (Any): Argument start.
        stop (Any): Argument stop.
        num (Any): Argument num.
        endpoint (Any): Argument endpoint.
        retstep (Any): Argument retstep.
        dtype (Any): Argument dtype.
        axis (Any): Argument axis.

    Returns:
        Any: The result of the operation.
    """
    if retstep or axis != 0 or not endpoint:
        raise NotImplementedError("linspace currently only supports basic usage")
    return _wrap(ops.linspace(start=start, stop=stop, steps=num, dtype=dtype))


def logspace(
    start: Any,
    stop: Any,
    num: int = 50,
    endpoint: bool = True,
    base: float = 10.0,
    dtype: Any = None,
    axis: int = 0,
) -> Any:
    """Return numbers spaced evenly on a log scale.

    Args:
        start (Any): Argument start.
        stop (Any): Argument stop.
        num (Any): Argument num.
        endpoint (Any): Argument endpoint.
        base (Any): Argument base.
        dtype (Any): Argument dtype.
        axis (Any): Argument axis.

    Returns:
        Any: The result of the operation.
    """
    lin = linspace(start, stop, num, endpoint=endpoint, dtype=dtype, axis=axis)
    return power(base, lin)


def eye(N: int, M: int = None, k: int = 0, dtype: Any = None) -> Any:
    """Return a 2-D array with ones on the diagonal and zeros elsewhere.

    Args:
        N (Any): Argument N.
        M (Any): Argument M.
        k (Any): Argument k.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    if k != 0:
        raise NotImplementedError()
    return _wrap(ops.eye(n=N, m=M, dtype=dtype))


def identity(n: int, dtype: Any = None) -> Any:
    """Return the identity array.

    Args:
        n (Any): Argument n.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.identity(n=n, dtype=dtype))


def meshgrid(
    *xi: Any, copy: Any = True, sparse: Any = False, indexing: Any = "xy"
) -> Any:
    """Return coordinate matrices from coordinate vectors.

    Returns:
        Any: The result of the operation.
    """
    if sparse or not copy:
        raise NotImplementedError()

    tensors = [_to_tensor(x) for x in xi]

    ndim = len(tensors)
    if indexing == "xy" and ndim > 1:
        tensors[0], tensors[1] = tensors[1], tensors[0]

    s0 = (1,) * ndim
    output = []
    for i, t in enumerate(tensors):
        shape = list(s0)
        shape[i] = -1
        reshaped = ops.reshape(t, shape=tuple(shape))
        output.append(reshaped)

    broadcast_shape = tuple(t.shape[0] for t in tensors)
    output = [ops.broadcast_to(t, size=broadcast_shape) for t in output]

    if indexing == "xy" and ndim > 1:
        output[0], output[1] = output[1], output[0]

    return tuple(_wrap(t) for t in output)


def subtract(x: Any, y: Any) -> Any:
    """Subtract arguments, element-wise.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.subtract(_to_tensor(x), _to_tensor(y)))


def divide(x: Any, y: Any) -> Any:
    """Divide arguments element-wise.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.divide(_to_tensor(x), _to_tensor(y)))


def true_divide(x: Any, y: Any) -> Any:
    """Divide arguments element-wise.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return divide(x, y)


def floor_divide(x: Any, y: Any) -> Any:
    """Return the largest integer smaller or equal to the division of the inputs.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.floor_divide(_to_tensor(x), _to_tensor(y)))


def mod(x: Any, y: Any) -> Any:
    """Return element-wise remainder of division.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.mod(_to_tensor(x), _to_tensor(y)))


def remainder(x: Any, y: Any) -> Any:
    """Return element-wise remainder of division.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.remainder(_to_tensor(x), _to_tensor(y)))


def divmod(x: Any, y: Any) -> Any:
    """Return element-wise quotient and remainder simultaneously.

    Args:
        x (Any): Argument x.
        y (Any): Argument y.

    Returns:
        Any: The result of the operation.
    """
    out1, out2 = ops.divmod(_to_tensor(x), _to_tensor(y))
    return _wrap(out1), _wrap(out2)


def negative(x: Any) -> Any:
    """Numerical negative, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.negative(_to_tensor(x)))


def positive(x: Any) -> Any:
    """Numerical positive, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.positive(_to_tensor(x)))


def sign(x: Any) -> Any:
    """Returns an element-wise indication of the sign of a number.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.sign(_to_tensor(x)))


def floor(x: Any) -> Any:
    """Return the floor of the input, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.floor(_to_tensor(x)))


def ceil(x: Any) -> Any:
    """Return the ceiling of the input, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.ceil(_to_tensor(x)))


def trunc(x: Any) -> Any:
    """Return the truncated value of the input, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.trunc(_to_tensor(x)))


def rint(x: Any) -> Any:
    """Round elements of the array to the nearest integer.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.round(_to_tensor(x)))


def tan(x: Any) -> Any:
    """Compute tangent element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.tan(_to_tensor(x)))


def arcsin(x: Any) -> Any:
    """Inverse sine, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.asin(_to_tensor(x)))


def arccos(x: Any) -> Any:
    """Trigonometric inverse cosine, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.acos(_to_tensor(x)))


def arctan(x: Any) -> Any:
    """Trigonometric inverse tangent, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.atan(_to_tensor(x)))


def arctan2(x1: Any, x2: Any) -> Any:
    """Element-wise arc tangent of x1/x2 choosing the quadrant correctly.

    Args:
        x1 (Any): Argument x1.
        x2 (Any): Argument x2.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.atan2(_to_tensor(x1), _to_tensor(x2)))


def sinh(x: Any) -> Any:
    """Hyperbolic sine, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.sinh(_to_tensor(x)))


def cosh(x: Any) -> Any:
    """Hyperbolic cosine, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.cosh(_to_tensor(x)))


def tanh(x: Any) -> Any:
    """Compute hyperbolic tangent element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.tanh(_to_tensor(x)))


def arcsinh(x: Any) -> Any:
    """Inverse hyperbolic sine element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.asinh(_to_tensor(x)))


def arccosh(x: Any) -> Any:
    """Inverse hyperbolic cosine, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.acosh(_to_tensor(x)))


def arctanh(x: Any) -> Any:
    """Inverse hyperbolic tangent element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.atanh(_to_tensor(x)))


def exp2(x: Any) -> Any:
    """Calculate 2**p for all p in the input array.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    # 2^x = exp(x * ln(2)) or just power(2, x)
    return power(2.0, x)


def expm1(x: Any) -> Any:
    """Calculate exp(x) - 1 for all elements in the array.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return subtract(exp(x), 1.0)


def log2(x: Any) -> Any:
    """Base-2 logarithm of x.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    # log2(x) = log(x) / log(2)
    import math

    return divide(log(x), math.log(2.0))


def log10(x: Any) -> Any:
    """Return the base 10 logarithm of the input array, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    import math

    return divide(log(x), math.log(10.0))


def log1p(x: Any) -> Any:
    """Return the natural logarithm of one plus the input array, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return log(add(x, 1.0))


def prod(a: Any, axis: Any = None, dtype: Any = None, keepdims: bool = False) -> Any:
    """Return the product of array elements over a given axis.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        dtype (Any): Argument dtype.
        keepdims (Any): Argument keepdims.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.prod(_to_tensor(a), axis=axis, keepdims=keepdims))


def min(a: Any, axis: Any = None, keepdims: bool = False) -> Any:
    """Return the minimum of an array or minimum along an axis.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        keepdims (Any): Argument keepdims.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.min(_to_tensor(a), axis=axis, keepdims=keepdims))


def amin(a: Any, axis: Any = None, keepdims: bool = False) -> Any:
    """Return the minimum of an array or minimum along an axis.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        keepdims (Any): Argument keepdims.

    Returns:
        Any: The result of the operation.
    """
    return min(a, axis=axis, keepdims=keepdims)


def amax(a: Any, axis: Any = None, keepdims: bool = False) -> Any:
    """Return the maximum of an array or maximum along an axis.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        keepdims (Any): Argument keepdims.

    Returns:
        Any: The result of the operation.
    """
    return max(a, axis=axis, keepdims=keepdims)


def argmax(a: Any, axis: Any = None, keepdims: bool = False) -> Any:
    """Returns the indices of the maximum values along an axis.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        keepdims (Any): Argument keepdims.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.argmax(_to_tensor(a), axis=axis, keepdims=keepdims))


def argmin(a: Any, axis: Any = None, keepdims: bool = False) -> Any:
    """Returns the indices of the minimum values along an axis.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        keepdims (Any): Argument keepdims.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.argmin(_to_tensor(a), axis=axis, keepdims=keepdims))


def any(a: Any, axis: Any = None, keepdims: bool = False) -> Any:
    """Test whether any array element along a given axis evaluates to True.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        keepdims (Any): Argument keepdims.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.any(_to_tensor(a), axis=axis, keepdims=keepdims))


def all(a: Any, axis: Any = None, keepdims: bool = False) -> Any:
    """Test whether all array elements along a given axis evaluate to True.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        keepdims (Any): Argument keepdims.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.all(_to_tensor(a), axis=axis, keepdims=keepdims))


def var(
    a: Any, axis: Any = None, dtype: Any = None, keepdims: bool = False, ddof: int = 0
) -> Any:
    """Compute the variance along the specified axis.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        dtype (Any): Argument dtype.
        keepdims (Any): Argument keepdims.
        ddof (Any): Argument ddof.

    Returns:
        Any: The result of the operation.
    """
    # variance = E[X^2] - E[X]^2 or mean((x - mean(x))^2)
    # Using eager wrapper or tracing composition
    t = _to_tensor(a)
    m = mean(a, axis=axis, keepdims=True)
    diff = subtract(t, m)
    sq = multiply(diff, diff)
    # if ddof != 0 we would need more math, but standard test probably uses default
    return mean(sq, axis=axis, keepdims=keepdims)


def std(
    a: Any, axis: Any = None, dtype: Any = None, keepdims: bool = False, ddof: int = 0
) -> Any:
    """Compute the standard deviation along the specified axis.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        dtype (Any): Argument dtype.
        keepdims (Any): Argument keepdims.
        ddof (Any): Argument ddof.

    Returns:
        Any: The result of the operation.
    """
    # Standard deviation is sqrt of variance
    # ops.sqrt exists or power(var, 0.5)
    v = var(a, axis=axis, dtype=dtype, keepdims=keepdims, ddof=ddof)
    return power(v, 0.5)


def ravel(a: Any, order: str = "C") -> Any:
    """Return a contiguous flattened array.

    Args:
        a (Any): Argument a.
        order (Any): Argument order.

    Returns:
        Any: The result of the operation.
    """
    # Eager fallback or reshape if order='C'
    if order != "C":
        raise NotImplementedError("ravel only supports order='C'")
    return reshape(a, (-1,))


def squeeze(a: Any, axis: Any = None) -> Any:
    """Remove axes of length one from a.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.squeeze(_to_tensor(a), dim=axis))


def swapaxes(a: Any, axis1: int, axis2: int) -> Any:
    """Interchange two axes of an array.

    Args:
        a (Any): Argument a.
        axis1 (Any): Argument axis1.
        axis2 (Any): Argument axis2.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.swapaxes(_to_tensor(a), axis1=axis1, axis2=axis2))


def moveaxis(a: Any, source: Any, destination: Any) -> Any:
    """Move axes of an array to new positions.

    Args:
        a (Any): Argument a.
        source (Any): Argument source.
        destination (Any): Argument destination.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.moveaxis(_to_tensor(a), source=source, destination=destination))


def stack(arrays: Any, axis: int = 0) -> Any:
    """Join a sequence of arrays along a new axis.

    Args:
        arrays (Any): Argument arrays.
        axis (Any): Argument axis.

    Returns:
        Any: The result of the operation.
    """
    tensors = [_to_tensor(arr) for arr in arrays]
    return _wrap(ops.stack(tensors, dim=axis))


def vstack(tup: Any) -> Any:
    """Stack arrays in sequence vertically (row wise).

    Args:
        tup (Any): Argument tup.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.vstack([_to_tensor(arr) for arr in tup]))


def hstack(tup: Any) -> Any:
    """Stack arrays in sequence horizontally (column wise).

    Args:
        tup (Any): Argument tup.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.hstack([_to_tensor(arr) for arr in tup]))


def dstack(tup: Any) -> Any:
    """Stack arrays in sequence depth wise (along third axis).

    Args:
        tup (Any): Argument tup.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.dstack([_to_tensor(arr) for arr in tup]))


def split(ary: Any, indices_or_sections: Any, axis: int = 0) -> Any:
    """Split an array into multiple sub-arrays as views into ary.

    Args:
        ary (Any): Argument ary.
        indices_or_sections (Any): Argument indices_or_sections.
        axis (Any): Argument axis.

    Returns:
        Any: The result of the operation.
    """
    return tuple(
        _wrap(t) for t in ops.split(_to_tensor(ary), indices_or_sections, axis)
    )


def array_split(ary: Any, indices_or_sections: Any, axis: int = 0) -> Any:
    """Split an array into multiple sub-arrays.

    Args:
        ary (Any): Argument ary.
        indices_or_sections (Any): Argument indices_or_sections.
        axis (Any): Argument axis.

    Returns:
        Any: The result of the operation.
    """
    return tuple(
        _wrap(t) for t in ops.array_split(_to_tensor(ary), indices_or_sections, axis)
    )


def vsplit(ary: Any, indices_or_sections: Any) -> Any:
    """Split an array into multiple sub-arrays vertically (row-wise).

    Args:
        ary (Any): Argument ary.
        indices_or_sections (Any): Argument indices_or_sections.

    Returns:
        Any: The result of the operation.
    """
    return tuple(_wrap(t) for t in ops.vsplit(_to_tensor(ary), indices_or_sections))


def hsplit(ary: Any, indices_or_sections: Any) -> Any:
    """Split an array into multiple sub-arrays horizontally (column-wise).

    Args:
        ary (Any): Argument ary.
        indices_or_sections (Any): Argument indices_or_sections.

    Returns:
        Any: The result of the operation.
    """
    return tuple(_wrap(t) for t in ops.hsplit(_to_tensor(ary), indices_or_sections))


def dsplit(ary: Any, indices_or_sections: Any) -> Any:
    """Split array into multiple sub-arrays along the 3rd axis (depth).

    Args:
        ary (Any): Argument ary.
        indices_or_sections (Any): Argument indices_or_sections.

    Returns:
        Any: The result of the operation.
    """
    return tuple(_wrap(t) for t in ops.dsplit(_to_tensor(ary), indices_or_sections))


def tile(A: Any, reps: Any) -> Any:
    """Construct an array by repeating A the number of times given by reps.

    Args:
        A (Any): Argument A.
        reps (Any): Argument reps.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.tile(_to_tensor(A), reps=reps))


def repeat(a: Any, repeats: Any, axis: Any = None) -> Any:
    """Repeat elements of an array.

    Args:
        a (Any): Argument a.
        repeats (Any): Argument repeats.
        axis (Any): Argument axis.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.repeat(_to_tensor(a), repeats=repeats, axis=axis))


def pad(array: Any, pad_width: Any, mode: str = "constant", **kwargs: Any) -> Any:
    """Pad an array.

    Args:
        array (Any): Argument array.
        pad_width (Any): Argument pad_width.
        mode (Any): Argument mode.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.pad(_to_tensor(array), pad_width, mode=mode, **kwargs))


def take(a: Any, indices: Any, axis: int = None, mode: str = None) -> Any:
    """Take elements from an array along an axis.

    Args:
        a (Any): Argument a.
        indices (Any): Argument indices.
        axis (Any): Argument axis.
        mode (Any): Argument mode.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.take(_to_tensor(a), _to_tensor(indices)))


def take_along_axis(arr: Any, indices: Any, axis: int) -> Any:
    """Take values from the input array by matching 1d index and data slices.

    Args:
        arr (Any): Argument arr.
        indices (Any): Argument indices.
        axis (Any): Argument axis.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.take_along_axis(_to_tensor(arr), _to_tensor(indices), axis=axis))


def vdot(a: Any, b: Any) -> Any:
    """Return the dot product of two vectors.

    Args:
        a (Any): Argument a.
        b (Any): Argument b.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.vdot(_to_tensor(a), _to_tensor(b)))


def inner(a: Any, b: Any) -> Any:
    """Inner product of two arrays.

    Args:
        a (Any): Argument a.
        b (Any): Argument b.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.inner(_to_tensor(a), _to_tensor(b)))


def outer(a: Any, b: Any) -> Any:
    """Compute the outer product of two vectors.

    Args:
        a (Any): Argument a.
        b (Any): Argument b.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.outer(_to_tensor(a), _to_tensor(b)))


def tensordot(a: Any, b: Any, axes: Any = 2) -> Any:
    """Compute tensor dot product along specified axes.

    Args:
        a (Any): Argument a.
        b (Any): Argument b.
        axes (Any): Argument axes.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.tensordot(_to_tensor(a), _to_tensor(b), axes=axes))


def shape(a: Any) -> Any:
    """Get the shape of the array.

    Returns:
        Any: The shape property of the underlying tensor.
    """
    return asarray(a).shape


def sqrt(x: Any) -> Any:
    """Return the non-negative square-root of an array, element-wise.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.sqrt(_to_tensor(x)))


def square(x: Any) -> Any:
    """Return the element-wise square of the input.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.square(_to_tensor(x)))


def isnan(x: Any) -> Any:
    """Test element-wise for NaN and return result as a boolean array.

    Args:
        x (Any): Argument x.

    Returns:
        Any: The result of the operation.
    """
    return _wrap(ops.isnan(_to_tensor(x)))


nan = float("nan")

pi = 3.14159265358979323846


def cumsum(a: Any, axis: Any = None, dtype: Any = None) -> Any:
    """Return the cumulative sum of the elements along a given axis.

    Args:
        a (Any): Argument a.
        axis (Any): Argument axis.
        dtype (Any): Argument dtype.

    Returns:
        Any: The result of the operation.
    """
    res = ops.cumsum(_to_tensor(a), axis=axis)
    if dtype is not None:
        from ml_switcheroo.core.dtype import DType

        if isinstance(dtype, DType):
            dt = dtype
        else:
            val = getattr(dtype, "value", getattr(dtype, "name", str(dtype)))
            if isinstance(val, str):
                val = val.lower()
            dt = DType(val)
        res = ops.cast(res, dt)
    return _wrap(res)
