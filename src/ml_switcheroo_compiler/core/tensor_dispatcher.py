"""Tensor dispatcher."""

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.tracing import ProxyTensor, _tracer
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node
from ml_switcheroo_compiler.ops.binary import add, subtract
import uuid

from ml_switcheroo_ir import LogicalNode

UUID_LENGTH = 6


def dispatch_getitem(tensor: object, key: object) -> object:
    """Dispatch getitem."""
    arr = tensor.__array__()
    if hasattr(key, "data"):
        key = key.data
    elif isinstance(key, tuple):
        key = tuple(getattr(k, "data", k) for k in key)

    res = arr[key]
    if config.eager_mode:
        from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

        return Tensor(res, TensorConfig(getattr(res, "shape", ()), tensor.dtype, tensor.device))

    nid = f"getitem_{uuid.uuid4().hex[:UUID_LENGTH]}"
    input_id = getattr(tensor.data, "id", "const")

    node = LogicalNode(
        id=nid,
        op_type="GetItem",
        inputs=[input_id],
        attributes={"key": str(key)},
        shape_metadata=(),
    )
    if _tracer.is_tracing:
        _tracer.add_node(node)
    else:
        raise RuntimeError("Cannot add node: not currently tracing.")
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    return Tensor(
        ProxyTensor(nid, (), tensor.dtype.value), TensorConfig((), tensor.dtype, tensor.device)
    )


def dispatch_setitem(tensor: object, key: object, value: object) -> None:
    """Dispatch setitem."""
    if config.eager_mode:
        val = getattr(value, "data", value)
        tensor.data[key] = val
    else:
        msg = (
            "Tensor object does not support item assignment in tracing "
            "mode. Use .at[...].set(...) instead."
        )
        raise TypeError(msg)


def dispatch_assign(variable: object, value: object) -> object:
    """Dispatch assign."""
    if config.eager_mode:
        backend = get_active_backend()
        variable._data = backend.execute_op("Assign", variable._data, value.data)
    else:
        _emit_shape_node("Assign", [variable, value], {}, variable.shape, variable.dtype)
    return variable


def dispatch_assign_add(variable: object, value: object) -> object:
    """Dispatch assign_add."""
    if config.eager_mode:  # pragma: no branch
        backend = get_active_backend()  # pragma: no cover
        variable._data = backend.execute_op("Add", variable._data, value.data)  # pragma: no cover
    else:
        new_val = add(variable, value)
        _emit_shape_node("Assign", [variable, new_val], {}, variable.shape, variable.dtype)
    return variable


def dispatch_assign_sub(variable: object, value: object) -> object:
    """Dispatch assign_sub."""
    if config.eager_mode:  # pragma: no branch
        backend = get_active_backend()  # pragma: no cover
        variable._data = backend.execute_op(
            "Subtract", variable._data, value.data
        )  # pragma: no cover
    else:
        new_val = subtract(variable, value)
        _emit_shape_node("Assign", [variable, new_val], {}, variable.shape, variable.dtype)
    return variable


def dispatch_backward(tensor: object, *args: object, **kwargs: object) -> object:
    """Dispatch backward."""
    from ml_switcheroo_compiler.grad import backward

    return backward(tensor, *args, **kwargs)


def dispatch_reshape(tensor: object, shape: object) -> object:
    """Dispatch reshape."""
    from ml_switcheroo_compiler.ops.shape import reshape

    return reshape(tensor, shape)


def dispatch_eval(tensor: object) -> object:
    """Dispatch eval."""
    if config.eager_mode or not hasattr(tensor.data, "id"):
        return tensor

    if _tracer.is_tracing and _tracer.active_graph:
        graph = _tracer.active_graph
        if tensor.data.id not in graph.outputs:
            graph.outputs.append(tensor.data.id)
    return tensor
