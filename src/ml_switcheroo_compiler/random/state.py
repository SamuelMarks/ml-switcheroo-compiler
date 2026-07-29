"""Random operations."""

from __future__ import annotations

import uuid

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

# In tracing mode, we need to map to OpDefs just like core tensor ops do through get_op
from ml_switcheroo_compiler.ops.base import get_op
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

# It's an OpDef class


def _emit_random_node(
    op_type: str,
    inputs: list[Tensor],
    shape: tuple[int, ...],
    dtype: dtypes.DType,
    attributes: dict | None = None,
) -> Tensor:
    """Execute _emit_random_node.

    Args:
        op_type (str): The op_type parameter for the operation.
        inputs (list[Tensor]): The inputs parameter for the operation.
        shape (tuple[int, ...]): The target shape.
        dtype (dtypes.DType): The target data type.
        attributes (dict | None): The attributes parameter for the operation.

    Returns:
        Tensor: The result.
    """
    if config.eager_mode:
        attrs = dict(attributes) if attributes is not None else {}
        if "shape" not in attrs:
            attrs["shape"] = shape
        if "dtype" not in attrs:
            attrs["dtype"] = getattr(dtype, "value", dtype)
        res = _dispatch_random_eager(op_type.lower(), op_type, *inputs, **attrs)
        return Tensor(res, TensorConfig(shape, dtype, config.default_device))

    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type=op_type,
        inputs=[inp.data.id for inp in inputs],
        attributes=attributes or {},
        shape_metadata=shape,
    )
    global_tracing_state.add_node(node)
    proxy = ProxyTensor(id=out_id, shape=shape, dtype=dtype.value)
    return Tensor(proxy, TensorConfig(shape, dtype, config.default_device))


def PRNGKey(seed: int) -> Tensor:
    """Creates a PRNG key given an integer seed.

    Args:
        seed (int): The random seed.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    if config.eager_mode:
        res = _dispatch_random_eager("prng_key", "PRNGKey", seed)
        return Tensor(res, TensorConfig((2,), dtypes.DType.UInt32, config.default_device))

    # Trace as a creation node
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="PRNGKey",
        inputs=[],
        attributes={"seed": seed},
        shape_metadata=(2,),
    )
    if global_tracing_state.is_tracing:
        global_tracing_state.add_node(node)
    proxy = ProxyTensor(id=out_id, shape=(2,), dtype="uint32")
    return Tensor(proxy, TensorConfig((2,), dtypes.DType.UInt32, config.default_device))


def split(key: Tensor, num: int = 2) -> Tensor:
    """Splits a PRNG key into num new keys.

    Args:
        key (Tensor): The PRNG key.
        num (int): The num parameter for the operation.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    if config.eager_mode:
        res = _dispatch_random_eager("split", "RandomSplit", key.data, num=num)
        return Tensor(res, TensorConfig((num, 2), dtypes.DType.UInt32, config.default_device))
    return _emit_random_node("RandomSplit", [key], (num, 2), dtypes.DType.UInt32, {"num": num})


def fold_in(key: Tensor, data: int) -> Tensor:
    """Folds in data to a PRNG key to derive a new key.

    Args:
        key (Tensor): The PRNG key.
        data (int): The data parameter for the operation.

    Returns:
        Tensor: A tensor containing the result of the operation.
    """
    if config.eager_mode:
        res = _dispatch_random_eager("fold_in", "RandomFoldIn", key.data, data=data)
        return Tensor(res, TensorConfig((2,), dtypes.DType.UInt32, config.default_device))
    return _emit_random_node("RandomFoldIn", [key], (2,), dtypes.DType.UInt32, {"data": data})


def _dispatch_random_eager(func_name: str, op_name: str, *args: object, **kwargs: object) -> object:
    """Helper to dispatch random functions in eager mode."""
    backend = get_active_backend()
    return backend.execute_op(op_name, *args, **kwargs)


def _dispatch_random(func_name: str, *args: object, **kwargs: object) -> object:
    """Evaluate and process the dispatch random operation.

    Args:
        func_name (str): Required parameter for func_name.
        *args (Any): Variable positional arguments.
        **kwargs (Any): Arbitrary keyword arguments.

    Returns:
        object: The evaluated or processed output.
    """
    op_name = "".join(word.capitalize() for word in func_name.split("_"))
    if config.eager_mode:
        return _dispatch_random_eager(func_name, op_name, *args, **kwargs)

    op_cls = get_op(op_name)
    if op_cls:
        return op_cls()(*args, **kwargs)
    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    return _emit_shape_node(op_name, list(args), kwargs, (), "float32")


def key(*args: object, **kwargs: object) -> object:
    """Execute key."""
    return _dispatch_random("key", *args, **kwargs)


def key_data(*args: object, **kwargs: object) -> object:
    """Execute key_data."""
    return _dispatch_random("key_data", *args, **kwargs)


def key_impl(*args: object, **kwargs: object) -> object:
    """Execute key_impl."""
    return _dispatch_random("key_impl", *args, **kwargs)


def wrap_key_data(*args: object, **kwargs: object) -> object:
    """Execute wrap_key_data."""
    return _dispatch_random("wrap_key_data", *args, **kwargs)


def clone(*args: object, **kwargs: object) -> object:
    """Execute clone."""
    return _dispatch_random("clone", *args, **kwargs)


def bits(*args: object, **kwargs: object) -> object:
    """Execute bits."""
    return _dispatch_random("bits", *args, **kwargs)


def rng_bit_generator(key: object, shape: object, dtype: object = None) -> object:
    """Generates random bits.

    Args:
        key (object): PRNGKey.
        shape (object): Shape.
        dtype (object): Data type.

    Returns:
        object: Random bits.
    """
    return _dispatch_random("rng_bit_generator", key, shape=shape, dtype=dtype)


def rng_uniform(a: object, b: object, shape: object, dtype: object = None) -> object:
    """Generates uniform random values.

    Args:
        a (object): Lower bound.
        b (object): Upper bound.
        shape (object): Shape.
        dtype (object): Data type.

    Returns:
        object: Random values.
    """
    return _dispatch_random("rng_uniform", a, b, shape=shape, dtype=dtype)


def _get_numpy_rng(*args: object, **kwargs: object) -> object:
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    backend_cls = BackendRegistry.get("numpy")
    return backend_cls.get_numpy_rng(*args, **kwargs)
