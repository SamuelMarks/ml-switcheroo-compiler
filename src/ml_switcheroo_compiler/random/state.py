"""Random operations."""

from __future__ import annotations

from __future__ import annotations
from ml_switcheroo_compiler.backends.registry import get_active_backend
import uuid
import numpy as np
from ml_switcheroo_ir import LogicalNode
from ml_switcheroo_compiler.core import dtype as dtypes
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, _tracer


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
        raise NotImplementedError(f"{op_type} not implemented in eager mode via _emit_random_node")

    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type=op_type,
        inputs=[inp.data.id for inp in inputs],
        attributes=attributes or {},
        shape_metadata=shape,
    )
    _tracer.add_node(node)
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
        return Tensor(
            np.array([0, seed], dtype=np.uint32),
            TensorConfig((2,), dtypes.DType.UInt32, config.default_device),
        )

    # Trace as a creation node
    out_id = str(uuid.uuid4())
    node = LogicalNode(
        id=out_id,
        op_type="PRNGKey",
        inputs=[],
        attributes={"seed": seed},
        shape_metadata=(2,),
    )
    if _tracer.is_tracing:  # pragma: no branch
        _tracer.add_node(node)
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
        return Tensor(
            np.random.randint(0, 2**32, size=(num, 2), dtype=np.uint32),
            TensorConfig((num, 2), dtypes.DType.UInt32, config.default_device),
        )
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
        return Tensor(
            np.array([key.data[0] + data, key.data[1]], dtype=np.uint32),
            TensorConfig((2,), dtypes.DType.UInt32, config.default_device),
        )
    return _emit_random_node("RandomFoldIn", [key], (2,), dtypes.DType.UInt32, {"data": data})


def _dispatch_random(func_name: str, *args: object, **kwargs: object) -> object:
    """Function docstring.

    Args:
        func_name: Arg.
        args: Arg.
        kwargs: Arg.
    """
    if config.eager_mode:
        backend = get_active_backend()
        if hasattr(backend.module, "random") and hasattr(
            backend.module.random, func_name
        ):  # pragma: no branch
            return getattr(backend.module.random, func_name)(*args, **kwargs)  # pragma: no cover
        raise NotImplementedError(  # pragma: no cover
            f"{func_name} is not supported in eager mode without backend support."
        )
    raise NotImplementedError(f"{func_name} is not fully supported in tracing mode.")


def key(*args: object, **kwargs: object) -> object:
    """Execute key."""
    return _dispatch_random("key", *args, **kwargs)


def key_data(*args: object, **kwargs: object) -> object:
    """Execute key_data."""
    if config.eager_mode:
        backend = get_active_backend()
        if hasattr(backend.module, "random") and hasattr(
            backend.module.random, "key_data"
        ):  # pragma: no branch
            return backend.module.random.key_data(*args, **kwargs)  # pragma: no cover
        raise NotImplementedError(  # pragma: no cover
            "key_data is not supported in eager mode without backend support."
        )
    raise NotImplementedError("key_data is not fully supported in tracing mode.")


def key_impl(*args: object, **kwargs: object) -> object:
    """Execute key_impl."""
    if config.eager_mode:
        backend = get_active_backend()
        if hasattr(backend.module, "random") and hasattr(
            backend.module.random, "key_impl"
        ):  # pragma: no branch
            return backend.module.random.key_impl(*args, **kwargs)  # pragma: no cover
        raise NotImplementedError(  # pragma: no cover
            "key_impl is not supported in eager mode without backend support."
        )
    raise NotImplementedError("key_impl is not fully supported in tracing mode.")


def wrap_key_data(*args: object, **kwargs: object) -> object:
    """Execute wrap_key_data."""
    if config.eager_mode:
        backend = get_active_backend()
        if hasattr(backend.module, "random") and hasattr(
            backend.module.random, "wrap_key_data"
        ):  # pragma: no branch
            return backend.module.random.wrap_key_data(*args, **kwargs)  # pragma: no cover
        raise NotImplementedError(  # pragma: no cover
            "wrap_key_data is not supported in eager mode without backend support."
        )
    raise NotImplementedError("wrap_key_data is not fully supported in tracing mode.")


def clone(*args: object, **kwargs: object) -> object:
    """Execute clone."""
    return _dispatch_random("clone", *args, **kwargs)


def bits(*args: object, **kwargs: object) -> object:
    """Execute bits."""
    return _dispatch_random("bits", *args, **kwargs)
