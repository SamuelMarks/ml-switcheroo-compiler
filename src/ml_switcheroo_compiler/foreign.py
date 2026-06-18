"""Foreign module integration."""

from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node
from ml_switcheroo_compiler.core.config import config


def torch_to_ir(module: object, *inputs: object, **kwargs: object) -> object:
    """Convert torch module to IR."""
    if config.eager_mode:
        return module(*[i.data if isinstance(i, Tensor) else i for i in inputs], **kwargs)
    return _emit_shape_node(
        "TorchModule", list(inputs), {"module": module, "kwargs": kwargs}, (), inputs[0].dtype
    )


def jaxpr_to_ir(
    call_fn: object, params: object, state: object, *inputs: object, **kwargs: object
) -> object:
    """Convert jaxpr to IR."""
    if config.eager_mode:
        if state is not None:
            return call_fn(
                {"params": params, **state},
                *[i.data if isinstance(i, Tensor) else i for i in inputs],
                **kwargs,
            )
        return call_fn(
            {"params": params}, *[i.data if isinstance(i, Tensor) else i for i in inputs], **kwargs
        )
    return _emit_shape_node(
        "Jaxpr",
        list(inputs),
        {"call_fn": call_fn, "params": params, "state": state, "kwargs": kwargs},
        (),
        inputs[0].dtype,
    )


def flax_to_ir(
    module: object, variables: object, method: object, *inputs: object, **kwargs: object
) -> object:
    """Convert flax module to IR."""
    if config.eager_mode:
        fn = getattr(module, method) if method else module.apply
        return fn(variables, *[i.data if isinstance(i, Tensor) else i for i in inputs], **kwargs)
    return _emit_shape_node(
        "FlaxModule",
        list(inputs),
        {"module": module, "variables": variables, "method": method, "kwargs": kwargs},
        (),
        inputs[0].dtype,
    )


def tf_to_ir(module: object, *inputs: object, **kwargs: object) -> object:
    """Convert tf module to IR."""
    if config.eager_mode:
        return module(*[i.data if isinstance(i, Tensor) else i for i in inputs], **kwargs)
    return _emit_shape_node(
        "TFModule", list(inputs), {"module": module, "kwargs": kwargs}, (), inputs[0].dtype
    )
