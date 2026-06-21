"""Foreign module integration."""

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.tensor import Tensor

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("TorchModule")
class TorchModule(OpDef):
    """TorchModule op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        if args and hasattr(args[0], "shape"):
            return args[0].shape
        return ()


@register_op("Jaxpr")
class Jaxpr(OpDef):
    """Jaxpr op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        if args and hasattr(args[0], "shape"):
            return args[0].shape
        return ()


@register_op("FlaxModule")
class FlaxModule(OpDef):
    """FlaxModule op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        if args and hasattr(args[0], "shape"):
            return args[0].shape
        return ()


@register_op("TFModule")
class TFModule(OpDef):
    """TFModule op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape."""
        if args and hasattr(args[0], "shape"):
            return args[0].shape
        return ()


def torch_to_ir(module: object, *inputs: object, **kwargs: object) -> object:
    """Convert torch module to IR."""
    if config.eager_mode:
        return module(*[i.data if isinstance(i, Tensor) else i for i in inputs], **kwargs)
    from ml_switcheroo_compiler.ops.base import get_op

    return get_op("TorchModule")()(*inputs, module=module, **kwargs)


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
    from ml_switcheroo_compiler.ops.base import get_op

    return get_op("Jaxpr")()(*inputs, call_fn=call_fn, params=params, state=state, **kwargs)


def flax_to_ir(
    module: object, variables: object, method: object, *inputs: object, **kwargs: object
) -> object:
    """Convert flax module to IR."""
    if config.eager_mode:
        fn = getattr(module, method) if method else module.apply
        return fn(variables, *[i.data if isinstance(i, Tensor) else i for i in inputs], **kwargs)
    from ml_switcheroo_compiler.ops.base import get_op

    return get_op("FlaxModule")()(
        *inputs, module=module, variables=variables, method=method, **kwargs
    )


def tf_to_ir(module: object, *inputs: object, **kwargs: object) -> object:
    """Convert tf module to IR."""
    if config.eager_mode:
        return module(*[i.data if isinstance(i, Tensor) else i for i in inputs], **kwargs)
    from ml_switcheroo_compiler.ops.base import get_op

    return get_op("TFModule")()(*inputs, module=module, **kwargs)
