"""Foreign module integration."""

from ml_switcheroo_compiler.core.config import config  # pragma: no cover
from ml_switcheroo_compiler.core.tensor import Tensor  # pragma: no cover

from ml_switcheroo_compiler.ops.base import OpDef, register_op  # pragma: no cover


@register_op("TorchModule")  # pragma: no cover
class TorchModule(OpDef):  # pragma: no cover
    """TorchModule op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:  # pragma: no cover
        """Infer shape."""
        if args and hasattr(args[0], "shape"):  # pragma: no cover
            return args[0].shape  # pragma: no cover
        return ()  # pragma: no cover


@register_op("Jaxpr")  # pragma: no cover
class Jaxpr(OpDef):  # pragma: no cover
    """Jaxpr op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:  # pragma: no cover
        """Infer shape."""
        if args and hasattr(args[0], "shape"):  # pragma: no cover
            return args[0].shape  # pragma: no cover
        return ()  # pragma: no cover


@register_op("FlaxModule")  # pragma: no cover
class FlaxModule(OpDef):  # pragma: no cover
    """FlaxModule op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:  # pragma: no cover
        """Infer shape."""
        if args and hasattr(args[0], "shape"):  # pragma: no cover
            return args[0].shape  # pragma: no cover
        return ()  # pragma: no cover


@register_op("TFModule")  # pragma: no cover
class TFModule(OpDef):  # pragma: no cover
    """TFModule op."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:  # pragma: no cover
        """Infer shape."""
        if args and hasattr(args[0], "shape"):  # pragma: no cover
            return args[0].shape  # pragma: no cover
        return ()  # pragma: no cover


def torch_to_ir(module: object, *inputs: object, **kwargs: object) -> object:  # pragma: no cover
    """Convert torch module to IR."""
    if config.eager_mode:  # pragma: no cover
        return module(
            *[i.data if isinstance(i, Tensor) else i for i in inputs], **kwargs
        )  # pragma: no cover
    from ml_switcheroo_compiler.ops.base import get_op  # pragma: no cover

    return get_op("TorchModule")()(*inputs, module=module, **kwargs)  # pragma: no cover


def jaxpr_to_ir(  # pragma: no cover
    call_fn: object, params: object, state: object, *inputs: object, **kwargs: object
) -> object:
    """Convert jaxpr to IR."""
    if config.eager_mode:  # pragma: no cover
        if state is not None:  # pragma: no cover
            return call_fn(  # pragma: no cover
                {"params": params, **state},
                *[i.data if isinstance(i, Tensor) else i for i in inputs],
                **kwargs,
            )
        return call_fn(  # pragma: no cover
            {"params": params}, *[i.data if isinstance(i, Tensor) else i for i in inputs], **kwargs
        )
    from ml_switcheroo_compiler.ops.base import get_op  # pragma: no cover

    return get_op("Jaxpr")()(
        *inputs, call_fn=call_fn, params=params, state=state, **kwargs
    )  # pragma: no cover


def flax_to_ir(  # pragma: no cover
    module: object, variables: object, method: object, *inputs: object, **kwargs: object
) -> object:
    """Convert flax module to IR."""
    if config.eager_mode:  # pragma: no cover
        fn = getattr(module, method) if method else module.apply  # pragma: no cover
        return fn(
            variables, *[i.data if isinstance(i, Tensor) else i for i in inputs], **kwargs
        )  # pragma: no cover
    from ml_switcheroo_compiler.ops.base import get_op  # pragma: no cover

    return get_op("FlaxModule")()(  # pragma: no cover
        *inputs, module=module, variables=variables, method=method, **kwargs
    )


def tf_to_ir(module: object, *inputs: object, **kwargs: object) -> object:  # pragma: no cover
    """Convert tf module to IR."""
    if config.eager_mode:  # pragma: no cover
        return module(
            *[i.data if isinstance(i, Tensor) else i for i in inputs], **kwargs
        )  # pragma: no cover
    from ml_switcheroo_compiler.ops.base import get_op  # pragma: no cover

    return get_op("TFModule")()(*inputs, module=module, **kwargs)  # pragma: no cover
