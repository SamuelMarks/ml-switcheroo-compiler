"""Activations and advanced NN operations."""

# Base logic implementation
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef
from ml_switcheroo_compiler.ops.registry import get_op, register_op
from ml_switcheroo_compiler.ops.shape.joining import concatenate


def crelu(features: object, axis: object = -1, name: object = None) -> object:
    """Computes Concatenated ReLU."""
    negative = get_op("Negative")()

    from ml_switcheroo_compiler.ops.binary import maximum

    return maximum(concatenate([features, negative(features)], dim=axis), 0.0)


def isotonic_regression(y: object, sample_weights: object = None, increasing: object = True, name: object = None) -> object:
    """Solves isotonic regression problems."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("IsotonicRegression", y, sample_weights=sample_weights, increasing=increasing, name=name)

    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out1 = _emit_shape_node("IsotonicRegression", [y], {"sample_weights": sample_weights, "increasing": increasing, "name": name}, getattr(y, "shape", ()), getattr(y, "dtype", "float32"))
    return out1, Tensor(None, TensorConfig(getattr(y, "shape", ()), "int32", "cpu"))


__all__ = ["softplus", "softmax", "log_softmax", "sigmoid", "one_hot", "rrelu"]


def softplus(x: object) -> object:
    """Softplus activation."""
    from ml_switcheroo_compiler.ops.unary import exp, log1p

    return log1p(exp(x))


@register_op("Softmax")
class Softmax(OpDef):
    """Operator Softmax."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        if args and hasattr(args[0], "shape"):
            return args[0].shape
        return ()


def softmax(x: object, axis: int = -1, *args: object, **kwargs: object) -> object:
    """Softmax activation."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Softmax", x, axis=axis)
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    return emit_ir_node(None, "Softmax", [x], getattr(x, "shape_metadata", None), {"axis": axis})


@register_op("LogSoftmax")
class LogSoftmax(OpDef):
    """Operator LogSoftmax."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        if args and hasattr(args[0], "shape"):
            return args[0].shape
        return ()


def log_softmax(x: object, axis: int = -1, *args: object, **kwargs: object) -> object:
    """LogSoftmax activation."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("LogSoftmax", x, axis=axis)
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    return emit_ir_node(None, "LogSoftmax", [x], getattr(x, "shape_metadata", None), {"axis": axis})


@register_op("Sigmoid")
class Sigmoid(OpDef):
    """Operator Sigmoid."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        if args and hasattr(args[0], "shape"):
            return args[0].shape
        return ()


def sigmoid(x: object, *args: object, **kwargs: object) -> object:
    """Sigmoid activation."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Sigmoid", x)
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    return emit_ir_node(None, "Sigmoid", [x], getattr(x, "shape_metadata", None), {})


@register_op("OneHot")
class OneHot(OpDef):
    """Operator OneHot."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        if args and hasattr(args[0], "shape"):
            depth = kwargs.get("depth", args[1] if len(args) > 1 else 1)
            return args[0].shape + (depth,)
        return ()


def one_hot(indices: object, depth: int, *args: object, **kwargs: object) -> object:
    """OneHot encoding."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("OneHot", indices, depth, *args, **kwargs)
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    kwargs["depth"] = depth
    shape = getattr(indices, "shape", ()) + (depth,)
    import copy

    meta = copy.copy(getattr(indices, "shape_metadata", None))
    if meta is not None:
        meta.shape = shape
    return emit_ir_node(None, "OneHot", [indices], meta, kwargs)


@register_op("Rrelu")
class Rrelu(OpDef):
    """Operator Rrelu."""

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        if args and hasattr(args[0], "shape"):
            return args[0].shape
        return ()


def rrelu(x: object, *args: object, **kwargs: object) -> object:
    """Rrelu activation."""
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Rrelu", x, *args, **kwargs)
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    return emit_ir_node(None, "Rrelu", [x], getattr(x, "shape_metadata", None), kwargs)


@register_op("HardSilu")
class HardSilu(OpDef):
    """HardSilu activation."""

    op_name = "HardSilu"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape if args and hasattr(args[0], "shape") else ()


@register_op("HardSwish")
class HardSwish(OpDef):
    """HardSwish activation."""

    op_name = "HardSwish"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape if args and hasattr(args[0], "shape") else ()


@register_op("Squareplus")
class Squareplus(OpDef):
    """Squareplus activation."""

    op_name = "Squareplus"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return args[0].shape if args and hasattr(args[0], "shape") else ()
