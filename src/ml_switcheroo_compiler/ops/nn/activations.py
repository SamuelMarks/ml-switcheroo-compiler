# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Activations and advanced NN operations."""

# Base logic implementation
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef
from ml_switcheroo_compiler.ops.registry import get_op, register_op
from ml_switcheroo_compiler.ops.shape.joining import concatenate


def crelu(features: object, axis: object = -1, name: object = None) -> object:
    """Compute Concatenated ReLU.

    Args:
        features (object): The features parameter.
        axis (object): The axis parameter.
        name (object): The name parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    negative: object = get_op("Negative")()

    from ml_switcheroo_compiler.ops.binary import maximum

    return maximum(concatenate([features, negative(features)], axis=axis), 0.0)


def isotonic_regression(y: object, sample_weights: object = None, increasing: object = True, name: object = None) -> object:
    """Solves isotonic regression problems.

    Args:
        y (object): The y parameter.
        sample_weights (object): The sample_weights parameter.
        increasing (object): The increasing parameter.
        name (object): The name parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("IsotonicRegression", y, sample_weights=sample_weights, increasing=increasing, name=name)

    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out1: object = _emit_shape_node("IsotonicRegression", [y], {"sample_weights": sample_weights, "increasing": increasing, "name": name}, getattr(y, "shape", ()), getattr(y, "dtype", "float32"))
    return out1, Tensor(None, TensorConfig(getattr(y, "shape", ()), "int32", "cpu"))


__all__ = ["softplus", "softmax", "log_softmax", "sigmoid", "one_hot", "rrelu"]


def softplus(x: object) -> object:
    """Softplus activation.

    Args:
        x (object): The x parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.unary import exp, log1p

    return log1p(exp(x))


@register_op("Softmax")
class Softmax(OpDef):
    """Operator Softmax."""

    op_name: object = "Softmax"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes: object = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res: object = shapes[0]
        for s in shapes[1:]:
            res: object = broadcast_shapes(res, s)
        return res


def softmax(x: object, axis: int = -1, *args: object, **kwargs: object) -> object:
    """Softmax activation.

    Args:
        x (object): The x parameter.
        axis (int): The axis parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Softmax", x, axis=axis)
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    return emit_ir_node(None, "Softmax", [x], getattr(x, "shape_metadata", None), {"axis": axis})


@register_op("LogSoftmax")
class LogSoftmax(OpDef):
    """Operator LogSoftmax."""

    op_name: object = "LogSoftmax"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes: object = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res: object = shapes[0]
        for s in shapes[1:]:
            res: object = broadcast_shapes(res, s)
        return res


def log_softmax(x: object, axis: int = -1, *args: object, **kwargs: object) -> object:
    """LogSoftmax activation.

    Args:
        x (object): The x parameter.
        axis (int): The axis parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("LogSoftmax", x, axis=axis)
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    return emit_ir_node(None, "LogSoftmax", [x], getattr(x, "shape_metadata", None), {"axis": axis})


@register_op("Sigmoid")
class Sigmoid(OpDef):
    """Operator Sigmoid."""

    op_name: object = "Sigmoid"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes: object = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res: object = shapes[0]
        for s in shapes[1:]:
            res: object = broadcast_shapes(res, s)
        return res


def sigmoid(x: object, *args: object, **kwargs: object) -> object:
    """Sigmoid activation.

    Args:
        x (object): The x parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Sigmoid", x)
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    return emit_ir_node(None, "Sigmoid", [x], getattr(x, "shape_metadata", None), {})


@register_op("OneHot")
class OneHot(OpDef):
    """Operator OneHot."""

    op_name: object = "OneHot"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes: object = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res: object = shapes[0]
        for s in shapes[1:]:
            res: object = broadcast_shapes(res, s)
        return res


def one_hot(indices: object, depth: int, *args: object, **kwargs: object) -> object:
    """OneHot encoding.

    Args:
        indices (object): The indices parameter.
        depth (int): The depth parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("OneHot", indices, depth, *args, **kwargs)
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    kwargs["depth"] = depth
    shape: object = getattr(indices, "shape", ()) + (depth,)
    import copy

    meta: object = copy.copy(getattr(indices, "shape_metadata", None))
    if meta is not None:
        meta.shape = shape
    return emit_ir_node(None, "OneHot", [indices], meta, kwargs)


@register_op("Rrelu")
class Rrelu(OpDef):
    """Operator Rrelu."""

    op_name: object = "Rrelu"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes: object = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res: object = shapes[0]
        for s in shapes[1:]:
            res: object = broadcast_shapes(res, s)
        return res


def rrelu(x: object, *args: object, **kwargs: object) -> object:
    """Rrelu activation.

    Args:
        x (object): The x parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Rrelu", x, *args, **kwargs)
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    return emit_ir_node(None, "Rrelu", [x], getattr(x, "shape_metadata", None), kwargs)


@register_op("HardSilu")
class HardSilu(OpDef):
    """HardSilu activation."""

    op_name: object = "HardSilu"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        x: object = args[0] if len(args) > 0 else None
        return getattr(x, "shape", ())


@register_op("HardSwish")
class HardSwish(OpDef):
    """HardSwish activation."""

    op_name: object = "HardSwish"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        x: object = args[0] if len(args) > 0 else None
        return getattr(x, "shape", ())


@register_op("Squareplus")
class Squareplus(OpDef):
    """Squareplus activation."""

    op_name: object = "Squareplus"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        x: object = args[0] if len(args) > 0 else None
        return getattr(x, "shape", ())


def hard_silu(*args: object, **kwargs: object) -> object:
    """Hard SiLU activation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("HardSilu", *args, **kwargs)


def hard_swish(*args: object, **kwargs: object) -> object:
    """Hard Swish activation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("HardSwish", *args, **kwargs)


def mish(*args: object, **kwargs: object) -> object:
    """Mish activation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Mish", *args, **kwargs)


def squareplus(*args: object, **kwargs: object) -> object:
    """Squareplus activation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Squareplus", *args, **kwargs)
