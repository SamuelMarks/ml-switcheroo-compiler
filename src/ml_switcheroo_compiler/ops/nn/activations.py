# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Activations and advanced NN operations."""

from typing import Any

# Base logic implementation
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef
from ml_switcheroo_compiler.ops.registry import get_op, register_op
from ml_switcheroo_compiler.ops.shape.joining import concatenate


def crelu(features: Any, axis: Any = -1, name: Any = None) -> Any:
    """Compute Concatenated ReLU.

    Args:
        features (object): The features parameter.
        axis (object): The axis parameter.
        name (object): The name parameter.

    Returns: Any: Result.
    """
    negative = get_op("Negative")()

    from ml_switcheroo_compiler.ops.binary import maximum

    return maximum(concatenate([features, negative(features)], axis=axis), 0.0)


def isotonic_regression(y: Any, sample_weights: Any = None, increasing: Any = True, name: Any = None) -> Any:
    """Solves isotonic regression problems.

    Args:
        y (object): The y parameter.
        sample_weights (object): The sample_weights parameter.
        increasing (object): The increasing parameter.
        name (object): The name parameter.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("IsotonicRegression", y, sample_weights=sample_weights, increasing=increasing, name=name)

    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    out1 = _emit_shape_node("IsotonicRegression", [y], {"sample_weights": sample_weights, "increasing": increasing, "name": name}, getattr(y, "shape", ()), getattr(y, "dtype", "float32"))
    return out1, Tensor(None, TensorConfig(getattr(y, "shape", ()), "int32", "cpu"))


__all__ = ["softplus", "softmax", "log_softmax", "sigmoid", "one_hot", "rrelu"]


def softplus(x: Any) -> Any:
    """Softplus activation.

    Args:
        x (object): The x parameter.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.unary import exp, log1p

    return log1p(exp(x))


@register_op("Softmax")
class Softmax(OpDef):
    """Operator Softmax."""

    op_name = "Softmax"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


def softmax(x: Any, axis: int = -1, *args: Any, **kwargs: Any) -> Any:
    """Softmax activation.

    Args:
        x (object): The x parameter.
        axis (int): The axis parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
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

    op_name = "LogSoftmax"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


def log_softmax(x: Any, axis: int = -1, *args: Any, **kwargs: Any) -> Any:
    """LogSoftmax activation.

    Args:
        x (object): The x parameter.
        axis (int): The axis parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
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

    op_name = "Sigmoid"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


def sigmoid(x: Any, *args: Any, **kwargs: Any) -> Any:
    """Sigmoid activation.

    Args:
        x (object): The x parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
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

    op_name = "OneHot"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


def one_hot(indices: Any, depth: int, *args: Any, **kwargs: Any) -> Any:
    """OneHot encoding.

    Args:
        indices (object): The indices parameter.
        depth (int): The depth parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
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

    op_name = "Rrelu"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


def rrelu(x: Any, *args: Any, **kwargs: Any) -> Any:
    """Rrelu activation.

    Args:
        x (object): The x parameter.
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
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

    op_name = "HardSilu"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        x = args[0] if len(args) > 0 else None
        return getattr(x, "shape", ())


@register_op("HardSwish")
class HardSwish(OpDef):
    """HardSwish activation."""

    op_name = "HardSwish"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        x = args[0] if len(args) > 0 else None
        return getattr(x, "shape", ())


@register_op("Squareplus")
class Squareplus(OpDef):
    """Squareplus activation."""

    op_name = "Squareplus"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        x = args[0] if len(args) > 0 else None
        return getattr(x, "shape", ())


def hard_silu(*args: Any, **kwargs: Any) -> Any:
    """Hard SiLU activation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("HardSilu", *args, **kwargs)


def hard_swish(*args: Any, **kwargs: Any) -> Any:
    """Hard Swish activation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("HardSwish", *args, **kwargs)


def mish(*args: Any, **kwargs: Any) -> Any:
    """Mish activation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Mish", *args, **kwargs)


def squareplus(*args: Any, **kwargs: Any) -> Any:
    """Squareplus activation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Squareplus", *args, **kwargs)
