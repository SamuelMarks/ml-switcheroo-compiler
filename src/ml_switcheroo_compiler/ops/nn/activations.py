"""Activations and advanced NN operations."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.registry import EagerTensorProtocol, as_int, as_str
from ml_switcheroo_compiler.core.shape import TensorOrScalar

# Base logic implementation
from ml_switcheroo_compiler.core.tensor import Tensor, as_tensor_obj
from ml_switcheroo_compiler.ops.base import OpDef
from ml_switcheroo_compiler.ops.registry import get_op, register_op
from ml_switcheroo_compiler.ops.shape.joining import concatenate


def crelu(features: TensorOrScalar, axis: TensorOrScalar = -1, name: TensorOrScalar = None) -> TensorOrScalar:
    """Compute Concatenated ReLU.

    Args:
        features (TensorOrScalar): The features parameter.
        axis (TensorOrScalar): The axis parameter.
        name (TensorOrScalar): The name parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    negative = get_op("Negative")()

    from ml_switcheroo_compiler.ops.binary import maximum

    return maximum(concatenate([as_tensor_obj(features), negative(features)], axis=as_int(axis)), 0.0)


def isotonic_regression(y: TensorOrScalar, sample_weights: TensorOrScalar = None, increasing: TensorOrScalar = True, name: TensorOrScalar = None) -> TensorOrScalar:
    """Solves isotonic regression problems.

    Args:
        y (TensorOrScalar): The y parameter.
        sample_weights (TensorOrScalar): The sample_weights parameter.
        increasing (TensorOrScalar): The increasing parameter.
        name (TensorOrScalar): The name parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("IsotonicRegression", y, sample_weights=sample_weights, increasing=increasing, name=name)

    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    attrs = {"sample_weights": sample_weights, "increasing": increasing, "name": name}
    out1 = _emit_shape_node("IsotonicRegression", [y], attrs, getattr(y, "shape", ()), getattr(y, "dtype", "float32"))
    out2 = _emit_shape_node("IsotonicRegressionSegments", [y], attrs, getattr(y, "shape", ()), "int32")
    return out1, out2


__all__ = ["softplus", "softmax", "log_softmax", "sigmoid", "one_hot", "rrelu"]


def softplus(x: Tensor) -> Tensor:
    """Softplus activation.

    Args:
        x (Tensor): The x parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.unary import exp, log1p

    return log1p(exp(x))


@register_op("Softmax")
class Softmax(OpDef):
    """Operator Softmax."""

    op_name = "Softmax"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
        *args (TensorOrScalar): Positional args.
        **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


def softmax(x: TensorOrScalar, axis: int = -1, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Softmax activation.

    Args:
        x (TensorOrScalar): The x parameter.
        axis (int): The axis parameter.
        *args (TensorOrScalar): Positional args.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Softmax", x, axis=axis)
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    return emit_ir_node(None, "Softmax", [as_str(x)], getattr(x, "shape_metadata", None), {"axis": axis})


@register_op("LogSoftmax")
class LogSoftmax(OpDef):
    """Operator LogSoftmax."""

    op_name = "LogSoftmax"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


def log_softmax(x: TensorOrScalar, axis: int = -1, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """LogSoftmax activation.

    Args:
        x (TensorOrScalar): The x parameter.
        axis (int): The axis parameter.
        *args (TensorOrScalar): Positional args.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("LogSoftmax", x, axis=axis)
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    return emit_ir_node(None, "LogSoftmax", [as_str(x)], getattr(x, "shape_metadata", None), {"axis": axis})


@register_op("Sigmoid")
class Sigmoid(OpDef):
    """Operator Sigmoid."""

    op_name = "Sigmoid"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


def sigmoid(x: Tensor | str, *args: int | float | str | None, **kwargs: int | float | str | None) -> Tensor:
    """Sigmoid activation.

    Args:
        x (Tensor | str): The x parameter.
        *args (int | float | str | None): Positional args.
        **kwargs (int | float | str | None): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return as_tensor_obj(get_active_backend().execute_op("Sigmoid", x))
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    return as_tensor_obj(emit_ir_node(None, "Sigmoid", [str(x)], getattr(x, "shape_metadata", None), {}))


@register_op("OneHot")
class OneHot(OpDef):
    """Operator OneHot."""

    op_name = "OneHot"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes  # noqa: F401

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        indices_shape = shapes[0] if shapes else ()
        depth = kwargs.get("depth", args[1] if len(args) > 1 else 1)
        axis = kwargs.get("axis", -1)
        if axis == -1:
            return indices_shape + (depth,)
        else:
            return indices_shape[: int(as_int(axis))] + (depth,) + indices_shape[int(as_int(axis)) :]


def one_hot(indices: TensorOrScalar, depth: int, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """OneHot encoding.

    Args:
        indices (TensorOrScalar): The indices parameter.
        depth (int): The depth parameter.
        *args (TensorOrScalar): Positional args.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
            tuple[int, ...]: Result.
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
    return emit_ir_node(None, "OneHot", [as_str(indices)], meta, kwargs)


@register_op("Rrelu")
class Rrelu(OpDef):
    """Operator Rrelu."""

    op_name = "Rrelu"

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer shape.

        Args:
            *args (TensorOrScalar): Positional args.
            **kwargs (TensorOrScalar): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [getattr(a, "shape", ()) for a in args if hasattr(a, "shape")]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        return res


def rrelu(x: TensorOrScalar, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Rrelu activation.

    Args:
        x (TensorOrScalar): The x parameter.
        *args (TensorOrScalar): Positional args.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Rrelu", x, *args, **kwargs)
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    return emit_ir_node(None, "Rrelu", [as_str(x)], getattr(x, "shape_metadata", None), kwargs)


@register_op("HardSilu")
class HardSilu(OpDef):
    """HardSilu activation."""

    op_name = "HardSilu"

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> tuple[int, ...]:
        """Infer shape.

        Args:
            *args (EagerTensorProtocol): Positional args.
            **kwargs (EagerTensorProtocol): Keyword args.

        Returns:
            tuple[int, ...]: Result shape matching input.
        """
        x = args[0] if len(args) > 0 else kwargs.get("x", kwargs.get("input"))
        return tuple(getattr(x, "shape", getattr(x, "shape_metadata", x if isinstance(x, (list, tuple)) else ())))


@register_op("HardSwish")
class HardSwish(OpDef):
    """HardSwish activation."""

    op_name = "HardSwish"

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> tuple[int, ...]:
        """Infer shape.

        Args:
            *args (EagerTensorProtocol): Positional args.
            **kwargs (EagerTensorProtocol): Keyword args.

        Returns:
            tuple[int, ...]: Result shape matching input.
        """
        x = args[0] if len(args) > 0 else kwargs.get("x", kwargs.get("input"))
        return tuple(getattr(x, "shape", getattr(x, "shape_metadata", x if isinstance(x, (list, tuple)) else ())))


@register_op("Squareplus")
class Squareplus(OpDef):
    """Squareplus activation."""

    op_name = "Squareplus"

    def infer_shape(self, *args: EagerTensorProtocol, **kwargs: TensorOrScalar) -> tuple[int, ...]:
        """Infer shape.

        Args:
            *args (EagerTensorProtocol): Positional args.
            **kwargs (EagerTensorProtocol): Keyword args.

        Returns:
            tuple[int, ...]: Result shape matching input.
        """
        x = args[0] if len(args) > 0 else kwargs.get("x", kwargs.get("input"))
        return tuple(getattr(x, "shape", getattr(x, "shape_metadata", x if isinstance(x, (list, tuple)) else ())))


def hard_silu(*args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Hard SiLU activation.

    Args:
        *args (TensorOrScalar): Positional args.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("HardSilu", *args, **kwargs)


def hard_swish(*args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Hard Swish activation.

    Args:
        *args (TensorOrScalar): Positional args.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("HardSwish", *args, **kwargs)


def mish(*args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Mish activation.

    Args:
        *args (TensorOrScalar): Positional args.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Mish", *args, **kwargs)


def squareplus(*args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
    """Squareplus activation.

    Args:
        *args (TensorOrScalar): Positional args.
        **kwargs (TensorOrScalar): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Squareplus", *args, **kwargs)
