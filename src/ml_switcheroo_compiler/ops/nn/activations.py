"""Activations and advanced NN operations."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.registry import coerce_int, coerce_str
from ml_switcheroo_compiler.core.shape import (
    OpValue,
    ShapeMetadataProtocol,
    ShapeProtocol,
    extract_dtype,
    extract_shape,
)

# Base logic implementation
from ml_switcheroo_compiler.core.tensor import Tensor, ensure_tensor
from ml_switcheroo_compiler.ops.base import OpDef
from ml_switcheroo_compiler.ops.registry import get_op, register_op
from ml_switcheroo_compiler.ops.shape.joining import concatenate


@typing.runtime_checkable
class _MutableShapeMetaProtocol(typing.Protocol):
    """Protocol for metadata objects with a mutable `.shape` attribute."""

    shape: tuple[int, ...]


def _extract_shape_meta(obj: OpValue) -> OpValue:
    """Extract `.shape_metadata` from `obj` if available.

    Args:
        obj (OpValue): Input tensor or proxy.

    Returns:
        OpValue: Shape metadata or `None`.
    """
    return obj.shape_metadata if isinstance(obj, ShapeMetadataProtocol) else None


def _infer_unary_activation_shape(
    args: tuple[OpValue, ...],
    kwargs: dict[str, OpValue],
) -> tuple[int, ...]:
    """Infer output shape for a unary activation from `args` or `kwargs`.

    Args:
        args (tuple[OpValue, ...]): Positional arguments.
        kwargs (dict[str, OpValue]): Keyword arguments.

    Returns:
        tuple[int, ...]: Inferred integer shape tuple.
    """
    x = args[0] if len(args) > 0 else kwargs.get("x", kwargs.get("input"))
    fallback = tuple(coerce_int(d) for d in x) if isinstance(x, (list, tuple)) else ()
    return extract_shape(x, fallback)


def crelu(features: OpValue, axis: OpValue = -1, name: OpValue = None) -> OpValue:
    """Compute Concatenated ReLU.

    Args:
        features (OpValue): Input `features` argument (features).
        axis (OpValue): Input `axis` argument (axis).
        name (OpValue): Input `name` argument (name).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    negative = get_op("Negative")()

    from ml_switcheroo_compiler.ops.binary import maximum

    return maximum(concatenate([ensure_tensor(features), negative(features)], axis=coerce_int(axis)), 0.0)


def isotonic_regression(
    y: OpValue,
    sample_weights: OpValue = None,
    increasing: OpValue = True,
    name: OpValue = None,
) -> OpValue:
    """Solves isotonic regression problems.

    Args:
        y (OpValue): Input `y` argument (y).
        sample_weights (OpValue): Input `sample_weights` argument (sample weights).
        increasing (OpValue): Input `increasing` argument (increasing).
        name (OpValue): Input `name` argument (name).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op(
            "IsotonicRegression", y, sample_weights=sample_weights, increasing=increasing, name=name
        )

    from ml_switcheroo_compiler.ops.shape.utils import _emit_shape_node

    attrs = {"sample_weights": sample_weights, "increasing": increasing, "name": name}
    y_shape = extract_shape(y)
    out1 = _emit_shape_node("IsotonicRegression", [y], attrs, y_shape, extract_dtype(y, "float32"))
    out2 = _emit_shape_node("IsotonicRegressionSegments", [y], attrs, y_shape, "int32")
    return out1, out2


__all__ = ["softplus", "softmax", "log_softmax", "sigmoid", "one_hot", "rrelu"]


def softplus(x: Tensor) -> Tensor:
    """Softplus activation.

    Args:
        x (Tensor): Input `x` argument (x).

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.unary import exp, log1p

    return log1p(exp(x))


@register_op("Softmax")
class Softmax(OpDef):
    """Operator Softmax."""

    op_name = "Softmax"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [extract_shape(a) for a in args if isinstance(a, ShapeProtocol)]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        _typed_ret_1: OpValue = res
        return _typed_ret_1


def softmax(x: OpValue, axis: int = -1, *args: OpValue, **kwargs: OpValue) -> OpValue:
    """Softmax activation.

    Args:
        x (OpValue): Input `x` argument (x).
        axis (int): Input `axis` argument (axis).
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Softmax", x, axis=axis)
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    return emit_ir_node(None, "Softmax", [coerce_str(x)], _extract_shape_meta(x), {"axis": axis})


@register_op("LogSoftmax")
class LogSoftmax(OpDef):
    """Operator LogSoftmax."""

    op_name = "LogSoftmax"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [extract_shape(a) for a in args if isinstance(a, ShapeProtocol)]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        _typed_ret_2: OpValue = res
        return _typed_ret_2


def log_softmax(x: OpValue, axis: int = -1, *args: OpValue, **kwargs: OpValue) -> OpValue:
    """LogSoftmax activation.

    Args:
        x (OpValue): Input `x` argument (x).
        axis (int): Input `axis` argument (axis).
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("LogSoftmax", x, axis=axis)
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    return emit_ir_node(None, "LogSoftmax", [coerce_str(x)], _extract_shape_meta(x), {"axis": axis})


@register_op("Sigmoid")
class Sigmoid(OpDef):
    """Operator Sigmoid."""

    op_name = "Sigmoid"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [extract_shape(a) for a in args if isinstance(a, ShapeProtocol)]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        _typed_ret_3: OpValue = res
        return _typed_ret_3


def sigmoid(x: Tensor | str, *args: int | float | str | None, **kwargs: int | float | str | None) -> Tensor:
    """Sigmoid activation.

    Args:
        x (Tensor | str): Input `x` argument (x).
        *args (int | float | str | None): Positional args.
        **kwargs (int | float | str | None): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return ensure_tensor(get_active_backend().execute_op("Sigmoid", x))
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    return ensure_tensor(emit_ir_node(None, "Sigmoid", [str(x)], _extract_shape_meta(x), {}))


@register_op("OneHot")
class OneHot(OpDef):
    """Operator OneHot."""

    op_name = "OneHot"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes  # noqa: F401

        shapes = [extract_shape(a) for a in args if isinstance(a, ShapeProtocol)]
        indices_shape = shapes[0] if shapes else ()
        depth = kwargs.get("depth", args[1] if len(args) > 1 else 1)
        axis = kwargs.get("axis", -1)
        if axis == -1:
            return indices_shape + (depth,)
        else:
            return indices_shape[: int(coerce_int(axis))] + (depth,) + indices_shape[int(coerce_int(axis)) :]


def one_hot(indices: OpValue, depth: int, *args: OpValue, **kwargs: OpValue) -> OpValue:
    """OneHot encoding.

    Args:
        indices (OpValue): Input `indices` argument (indices).
        depth (int): Input `depth` argument (depth).
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("OneHot", indices, depth, *args, **kwargs)
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    kwargs["depth"] = depth
    shape: tuple[int, ...] = (
        tuple(int(d) for d in indices.shape if isinstance(d, int)) + (depth,)
        if isinstance(indices, ShapeProtocol)
        else (depth,)
    )
    import copy

    meta = copy.copy(_extract_shape_meta(indices))
    if isinstance(meta, _MutableShapeMetaProtocol):
        meta.shape = shape
    return emit_ir_node(None, "OneHot", [coerce_str(indices)], meta, kwargs)


@register_op("Rrelu")
class Rrelu(OpDef):
    """Operator Rrelu."""

    op_name = "Rrelu"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes

        shapes = [extract_shape(a) for a in args if isinstance(a, ShapeProtocol)]
        if not shapes:
            return ()
        res = shapes[0]
        for s in shapes[1:]:
            res = broadcast_shapes(res, s)
        _typed_ret_4: OpValue = res
        return _typed_ret_4


def rrelu(x: OpValue, *args: OpValue, **kwargs: OpValue) -> OpValue:
    """Rrelu activation.

    Args:
        x (OpValue): Input `x` argument (x).
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.core.config import config

    if config.eager_mode:
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        return get_active_backend().execute_op("Rrelu", x, *args, **kwargs)
    from ml_switcheroo_compiler.ops.base import emit_ir_node

    return emit_ir_node(None, "Rrelu", [coerce_str(x)], _extract_shape_meta(x), kwargs)


@register_op("HardSilu")
class HardSilu(OpDef):
    """HardSilu activation."""

    op_name = "HardSilu"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> tuple[int, ...]:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Result shape matching input.
        """
        return _infer_unary_activation_shape(args, kwargs)


@register_op("HardSwish")
class HardSwish(OpDef):
    """HardSwish activation."""

    op_name = "HardSwish"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> tuple[int, ...]:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Result shape matching input.
        """
        return _infer_unary_activation_shape(args, kwargs)


@register_op("Squareplus")
class Squareplus(OpDef):
    """Squareplus activation."""

    op_name = "Squareplus"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> tuple[int, ...]:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Result shape matching input.
        """
        return _infer_unary_activation_shape(args, kwargs)


def hard_silu(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Hard SiLU activation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("HardSilu", *args, **kwargs)


def hard_swish(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Hard Swish activation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("HardSwish", *args, **kwargs)


def mish(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Mish activation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Mish", *args, **kwargs)


def squareplus(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Squareplus activation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Squareplus", *args, **kwargs)
