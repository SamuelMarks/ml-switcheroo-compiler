"""Apply pooling reduction operations."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.backends.registry import ensure_int
from ml_switcheroo_compiler.core.constants import MATRIX_RANK
from ml_switcheroo_compiler.core.shape import ShapeProtocol, TensorOrScalar, extract_shape
from ml_switcheroo_compiler.ops.base import OpDef, register_op


@typing.runtime_checkable
class _IndexableSizeProtocol(typing.Protocol):
    """Protocol for indexable size sequences."""

    def __getitem__(self, index: int) -> TensorOrScalar:
        """Retrieve dimension size at `index`.

        Args:
            index (int): Spatial dimension index.

        Returns:
            TensorOrScalar: Spatial dimension size.
        """
        ...


@register_op("CTCLoss")
class CTCLoss(OpDef):
    """Connectionist Temporal Classification Loss."""

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the shape of the CTCLoss output.

        Args:
            *args: Positional arguments, expecting the log probabilities tensor as the first argument.
            **kwargs: Keyword arguments, optionally containing 'log_probs'.

        Returns:
            The inferred shape tuple (typically containing the batch size), or an empty tuple if it cannot be determined.
        """
        log_probs = args[0] if len(args) > 0 else kwargs.get("log_probs")
        if isinstance(log_probs, ShapeProtocol):
            return (log_probs.shape[1],) if len(log_probs.shape) >= MATRIX_RANK else ()
        return ()


@register_op("FractionalMaxPool2D")
class FractionalMaxPool2D(OpDef):
    """Fractional max pooling 2D."""

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the shape of the FractionalMaxPool2D output.

        Args:
            *args: Positional arguments, expecting the operand as the first argument and output_size as the second.
            **kwargs: Keyword arguments, optionally containing 'operand' and 'output_size'.

        Returns:
            The inferred shape tuple with updated spatial dimensions, or an empty tuple if it cannot be determined.
        """
        operand = args[0] if len(args) > 0 else kwargs.get("operand")
        output_size = args[1] if len(args) > 1 else kwargs.get("output_size")
        if isinstance(operand, ShapeProtocol) and isinstance(output_size, _IndexableSizeProtocol):
            s = list(operand.shape)
            s[-2], s[-1] = ensure_int(output_size[0]), ensure_int(output_size[1])
            return tuple(s)
        return ()


@register_op("AdaptiveAvgPool2D")
class AdaptiveAvgPool2D(OpDef):
    """Adaptive average pooling 2D."""

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the shape of the AdaptiveAvgPool2D output.

        Args:
            *args: Positional arguments, expecting the operand as the first argument and output_size as the second.
            **kwargs: Keyword arguments, optionally containing 'operand' and 'output_size'.

        Returns:
            The inferred shape tuple with updated spatial dimensions, or an empty tuple if it cannot be determined.
        """
        operand = args[0] if len(args) > 0 else kwargs.get("operand")
        output_size = args[1] if len(args) > 1 else kwargs.get("output_size")
        if isinstance(operand, ShapeProtocol) and isinstance(output_size, _IndexableSizeProtocol):
            s = list(operand.shape)
            s[-2], s[-1] = ensure_int(output_size[0]), ensure_int(output_size[1])
            return tuple(s)
        return ()


@register_op("AdaptiveMaxPool2D")
class AdaptiveMaxPool2D(OpDef):
    """Adaptive max pooling 2D."""

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the shape of the AdaptiveMaxPool2D output.

        Args:
            *args: Positional arguments, expecting the operand as the first argument and output_size as the second.
            **kwargs: Keyword arguments, optionally containing 'operand' and 'output_size'.

        Returns:
            The inferred shape tuple with updated spatial dimensions, or an empty tuple if it cannot be determined.
        """
        operand = args[0] if len(args) > 0 else kwargs.get("operand")
        output_size = args[1] if len(args) > 1 else kwargs.get("output_size")
        if isinstance(operand, ShapeProtocol) and isinstance(output_size, _IndexableSizeProtocol):
            s = list(operand.shape)
            s[-2], s[-1] = ensure_int(output_size[0]), ensure_int(output_size[1])
            return tuple(s)
        return ()


@register_op("Unfold")
class Unfold(OpDef):
    """Unfold (Im2Col) operator."""

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the shape of the Unfold output.

        Args:
            *args: Positional arguments for the Unfold operation.
            **kwargs: Keyword arguments for the Unfold operation.

        Returns:
            An empty tuple as the exact shape depends on strides and padding which are not inferred here.
        """
        return ()


@register_op("Fold")
class Fold(OpDef):
    """Fold (Col2Im) operator."""

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the shape of the Fold output.

        Args:
            *args: Positional arguments for the Fold operation.
            **kwargs: Keyword arguments for the Fold operation.

        Returns:
            An empty tuple as the exact shape depends on output size parameters which are not inferred here.
        """
        return ()


@register_op("FractionalMaxPool3D")
class FractionalMaxPool3D(OpDef):
    """Fractional max pooling 3D."""

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the shape of the FractionalMaxPool3D output.

        Args:
            *args: Positional arguments, expecting the operand as the first argument and output_size as the second.
            **kwargs: Keyword arguments, optionally containing 'operand' and 'output_size'.

        Returns:
            The inferred shape tuple with updated spatial dimensions, or an empty tuple if it cannot be determined.
        """
        operand = args[0] if len(args) > 0 else kwargs.get("operand")
        output_size = args[1] if len(args) > 1 else kwargs.get("output_size")
        if isinstance(operand, ShapeProtocol) and isinstance(output_size, _IndexableSizeProtocol):
            s = list(operand.shape)
            s[-3], s[-2], s[-1] = ensure_int(output_size[0]), ensure_int(output_size[1]), ensure_int(output_size[2])
            return tuple(s)
        return ()


@register_op("AdaptiveAvgPool3D")
class AdaptiveAvgPool3D(OpDef):
    """Adaptive average pooling 3D."""

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the shape of the AdaptiveAvgPool3D output.

        Args:
            *args: Positional arguments, expecting the operand as the first argument and output_size as the second.
            **kwargs: Keyword arguments, optionally containing 'operand' and 'output_size'.

        Returns:
            The inferred shape tuple with updated spatial dimensions, or an empty tuple if it cannot be determined.
        """
        operand = args[0] if len(args) > 0 else kwargs.get("operand")
        output_size = args[1] if len(args) > 1 else kwargs.get("output_size")
        if isinstance(operand, ShapeProtocol) and isinstance(output_size, _IndexableSizeProtocol):
            s = list(operand.shape)
            s[-3], s[-2], s[-1] = ensure_int(output_size[0]), ensure_int(output_size[1]), ensure_int(output_size[2])
            return tuple(s)
        return ()


@register_op("AdaptiveMaxPool3D")
class AdaptiveMaxPool3D(OpDef):
    """Adaptive max pooling 3D."""

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the shape of the AdaptiveMaxPool3D output.

        Args:
            *args: Positional arguments, expecting the operand as the first argument and output_size as the second.
            **kwargs: Keyword arguments, optionally containing 'operand' and 'output_size'.

        Returns:
            The inferred shape tuple with updated spatial dimensions, or an empty tuple if it cannot be determined.
        """
        operand = args[0] if len(args) > 0 else kwargs.get("operand")
        output_size = args[1] if len(args) > 1 else kwargs.get("output_size")
        if isinstance(operand, ShapeProtocol) and isinstance(output_size, _IndexableSizeProtocol):
            s = list(operand.shape)
            s[-3], s[-2], s[-1] = ensure_int(output_size[0]), ensure_int(output_size[1]), ensure_int(output_size[2])
            return tuple(s)
        return ()


@register_op("MaxUnpool1D")
class MaxUnpool1D(OpDef):
    """Max unpooling 1D."""

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the shape of the MaxUnpool1D output.

        Args:
            *args: Positional arguments, expecting the operand as the first argument and output_size as the third.
            **kwargs: Keyword arguments, optionally containing 'operand' and 'output_size'.

        Returns:
            The inferred shape tuple with updated spatial dimension, or an empty tuple if it cannot be determined.
        """
        operand = args[0] if len(args) > 0 else kwargs.get("operand")
        output_size = args[2] if len(args) > 2 else kwargs.get("output_size")
        if isinstance(output_size, _IndexableSizeProtocol):
            s = list(extract_shape(operand))
            s[-1] = ensure_int(output_size[0])
            return tuple(s)
        return ()


@register_op("MaxUnpool2D")
class MaxUnpool2D(OpDef):
    """Max unpooling 2D."""

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the shape of the MaxUnpool2D output.

        Args:
            *args: Positional arguments, expecting the operand as the first argument and output_size as the third.
            **kwargs: Keyword arguments, optionally containing 'operand' and 'output_size'.

        Returns:
            The inferred shape tuple with updated spatial dimensions, or an empty tuple if it cannot be determined.
        """
        operand = args[0] if len(args) > 0 else kwargs.get("operand")
        output_size = args[2] if len(args) > 2 else kwargs.get("output_size")
        if isinstance(output_size, _IndexableSizeProtocol):
            s = list(extract_shape(operand))
            s[-2], s[-1] = ensure_int(output_size[0]), ensure_int(output_size[1])
            return tuple(s)
        return ()


@register_op("MaxUnpool3D")
class MaxUnpool3D(OpDef):
    """Max unpooling 3D."""

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the shape of the MaxUnpool3D output.

        Args:
            *args: Positional arguments, expecting the operand as the first argument and output_size as the third.
            **kwargs: Keyword arguments, optionally containing 'operand' and 'output_size'.

        Returns:
            The inferred shape tuple with updated spatial dimensions, or an empty tuple if it cannot be determined.
        """
        operand = args[0] if len(args) > 0 else kwargs.get("operand")
        output_size = args[2] if len(args) > 2 else kwargs.get("output_size")
        if isinstance(output_size, _IndexableSizeProtocol):
            s = list(extract_shape(operand))
            s[-3], s[-2], s[-1] = ensure_int(output_size[0]), ensure_int(output_size[1]), ensure_int(output_size[2])
            return tuple(s)
        return ()


@register_op("AdaptiveMaxPool3D_Indices")
class AdaptiveMaxPool3D_Indices(OpDef):
    """Adaptive max pooling 3D indices."""

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the shape of the AdaptiveMaxPool3D indices output.

        Args:
            *args: Positional arguments, expecting the operand as the first argument and output_size as the second.
            **kwargs: Keyword arguments, optionally containing 'operand' and 'output_size'.

        Returns:
            The inferred shape tuple with updated spatial dimensions, or an empty tuple if it cannot be determined.
        """
        operand = args[0] if len(args) > 0 else kwargs.get("operand")
        output_size = args[1] if len(args) > 1 else kwargs.get("output_size")
        if isinstance(operand, ShapeProtocol) and isinstance(output_size, _IndexableSizeProtocol):
            s = list(operand.shape)
            s[-3], s[-2], s[-1] = ensure_int(output_size[0]), ensure_int(output_size[1]), ensure_int(output_size[2])
            return tuple(s)
        return ()


@register_op("FractionalMaxPool3D_Indices")
class FractionalMaxPool3D_Indices(OpDef):
    """Fractional max pooling 3D indices."""

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the shape of the FractionalMaxPool3D indices output.

        Args:
            *args: Positional arguments, expecting the operand as the first argument and output_size as the second.
            **kwargs: Keyword arguments, optionally containing 'operand' and 'output_size'.

        Returns:
            The inferred shape tuple with updated spatial dimensions, or an empty tuple if it cannot be determined.
        """
        operand = args[0] if len(args) > 0 else kwargs.get("operand")
        output_size = args[1] if len(args) > 1 else kwargs.get("output_size")
        if isinstance(operand, ShapeProtocol) and isinstance(output_size, _IndexableSizeProtocol):
            s = list(operand.shape)
            s[-3], s[-2], s[-1] = ensure_int(output_size[0]), ensure_int(output_size[1]), ensure_int(output_size[2])
            return tuple(s)
        return ()


@register_op("MaxPoolWithIndices")
class MaxPoolWithIndices(OpDef):
    """Max pooling with indices."""

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the shape of the MaxPoolWithIndices output.

        Args:
            *args: Positional arguments for the pooling operation.
            **kwargs: Keyword arguments for the pooling operation.

        Returns:
            An empty tuple as the exact shape depends on spatial arguments not inferred here.
        """
        return ()


@register_op("MaxPoolWithIndices_Indices")
class MaxPoolWithIndices_Indices(OpDef):
    """Max pooling with indices (indices part)."""

    def infer_shape(self, *args: TensorOrScalar, **kwargs: TensorOrScalar) -> TensorOrScalar:
        """Infer the shape of the MaxPoolWithIndices indices output.

        Args:
            *args: Positional arguments for the pooling operation.
            **kwargs: Keyword arguments for the pooling operation.

        Returns:
            An empty tuple as the exact shape depends on spatial arguments not inferred here.
        """
        return ()
