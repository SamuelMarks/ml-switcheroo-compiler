"""Reduction operations."""

import numpy as np
from typing import Any
from ml_switcheroo.ops.base import OpDef, register_op


class ReductionOp(OpDef):
    """Base class for reduction operations."""

    op_name: str = ""

    def infer_shape(
        self, x: object, axis: object = None, keepdims: bool = False, **kwargs: object
    ) -> object:
        """Docstring."""
        return None  # Symbolic shape inference will handle axis reduction logic

    def numpy_eval(
        self, x: object, axis: object = None, keepdims: bool = False, **kwargs: object
    ) -> object:
        """Docstring."""
        return getattr(np, self.op_name.lower())(
            x, axis=axis, keepdims=keepdims, **kwargs
        )

    def vjp(self, cotangent: object, x: object, **kwargs: object) -> tuple[Any, ...]:
        """Docstring."""
        return (f"broadcast_vjp({cotangent})",)

    def jvp(self, tangent: object, x: object, **kwargs: object) -> object:
        """Docstring."""
        return f"reduction_jvp({tangent})"

    def _format_args(self, x: str, **kwargs: object) -> str:
        """Docstring."""
        args = [x]
        if "axis" in kwargs and kwargs["axis"] is not None:
            args.append(f"axis={kwargs['axis']}")
        if "keepdims" in kwargs and kwargs["keepdims"]:
            args.append(f"keepdims={kwargs['keepdims']}")
        return ", ".join(args)

    def emit_jax(self, x: str, **kwargs: object) -> str:
        """Docstring."""
        return f"jnp.{self.op_name.lower()}({self._format_args(x, **kwargs)})"

    def emit_pytorch(self, x: str, **kwargs: object) -> str:
        """Docstring."""
        # PyTorch often uses dim instead of axis
        args = [x]
        if "axis" in kwargs and kwargs["axis"] is not None:
            args.append(f"dim={kwargs['axis']}")
        if "keepdims" in kwargs and kwargs["keepdims"]:
            args.append(f"keepdim={kwargs['keepdims']}")
        return f"torch.{self.op_name.lower()}({', '.join(args)})"

    def emit_mlx(self, x: str, **kwargs: object) -> str:
        """Docstring."""
        return f"mx.{self.op_name.lower()}({self._format_args(x, **kwargs)})"

    def emit_keras(self, x: str, **kwargs: object) -> str:
        """Docstring."""
        return f"keras.ops.{self.op_name.lower()}({self._format_args(x, **kwargs)})"

    def emit_tensorflow(self, x: str, **kwargs: object) -> str:
        """Docstring."""
        return f"tf.reduce_{self.op_name.lower()}({self._format_args(x, **kwargs)})"


@register_op("Sum")
class Sum(ReductionOp):
    """Docstring."""

    op_name = "Sum"


@register_op("Mean")
class Mean(ReductionOp):
    """Docstring."""

    op_name = "Mean"


@register_op("Max")
class Max(ReductionOp):
    """Docstring."""

    op_name = "Max"

    def numpy_eval(
        self, x: object, axis: object = None, keepdims: bool = False, **kwargs: object
    ) -> object:
        """Docstring."""
        return np.max(x, axis=axis, keepdims=keepdims, **kwargs)


@register_op("Min")
class Min(ReductionOp):
    """Docstring."""

    op_name = "Min"

    def numpy_eval(
        self, x: object, axis: object = None, keepdims: bool = False, **kwargs: object
    ) -> object:
        """Docstring."""
        return np.min(x, axis=axis, keepdims=keepdims, **kwargs)
