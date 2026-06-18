# pylint: disable=duplicate-code

"""Shape operations for Tensor objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ml_switcheroo_compiler.ops.base import OpDef, register_op

if TYPE_CHECKING:
    pass


@register_op("TopK")
class TopK(OpDef):
    """TopK operation."""

    op_name = "TopK"

    def infer_shape(self, x: object, k: object = None, **kwargs: object) -> object:
        """Infer shape.

        Args:
            x (object): The input x tensor.
            k (object): The k parameter for the operation.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        if k is None:
            k = kwargs.get("k", 1)
        if hasattr(k, "__array__") and not isinstance(k, tuple):
            k = k.__array__()
        if hasattr(k, "item"):
            k = int(k.item())
        else:
            try:
                k = int(k)
            except (ValueError, TypeError):
                pass

        if not hasattr(x, "shape") or not x.shape:
            return ()
        out_shape = list(x.shape)
        out_shape[-1] = k
        return tuple(out_shape)

    def emit_jax(self, *args: object, **kwargs: object) -> object:
        """Emit jax code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented TopK"

    def emit_keras(self, *args: object, **kwargs: object) -> object:
        """Emit keras code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented TopK"

    def emit_mlx(self, *args: object, **kwargs: object) -> object:
        """Emit mlx code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented TopK"

    def emit_pytorch(self, *args: object, **kwargs: object) -> object:
        """Emit pytorch code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented TopK"

    def emit_tensorflow(self, *args: object, **kwargs: object) -> object:
        """Emit tensorflow code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented TopK"


@register_op("ArgSort")
class ArgSort(OpDef):
    """ArgSort operation."""

    op_name = "ArgSort"

    def infer_shape(
        self,
        x: object,
        dimension: object = -1,
        is_stable: object = True,
        **kwargs: object,
    ) -> object:
        """Infer shape for ArgSort."""
        if isinstance(x, tuple) and hasattr(x, "shape"):
            return x.shape
        return getattr(x, "shape", ())

    def eval_numpy(self, x: object, **kwargs: object) -> object:
        """Evaluate ArgSort."""
        return "Not implemented ArgSort"

    def eval_jax(self, x: object, **kwargs: object) -> object:
        """Evaluate ArgSort."""
        return "Not implemented ArgSort"

    def eval_pytorch(self, x: object, **kwargs: object) -> object:
        """Evaluate ArgSort."""
        return "Not implemented ArgSort"

    def eval_mlx(self, x: object, **kwargs: object) -> object:
        """Evaluate ArgSort."""
        return "Not implemented ArgSort"

    def eval_tensorflow(self, x: object, **kwargs: object) -> object:
        """Evaluate ArgSort."""
        return "Not implemented ArgSort"


@register_op("Sort")
class Sort(OpDef):
    """Sort operation."""

    op_name = "Sort"

    def infer_shape(
        self,
        x: object,
        dimension: object = -1,
        is_stable: object = True,
        **kwargs: object,
    ) -> object:
        """Infer shape.

        Args:
            x (object): The input x tensor.
            dimension (object): The dimension parameter for the operation.
            is_stable (object): The is_stable parameter for the operation.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return getattr(x, "shape", ())

    def emit_jax(self, *args: object, **kwargs: object) -> object:
        """Emit jax code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented Sort"

    def emit_keras(self, *args: object, **kwargs: object) -> object:
        """Emit keras code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented Sort"

    def emit_mlx(self, *args: object, **kwargs: object) -> object:
        """Emit mlx code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented Sort"

    def emit_pytorch(self, *args: object, **kwargs: object) -> object:
        """Emit pytorch code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented Sort"

    def emit_tensorflow(self, *args: object, **kwargs: object) -> object:
        """Emit tensorflow code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented Sort"


@register_op("Where")
class Where(OpDef):
    """Where operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Where."""
        return ()


@register_op("Gather")
class Gather(OpDef):
    """Gather operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Gather."""
        return ()


@register_op("Take")
class Take(OpDef):
    """Take operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Take."""
        return ()


@register_op("TakeAlongAxis")
class TakeAlongAxis(OpDef):
    """TakeAlongAxis operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for TakeAlongAxis."""
        return ()


@register_op("GatherNd")
class GatherNd(OpDef):
    """GatherNd operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for GatherNd."""
        return ()


@register_op("Scatter")
class Scatter(OpDef):
    """Scatter operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Scatter."""
        return ()


@register_op("ScatterNd")
class ScatterNd(OpDef):
    """ScatterNd operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for ScatterNd."""
        return ()


@register_op("ScatterAdd")
class ScatterAdd(OpDef):
    """ScatterAdd operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for ScatterAdd."""
        return ()


@register_op("Vdot")
class Vdot(OpDef):
    """Vdot operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Vdot."""
        return ()


@register_op("SearchSorted")
class SearchSorted(OpDef):
    """SearchSorted operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for SearchSorted."""
        return ()


@register_op("Select")
class Select(OpDef):
    """Select operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Select."""
        return ()


@register_op("Assign")
class Assign(OpDef):
    """Assign operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Assign."""
        return ()


@register_op("AssignAdd")
class AssignAdd(OpDef):
    """AssignAdd operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for AssignAdd."""
        return ()


@register_op("AssignSub")
class AssignSub(OpDef):
    """AssignSub operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for AssignSub."""
        return ()


@register_op("TensorScatterUpdate")
class TensorScatterUpdate(OpDef):
    """TensorScatterUpdate operator definition."""

    def infer_shape(
        self, tensor: object, indices: object, updates: object, **kwargs: object
    ) -> object:
        """Infer shape for TensorScatterUpdate."""
        return getattr(tensor, "shape", ())
