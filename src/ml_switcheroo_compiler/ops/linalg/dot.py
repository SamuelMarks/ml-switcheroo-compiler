"""Dot product operations."""

from ml_switcheroo_compiler.core.constants import MAGIC_VAL_2
from ml_switcheroo_compiler.ops.base import OpDef, register_op


def _has_valid_shape(obj: object) -> bool:
    """Function docstring.

    Args:
        obj: Arg.
    """
    return hasattr(obj, "shape") and bool(obj.shape)


@register_op("Dot")
class Dot(OpDef):
    """Dot product operator.

    Computes the dot product of two arrays
    """

    def infer_shape(self, a: object, b: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            a (object): The first input tensor.
            b (object): The second input tensor.
            **kwargs (object): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        return None


@register_op("DotGeneral")
class DotGeneral(OpDef):
    """General dot product operator.

    Computes a generalized dot product matching JAX's lax.dot_general.
    """

    op_name = "DotGeneral"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): lhs, rhs, dimension_numbers.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        lhs = args[0] if len(args) > 0 else kwargs["lhs"]
        rhs = args[1] if len(args) > 1 else kwargs["rhs"]
        dimension_numbers = args[2] if len(args) > MAGIC_VAL_2 else kwargs["dimension_numbers"]
        if not _has_valid_shape(lhs) or not _has_valid_shape(rhs):
            return ()

        return self._compute_out_shape(lhs.shape, rhs.shape, dimension_numbers)

    def _compute_out_shape(
        self, lhs_shape: tuple, rhs_shape: tuple, dimension_numbers: tuple
    ) -> tuple:
        """Execute _compute_out_shape.

        Args:
            lhs_shape (Any): Argument lhs_shape.
            rhs_shape (Any): Argument rhs_shape.
            dimension_numbers (Any): Argument dimension_numbers.

        Returns:
        Any: The result.
        """
        contracting, batch = dimension_numbers
        lhs_contracting, rhs_contracting = contracting
        lhs_batch, rhs_batch = batch

        out_shape = [lhs_shape[b] for b in lhs_batch]
        out_shape.extend(
            [lhs_shape[i] for i in range(len(lhs_shape)) if i not in lhs_contracting + lhs_batch]
        )
        out_shape.extend(
            [rhs_shape[i] for i in range(len(rhs_shape)) if i not in rhs_contracting + rhs_batch]
        )

        return tuple(out_shape)

    def emit_jax(self, *args: object, **kwargs: object) -> object:
        """Emit jax code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented DotGeneral"

    def emit_keras(self, *args: object, **kwargs: object) -> object:
        """Emit keras code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented DotGeneral"

    def emit_mlx(self, *args: object, **kwargs: object) -> object:
        """Emit mlx code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented DotGeneral"

    def emit_pytorch(self, *args: object, **kwargs: object) -> object:
        """Emit pytorch code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented DotGeneral"

    def emit_tensorflow(self, *args: object, **kwargs: object) -> object:
        """Emit tensorflow code.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return "Not implemented DotGeneral"


@register_op("Tensordot")
class Tensordot(OpDef):
    """Tensordot operator.

    Computes tensor dot product along specified axes.
    """

    def infer_shape(self, a: object, b: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            a (object): The first input tensor.
            b (object): The second input tensor.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return ()


@register_op("Inner")
class Inner(OpDef):
    """Inner product operator.

    Computes the inner product of two vectors or matrices.
    """

    def infer_shape(self, a: object, b: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            a (object): The first input tensor.
            b (object): The second input tensor.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return ()


@register_op("Outer")
class Outer(OpDef):
    """Outer product operator.

    Computes the outer product of two vectors.
    """

    def infer_shape(self, a: object, b: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            a (object): The first input tensor.
            b (object): The second input tensor.
            **kwargs (object): Additional keyword arguments.

        Returns:
            object: The evaluated output resulting from this operation.
        """
        return ()
