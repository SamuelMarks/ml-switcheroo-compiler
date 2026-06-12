"""Defines stateful operations for reading and writing variables within the ML Switcheroo.

framework
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("ReadVariable")
class ReadVariable(OpDef):
    """An operation definition for reading the value of a stateful variable within the.

    computational graph
    """

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            *args (object): Variable length argument list
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return kwargs.get("shape", ())

    def numpy_eval(self, *args: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            *args (object): Variable length argument list
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        from ml_switcheroo_compiler.core.errors import CompilationError

        msg = "ReadVariable cannot be evaluated eagerly without a state manager."
        raise CompilationError(
            msg,
        )


@register_op("AssignVariable")
class AssignVariable(OpDef):
    """An operation definition for assigning a new value to a stateful variable within the.

    computational graph
    """

    def infer_shape(self, x: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            x (object): The x parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return x

    def numpy_eval(self, x: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The x parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        from ml_switcheroo_compiler.core.errors import CompilationError

        msg = "AssignVariable cannot be evaluated eagerly without a state manager."
        raise CompilationError(
            msg,
        )
