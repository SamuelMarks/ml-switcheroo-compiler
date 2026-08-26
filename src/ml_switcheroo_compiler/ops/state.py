# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Define stateful operations for reading and writing variables within the ML Switcheroo.

framework
"""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("ReadVariable")
class ReadVariable(OpDef):
    """Provide an operation definition for reading the value of a stateful variable within the.

    computational graph
    """

    def infer_shape(self, *args, **kwargs):
        """Infer the output shape of the operation.

        Args:
            *args (object): Additional keyword arguments.
            **kwargs (object): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        return kwargs.get("shape", ())


@register_op("AssignVariable")
class AssignVariable(OpDef):
    """Provide an operation definition for assigning a new value to a stateful variable within the.

    computational graph
    """

    def infer_shape(self, x, **kwargs):
        """Infer the output shape of the operation.

        Args:
            x (object): The first input tensor.
            **kwargs (object): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        return x
