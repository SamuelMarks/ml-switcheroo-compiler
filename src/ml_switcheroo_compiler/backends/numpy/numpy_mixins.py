# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Mixins."""

from ml_switcheroo_compiler.backends.common.audio_utils import (
    extract_mel_attributes,
    extract_stft_attributes,
)
from ml_switcheroo_compiler.backends.generator_utils import (
    _extract_extract_boxes_attributes,
    _extract_filter_attributes,
    _extract_vision_transform_attributes,
)
from ml_switcheroo_compiler.ir.core import IRNode


class NumpyScatterVisitor:
    """Provide AST visitor methods for scatter operations in NumPy."""

    def visit_TensorScatterUpdate(self, node: IRNode, input_vars: list[str], **kwargs) -> str:
        """Generate NumPy code for a tensor scatter update operation.

        Args:
            node: The IR node representing the tensor scatter update.
            input_vars: List of variable names corresponding to the inputs.
            **kwargs: Additional keyword arguments for the visitor.

        Returns:
            A string containing the generated NumPy expression.
        """
        return f"(lambda c, i, u: [c.__setitem__(tuple(np.moveaxis(np.asarray(i), -1, 0)), u), c][1])(np.copy({input_vars[0]}), {input_vars[1]}, {input_vars[2]})"

    def visit_TensorScatterAdd(self, node: IRNode, input_vars: list[str], **kwargs) -> str:
        """Generate NumPy code for a tensor scatter add operation.

        Args:
            node: The IR node representing the tensor scatter add.
            input_vars: List of variable names corresponding to the inputs.
            **kwargs: Additional keyword arguments for the visitor.

        Returns:
            A string containing the generated NumPy expression.
        """
        return f"(lambda c, i, u: [np.add.at(c, tuple(np.moveaxis(np.asarray(i), -1, 0)), u), c][1])(np.copy({input_vars[0]}), {input_vars[1]}, {input_vars[2]})"

    def visit_TensorScatterMax(self, node: IRNode, input_vars: list[str], **kwargs) -> str:
        """Generate NumPy code for a tensor scatter max operation.

        Args:
            node: The IR node representing the tensor scatter max.
            input_vars: List of variable names corresponding to the inputs.
            **kwargs: Additional keyword arguments for the visitor.

        Returns:
            A string containing the generated NumPy expression.
        """
        return f"(lambda c, i, u: [np.maximum.at(c, tuple(np.moveaxis(np.asarray(i), -1, 0)), u), c][1])(np.copy({input_vars[0]}), {input_vars[1]}, {input_vars[2]})"

    def visit_TensorScatterMin(self, node: IRNode, input_vars: list[str], **kwargs) -> str:
        """Generate NumPy code for a tensor scatter min operation.

        Args:
            node: The IR node representing the tensor scatter min.
            input_vars: List of variable names corresponding to the inputs.
            **kwargs: Additional keyword arguments for the visitor.

        Returns:
            A string containing the generated NumPy expression.
        """
        return f"(lambda c, i, u: [np.minimum.at(c, tuple(np.moveaxis(np.asarray(i), -1, 0)), u), c][1])(np.copy({input_vars[0]}), {input_vars[1]}, {input_vars[2]})"
