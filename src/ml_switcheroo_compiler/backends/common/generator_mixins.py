"""Shared AST generator mixins."""


class SharedASTGeneratorMixin:
    """Mixin for shared AST generation logic across backends."""

    def _get_backend_prefix(self) -> str:
        """Returns the backend prefix (e.g., 'jax', 'pt', 'mx')."""
        raise NotImplementedError

    def visit_GroupNorm(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate group norm."""
        groups = kwargs.get("groups")
        axis = kwargs.get("axis", -1)
        epsilon = kwargs.get("epsilon", 1e-5)

        weight_str = "None"
        bias_str = "None"
        if len(input_vars) > 1:
            weight_str = input_vars[1]
        if len(input_vars) > 2:
            bias_str = input_vars[2]

        return f"{self._get_backend_prefix()}_group_norm({input_vars[0]}, {groups}, {weight_str}, {bias_str}, {axis}, {epsilon})"

    def visit_GroupMean(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate group mean."""
        groups = kwargs.get("groups")
        axis = kwargs.get("axis", -1)
        return f"{self._get_backend_prefix()}_group_mean({input_vars[0]}, {groups}, {axis})"

    def visit_GroupVariance(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate group variance."""
        groups = kwargs.get("groups")
        axis = kwargs.get("axis", -1)
        return f"{self._get_backend_prefix()}_group_variance({input_vars[0]}, {groups}, {axis})"
