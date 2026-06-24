"""Shared AST generator mixins."""

from dataclasses import dataclass


@dataclass
class GroupNormConfig:  # pylint: disable=too-many-instance-attributes
    """Configuration for group norm code generation."""

    prefix: str
    module: str
    reshape: str
    mean: str
    var: str
    sqrt: str
    dim_arg: str
    keepdim_arg: str
    unbiased_arg: str = ""


class SharedASTGeneratorMixin:
    """Shared AST generator mixin."""

    def visit_TimeDistributed(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate time distributed."""
        # Fallback implementation: we assume the frontend has provided a TimeDistributed node.
        # Natively, backends might want to generate a loop or a vmap.
        # For simplicity in this mixin, we return a function call to a backend-specific time_distributed utility.
        return f"{self._get_backend_prefix()}_time_distributed({input_vars[0]}, '{node.attributes.get('wrapped_op_name', '')}')"  # pragma: no cover

    def visit_ActivityRegularization(
        self, node: object, input_vars: list[str], **kwargs: object
    ) -> str:
        """Evaluate activity regularization."""
        # It's an identity op as loss is handled externally
        return input_vars[0]  # pragma: no cover

    """Mixin for shared AST generation logic across backends."""

    def _get_backend_prefix(self) -> str:
        """Returns the backend prefix (e.g., 'jax', 'pt', 'mx')."""
        raise NotImplementedError  # pragma: no cover

    def visit_GroupNorm(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate group norm."""
        MIN_ARGS_FOR_WEIGHT = 1  # pragma: no cover
        MIN_ARGS_FOR_BIAS = 2  # pragma: no cover

        groups = kwargs.get("groups")  # pragma: no cover
        axis = kwargs.get("axis", -1)  # pragma: no cover
        epsilon = kwargs.get("epsilon", 1e-5)  # pragma: no cover

        weight_str = "None"  # pragma: no cover
        bias_str = "None"  # pragma: no cover
        if len(input_vars) > MIN_ARGS_FOR_WEIGHT:  # pragma: no cover
            weight_str = input_vars[1]  # pragma: no cover
        if len(input_vars) > MIN_ARGS_FOR_BIAS:  # pragma: no cover
            bias_str = input_vars[2]  # pragma: no cover

        return f"{self._get_backend_prefix()}_group_norm({input_vars[0]}, {groups}, {weight_str}, {bias_str}, {axis}, {epsilon})"  # pragma: no cover

    def visit_GroupMean(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate group mean."""
        groups = kwargs.get("groups")  # pragma: no cover
        axis = kwargs.get("axis", -1)  # pragma: no cover
        return f"{self._get_backend_prefix()}_group_mean({input_vars[0]}, {groups}, {axis})"  # pragma: no cover

    def visit_GroupVariance(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate group variance."""
        groups = kwargs.get("groups")  # pragma: no cover
        axis = kwargs.get("axis", -1)  # pragma: no cover
        return f"{self._get_backend_prefix()}_group_variance({input_vars[0]}, {groups}, {axis})"  # pragma: no cover

    def _get_group_norm_code(self, config: GroupNormConfig) -> list[str]:
        """Generate group norm helper functions."""
        return [
            f"def {config.prefix}_group_norm(x, groups, weight=None, bias=None, axis=-1, epsilon=1e-5):",
            f"    import {config.module}",
            "    shape = list(x.shape)",
            "    ndims = len(shape)",
            "    if axis < 0: axis += ndims",
            "    C = shape[axis]",
            "    reshaped_dims = shape.copy()",
            "    reshaped_dims[axis:axis+1] = [groups, C // groups]",
            f"    reshaped_x = {config.reshape}(x, reshaped_dims)",
            "    reduction_axes = tuple(i for i in range(len(reshaped_dims)) if i != 0 and i != axis)",
            f"    mean = {config.mean}(reshaped_x, {config.dim_arg}=reduction_axes, {config.keepdim_arg}=True)",
            f"    var = {config.var}(reshaped_x, {config.dim_arg}=reduction_axes, {config.keepdim_arg}=True{config.unbiased_arg})",
            f"    normalized = (reshaped_x - mean) / {config.sqrt}(var + epsilon)",
            f"    out = {config.reshape}(normalized, shape)",
            "    if weight is not None:",
            "        w_shape = [1] * ndims",
            "        w_shape[axis] = C",
            f"        weight = {config.reshape}(weight, w_shape)",
            "        out = out * weight",
            "    if bias is not None:",
            "        b_shape = [1] * ndims",
            "        b_shape[axis] = C",
            f"        bias = {config.reshape}(bias, b_shape)",
            "        out = out + bias",
            "    return out",
            f"def {config.prefix}_group_mean(x, groups, axis=-1):",
            f"    import {config.module}",
            "    shape = list(x.shape)",
            "    ndims = len(shape)",
            "    if axis < 0: axis += ndims",
            "    C = shape[axis]",
            "    reshaped_dims = shape.copy()",
            "    reshaped_dims[axis:axis+1] = [groups, C // groups]",
            f"    reshaped_x = {config.reshape}(x, reshaped_dims)",
            "    reduction_axes = tuple(i for i in range(len(reshaped_dims)) if i != 0 and i != axis)",
            f"    return {config.mean}(reshaped_x, {config.dim_arg}=reduction_axes, {config.keepdim_arg}=True)",
            f"def {config.prefix}_group_variance(x, groups, axis=-1):",
            f"    import {config.module}",
            "    shape = list(x.shape)",
            "    ndims = len(shape)",
            "    if axis < 0: axis += ndims",
            "    C = shape[axis]",
            "    reshaped_dims = shape.copy()",
            "    reshaped_dims[axis:axis+1] = [groups, C // groups]",
            f"    reshaped_x = {config.reshape}(x, reshaped_dims)",
            "    reduction_axes = tuple(i for i in range(len(reshaped_dims)) if i != 0 and i != axis)",
            f"    return {config.var}(reshaped_x, {config.dim_arg}=reduction_axes, {config.keepdim_arg}=True{config.unbiased_arg})",
        ]
