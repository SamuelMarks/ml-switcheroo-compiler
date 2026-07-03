"""Mixin module."""

from __future__ import annotations

from dataclasses import dataclass

from .common import CommonASTVisitor


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


class NNASTVisitor(CommonASTVisitor):
    # pylint: disable=abstract-method
    """Neural Network AST generator mixin."""

    def visit_AdaptiveAvgPool2D(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AdaptiveAvgPool2D."""
        pfx = self.generator._get_backend_prefix()
        output_size = kwargs.get("output_size", (1, 1))
        return f"{pfx}_adaptive_avg_pool2d({input_vars[0]}, {output_size})"

    def visit_AdaptiveMaxPool2D(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AdaptiveMaxPool2D."""
        pfx = self.generator._get_backend_prefix()
        output_size = kwargs.get("output_size", (1, 1))
        return f"{pfx}_adaptive_max_pool2d({input_vars[0]}, {output_size})"

    def visit_ActivityRegularization(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate activity regularization."""
        # It's an identity op as loss is handled externally
        return input_vars[0]  # pragma: no cover

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

        return f"{self.generator._get_backend_prefix()}_group_norm({input_vars[0]}, {groups}, {weight_str}, {bias_str}, {axis}, {epsilon})"  # pragma: no cover

    def visit_GroupMean(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate group mean."""
        groups = kwargs.get("groups")  # pragma: no cover
        axis = kwargs.get("axis", -1)  # pragma: no cover
        return f"{self.generator._get_backend_prefix()}_group_mean({input_vars[0]}, {groups}, {axis})"  # pragma: no cover

    def visit_GroupVariance(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate group variance."""
        groups = kwargs.get("groups")  # pragma: no cover
        axis = kwargs.get("axis", -1)  # pragma: no cover
        return f"{self.generator._get_backend_prefix()}_group_variance({input_vars[0]}, {groups}, {axis})"  # pragma: no cover

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

    def visit_AlphaDropout(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AlphaDropout."""
        pfx = self.generator._get_backend_prefix()
        rate = kwargs.get("rate", 0.5)
        config = kwargs.get("config", None)
        if config:
            training = getattr(config, "training", False)
            noise_shape = getattr(config, "noise_shape", None)
            seed = getattr(config, "seed", None)
            config_str = f"config={pfx}_DropoutConfig(training={training}, noise_shape={noise_shape}, seed={seed})"
            return f"{pfx}_alpha_dropout({input_vars[0]}, rate={rate}, {config_str})"
        return f"{pfx}_alpha_dropout({input_vars[0]}, rate={rate})"

    def visit_Dropout2d(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Dropout2d."""
        return f"{self.generator._get_backend_prefix()}.dropout2d({input_vars[0]}, p={node.attributes.get('p', 0.5)})"

    def visit_Dropout3d(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Dropout3d."""
        return f"{self.generator._get_backend_prefix()}.dropout3d({input_vars[0]}, p={node.attributes.get('p', 0.5)})"

    def visit_Gru(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Gru."""
        return f"{self.generator._get_backend_prefix()}.gru({', '.join(input_vars)})"

    def visit_ConvGeneralDilated(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate ConvGeneralDilated."""
        lhs = input_vars[0]
        rhs = input_vars[1]
        config = node.attributes.get("config")
        pfx = self.generator._get_backend_prefix()

        # We need to extract config fields. They might be in a dict if serialized, or an object.
        if hasattr(config, "window_strides"):
            strides = config.window_strides
            padding = config.padding
            lhs_dilation = config.lhs_dilation
            rhs_dilation = config.rhs_dilation
            groups = getattr(config, "feature_group_count", 1)
        else:
            strides = 1
            padding = 0
            lhs_dilation = 1
            rhs_dilation = 1
            groups = 1

        if pfx == "jax":
            # JAX uses dimension_numbers. We assume NHWC by default if not specified.
            return f"jax.lax.conv_general_dilated({lhs}, {rhs}, window_strides={strides}, padding={padding}, lhs_dilation={lhs_dilation}, rhs_dilation={rhs_dilation}, feature_group_count={groups})"
        elif pfx in ("mlx", "mx"):
            return f"mx.conv_general({lhs}, {rhs}, strides={strides}, padding={padding}, kernel_dilation={rhs_dilation}, input_dilation={lhs_dilation}, groups={groups})"
        else:
            # For PyTorch/Numpy, we emit a mock or simplified version
            return f"{pfx}_conv_general_dilated({lhs}, {rhs}, {strides}, {padding}, {lhs_dilation}, {rhs_dilation}, {groups})"

    def visit_ConvTranspose(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate ConvTranspose."""
        lhs = input_vars[0]
        rhs = input_vars[1]
        strides = node.attributes.get("strides", 1)
        padding = node.attributes.get("padding", "VALID")
        lhs_dilation = node.attributes.get("lhs_dilation", 1)
        rhs_dilation = node.attributes.get("rhs_dilation", 1)
        groups = node.attributes.get("groups", 1)
        pfx = self.generator._get_backend_prefix()

        if pfx == "jax":
            return f"jax.lax.conv_transpose({lhs}, {rhs}, strides={strides}, padding='{padding}', rhs_dilation={rhs_dilation})"
        elif pfx in ("mlx", "mx"):
            return f"mx.conv_transpose({lhs}, {rhs}, strides={strides}, padding='{padding}', kernel_dilation={rhs_dilation}, input_dilation={lhs_dilation}, groups={groups})"
        else:
            return f"{pfx}_conv_transpose({lhs}, {rhs}, {strides}, '{padding}', {lhs_dilation}, {rhs_dilation}, {groups})"
