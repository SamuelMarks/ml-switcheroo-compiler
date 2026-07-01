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

    def visit_AdaptiveAvgPool2D(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AdaptiveAvgPool2D."""
        pfx = self._get_backend_prefix()
        output_size = kwargs.get("output_size", (1, 1))
        return f"{pfx}_adaptive_avg_pool2d({input_vars[0]}, {output_size})"

    def visit_AdaptiveMaxPool2D(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AdaptiveMaxPool2D."""
        pfx = self._get_backend_prefix()
        output_size = kwargs.get("output_size", (1, 1))
        return f"{pfx}_adaptive_max_pool2d({input_vars[0]}, {output_size})"

    def visit_AddN(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AddN."""
        if not input_vars:
            return "0.0"
        return " + ".join(input_vars)

    def visit_AccumulateN(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AccumulateN."""
        return self.visit_AddN(node, input_vars, **kwargs)

    def visit_Scan(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Scan."""
        pfx = self._get_backend_prefix()
        # Natively, backends implement this as a specific scan.
        return f"{pfx}_scan({', '.join(input_vars)})"

    def visit_Switch(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Switch."""
        pfx = self._get_backend_prefix()
        # Fallback to a custom runner
        return f"{pfx}_switch({', '.join(input_vars)})"

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

    def visit_AdjustBrightness(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AdjustBrightness."""
        pfx = self._get_backend_prefix()
        delta = kwargs.get("delta", 0.0)
        return f"{pfx}_adjust_brightness({input_vars[0]}, {delta})"

    def visit_AdjustContrast(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AdjustContrast."""
        pfx = self._get_backend_prefix()
        contrast_factor = kwargs.get("contrast_factor", 1.0)
        return f"{pfx}_adjust_contrast({input_vars[0]}, {contrast_factor})"

    def visit_AdjustHue(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AdjustHue."""
        pfx = self._get_backend_prefix()
        delta = kwargs.get("delta", 0.0)
        return f"{pfx}_adjust_hue({input_vars[0]}, {delta})"

    def visit_AdjustSaturation(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AdjustSaturation."""
        pfx = self._get_backend_prefix()
        saturation_factor = kwargs.get("saturation_factor", 1.0)
        return f"{pfx}_adjust_saturation({input_vars[0]}, {saturation_factor})"

    def visit_AffineGenerator(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AffineGenerator."""
        pfx = self._get_backend_prefix()
        batch_size = kwargs.get("batch_size", 1)
        return f"{pfx}_affine_generator({batch_size}, {', '.join(input_vars)})"

    def visit_AffineGrid(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AffineGrid."""
        pfx = self._get_backend_prefix()
        size = kwargs.get("size", ())
        align_corners = kwargs.get("align_corners", False)
        return f"{pfx}_affine_grid({input_vars[0]}, size={size}, align_corners={align_corners})"

    def visit_AffineTransform(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AffineTransform."""
        pfx = self._get_backend_prefix()
        interpolation = kwargs.get("interpolation", "nearest")
        return f"{pfx}_affine_transform({input_vars[0]}, {input_vars[1]}, interpolation='{interpolation}')"

    def visit_AllGather(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AllGather."""
        pfx = self._get_backend_prefix()
        axis = kwargs.get("axis", 0)
        return f"{pfx}_all_gather({input_vars[0]}, axis={axis})"

    def visit_AllReduce(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AllReduce."""
        pfx = self._get_backend_prefix()
        op = kwargs.get("op", "sum")
        return f"{pfx}_all_reduce({input_vars[0]}, op='{op}')"

    def visit_AllToAll(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AllToAll."""
        pfx = self._get_backend_prefix()
        split_axis = kwargs.get("split_axis", 0)
        concat_axis = kwargs.get("concat_axis", 0)
        axis_name = kwargs.get("axis_name", "")
        return f"{pfx}_all_to_all({input_vars[0]}, split_axis={split_axis}, concat_axis={concat_axis}, axis_name='{axis_name}')"

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

    def visit_AlphaDropout(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AlphaDropout."""
        pfx = self._get_backend_prefix()
        rate = kwargs.get("rate", 0.5)
        config = kwargs.get("config", None)
        if config:
            training = getattr(config, "training", False)
            noise_shape = getattr(config, "noise_shape", None)
            seed = getattr(config, "seed", None)
            config_str = f"config={pfx}_DropoutConfig(training={training}, noise_shape={noise_shape}, seed={seed})"
            return f"{pfx}_alpha_dropout({input_vars[0]}, rate={rate}, {config_str})"
        return f"{pfx}_alpha_dropout({input_vars[0]}, rate={rate})"

    def visit_Angle(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Angle."""
        pfx = self._get_backend_prefix()
        return f"{pfx}_angle({input_vars[0]})"

    def visit_ApproxMaxK(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate ApproxMaxK."""
        pfx = self._get_backend_prefix()
        k = kwargs.get("k", 1)
        reduction_dimension = kwargs.get("reduction_dimension", -1)
        return f"{pfx}_approx_max_k({input_vars[0]}, k={k}, reduction_dimension={reduction_dimension})[0]"

    def visit_ApproxMaxKIndices(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate ApproxMaxKIndices."""
        pfx = self._get_backend_prefix()
        k = kwargs.get("k", 1)
        reduction_dimension = kwargs.get("reduction_dimension", -1)
        return f"{pfx}_approx_max_k({input_vars[0]}, k={k}, reduction_dimension={reduction_dimension})[1]"

    def visit_ApproxMinK(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate ApproxMinK."""
        pfx = self._get_backend_prefix()
        k = kwargs.get("k", 1)
        reduction_dimension = kwargs.get("reduction_dimension", -1)
        return f"{pfx}_approx_min_k({input_vars[0]}, k={k}, reduction_dimension={reduction_dimension})[0]"

    def visit_ApproxMinKIndices(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate ApproxMinKIndices."""
        pfx = self._get_backend_prefix()
        k = kwargs.get("k", 1)
        reduction_dimension = kwargs.get("reduction_dimension", -1)
        return f"{pfx}_approx_min_k({input_vars[0]}, k={k}, reduction_dimension={reduction_dimension})[1]"

    def visit_ArgSort(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate ArgSort."""
        pfx = self._get_backend_prefix()
        dimension = kwargs.get("dimension", -1)
        return f"{pfx}_argsort({input_vars[0]}, dimension={dimension})"

    def visit_Argwhere(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Argwhere."""
        pfx = self._get_backend_prefix()
        return f"{pfx}_argwhere({input_vars[0]})"

    def visit_Argpartition(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Argpartition."""
        pfx = self._get_backend_prefix()
        kth = kwargs.get("kth")
        axis = kwargs.get("axis", -1)
        return f"{pfx}_argpartition({input_vars[0]}, kth={kth}, axis={axis})"

    def visit_AsString(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AsString."""
        pfx = self._get_backend_prefix()
        return f"{pfx}_as_string({input_vars[0]})"

    def visit_Assert(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Assert."""
        pfx = self._get_backend_prefix()
        data = kwargs.get("data", ["Assertion failed."])
        return f"{pfx}_assert({input_vars[0]}, data={data})"

    def visit_Assign(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Assign."""
        pfx = self._get_backend_prefix()
        return f"{pfx}_assign({input_vars[0]}, {input_vars[1]})"

    def visit_AssignAdd(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AssignAdd."""
        pfx = self._get_backend_prefix()
        return f"{pfx}_assign_add({input_vars[0]}, {input_vars[1]})"

    def visit_AssignSub(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AssignSub."""
        pfx = self._get_backend_prefix()
        return f"{pfx}_assign_sub({input_vars[0]}, {input_vars[1]})"

    def visit_AssociativeScan(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AssociativeScan."""
        pfx = self._get_backend_prefix()
        return f"{pfx}_associative_scan({', '.join(input_vars)})"

    def visit_AugMix(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AugMix."""
        pfx = self._get_backend_prefix()
        return f"{pfx}_augmix({input_vars[0]})"

    def visit_AutoContrast(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AutoContrast."""
        pfx = self._get_backend_prefix()
        return f"{pfx}_auto_contrast({input_vars[0]})"

    def visit_AxisIndex(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AxisIndex."""
        pfx = self._get_backend_prefix()
        axis_name = kwargs.get("axis_name", "")
        return f"{pfx}_axis_index(axis_name='{axis_name}')"

    def visit_Ball(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Ball."""
        pfx = self._get_backend_prefix()
        return f"{pfx}_ball()"

    def visit_BandPart(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate BandPart."""
        pfx = self._get_backend_prefix()
        num_lower = kwargs.get("num_lower", -1)
        num_upper = kwargs.get("num_upper", -1)
        return f"{pfx}_band_part({input_vars[0]}, {num_lower}, {num_upper})"

    def visit_BandedTriangularSolve(
        self, node: object, input_vars: list[str], **kwargs: object
    ) -> str:
        """Evaluate BandedTriangularSolve."""
        pfx = self._get_backend_prefix()
        lower = kwargs.get("lower", False)
        adjoint = kwargs.get("adjoint", False)
        return f"{pfx}_banded_triangular_solve({input_vars[0]}, {input_vars[1]}, lower={lower}, adjoint={adjoint})"

    def visit_Bartlett(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Bartlett."""
        pfx = self._get_backend_prefix()
        return f"{pfx}_bartlett({input_vars[0]})"

    def visit_BesselI0(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate BesselI0."""
        pfx = self._get_backend_prefix()
        return f"{pfx}_bessel_i0({input_vars[0]})"

    def visit_BesselI0e(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate BesselI0e."""
        pfx = self._get_backend_prefix()
        return f"{pfx}_bessel_i0e({input_vars[0]})"

    def visit_BesselI1(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate BesselI1."""
        pfx = self._get_backend_prefix()
        return f"{pfx}_bessel_i1({input_vars[0]})"

    def visit_BesselI1e(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate BesselI1e."""
        pfx = self._get_backend_prefix()
        return f"{pfx}_bessel_i1e({input_vars[0]})"

    def visit_BesselJ0(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate BesselJ0."""
        pfx = self._get_backend_prefix()
        return f"{pfx}_bessel_j0({input_vars[0]})"

    def visit_BesselJ1(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate BesselJ1."""
        pfx = self._get_backend_prefix()
        return f"{pfx}_bessel_j1({input_vars[0]})"

    def visit_BesselJn(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate BesselJn."""
        pfx = self._get_backend_prefix()
        return f"{pfx}_bessel_jn({input_vars[0]}, {input_vars[1]})"

    def visit_BesselK0(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate BesselK0."""
        pfx = self._get_backend_prefix()
        return f"{pfx}_bessel_k0({input_vars[0]})"

    def visit_BesselK0e(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate BesselK0e."""
        pfx = self._get_backend_prefix()
        return f"{pfx}_bessel_k0e({input_vars[0]})"

    def visit_BesselK1(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate BesselK1."""
        pfx = self._get_backend_prefix()
        return f"{pfx}_bessel_k1({input_vars[0]})"

    def visit_BesselK1e(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate BesselK1e."""
        pfx = self._get_backend_prefix()
        return f"{pfx}_bessel_k1e({input_vars[0]})"

    def visit_BesselY0(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate BesselY0."""
        pfx = self._get_backend_prefix()
        return f"{pfx}_bessel_y0({input_vars[0]})"

    def visit_BesselY1(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate BesselY1."""
        pfx = self._get_backend_prefix()
        return f"{pfx}_bessel_y1({input_vars[0]})"

    def visit_Beta(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Beta."""
        pfx = self._get_backend_prefix()
        return f"{pfx}_beta({input_vars[0]}, {input_vars[1]})"

    def visit_Betainc(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Betainc."""
        pfx = self._get_backend_prefix()
        return f"{pfx}_betainc({input_vars[0]}, {input_vars[1]}, {input_vars[2]})"

    def visit_TopK(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate TopK."""
        k = node.attributes.get("k", 1)
        k_val = k.expr if hasattr(k, "expr") else str(k)
        is_idx = node.attributes.get("return_indices", False)
        pfx = self._get_backend_prefix()
        var = input_vars[0]

        if pfx == "jax":
            idx = 1 if is_idx else 0
            return f"jax.lax.top_k({var}, {k_val})[{idx}]"
        elif pfx in ("torch", "pt"):
            idx_attr = "indices" if is_idx else "values"
            return f"torch.topk({var}, {k_val}, dim=-1).{idx_attr}"
        elif pfx == "tf":
            idx = 1 if is_idx else 0
            return f"tf.math.top_k({var}, k={k_val})[{idx}]"
        elif pfx in ("keras", "keras.ops"):
            idx = 1 if is_idx else 0
            return f"keras.ops.top_k({var}, {k_val})[{idx}]"
        else:
            op_pfx = "mx" if pfx == "mlx" else pfx
            op_fn = f"{op_pfx}.argsort" if is_idx else f"{op_pfx}.sort"
            return f"{op_fn}({var}, axis=-1)[..., -({k_val}):][..., ::-1]"

    def visit_Meshgrid(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Meshgrid."""
        idx = node.attributes.get("output_index", 0)
        indexing = node.attributes.get("indexing", "ij")
        pfx = self._get_backend_prefix()
        inputs_str = ", ".join(input_vars)
        if pfx == "mlx":
            return f"mx.meshgrid({inputs_str}, indexing='{indexing}')[{idx}]"
        elif pfx == "jax":
            return f"jnp.meshgrid({inputs_str}, indexing='{indexing}')[{idx}]"
        elif pfx == "torch" or pfx == "pt":
            return f"torch.meshgrid({inputs_str}, indexing='{indexing}')[{idx}]"
        else:
            return f"np.meshgrid({inputs_str}, indexing='{indexing}')[{idx}]"

    def visit_Slice(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Slice."""
        dim = node.attributes.get("dim")
        start = node.attributes.get("start")
        end = node.attributes.get("end")
        step = node.attributes.get("step", 1)

        start_str = "None" if start is None else str(start)
        end_str = "None" if end is None else str(end)
        step_str = "None" if step is None else str(step)

        if dim < 0:
            return f"{input_vars[0]}[(..., slice({start_str}, {end_str}, {step_str})) + (slice(None),) * ({-dim - 1})]"
        else:
            return f"{input_vars[0]}[(slice(None),) * ({dim}) + (slice({start_str}, {end_str}, {step_str}),) + (...,)]"

    def visit_DynamicSlice(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate DynamicSlice."""
        operand = input_vars[0]
        starts = input_vars[1:]
        slice_sizes = node.attributes.get("slice_sizes", [])
        pfx = self._get_backend_prefix()

        starts_str = ", ".join(starts)
        if pfx == "jax":
            sizes_str = ", ".join(map(str, slice_sizes))
            return f"jax.lax.dynamic_slice({operand}, ({starts_str},), ({sizes_str},))"
        elif pfx == "tf":
            sizes_str = ", ".join(map(str, slice_sizes))
            return f"tf.slice({operand}, [{starts_str}], [{sizes_str}])"
        else:
            return f"{operand}[tuple(slice(s, s + sz) for s, sz in zip([{starts_str}], {list(slice_sizes)}))]"

    def visit_DynamicUpdateSlice(
        self, node: object, input_vars: list[str], **kwargs: object
    ) -> str:
        """Evaluate DynamicUpdateSlice."""
        operand = input_vars[0]
        update = input_vars[1]
        starts = input_vars[2:]
        pfx = self._get_backend_prefix()

        starts_str = ", ".join(starts)
        if pfx == "jax":
            return f"jax.lax.dynamic_update_slice({operand}, {update}, ({starts_str},))"
        elif pfx == "tf":
            return f"tf.tensor_scatter_nd_update({operand}, tf.stack([{starts_str}], axis=-1), {update})"
        else:
            copy_meth = "clone()" if pfx in ("torch", "pt") else "copy()"
            return f"(lambda out: [out.__setitem__(tuple(slice(s, s + sz) for s, sz in zip([{starts_str}], {update}.shape)), {update}), out][1])({operand}.{copy_meth})"

    def visit_Quantize(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Quantize."""
        group_size = node.attributes.get("group_size", 64)
        bits = node.attributes.get("bits", 4)
        idx = node.attributes.get("return_idx", 0)
        pfx = self._get_backend_prefix()

        if pfx in ("mlx", "mx"):
            return f"mx.quantize({input_vars[0]}, group_size={group_size}, bits={bits})[{idx}]"
        return f"{input_vars[0]}"

    def visit_Dropout2d(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Dropout2d."""
        return f"{self._get_backend_prefix()}.dropout2d({input_vars[0]}, p={node.attributes.get('p', 0.5)})"

    def visit_Dropout3d(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Dropout3d."""
        return f"{self._get_backend_prefix()}.dropout3d({input_vars[0]}, p={node.attributes.get('p', 0.5)})"

    def visit_GatherMm(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate GatherMm."""
        args_str = f"{input_vars[0]}, {input_vars[1]}"
        if "lhs_indices" in node.attributes:
            args_str += f", lhs_indices={input_vars[node.attributes['lhs_indices']]}"
        if "rhs_indices" in node.attributes:
            args_str += f", rhs_indices={input_vars[node.attributes['rhs_indices']]}"
        return f"{self._get_backend_prefix()}.gather_mm({args_str})"

    def visit_SegmentedMm(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate SegmentedMm."""
        return f"{self._get_backend_prefix()}.segmented_mm({input_vars[0]}, {input_vars[1]}, {input_vars[node.attributes.get('segments', 2)]})"

    def visit_PutAlongAxis(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate PutAlongAxis."""
        return f"{self._get_backend_prefix()}.put_along_axis({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, axis={node.attributes.get('axis', None)})"

    def visit_Logcumsumexp(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Logcumsumexp."""
        return f"{self._get_backend_prefix()}.logcumsumexp({input_vars[0]}, axis={node.attributes.get('axis', None)})"

    def visit_Gru(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Gru."""
        return f"{self._get_backend_prefix()}.gru({', '.join(input_vars)})"

    def visit_GetItem(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate GetItem."""
        key = node.attributes.get("key", "")
        return f"{input_vars[0]}[{key}]"

    def visit_BlockMaskedMm(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate BlockMaskedMm."""
        a = input_vars[0]
        b = input_vars[1]
        out = f"{self._get_backend_prefix()}.matmul({a}, {b})"
        return out

    def visit_QuantizedMatmul(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate QuantizedMatmul."""
        transpose = node.attributes.get("transpose", True)
        group_size = node.attributes.get("group_size", 64)
        bits = node.attributes.get("bits", 4)
        pfx = self._get_backend_prefix()

        x, w, scales, biases = input_vars[0], input_vars[1], input_vars[2], input_vars[3]
        if pfx in ("mlx", "mx"):
            return f"mx.quantized_matmul({x}, {w}, {scales}, {biases}, transpose={transpose}, group_size={group_size}, bits={bits})"
        return f"{pfx}.matmul({x}, {w}.T if {transpose} else {w})"

    def visit_GatherQMM(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate GatherQMM."""
        transpose = node.attributes.get("transpose", True)
        group_size = node.attributes.get("group_size", 64)
        bits = node.attributes.get("bits", 4)
        pfx = self._get_backend_prefix()

        x, w, scales, biases, indices = (
            input_vars[0],
            input_vars[1],
            input_vars[2],
            input_vars[3],
            input_vars[4],
        )
        if pfx in ("mlx", "mx"):
            return f"mx.gather_qmm({x}, {w}, {scales}, {biases}, {indices}, transpose={transpose}, group_size={group_size}, bits={bits})"
        return f"{pfx}.matmul({x}, {w}[{indices}].T if {transpose} else {w}[{indices}])"

    def visit_ConvGeneralDilated(
        self, node: object, input_vars: list[str], **kwargs: object
    ) -> str:
        """Evaluate ConvGeneralDilated."""
        lhs = input_vars[0]
        rhs = input_vars[1]
        config = node.attributes.get("config")
        pfx = self._get_backend_prefix()

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
        pfx = self._get_backend_prefix()

        if pfx == "jax":
            return f"jax.lax.conv_transpose({lhs}, {rhs}, strides={strides}, padding='{padding}', rhs_dilation={rhs_dilation})"
        elif pfx in ("mlx", "mx"):
            return f"mx.conv_transpose({lhs}, {rhs}, strides={strides}, padding='{padding}', kernel_dilation={rhs_dilation}, input_dilation={lhs_dilation}, groups={groups})"
        else:
            return f"{pfx}_conv_transpose({lhs}, {rhs}, {strides}, '{padding}', {lhs_dilation}, {rhs_dilation}, {groups})"
