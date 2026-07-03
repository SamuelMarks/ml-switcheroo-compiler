"""Mixin module."""

from __future__ import annotations

from .common import CommonASTVisitor


class ArrayASTVisitor(CommonASTVisitor):
    # pylint: disable=abstract-method
    """Array and shape manipulation AST generator mixin."""

    def visit_ApproxMaxK(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate ApproxMaxK."""
        pfx = self.generator._get_backend_prefix()
        k = kwargs.get("k", 1)
        reduction_dimension = kwargs.get("reduction_dimension", -1)
        return f"{pfx}_approx_max_k({input_vars[0]}, k={k}, reduction_dimension={reduction_dimension})[0]"

    def visit_ApproxMaxKIndices(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate ApproxMaxKIndices."""
        pfx = self.generator._get_backend_prefix()
        k = kwargs.get("k", 1)
        reduction_dimension = kwargs.get("reduction_dimension", -1)
        return f"{pfx}_approx_max_k({input_vars[0]}, k={k}, reduction_dimension={reduction_dimension})[1]"

    def visit_ApproxMinK(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate ApproxMinK."""
        pfx = self.generator._get_backend_prefix()
        k = kwargs.get("k", 1)
        reduction_dimension = kwargs.get("reduction_dimension", -1)
        return f"{pfx}_approx_min_k({input_vars[0]}, k={k}, reduction_dimension={reduction_dimension})[0]"

    def visit_ApproxMinKIndices(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate ApproxMinKIndices."""
        pfx = self.generator._get_backend_prefix()
        k = kwargs.get("k", 1)
        reduction_dimension = kwargs.get("reduction_dimension", -1)
        return f"{pfx}_approx_min_k({input_vars[0]}, k={k}, reduction_dimension={reduction_dimension})[1]"

    def visit_ArgSort(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate ArgSort."""
        pfx = self.generator._get_backend_prefix()
        dimension = kwargs.get("dimension", -1)
        return f"{pfx}_argsort({input_vars[0]}, dimension={dimension})"

    def visit_Argwhere(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Argwhere."""
        pfx = self.generator._get_backend_prefix()
        return f"{pfx}_argwhere({input_vars[0]})"

    def visit_Argpartition(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Argpartition."""
        pfx = self.generator._get_backend_prefix()
        kth = kwargs.get("kth")
        axis = kwargs.get("axis", -1)
        return f"{pfx}_argpartition({input_vars[0]}, kth={kth}, axis={axis})"

    def visit_AsString(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AsString."""
        pfx = self.generator._get_backend_prefix()
        return f"{pfx}_as_string({input_vars[0]})"

    def visit_AxisIndex(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate AxisIndex."""
        pfx = self.generator._get_backend_prefix()
        axis_name = kwargs.get("axis_name", "")
        return f"{pfx}_axis_index(axis_name='{axis_name}')"

    def visit_TopK(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate TopK."""
        k = node.attributes.get("k", 1)
        k_val = k.expr if hasattr(k, "expr") else str(k)
        is_idx = node.attributes.get("return_indices", False)
        pfx = self.generator._get_backend_prefix()
        var = input_vars[0]

        native = self._topk_native_dispatch(pfx, var, k_val, is_idx)
        if native is not None:
            return native

        return self._topk_sort_fallback(pfx, var, k_val, is_idx)

    def _topk_native_dispatch(self, pfx: str, var: str, k_val: str, is_idx: bool) -> str | None:
        """Gets native top_k AST string if supported."""
        idx_int = 1 if is_idx else 0
        idx_str = "indices" if is_idx else "values"

        dispatch = {
            "jax": f"jax.lax.top_k({var}, {k_val})[{idx_int}]",
            "torch": f"torch.topk({var}, {k_val}, dim=-1).{idx_str}",
            "pt": f"torch.topk({var}, {k_val}, dim=-1).{idx_str}",
            "tf": f"tf.math.top_k({var}, k={k_val})[{idx_int}]",
            "keras": f"keras.ops.top_k({var}, {k_val})[{idx_int}]",
            "keras.ops": f"keras.ops.top_k({var}, {k_val})[{idx_int}]",
        }
        return dispatch.get(pfx)

    def _topk_sort_fallback(self, pfx: str, var: str, k_val: str, is_idx: bool) -> str:
        """Generates fallback top_k using sort/argsort and slicing."""
        op_pfx = "mx" if pfx == "mlx" else pfx
        op_fn = f"{op_pfx}.argsort" if is_idx else f"{op_pfx}.sort"
        return f"{op_fn}({var}, axis=-1)[..., -({k_val}):][..., ::-1]"

    def visit_Meshgrid(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate Meshgrid."""
        idx = node.attributes.get("output_index", 0)
        indexing = node.attributes.get("indexing", "ij")
        pfx = self.generator._get_backend_prefix()
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
        pfx = self.generator._get_backend_prefix()

        starts_str = ", ".join(starts)
        if pfx == "jax":
            sizes_str = ", ".join(map(str, slice_sizes))
            return f"jax.lax.dynamic_slice({operand}, ({starts_str},), ({sizes_str},))"
        elif pfx == "tf":
            sizes_str = ", ".join(map(str, slice_sizes))
            return f"tf.slice({operand}, [{starts_str}], [{sizes_str}])"
        else:
            return f"{operand}[tuple(slice(s, s + sz) for s, sz in zip([{starts_str}], {list(slice_sizes)}))]"

    def visit_DynamicUpdateSlice(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate DynamicUpdateSlice."""
        operand = input_vars[0]
        update = input_vars[1]
        starts = input_vars[2:]
        pfx = self.generator._get_backend_prefix()

        starts_str = ", ".join(starts)
        if pfx == "jax":
            return f"jax.lax.dynamic_update_slice({operand}, {update}, ({starts_str},))"
        elif pfx == "tf":
            return f"tf.tensor_scatter_nd_update({operand}, tf.stack([{starts_str}], axis=-1), {update})"
        else:
            copy_meth = "clone()" if pfx in ("torch", "pt") else "copy()"
            return f"(lambda out: [out.__setitem__(tuple(slice(s, s + sz) for s, sz in zip([{starts_str}], {update}.shape)), {update}), out][1])({operand}.{copy_meth})"

    def visit_GetItem(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate GetItem."""
        key = node.attributes.get("key", "")
        return f"{input_vars[0]}[{key}]"

    def visit_PutAlongAxis(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Evaluate PutAlongAxis."""
        return f"{self.generator._get_backend_prefix()}.put_along_axis({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, axis={node.attributes.get('axis', None)})"
