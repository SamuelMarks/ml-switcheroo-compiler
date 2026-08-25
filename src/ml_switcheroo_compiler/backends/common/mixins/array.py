"""Module array.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Provide mixin module."""

from .common import CommonASTVisitor


class ArrayASTVisitor(CommonASTVisitor):
    # pylint: disable=abstract-method
    """Convert to array and shape manipulation AST generator mixin."""

    def visit_ApproxMaxK(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate the AST string for an approximate maximum k-elements operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of string variables representing the inputs to the operation.
            **kwargs: Additional keyword arguments, including 'k' and 'reduction_dimension'.

        Returns:
            A string containing the backend-specific code for the approximate max k operation.
        """
        pfx: object = self.generator.get_fallback_prefix()
        k: object = kwargs.get("k", 1)
        reduction_dimension: object = kwargs.get("reduction_dimension", -1)
        return f"{pfx}_approx_max_k({input_vars[0]}, k={k}, reduction_dimension={reduction_dimension})[0]"

    def visit_ApproxMaxKIndices(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate the AST string for retrieving the indices of an approximate maximum k-elements operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of string variables representing the inputs to the operation.
            **kwargs: Additional keyword arguments, including 'k' and 'reduction_dimension'.

        Returns:
            A string containing the backend-specific code to get the indices of the approximate max k elements.
        """
        pfx: object = self.generator.get_fallback_prefix()
        k: object = kwargs.get("k", 1)
        reduction_dimension: object = kwargs.get("reduction_dimension", -1)
        return f"{pfx}_approx_max_k({input_vars[0]}, k={k}, reduction_dimension={reduction_dimension})[1]"

    def visit_ApproxMinK(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate the AST string for an approximate minimum k-elements operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of string variables representing the inputs to the operation.
            **kwargs: Additional keyword arguments, including 'k' and 'reduction_dimension'.

        Returns:
            A string containing the backend-specific code for the approximate min k operation.
        """
        pfx: object = self.generator.get_fallback_prefix()
        k: object = kwargs.get("k", 1)
        reduction_dimension: object = kwargs.get("reduction_dimension", -1)
        return f"{pfx}_approx_min_k({input_vars[0]}, k={k}, reduction_dimension={reduction_dimension})[0]"

    def visit_ApproxMinKIndices(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate the AST string for retrieving the indices of an approximate minimum k-elements operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of string variables representing the inputs to the operation.
            **kwargs: Additional keyword arguments, including 'k' and 'reduction_dimension'.

        Returns:
            A string containing the backend-specific code to get the indices of the approximate min k elements.
        """
        pfx: object = self.generator.get_fallback_prefix()
        k: object = kwargs.get("k", 1)
        reduction_dimension: object = kwargs.get("reduction_dimension", -1)
        return f"{pfx}_approx_min_k({input_vars[0]}, k={k}, reduction_dimension={reduction_dimension})[1]"

    def visit_ArgSort(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate the AST string for computing the indices that would sort an array.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of string variables representing the inputs to the operation.
            **kwargs: Additional keyword arguments, including 'dimension'.

        Returns:
            A string containing the backend-specific code for the argsort operation.
        """
        pfx: object = self.generator.get_fallback_prefix()
        dimension: object = kwargs.get("dimension", -1)
        return f"{pfx}_argsort({input_vars[0]}, dimension={dimension})"

    def visit_Argwhere(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate the AST string for finding the indices of array elements that are non-zero.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of string variables representing the inputs to the operation.
            **kwargs: Additional keyword arguments.

        Returns:
            A string containing the backend-specific code for the argwhere operation.
        """
        pfx: object = self.generator.get_fallback_prefix()
        return f"{pfx}_argwhere({input_vars[0]})"

    def visit_Argpartition(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate the AST string for performing an indirect partition along the given axis.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of string variables representing the inputs to the operation.
            **kwargs: Additional keyword arguments, including 'kth' and 'axis'.

        Returns:
            A string containing the backend-specific code for the argpartition operation.
        """
        pfx: object = self.generator.get_fallback_prefix()
        kth: object = kwargs.get("kth")
        axis: object = kwargs.get("axis", -1)
        return f"{pfx}_argpartition({input_vars[0]}, kth={kth}, axis={axis})"

    def visit_AsString(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate the AST string for converting a tensor to its string representation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of string variables representing the inputs to the operation.
            **kwargs: Additional keyword arguments.

        Returns:
            A string containing the backend-specific code for string conversion.
        """
        pfx: object = self.generator.get_fallback_prefix()
        return f"{pfx}_as_string({input_vars[0]})"

    def visit_AxisIndex(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate the AST string for returning the index along a mapped axis.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of string variables representing the inputs to the operation.
            **kwargs: Additional keyword arguments, including 'axis_name'.

        Returns:
            A string containing the backend-specific code for retrieving the axis index.
        """
        pfx: object = self.generator.get_fallback_prefix()
        axis_name: object = kwargs.get("axis_name", "")
        return f"{pfx}_axis_index(axis_name='{axis_name}')"

    def visit_TopK(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate the AST string for finding the k largest entries for the given dimensions.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of string variables representing the inputs to the operation.
            **kwargs: Additional keyword arguments.

        Returns:
            A string containing the backend-specific code for computing the top k elements.
        """
        k: object = node.attributes.get("k", 1)
        k_val: object = k.expr if hasattr(k, "expr") else str(k)
        is_idx: object = node.attributes.get("return_indices", False)
        pfx: object = self.generator.get_fallback_prefix()
        var: object = input_vars[0]

        native: object = self._topk_native_dispatch(pfx, var, k_val, is_idx)
        if native is not None:
            return native

        return self._topk_sort_fallback(pfx, var, k_val, is_idx)

    def _topk_native_dispatch(self, pfx: str, var: str, k_val: str, is_idx: bool) -> str | None:
        """Retrieve the native top-k implementation string if it is supported by the backend.

        Args:
            pfx: A string indicating the backend prefix.
            var: The input variable string.
            k_val: A string representation of the k value.
            is_idx: A boolean indicating whether to return indices instead of values.

        Returns:
            A string with the native backend call, or None if no native function is defined.
        """
        idx_int: object = 1 if is_idx else 0
        idx_str: object = "indices" if is_idx else "values"

        dispatch: object = {
            "jax": f"jax.lax.top_k({var}, {k_val})[{idx_int}]",
            "torch": f"torch.topk({var}, {k_val}, dim=-1).{idx_str}",
            "pt": f"torch.topk({var}, {k_val}, dim=-1).{idx_str}",
            "tf": f"tf.math.top_k({var}, k={k_val})[{idx_int}]",
            "keras": f"keras.ops.top_k({var}, {k_val})[{idx_int}]",
            "keras.ops": f"keras.ops.top_k({var}, {k_val})[{idx_int}]",
        }
        return dispatch.get(pfx)

    def _topk_sort_fallback(self, pfx: str, var: str, k_val: str, is_idx: bool) -> str:
        """Generate a fallback implementation for top-k using sorting and slicing.

        Args:
            pfx: A string indicating the backend prefix.
            var: The input variable string.
            k_val: A string representation of the k value.
            is_idx: A boolean indicating whether to return indices instead of values.

        Returns:
            A string computing the top-k values or indices using a sort fallback.
        """
        op_pfx: object = "mx" if pfx == "mlx" else pfx
        op_fn: object = f"{op_pfx}.argsort" if is_idx else f"{op_pfx}.sort"
        return f"{op_fn}({var}, axis=-1)[..., -({k_val}):][..., ::-1]"

    def visit_Meshgrid(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate the AST string for creating coordinate matrices from coordinate vectors.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of string variables representing the inputs to the operation.
            **kwargs: Additional keyword arguments.

        Returns:
            A string containing the backend-specific code for generating a meshgrid.
        """
        idx: object = node.attributes.get("output_index", 0)
        indexing: object = node.attributes.get("indexing", "ij")
        pfx: object = self.generator.get_fallback_prefix()
        inputs_str: object = ", ".join(input_vars)
        if pfx == "mlx":
            return f"mx.meshgrid({inputs_str}, indexing='{indexing}')[{idx}]"
        elif pfx == "jax":
            return f"jnp.meshgrid({inputs_str}, indexing='{indexing}')[{idx}]"
        elif pfx == "torch" or pfx == "pt":
            return f"torch.meshgrid({inputs_str}, indexing='{indexing}')[{idx}]"
        else:
            fallback: object = getattr(self.generator, "get_fallback_prefix", lambda: "numpy")()
            return f"{fallback}.meshgrid({inputs_str}, indexing='{indexing}')[{idx}]"

    def visit_Slice(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate the AST string for a tensor slicing operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of string variables representing the inputs to the operation.
            **kwargs: Additional keyword arguments.

        Returns:
            A string containing the backend-specific code for the slice operation.
        """
        dim: object = node.attributes.get("dim")
        start: object = node.attributes.get("start")
        end: object = node.attributes.get("end")
        step: object = node.attributes.get("step", 1)

        start_str: object = "None" if start is None else str(start)
        end_str: object = "None" if end is None else str(end)
        step_str: object = "None" if step is None else str(step)

        if dim < 0:
            return f"{input_vars[0]}[(..., slice({start_str}, {end_str}, {step_str})) + (slice(None),) * ({-dim - 1})]"
        else:
            return f"{input_vars[0]}[(slice(None),) * ({dim}) + (slice({start_str}, {end_str}, {step_str}),) + (...,)]"

    def visit_DynamicSlice(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate the AST string for dynamically slicing a tensor.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of string variables representing the inputs to the operation.
            **kwargs: Additional keyword arguments.

        Returns:
            A string containing the backend-specific code for dynamic slicing.
        """
        operand: object = input_vars[0]
        starts: object = input_vars[1:]
        slice_sizes: object = node.attributes.get("slice_sizes", [])
        pfx: object = self.generator.get_fallback_prefix()

        starts_str: object = ", ".join(starts)
        if pfx == "jax":
            sizes_str: object = ", ".join(map(str, slice_sizes))
            return f"jax.lax.dynamic_slice({operand}, ({starts_str},), ({sizes_str},))"
        elif pfx == "tf":
            sizes_str: object = ", ".join(map(str, slice_sizes))
            return f"tf.slice({operand}, [{starts_str}], [{sizes_str}])"
        else:
            return f"{operand}[tuple(slice(s, s + sz) for s, sz in zip([{starts_str}], {list(slice_sizes)}))]"

    def visit_DynamicUpdateSlice(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate the AST string for dynamically updating a slice of a tensor.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of string variables representing the inputs to the operation.
            **kwargs: Additional keyword arguments.

        Returns:
            A string containing the backend-specific code for the dynamic update slice operation.
        """
        operand: object = input_vars[0]
        update: object = input_vars[1]
        starts: object = input_vars[2:]
        pfx: object = self.generator.get_fallback_prefix()

        starts_str: object = ", ".join(starts)
        if pfx == "jax":
            return f"jax.lax.dynamic_update_slice({operand}, {update}, ({starts_str},))"
        elif pfx == "tf":
            return f"tf.tensor_scatter_nd_update({operand}, tf.stack([{starts_str}], axis=-1), {update})"
        else:
            copy_meth: object = "clone()" if pfx in ("torch", "pt") else "copy()"
            return f"(lambda out: [out.__setitem__(tuple(slice(s, s + sz) for s, sz in zip([{starts_str}], {update}.shape)), {update}), out][1])({operand}.{copy_meth})"

    def visit_GetItem(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate the AST string for fetching an item from an array via an index or slice.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of string variables representing the inputs to the operation.
            **kwargs: Additional keyword arguments.

        Returns:
            A string containing the backend-specific code to get an item from the array.
        """
        key: object = node.attributes.get("key", "")
        return f"{input_vars[0]}[{key}]"

    def visit_PutAlongAxis(self, node: object, input_vars: list[str], **kwargs: object) -> str:
        """Generate the AST string for placing values into a destination array along a specified axis.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of string variables representing the inputs to the operation.
            **kwargs: Additional keyword arguments.

        Returns:
            A string containing the backend-specific code for putting elements along a specified axis.
        """
        return f"{self.generator.get_fallback_prefix()}.put_along_axis({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, axis={node.attributes.get('axis', None)})"
