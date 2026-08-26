"""Module array.py."""

from __future__ import annotations

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Provide mixin module."""

from typing import Any

from ml_switcheroo_compiler.ir.core import IRNode

from .common import CommonASTVisitor


class ArrayASTVisitor(CommonASTVisitor):
    # pylint: disable=abstract-method
    """Convert to array and shape manipulation AST generator mixin."""

    def visit_ApproxMaxK(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Generate the AST string for an approximate maximum k-elements operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of string variables representing the inputs to the operation.
            **kwargs: Additional keyword arguments, including 'k' and 'reduction_dimension'.

        Returns:
            A string containing the backend-specific code for the approximate max k operation.
        """
        pfx = self.generator.get_fallback_prefix()
        k = kwargs.get("k", 1)
        reduction_dimension = kwargs.get("reduction_dimension", -1)
        return f"{pfx}_approx_max_k({input_vars[0]}, k={k}, reduction_dimension={reduction_dimension})[0]"

    def visit_ApproxMaxKIndices(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Generate the AST string for retrieving the indices of an approximate maximum k-elements operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of string variables representing the inputs to the operation.
            **kwargs: Additional keyword arguments, including 'k' and 'reduction_dimension'.

        Returns:
            A string containing the backend-specific code to get the indices of the approximate max k elements.
        """
        pfx = self.generator.get_fallback_prefix()
        k = kwargs.get("k", 1)
        reduction_dimension = kwargs.get("reduction_dimension", -1)
        return f"{pfx}_approx_max_k({input_vars[0]}, k={k}, reduction_dimension={reduction_dimension})[1]"

    def visit_ApproxMinK(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Generate the AST string for an approximate minimum k-elements operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of string variables representing the inputs to the operation.
            **kwargs: Additional keyword arguments, including 'k' and 'reduction_dimension'.

        Returns:
            A string containing the backend-specific code for the approximate min k operation.
        """
        pfx = self.generator.get_fallback_prefix()
        k = kwargs.get("k", 1)
        reduction_dimension = kwargs.get("reduction_dimension", -1)
        return f"{pfx}_approx_min_k({input_vars[0]}, k={k}, reduction_dimension={reduction_dimension})[0]"

    def visit_ApproxMinKIndices(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Generate the AST string for retrieving the indices of an approximate minimum k-elements operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of string variables representing the inputs to the operation.
            **kwargs: Additional keyword arguments, including 'k' and 'reduction_dimension'.

        Returns:
            A string containing the backend-specific code to get the indices of the approximate min k elements.
        """
        pfx = self.generator.get_fallback_prefix()
        k = kwargs.get("k", 1)
        reduction_dimension = kwargs.get("reduction_dimension", -1)
        return f"{pfx}_approx_min_k({input_vars[0]}, k={k}, reduction_dimension={reduction_dimension})[1]"

    def visit_ArgSort(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Generate the AST string for computing the indices that would sort an array.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of string variables representing the inputs to the operation.
            **kwargs: Additional keyword arguments, including 'dimension'.

        Returns:
            A string containing the backend-specific code for the argsort operation.
        """
        pfx = self.generator.get_fallback_prefix()
        dimension = kwargs.get("dimension", -1)
        return f"{pfx}_argsort({input_vars[0]}, dimension={dimension})"

    def visit_Argwhere(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Generate the AST string for finding the indices of array elements that are non-zero.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of string variables representing the inputs to the operation.
            **kwargs: Additional keyword arguments.

        Returns:
            A string containing the backend-specific code for the argwhere operation.
        """
        pfx = self.generator.get_fallback_prefix()
        return f"{pfx}_argwhere({input_vars[0]})"

    def visit_Argpartition(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Generate the AST string for performing an indirect partition along the given axis.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of string variables representing the inputs to the operation.
            **kwargs: Additional keyword arguments, including 'kth' and 'axis'.

        Returns:
            A string containing the backend-specific code for the argpartition operation.
        """
        pfx = self.generator.get_fallback_prefix()
        kth = kwargs.get("kth")
        axis = kwargs.get("axis", -1)
        return f"{pfx}_argpartition({input_vars[0]}, kth={kth}, axis={axis})"

    def visit_AsString(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Generate the AST string for converting a tensor to its string representation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of string variables representing the inputs to the operation.
            **kwargs: Additional keyword arguments.

        Returns:
            A string containing the backend-specific code for string conversion.
        """
        pfx = self.generator.get_fallback_prefix()
        return f"{pfx}_as_string({input_vars[0]})"

    def visit_AxisIndex(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Generate the AST string for returning the index along a mapped axis.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of string variables representing the inputs to the operation.
            **kwargs: Additional keyword arguments, including 'axis_name'.

        Returns:
            A string containing the backend-specific code for retrieving the axis index.
        """
        pfx = self.generator.get_fallback_prefix()
        axis_name = kwargs.get("axis_name", "")
        return f"{pfx}_axis_index(axis_name='{axis_name}')"

    def visit_TopK(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Generate the AST string for finding the k largest entries for the given dimensions.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of string variables representing the inputs to the operation.
            **kwargs: Additional keyword arguments.

        Returns:
            A string containing the backend-specific code for computing the top k elements.
        """
        k = node.attributes.get("k", 1)
        k_val = k.expr if hasattr(k, "expr") else str(k)
        is_idx = node.attributes.get("return_indices", False)
        pfx = self.generator.get_fallback_prefix()
        var = input_vars[0]

        native = self._topk_native_dispatch(pfx, var, k_val, bool(is_idx))
        if native is not None:
            return native

        return self._topk_sort_fallback(pfx, var, k_val, bool(is_idx))

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
        """Generate a fallback implementation for top-k using sorting and slicing.

        Args:
            pfx: A string indicating the backend prefix.
            var: The input variable string.
            k_val: A string representation of the k value.
            is_idx: A boolean indicating whether to return indices instead of values.

        Returns:
            A string computing the top-k values or indices using a sort fallback.
        """
        op_pfx = "mx" if pfx == "mlx" else pfx
        op_fn = f"{op_pfx}.argsort" if is_idx else f"{op_pfx}.sort"
        return f"{op_fn}({var}, axis=-1)[..., -({k_val}):][..., ::-1]"

    def visit_Meshgrid(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Generate the AST string for creating coordinate matrices from coordinate vectors.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of string variables representing the inputs to the operation.
            **kwargs: Additional keyword arguments.

        Returns:
            A string containing the backend-specific code for generating a meshgrid.
        """
        idx = node.attributes.get("output_index", 0)
        indexing = node.attributes.get("indexing", "ij")
        pfx = self.generator.get_fallback_prefix()
        inputs_str = ", ".join(input_vars)
        if pfx == "mlx":
            return f"mx.meshgrid({inputs_str}, indexing='{indexing}')[{idx}]"
        elif pfx == "jax":
            return f"jnp.meshgrid({inputs_str}, indexing='{indexing}')[{idx}]"
        elif pfx in ("torch", "pt"):
            return f"torch.meshgrid({inputs_str}, indexing='{indexing}')[{idx}]"
        else:
            fallback = getattr(self.generator, "get_fallback_prefix", lambda: "numpy")()
            return f"{fallback}.meshgrid({inputs_str}, indexing='{indexing}')[{idx}]"

    def visit_Slice(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Generate the AST string for a tensor slicing operation.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of string variables representing the inputs to the operation.
            **kwargs: Additional keyword arguments.

        Returns:
            A string containing the backend-specific code for the slice operation.
        """
        dim = int(node.attributes.get("dim", 0))
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

    def visit_DynamicSlice(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Generate the AST string for dynamically slicing a tensor.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of string variables representing the inputs to the operation.
            **kwargs: Additional keyword arguments.

        Returns:
            A string containing the backend-specific code for dynamic slicing.
        """
        operand = input_vars[0]
        starts = input_vars[1:]
        slice_sizes = list(node.attributes.get("slice_sizes", []))
        pfx = self.generator.get_fallback_prefix()

        starts_str = ", ".join(starts)
        if pfx == "jax":
            sizes_str = ", ".join(map(str, slice_sizes))
            return f"jax.lax.dynamic_slice({operand}, ({starts_str},), ({sizes_str},))"
        elif pfx == "tf":
            sizes_str = ", ".join(map(str, slice_sizes))
            return f"tf.slice({operand}, [{starts_str}], [{sizes_str}])"
        else:
            return f"{operand}[tuple(slice(s, s + sz) for s, sz in zip([{starts_str}], {slice_sizes}))]"

    def visit_DynamicUpdateSlice(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Generate the AST string for dynamically updating a slice of a tensor.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of string variables representing the inputs to the operation.
            **kwargs: Additional keyword arguments.

        Returns:
            A string containing the backend-specific code for the dynamic update slice operation.
        """
        operand = input_vars[0]
        update = input_vars[1]
        starts = input_vars[2:]
        pfx = self.generator.get_fallback_prefix()

        starts_str = ", ".join(starts)
        if pfx == "jax":
            return f"jax.lax.dynamic_update_slice({operand}, {update}, ({starts_str},))"
        elif pfx == "tf":
            return f"tf.tensor_scatter_nd_update({operand}, tf.stack([{starts_str}], axis=-1), {update})"
        else:
            copy_meth = "clone()" if pfx in ("torch", "pt") else "copy()"
            return f"(lambda out: [out.__setitem__(tuple(slice(s, s + sz) for s, sz in zip([{starts_str}], {update}.shape)), {update}), out][1])({operand}.{copy_meth})"

    def visit_GetItem(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Generate the AST string for fetching an item from an array via an index or slice.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of string variables representing the inputs to the operation.
            **kwargs: Additional keyword arguments.

        Returns:
            A string containing the backend-specific code to get an item from the array.
        """
        key = str(node.attributes.get("key", ""))
        return f"{input_vars[0]}[{key}]"

    def visit_PutAlongAxis(self, node: IRNode, input_vars: list[str], **kwargs: Any) -> str:
        """Generate the AST string for placing values into a destination array along a specified axis.

        Args:
            node: The AST node representing the operation.
            input_vars: A list of string variables representing the inputs to the operation.
            **kwargs: Additional keyword arguments.

        Returns:
            A string containing the backend-specific code for putting elements along a specified axis.
        """
        return f"{self.generator.get_fallback_prefix()}.put_along_axis({input_vars[0]}, {input_vars[1]}, {input_vars[2]}, axis={node.attributes.get('axis', None)})"
