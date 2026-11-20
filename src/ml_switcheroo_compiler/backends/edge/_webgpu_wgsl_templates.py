"""Declarative WGSL shader template argument formatting and kernel emission."""

from __future__ import annotations

import typing  # noqa: F401

from ml_switcheroo_ir import IRNode

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_int,
    coerce_int_list,
    coerce_str,
)
from ml_switcheroo_compiler.backends.edge._webgpu_base import (
    _node_attrs_dict,
    _node_inputs_list,
    _WebGPUGenBase,
)
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module


class _WebGPUWgslTemplates(_WebGPUGenBase):
    """WGSL shader template argument population and workgroup dispatch formatting."""

    def _build_offset_format_args(
        self,
        node: IRNode,
        shape: list[int],
        out_strides: list[int],
        input_nodes: list[IRNode | None],
        in0_shape: list[int],
        in0_strides: list[int],
        in1_shape: list[int],
        in1_strides: list[int],
    ) -> dict[str, str]:
        """Build dynamic offset WGSL code snippets for output and input tensors.

        Args:
            node (IRNode): Current operation node.
            shape (list[int]): Output tensor shape.
            out_strides (list[int]): Output tensor strides.
            input_nodes (list[IRNode | None]): Resolved input IR nodes.
            in0_shape (list[int]): First input shape.
            in0_strides (list[int]): First input strides.
            in1_shape (list[int]): Second input shape.
            in1_strides (list[int]): Second input strides.

        Returns:
            dict[str, str]: Mapping of offset code placeholders to WGSL code strings.
        """
        out_code: str = self._emit_node_offset_code(node, shape, out_strides, "out_offset")
        in0_code: str = (
            self._emit_node_offset_code(input_nodes[0], in0_shape, in0_strides, "in0_offset")
            if len(input_nodes) > 0
            else ""
        )
        in1_code: str = (
            self._emit_node_offset_code(input_nodes[1], in1_shape, in1_strides, "in1_offset")
            if len(input_nodes) > 1
            else ""
        )
        in2_shape, in2_strides = (
            self._get_shape_and_strides(input_nodes[2]) if len(input_nodes) > 2 and input_nodes[2] else ([], [])
        )
        in2_code: str = (
            self._emit_node_offset_code(input_nodes[2], in2_shape, in2_strides, "in2_offset")
            if len(input_nodes) > 2
            else ""
        )
        return {
            "out_offset_code": out_code,
            "in0_offset_code": in0_code,
            "in1_offset_code": in1_code,
            "in2_offset_code": in2_code,
        }

    def _populate_tensor_dim_args(
        self,
        expr_format_args: dict[str, BackendArg],
        shape: list[int],
        in0_shape: list[int],
        in1_shape: list[int],
    ) -> None:
        """Populate 4D spatial tensor dimension arguments for convolution and pooling templates.

        Args:
            expr_format_args (dict[str, BackendArg]): Template format arguments dictionary to update.
            shape (list[int]): Output tensor shape.
            in0_shape (list[int]): First input tensor shape.
            in1_shape (list[int]): Second input tensor shape.

        Returns:
            None: Updates expr_format_args in-place.
        """
        if len(shape) >= 4:
            expr_format_args["out_height"] = shape[2]
            expr_format_args["out_width"] = shape[3]
            expr_format_args["out_channels"] = shape[1]

        if len(in0_shape) >= 4:
            expr_format_args["in_height"] = in0_shape[2]
            expr_format_args["in_width"] = in0_shape[3]
            expr_format_args["in_channels"] = in0_shape[1]

        if len(in1_shape) >= 4:
            expr_format_args["filter_h"] = in1_shape[2]
            expr_format_args["filter_w"] = in1_shape[3]

    def _populate_window_and_stride_args(
        self,
        expr_format_args: dict[str, BackendArg],
        attrs: dict[str, BackendArg],
        mapping: dict[str, str],
    ) -> None:
        """Populate window size, stride, and default template format arguments.

        Args:
            expr_format_args (dict[str, BackendArg]): Template format arguments dictionary to update.
            attrs (dict[str, BackendArg]): Node attributes dictionary.
            mapping (dict[str, str]): Operation WGSL template mapping.

        Returns:
            None: Updates expr_format_args in-place.
        """
        if "window_size" in attrs:
            win = attrs["window_size"]
            if isinstance(win, (list, tuple)):
                expr_format_args["window_h"] = win[0]
                expr_format_args["window_w"] = win[1] if len(win) > 1 else win[0]
            else:
                expr_format_args["window_h"] = win
                expr_format_args["window_w"] = win

        stride: BackendArg = attrs.get("stride", 1)
        if isinstance(stride, (list, tuple)):
            expr_format_args["stride_h"] = stride[0]
            expr_format_args["stride_w"] = stride[1] if len(stride) > 1 else stride[0]
        else:
            expr_format_args["stride_h"] = stride
            expr_format_args["stride_w"] = stride

        default_one_keys: tuple[str, ...] = (
            "out_height",
            "out_width",
            "out_channels",
            "channels",
            "in_height",
            "in_width",
            "in_channels",
            "filter_h",
            "filter_w",
            "window_h",
            "window_w",
            "stride_h",
            "stride_w",
        )
        for k in default_one_keys:
            expr_format_args.setdefault(k, 1)

        expr_format_args.setdefault("start_idx", 0)
        expr_format_args.setdefault("post_loop_code", "")
        expr_format_args.setdefault("result_var", "acc")
        expr_format_args.setdefault("loop_code", "acc = acc + buf_in0_f32[i];")
        expr_format_args.setdefault("init_code", "var acc: f32 = 0.0f;")

        if mapping.get("template") == "tiled_matmul":
            expr_format_args["TILE_M"] = 16
            expr_format_args["TILE_N"] = 16
            expr_format_args["TILE_K"] = 16

    def _format_dispatch_dims(
        self,
        mapping: dict[str, str],
        expr_format_args: dict[str, BackendArg],
        nelem: int,
        wg_size: list[int],
    ) -> tuple[str, str, str]:
        """Format the (x, y, z) WebGPU workgroup dispatch dimension expressions.

        Args:
            mapping (dict[str, str]): Operation WGSL mapping dictionary.
            expr_format_args (dict[str, BackendArg]): Populated template format arguments.
            nelem (int): Total number of output elements.
            wg_size (list[int]): Workgroup size [x, y, z].

        Returns:
            tuple[str, str, str]: Formatted dispatch_x, dispatch_y, and dispatch_z expressions.
        """
        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_webgpu_ops

        raw_rules = get_webgpu_ops().get("dispatch_rules", {})
        dispatch_rules: dict[str, dict[str, str]] = (
            {str(k): {str(sk): str(sv) for sk, sv in v.items()} for k, v in raw_rules.items() if isinstance(v, dict)}
            if isinstance(raw_rules, dict)
            else {}
        )
        rule: dict[str, str] = dispatch_rules.get(mapping.get("template", "default"), dispatch_rules.get("default", {}))

        fmt_args: dict[str, BackendArg] = dict(expr_format_args)
        fmt_args["nelem"] = nelem
        fmt_args["wg_x"], fmt_args["wg_y"], fmt_args["wg_z"] = wg_size
        dispatch_x: str = rule.get("x", "1").format(**fmt_args)
        dispatch_y: str = rule.get("y", "1").format(**fmt_args)
        dispatch_z: str = rule.get("z", "1").format(**fmt_args)
        return dispatch_x, dispatch_y, dispatch_z

    def _get_wgsl_for_op(
        self, node: IRNode, shape: list[int], nelem: int, clean_id: str
    ) -> tuple[list[str], str, str, str]:
        """Generate WGSL compute shader lines and dispatch dimensions for an IR node.

        Args:
            node (IRNode): The IR node to lower.
            shape (list[int]): Output tensor shape.
            nelem (int): Number of output elements.
            clean_id (str): Sanitized node identifier.

        Returns:
            tuple[list[str], str, str, str]: Generated WGSL lines and (x, y, z) dispatch strings.
        """
        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_wgsl_template
        from ml_switcheroo_compiler.backends.edge.wgsl_ast import WGSLFunction, WGSLRaw

        op_type: str = coerce_str(adapt_backend_module(node).get_symbol("op_type", ""))
        resolved_op: str = self._resolve_wgsl_op_name(op_type)

        mapping: dict[str, str] = self._resolve_wgsl_mapping(node, op_type, resolved_op)
        inputs: list[str] = _node_inputs_list(node)
        input_nodes: list[IRNode | None] = [
            next((n for n in self.sorted_nodes if adapt_backend_module(n).get_symbol("id", None) == inp), None)
            for inp in inputs
        ]

        in0_shape, in0_strides = (
            self._get_shape_and_strides(input_nodes[0]) if len(input_nodes) > 0 and input_nodes[0] else ([], [])
        )
        in1_shape, in1_strides = (
            self._get_shape_and_strides(input_nodes[1]) if len(input_nodes) > 1 and input_nodes[1] else ([], [])
        )
        _, out_strides = self._get_shape_and_strides(node)

        template = get_wgsl_template(mapping["template"])
        wg_size: list[int] = list(coerce_int_list(template.get("workgroup_size", [64, 1, 1])))
        wg_x, wg_y, wg_z = wg_size

        outer_m: int = self._num_elements(shape[:-1]) if len(shape) > 1 else 1
        inner_n: int = shape[-1] if len(shape) > 0 else 1

        raw_nelem_in = adapt_backend_module(node).get_symbol("inputs_nelem", [1])
        nelem_in_val = (
            int(coerce_int(raw_nelem_in[0])) if isinstance(raw_nelem_in, (list, tuple)) and raw_nelem_in else 1
        )
        expr_format_args: dict[str, BackendArg] = {
            "nelem": nelem,
            "TILE_SIZE": 16,
            "K": in0_shape[1] if len(in0_shape) > 1 else 1,
            "N": inner_n,
            "M": outer_m,
            "clean_id": clean_id,
            "nelem_in": nelem_in_val,
            "expr": mapping.get("expr", "0.0"),
        }
        expr_format_args.update(
            self._build_offset_format_args(
                node, shape, out_strides, input_nodes, in0_shape, in0_strides, in1_shape, in1_strides
            )
        )
        expr_format_args.update(mapping)

        attrs: dict[str, BackendArg] = _node_attrs_dict(node)
        expr_format_args.update(attrs)
        self._populate_tensor_dim_args(expr_format_args, shape, in0_shape, in1_shape)
        self._populate_window_and_stride_args(expr_format_args, attrs, mapping)

        formatted_body: str = str(template["body"]).format(**expr_format_args) if template.get("body") else ""
        global_code_list: list[str] = (
            [str(template["global_code"]).format(**expr_format_args)] if template.get("global_code") else []
        )

        func: WGSLFunction = WGSLFunction(
            f"compute_{clean_id}",
            [
                "@builtin(global_invocation_id) global_id: vec3<u32>",
                "@builtin(local_invocation_id) local_id: vec3<u32>",
            ],
            [WGSLRaw(formatted_body)],
            [f"@compute @workgroup_size({wg_x}, {wg_y}, {wg_z})"],
        )
        wgsl_str: list[str] = global_code_list + self.emitter.emit(func).split("\n")
        dispatch_x, dispatch_y, dispatch_z = self._format_dispatch_dims(mapping, expr_format_args, nelem, wg_size)
        return wgsl_str, dispatch_x, dispatch_y, dispatch_z
