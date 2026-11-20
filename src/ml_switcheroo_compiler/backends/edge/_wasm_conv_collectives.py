"""2D/3D convolution and distributed collective visitors for WasmCodeGenerator."""

from __future__ import annotations

from ml_switcheroo_ir import IRNode

from ml_switcheroo_compiler.backends.eager_registry import (
    coerce_float,
    coerce_int,
    coerce_str,
)
from ml_switcheroo_compiler.backends.edge._wasm_base import (
    WasmAttrType,
    _node_attrs,
    _WasmGenBase,
)
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module


class _WasmConvCollectives(_WasmGenBase):
    """WASM emission layer for Conv2D, Conv3D, and collective communication ops."""

    def visit_Conv2D(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate Conv2D WASM.

        Args:
            node: Input `node` argument (node).
            op_type: Input `op_type` argument (op type).
            clean_id: Input `clean_id` argument (clean id).
            inputs: Input `inputs` argument (inputs).
            shape: Input `shape` argument (shape).
            nelem: Input `nelem` argument (nelem).
        """
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        attrs: dict[str, WasmAttrType] = _node_attrs(node)
        if attrs.get("folded_batch_norm"):
            raw_bn = attrs.get("bn_inputs", [])
            bn_inputs: list[str] = [str(x) for x in raw_bn] if isinstance(raw_bn, (list, tuple)) else []
            if len(bn_inputs) == 4:
                scale, _, _, var = [self.var_map.get(inp, inp) for inp in bn_inputs]
                eps: float = float(coerce_float(attrs.get("epsilon", 1e-5)))
                self.add_line(f"// Folded BatchNorm for {clean_id}")
                self.add_line(f"float mult_{clean_id} = {scale} / std::sqrt({var} + {eps});")

        template: dict[str, str] = get_wasm_template("im2col_conv2d" if attrs.get("tiling") else "conv2d")
        in0_shape, w_shape = self._resolve_node_input_shapes(node, 4)
        norm_shape = self._normalize_ndim_shape(shape, 4)

        stride: WasmAttrType = attrs.get("stride", 1)
        stride_h: int = int(coerce_int(stride[0])) if isinstance(stride, (tuple, list)) else int(coerce_int(stride))
        stride_w: int = int(coerce_int(stride[1])) if isinstance(stride, (tuple, list)) else int(coerce_int(stride))

        expr_args: dict[str, WasmAttrType] = {
            "B": norm_shape[0],
            "out_channels": norm_shape[1],
            "out_height": norm_shape[2],
            "out_width": norm_shape[3],
            "in_channels": in0_shape[1],
            "in_height": in0_shape[2],
            "in_width": in0_shape[3],
            "filter_h": w_shape[2],
            "filter_w": w_shape[3],
            "stride_h": stride_h,
            "stride_w": stride_w,
            "clean_id": clean_id,
            "in0": inputs[0] if len(inputs) > 0 else "dummy",
            "in1": inputs[1] if len(inputs) > 1 else "dummy",
        }
        body: str = template["body"].format(**expr_args)
        for line in body.split("\n"):
            self.add_line(f"    {line}")

    def visit_Conv3D(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Generate Conv3D WASM SIMD.

        Args:
            node (IRNode): The IR node representing Conv3D.
            op_type (str): The operation type name.
            clean_id (str): Sanitized identifier string for output buffer.
            inputs (list[str]): Input identifiers.
            shape (list[int]): Target output shape dimensions.
            nelem (int): Total number of output elements.
        """
        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_template

        template: dict[str, str] = get_wasm_template("Conv3D")
        in0_shape, w_shape = self._resolve_node_input_shapes(node, 5)
        norm_shape = self._normalize_ndim_shape(shape, 5)

        attrs: dict[str, WasmAttrType] = _node_attrs(node)
        raw_stride = attrs.get("stride", attrs.get("strides", 1))
        stride_d, stride_h, stride_w = self._parse_conv3d_spatial_triplet(raw_stride, 1)

        raw_pad = attrs.get("padding", attrs.get("pad", 0))
        pad_d, pad_h, pad_w = self._parse_conv3d_padding(
            raw_pad, norm_shape, in0_shape, w_shape, (stride_d, stride_h, stride_w)
        )

        raw_dil = attrs.get("dilation", attrs.get("dilations", 1))
        dilation_d, dilation_h, dilation_w = self._parse_conv3d_spatial_triplet(raw_dil, 1)

        expr_args: dict[str, WasmAttrType] = {
            "B": norm_shape[0],
            "out_channels": norm_shape[1],
            "out_depth": norm_shape[2],
            "out_height": norm_shape[3],
            "out_width": norm_shape[4],
            "in_channels": in0_shape[1],
            "in_depth": in0_shape[2],
            "in_height": in0_shape[3],
            "in_width": in0_shape[4],
            "filter_d": w_shape[2],
            "filter_h": w_shape[3],
            "filter_w": w_shape[4],
            "stride_d": stride_d,
            "stride_h": stride_h,
            "stride_w": stride_w,
            "pad_d": pad_d,
            "pad_h": pad_h,
            "pad_w": pad_w,
            "dilation_d": dilation_d,
            "dilation_h": dilation_h,
            "dilation_w": dilation_w,
            "clean_id": clean_id,
            "in0": inputs[0] if len(inputs) > 0 else "dummy",
            "in1": inputs[1] if len(inputs) > 1 else "dummy",
        }
        body: str = template["body"].format(**expr_args)
        for line in body.split("\n"):
            self.add_line(f"    {line}")

    def visit_AllReduce(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Emit WebRTC AllReduce.

        Args:
            node: Input `node` argument (node).
            op_type: Input `op_type` argument (op type).
            clean_id: Input `clean_id` argument (clean id).
            inputs: Input `inputs` argument (inputs).
            shape: Input `shape` argument (shape).
            nelem: Input `nelem` argument (nelem).
        """
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: str = coerce_str(adapt_backend_module(node).get_symbol("id", "op"))
        js_code: str = emit_webrtc_op("AllReduce", "buf_" + inputs[0] if inputs else "buf_in0", op_id)
        self.add_line(f"// JS Orcherstrator: \n// {js_code.replace(chr(10), chr(10) + '// ')}")

    def visit_AllGather(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Emit WebRTC AllGather.

        Args:
            node: Input `node` argument (node).
            op_type: Input `op_type` argument (op type).
            clean_id: Input `clean_id` argument (clean id).
            inputs: Input `inputs` argument (inputs).
            shape: Input `shape` argument (shape).
            nelem: Input `nelem` argument (nelem).
        """
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: str = coerce_str(adapt_backend_module(node).get_symbol("id", "op"))
        js_code: str = emit_webrtc_op("AllGather", "buf_" + inputs[0] if inputs else "buf_in0", op_id)
        self.add_line(f"// JS Orcherstrator: \n// {js_code.replace(chr(10), chr(10) + '// ')}")

    def visit_AllToAll(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Emit WebRTC AllToAll.

        Args:
            node: Input `node` argument (node).
            op_type: Input `op_type` argument (op type).
            clean_id: Input `clean_id` argument (clean id).
            inputs: Input `inputs` argument (inputs).
            shape: Input `shape` argument (shape).
            nelem: Input `nelem` argument (nelem).
        """
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: str = coerce_str(adapt_backend_module(node).get_symbol("id", "op"))
        js_code: str = emit_webrtc_op("AllToAll", "buf_" + inputs[0] if inputs else "buf_in0", op_id)
        self.add_line(f"// JS Orcherstrator: \n// {js_code.replace(chr(10), chr(10) + '// ')}")

    def visit_ReduceScatter(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Emit WebRTC ReduceScatter.

        Args:
            node: Input `node` argument (node).
            op_type: Input `op_type` argument (op type).
            clean_id: Input `clean_id` argument (clean id).
            inputs: Input `inputs` argument (inputs).
            shape: Input `shape` argument (shape).
            nelem: Input `nelem` argument (nelem).
        """
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: str = coerce_str(adapt_backend_module(node).get_symbol("id", "op"))
        js_code: str = emit_webrtc_op("ReduceScatter", "buf_" + inputs[0] if inputs else "buf_in0", op_id)
        self.add_line(f"// JS Orcherstrator: \n// {js_code.replace(chr(10), chr(10) + '// ')}")

    def visit_Broadcast(
        self, node: IRNode, op_type: str, clean_id: str, inputs: list[str], shape: list[int], nelem: int
    ) -> None:
        """Emit WebRTC Broadcast.

        Args:
            node: Input `node` argument (node).
            op_type: Input `op_type` argument (op type).
            clean_id: Input `clean_id` argument (clean id).
            inputs: Input `inputs` argument (inputs).
            shape: Input `shape` argument (shape).
            nelem: Input `nelem` argument (nelem).
        """
        from ml_switcheroo_compiler.backends.edge.webgpu_webrtc import emit_webrtc_op

        op_id: str = coerce_str(adapt_backend_module(node).get_symbol("id", "op"))
        js_code: str = emit_webrtc_op("Broadcast", "buf_" + inputs[0] if inputs else "buf_in0", op_id)
        self.add_line(f"// JS Orcherstrator: \n// {js_code.replace(chr(10), chr(10) + '// ')}")
