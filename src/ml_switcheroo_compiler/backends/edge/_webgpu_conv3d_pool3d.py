"""2D/depthwise/3D convolution and MaxPool3D WGSL shader visitors."""

from __future__ import annotations

import typing  # noqa: F401

from ml_switcheroo_ir import IRNode

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_int,
    coerce_int_list,
)
from ml_switcheroo_compiler.backends.edge._webgpu_base import (
    _node_attrs_dict,
)
from ml_switcheroo_compiler.backends.edge._webgpu_matmul_pool_telem import _WebGPUMatMulPoolTelem
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module


class _WebGPUConv3DPool3D(_WebGPUMatMulPoolTelem):
    """WebGPU WGSL emission layer for Conv2D, DepthwiseConv2D, Conv3D, and MaxPool3D."""

    def visit_Conv2D(
        self, node: IRNode, input_vars: list[str], **kwargs: BackendArg
    ) -> tuple[list[str], str, str, str]:
        """Emit WGSL for Conv2D.

        Args:
            node: Input `node` argument (node).
            input_vars: Input `input_vars` argument (input vars).
            **kwargs: Input `kwargs` argument (kwargs).

        Returns:
            tuple[list[str], str, str, str]: Computed result.
        """
        shape = coerce_int_list(kwargs.get("shape", []))
        nelem = int(coerce_int(kwargs.get("nelem", 1)))  # noqa: F841
        clean_id = str(kwargs.get("clean_id", ""))

        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_wgsl_template

        template = get_wgsl_template("conv2d")

        in0_shape, in0_strides = self._get_shape_and_strides(
            next(
                (n for n in self.sorted_nodes if adapt_backend_module(n).get_symbol("id", None) == input_vars[0]), None
            )
        )
        in1_shape, in1_strides = self._get_shape_and_strides(
            next(
                (n for n in self.sorted_nodes if adapt_backend_module(n).get_symbol("id", None) == input_vars[1]), None
            )
        )

        out_width: int = shape[3] if len(shape) == 4 else 1
        out_height: int = shape[2] if len(shape) == 4 else 1
        out_channels: int = shape[1] if len(shape) == 4 else 1
        in_channels: int = in0_shape[1] if len(in0_shape) == 4 else 1
        in_width: int = in0_shape[3] if len(in0_shape) == 4 else 1

        raw_strides = node.attributes.get("strides", (1, 1))
        stride_val = raw_strides if isinstance(raw_strides, (list, tuple)) else (1, 1)
        stride_h: int = int(stride_val[0])
        stride_w: int = int(stride_val[1])
        filter_h: int = in1_shape[2] if len(in1_shape) >= 4 else 1
        filter_w: int = in1_shape[3] if len(in1_shape) >= 4 else 1

        body: str = str(template["body"]).format(
            out_width=out_width,
            out_height=out_height,
            in_channels=in_channels,
            out_channels=out_channels,
            stride_h=stride_h,
            stride_w=stride_w,
            filter_h=filter_h,
            filter_w=filter_w,
            in_width=in_width,
        )

        from ml_switcheroo_compiler.backends.edge.wgsl_ast import WGSLFunction, WGSLRaw

        func: WGSLFunction = WGSLFunction(
            f"compute_{clean_id}",
            [
                "@builtin(global_invocation_id) global_id: vec3<u32>",
                "@builtin(local_invocation_id) local_id: vec3<u32>",
            ],
            [WGSLRaw(body)],
            ["@compute @workgroup_size(16, 16, 1)"],
        )
        wgsl_str: list[str] = self.emitter.emit(func).split("\n")

        return wgsl_str, f"Math.ceil({out_width} / 16)", f"Math.ceil({out_height} / 16)", "1"

    def visit_DepthwiseConv2D(
        self, node: IRNode, input_vars: list[str], **kwargs: BackendArg
    ) -> tuple[list[str], str, str, str]:
        """Emit WGSL for DepthwiseConv2D.

        Args:
            node: Input `node` argument (node).
            input_vars: Input `input_vars` argument (input vars).
            **kwargs: Input `kwargs` argument (kwargs).

        Returns:
            tuple[list[str], str, str, str]: Computed result.
        """
        shape = coerce_int_list(kwargs.get("shape", []))
        clean_id = str(kwargs.get("clean_id", ""))

        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_wgsl_template
        from ml_switcheroo_compiler.backends.edge.wgsl_ast import WGSLFunction, WGSLRaw

        template = get_wgsl_template("DepthwiseConv2D")
        in0_shape, _ = self._get_shape_and_strides(
            next(
                (n for n in self.sorted_nodes if adapt_backend_module(n).get_symbol("id", None) == input_vars[0]), None
            )
        )
        in1_shape, _ = self._get_shape_and_strides(
            next(
                (n for n in self.sorted_nodes if adapt_backend_module(n).get_symbol("id", None) == input_vars[1]), None
            )
        )

        out_width: int = shape[3] if len(shape) == 4 else 1
        out_height: int = shape[2] if len(shape) == 4 else 1
        channels: int = shape[1] if len(shape) == 4 else 1
        in_width: int = in0_shape[3] if len(in0_shape) == 4 else 1

        raw_strides = _node_attrs_dict(node).get("strides", (1, 1))
        stride_val = raw_strides if isinstance(raw_strides, (list, tuple)) else (1, 1)
        stride_h: int = int(coerce_int(stride_val[0]))
        stride_w: int = int(coerce_int(stride_val[1]))
        filter_h: int = in1_shape[2] if len(in1_shape) >= 4 else 1
        filter_w: int = in1_shape[3] if len(in1_shape) >= 4 else 1

        body: str = str(template["body"]).format(
            out_width=out_width,
            out_height=out_height,
            channels=channels,
            stride_h=stride_h,
            stride_w=stride_w,
            filter_h=filter_h,
            filter_w=filter_w,
            in_width=in_width,
        )

        func = WGSLFunction(
            f"compute_{clean_id}",
            [
                "@builtin(global_invocation_id) global_id: vec3<u32>",
                "@builtin(local_invocation_id) local_id: vec3<u32>",
            ],
            [WGSLRaw(body)],
            ["@compute @workgroup_size(16, 16, 1)"],
        )
        return self.emitter.emit(func).split("\n"), f"Math.ceil({out_width} / 16)", f"Math.ceil({out_height} / 16)", "1"

    def visit_Conv3D(
        self, node: IRNode, input_vars: list[str], **kwargs: BackendArg
    ) -> tuple[list[str], str, str, str]:
        """Emit WGSL for Conv3D.

        Args:
            node: Input `node` argument (node).
            input_vars: Input `input_vars` argument (input vars).
            **kwargs: Input `kwargs` argument (kwargs).

        Returns:
            tuple[list[str], str, str, str]: Computed result.
        """
        shape = coerce_int_list(kwargs.get("shape", []))
        clean_id = str(kwargs.get("clean_id", ""))

        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_wgsl_template
        from ml_switcheroo_compiler.backends.edge.wgsl_ast import WGSLFunction, WGSLRaw

        template = get_wgsl_template("Conv3D")
        in0_shape, _ = self._get_shape_and_strides(
            next(
                (n for n in self.sorted_nodes if adapt_backend_module(n).get_symbol("id", None) == input_vars[0]), None
            )
        )
        in1_shape, _ = self._get_shape_and_strides(
            next(
                (n for n in self.sorted_nodes if adapt_backend_module(n).get_symbol("id", None) == input_vars[1]), None
            )
        )

        out_w = shape[4] if len(shape) == 5 else 1
        out_h = shape[3] if len(shape) == 5 else 1
        out_d = shape[2] if len(shape) == 5 else 1
        out_channels = shape[1] if len(shape) == 5 else 1
        in_channels = in0_shape[1] if len(in0_shape) == 5 else 1
        in_w = in0_shape[4] if len(in0_shape) == 5 else 1
        in_h = in0_shape[3] if len(in0_shape) == 5 else 1
        in_d = in0_shape[2] if len(in0_shape) == 5 else 1

        raw_strides = _node_attrs_dict(node).get("strides", (1, 1, 1))
        stride_val = raw_strides if isinstance(raw_strides, (list, tuple)) else (1, 1, 1)
        stride_d = int(coerce_int(stride_val[0]))
        stride_h = int(coerce_int(stride_val[1]))
        stride_w = int(coerce_int(stride_val[2]))
        filter_d = in1_shape[2] if len(in1_shape) >= 5 else 1
        filter_h = in1_shape[3] if len(in1_shape) >= 5 else 1
        filter_w = in1_shape[4] if len(in1_shape) >= 5 else 1

        body = str(template["body"]).format(
            out_w=out_w,
            out_h=out_h,
            out_d=out_d,
            in_channels=in_channels,
            out_channels=out_channels,
            stride_d=stride_d,
            stride_h=stride_h,
            stride_w=stride_w,
            filter_d=filter_d,
            filter_h=filter_h,
            filter_w=filter_w,
            in_d=in_d,
            in_h=in_h,
            in_w=in_w,
        )

        func = WGSLFunction(
            f"compute_{clean_id}",
            [
                "@builtin(global_invocation_id) global_id: vec3<u32>",
                "@builtin(local_invocation_id) local_id: vec3<u32>",
            ],
            [WGSLRaw(body)],
            ["@compute @workgroup_size(8, 8, 4)"],
        )
        return (
            self.emitter.emit(func).split("\n"),
            f"Math.ceil({out_w} / 8)",
            f"Math.ceil({out_h} / 8)",
            f"Math.ceil({out_d} / 4)",
        )

    def visit_MaxPool3D(
        self, node: IRNode, input_vars: list[str], **kwargs: BackendArg
    ) -> tuple[list[str], str, str, str]:
        """Emit WGSL for MaxPool3D.

        Args:
            node: Input `node` argument (node).
            input_vars: Input `input_vars` argument (input vars).
            **kwargs: Input `kwargs` argument (kwargs).

        Returns:
            tuple[list[str], str, str, str]: Computed result.
        """
        shape = coerce_int_list(kwargs.get("shape", []))
        clean_id = str(kwargs.get("clean_id", ""))

        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_wgsl_template
        from ml_switcheroo_compiler.backends.edge.wgsl_ast import WGSLFunction, WGSLRaw

        template = get_wgsl_template("MaxPool3D")
        in0_shape, _ = self._get_shape_and_strides(
            next(
                (n for n in self.sorted_nodes if adapt_backend_module(n).get_symbol("id", None) == input_vars[0]), None
            )
        )

        out_w = shape[4] if len(shape) == 5 else 1
        out_h = shape[3] if len(shape) == 5 else 1
        out_d = shape[2] if len(shape) == 5 else 1
        channels = shape[1] if len(shape) == 5 else 1
        in_w = in0_shape[4] if len(in0_shape) == 5 else 1
        in_h = in0_shape[3] if len(in0_shape) == 5 else 1
        in_d = in0_shape[2] if len(in0_shape) == 5 else 1

        raw_strides = _node_attrs_dict(node).get("strides", (1, 1, 1))
        stride_val = raw_strides if isinstance(raw_strides, (list, tuple)) else (1, 1, 1)
        stride_d = int(coerce_int(stride_val[0]))
        stride_h = int(coerce_int(stride_val[1]))
        stride_w = int(coerce_int(stride_val[2]))

        raw_win = _node_attrs_dict(node).get("window", (2, 2, 2))
        win_val = raw_win if isinstance(raw_win, (list, tuple)) else (2, 2, 2)
        window_d = int(coerce_int(win_val[0]))
        window_h = int(coerce_int(win_val[1]))
        window_w = int(coerce_int(win_val[2]))

        body = str(template["body"]).format(
            out_w=out_w,
            out_h=out_h,
            out_d=out_d,
            channels=channels,
            stride_d=stride_d,
            stride_h=stride_h,
            stride_w=stride_w,
            window_d=window_d,
            window_h=window_h,
            window_w=window_w,
            in_d=in_d,
            in_h=in_h,
            in_w=in_w,
        )

        func = WGSLFunction(
            f"compute_{clean_id}",
            [
                "@builtin(global_invocation_id) global_id: vec3<u32>",
                "@builtin(local_invocation_id) local_id: vec3<u32>",
            ],
            [WGSLRaw(body)],
            ["@compute @workgroup_size(8, 8, 4)"],
        )
        return (
            self.emitter.emit(func).split("\n"),
            f"Math.ceil({out_w} / 8)",
            f"Math.ceil({out_h} / 8)",
            f"Math.ceil({out_d} / 4)",
        )
