"""Tiled matrix multiplication, shared-halo convolution, and dynamic pooling for CppGenerator."""

from __future__ import annotations

from collections.abc import Callable, Sequence  # noqa: F401

from ml_switcheroo_ir import IRGraph, IRNode

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
)
from ml_switcheroo_compiler.backends.llvm_cpp._cpp_gen_base import (
    AttrType,
    _node_attrs_dict,
    _node_inputs_list,
)
from ml_switcheroo_compiler.backends.llvm_cpp._cpp_gen_pool_softmax import _CppGenPoolSoftmax


class _CppGenTiledDynamic(_CppGenPoolSoftmax):
    """C++17 emission layer for tiled matmul, halo Conv2D, and dynamic 2D pooling."""

    def visit_TiledMatMul2D(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate TiledMatMul2D LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): Input `graph_to_use` argument (graph to use).
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | BackendArg] = get_cpp_template("tiled_matmul_2d")
        inputs_list: list[str] = _node_inputs_list(node)
        input_nodes: list[IRNode | None] = [graph_to_use.nodes.get(inp) for inp in inputs_list] if graph_to_use else []
        in0_shape: list[int] = self._get_shape(input_nodes[0]) if len(input_nodes) > 0 and input_nodes[0] else [1, 1]
        shape: list[int] = self._get_shape(node)
        M: int = shape[0] if len(shape) > 0 else 1
        N: int = shape[1] if len(shape) > 1 else 1
        K: int = in0_shape[1] if len(in0_shape) > 1 else 1

        expr_args: dict[str, AttrType] = {
            "clean_id": node.id.replace("-", "_"),
            "in0": inputs_list[0] if len(inputs_list) > 0 else "dummy",
            "in1": inputs_list[1] if len(inputs_list) > 1 else (inputs_list[0] if len(inputs_list) > 0 else "dummy"),
            "M": M,
            "N": N,
            "K": K,
            "out_shape_str": "{" + ", ".join(map(str, shape)) + "}",
        }
        body: str = str(template["body"]).format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_Conv2DSharedHalo(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate Conv2DSharedHalo LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): Input `graph_to_use` argument (graph to use).
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | BackendArg] = get_cpp_template("conv2d_shared_halo")
        inputs_list: list[str] = _node_inputs_list(node)
        input_nodes: list[IRNode | None] = [graph_to_use.nodes.get(inp) for inp in inputs_list] if graph_to_use else []
        in0_shape: list[int] = (
            self._get_shape(input_nodes[0]) if len(input_nodes) > 0 and input_nodes[0] else [1, 1, 1, 1]
        )
        w_shape: list[int] = (
            self._get_shape(input_nodes[1])
            if len(input_nodes) > 1 and input_nodes[1]
            else ([1, 1, 1, 1] if len(in0_shape) >= 4 else [1, 1, 3, 3])
        )
        shape: list[int] = self._get_shape(node)
        if len(in0_shape) < 4:
            in0_shape = [1] * (4 - len(in0_shape)) + in0_shape
        if len(w_shape) < 4:
            w_shape = [1] * (4 - len(w_shape)) + w_shape
        if len(shape) < 4:
            shape = [1] * (4 - len(shape)) + shape

        attrs: dict[str, AttrType] = _node_attrs_dict(node)
        stride: AttrType = attrs.get("stride", 1)
        stride_h: int = (
            int(stride[0])
            if isinstance(stride, (tuple, list)) and isinstance(stride[0], (int, float, str))
            else (int(stride) if isinstance(stride, (int, float, str)) else 1)
        )
        stride_w: int = (
            int(stride[1])
            if isinstance(stride, (tuple, list)) and len(stride) > 1 and isinstance(stride[1], (int, float, str))
            else stride_h
        )

        expr_args: dict[str, AttrType] = {
            "B": shape[0],
            "out_channels": shape[1],
            "out_height": shape[2],
            "out_width": shape[3],
            "in_channels": in0_shape[1],
            "in_height": in0_shape[2],
            "in_width": in0_shape[3],
            "filter_h": w_shape[2],
            "filter_w": w_shape[3],
            "stride_h": stride_h,
            "stride_w": stride_w,
            "clean_id": node.id.replace("-", "_"),
            "in0": inputs_list[0] if len(inputs_list) > 0 else "dummy",
            "in1": inputs_list[1] if len(inputs_list) > 1 else (inputs_list[0] if len(inputs_list) > 0 else "dummy"),
            "out_shape_str": "{" + ", ".join(map(str, shape)) + "}",
        }
        body: str = str(template["body"]).format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_MaxPool2DDynamic(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate MaxPool2DDynamic LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): Input `graph_to_use` argument (graph to use).
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | BackendArg] = get_cpp_template("maxpool2d_dynamic")
        inputs_list: list[str] = _node_inputs_list(node)
        input_nodes: list[IRNode | None] = [graph_to_use.nodes.get(inp) for inp in inputs_list] if graph_to_use else []
        in0_shape: list[int] = (
            self._get_shape(input_nodes[0]) if len(input_nodes) > 0 and input_nodes[0] else [1, 1, 1, 1]
        )
        shape: list[int] = self._get_shape(node)
        if len(in0_shape) < 4:
            in0_shape = [1] * (4 - len(in0_shape)) + in0_shape
        if len(shape) < 4:
            shape = [1] * (4 - len(shape)) + shape

        attrs: dict[str, AttrType] = _node_attrs_dict(node)
        ksize: AttrType = attrs.get("kernel_size", [2, 2])
        kh: int = (
            int(ksize[0])
            if isinstance(ksize, (list, tuple)) and len(ksize) > 0 and isinstance(ksize[0], (int, float, str))
            else (int(ksize) if isinstance(ksize, int) else 2)
        )
        kw: int = (
            int(ksize[1])
            if isinstance(ksize, (list, tuple)) and len(ksize) > 1 and isinstance(ksize[1], (int, float, str))
            else kh
        )
        stride: AttrType = attrs.get("stride", [kh, kw])
        stride_h: int = (
            int(stride[0])
            if isinstance(stride, (list, tuple)) and len(stride) > 0 and isinstance(stride[0], (int, float, str))
            else (int(stride) if isinstance(stride, int) else kh)
        )
        stride_w: int = (
            int(stride[1])
            if isinstance(stride, (list, tuple)) and len(stride) > 1 and isinstance(stride[1], (int, float, str))
            else stride_h
        )
        raw_dh = attrs.get("dilation_h", 1)
        raw_dw = attrs.get("dilation_w", 1)
        raw_ph = attrs.get("pad_h", 0)
        raw_pw = attrs.get("pad_w", 0)

        expr_args: dict[str, AttrType] = {
            "B": shape[0],
            "channels": shape[1],
            "in_height": in0_shape[2],
            "in_width": in0_shape[3],
            "out_height": shape[2],
            "out_width": shape[3],
            "filter_h": kh,
            "filter_w": kw,
            "stride_h": stride_h,
            "stride_w": stride_w,
            "dilation_h": int(raw_dh) if isinstance(raw_dh, (int, float, str)) else 1,
            "dilation_w": int(raw_dw) if isinstance(raw_dw, (int, float, str)) else 1,
            "pad_h": int(raw_ph) if isinstance(raw_ph, (int, float, str)) else 0,
            "pad_w": int(raw_pw) if isinstance(raw_pw, (int, float, str)) else 0,
            "clean_id": node.id.replace("-", "_"),
            "in0": inputs_list[0] if len(inputs_list) > 0 else "dummy",
            "out_shape_str": "{" + ", ".join(map(str, shape)) + "}",
        }
        body: str = str(template["body"]).format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_AvgPool2DDynamic(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate AvgPool2DDynamic LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): Input `graph_to_use` argument (graph to use).
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | BackendArg] = get_cpp_template("avgpool2d_dynamic")
        inputs_list: list[str] = _node_inputs_list(node)
        input_nodes: list[IRNode | None] = [graph_to_use.nodes.get(inp) for inp in inputs_list] if graph_to_use else []
        in0_shape: list[int] = (
            self._get_shape(input_nodes[0]) if len(input_nodes) > 0 and input_nodes[0] else [1, 1, 1, 1]
        )
        shape: list[int] = self._get_shape(node)
        if len(in0_shape) < 4:
            in0_shape = [1] * (4 - len(in0_shape)) + in0_shape
        if len(shape) < 4:
            shape = [1] * (4 - len(shape)) + shape

        attrs: dict[str, AttrType] = _node_attrs_dict(node)
        ksize: AttrType = attrs.get("kernel_size", [2, 2])
        kh: int = (
            int(ksize[0])
            if isinstance(ksize, (list, tuple)) and len(ksize) > 0 and isinstance(ksize[0], (int, float, str))
            else (int(ksize) if isinstance(ksize, int) else 2)
        )
        kw: int = (
            int(ksize[1])
            if isinstance(ksize, (list, tuple)) and len(ksize) > 1 and isinstance(ksize[1], (int, float, str))
            else kh
        )
        stride: AttrType = attrs.get("stride", [kh, kw])
        stride_h: int = (
            int(stride[0])
            if isinstance(stride, (list, tuple)) and len(stride) > 0 and isinstance(stride[0], (int, float, str))
            else (int(stride) if isinstance(stride, int) else kh)
        )
        stride_w: int = (
            int(stride[1])
            if isinstance(stride, (list, tuple)) and len(stride) > 1 and isinstance(stride[1], (int, float, str))
            else stride_h
        )
        raw_dh = attrs.get("dilation_h", 1)
        raw_dw = attrs.get("dilation_w", 1)
        raw_ph = attrs.get("pad_h", 0)
        raw_pw = attrs.get("pad_w", 0)

        expr_args: dict[str, AttrType] = {
            "B": shape[0],
            "channels": shape[1],
            "in_height": in0_shape[2],
            "in_width": in0_shape[3],
            "out_height": shape[2],
            "out_width": shape[3],
            "filter_h": kh,
            "filter_w": kw,
            "stride_h": stride_h,
            "stride_w": stride_w,
            "dilation_h": int(raw_dh) if isinstance(raw_dh, (int, float, str)) else 1,
            "dilation_w": int(raw_dw) if isinstance(raw_dw, (int, float, str)) else 1,
            "pad_h": int(raw_ph) if isinstance(raw_ph, (int, float, str)) else 0,
            "pad_w": int(raw_pw) if isinstance(raw_pw, (int, float, str)) else 0,
            "clean_id": node.id.replace("-", "_"),
            "in0": inputs_list[0] if len(inputs_list) > 0 else "dummy",
            "out_shape_str": "{" + ", ".join(map(str, shape)) + "}",
        }
        body: str = str(template["body"]).format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")
