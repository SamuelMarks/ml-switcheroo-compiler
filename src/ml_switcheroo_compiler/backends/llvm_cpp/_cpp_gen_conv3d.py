"""2D/3D convolution, scan, attention, and 3D pooling visitors for CppGenerator."""

from __future__ import annotations

from collections.abc import Callable, Sequence  # noqa: F401

from ml_switcheroo_ir import IRGraph, IRNode

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_int_or_tuple,
)
from ml_switcheroo_compiler.backends.llvm_cpp._cpp_gen_base import (
    AttrType,
    _CppGenBase,
    _node_attrs_dict,
    _node_inputs_list,
)


class _CppGenConv3D(_CppGenBase):
    """C++17 emission layer for Conv2D, Scan, Attention, Conv3D, MaxPool3D, and AvgPool3D."""

    def visit_Conv2D(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate Conv2D LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): Input `graph_to_use` argument (graph to use).
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | BackendArg] = get_cpp_template("conv2d")

        inputs_list: list[str] = _node_inputs_list(node)
        input_nodes: list[IRNode | None] = []
        if graph_to_use:
            input_nodes = [graph_to_use.nodes.get(inp) for inp in inputs_list]
        in0_shape: list[int] = (
            self._get_shape(input_nodes[0]) if len(input_nodes) > 0 and input_nodes[0] else [1, 1, 1, 1]
        )
        w_shape: list[int] = (
            self._get_shape(input_nodes[1]) if len(input_nodes) > 1 and input_nodes[1] else [1, 1, 1, 1]
        )
        shape: list[int] = self._get_shape(node)

        if len(in0_shape) < 4:
            in0_shape = [1] * (4 - len(in0_shape)) + in0_shape
        if len(w_shape) < 4:
            w_shape = [1] * (4 - len(w_shape)) + w_shape
        if len(shape) < 4:
            shape = [1] * (4 - len(shape)) + shape

        attrs: dict[str, AttrType] = _node_attrs_dict(node)
        stride: int | tuple[int, ...] | list[int] = coerce_int_or_tuple(attrs.get("stride", 1))
        stride_h: int = int(stride[0]) if isinstance(stride, (tuple, list)) else int(stride)
        stride_w: int = int(stride[1]) if isinstance(stride, (tuple, list)) else int(stride)

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
            "in1": inputs_list[1] if len(inputs_list) > 1 else "dummy",
            "out_shape_str": "{" + ", ".join(map(str, shape)) + "}",
        }
        body: str = str(template["body"]).format(**expr_args)

        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_Scan(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate Scan LLVM CPP.

        Args:
            node: Input `node` argument (node).
            graph_to_use: Input `graph_to_use` argument (graph to use).
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | BackendArg] = get_cpp_template("scan")
        out_shape: list[int] = self._get_shape(node)
        out_shape_str = "{" + ",".join(map(str, out_shape)) + "}"
        in_list = _node_inputs_list(node)
        expr_args: dict[str, AttrType] = {
            "clean_id": node.id.replace("-", "_"),
            "out_shape_str": out_shape_str,
            "in0": in_list[0] if in_list else "dummy",
        }
        body: str = str(template["body"]).format(**expr_args)
        for line in body.split("\n"):
            if line.strip():
                self.lines.append(f"    {line}")

    def visit_Attention(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate Attention LLVM CPP.

        Args:
            node: Input `node` argument (node).
            graph_to_use: Input `graph_to_use` argument (graph to use).
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | BackendArg] = get_cpp_template("attention")
        out_shape: list[int] = self._get_shape(node)
        out_shape_str = "{" + ",".join(map(str, out_shape)) + "}"
        expr_args: dict[str, AttrType] = {
            "clean_id": node.id.replace("-", "_"),
            "out_shape_str": out_shape_str,
        }
        body: str = str(template["body"]).format(**expr_args)
        for line in body.split("\n"):
            if line.strip():
                self.lines.append(f"    {line}")

    def visit_Conv3D(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate Conv3D LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): Input `graph_to_use` argument (graph to use).
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | BackendArg] = get_cpp_template("conv3d")
        inputs_list: list[str] = _node_inputs_list(node)
        input_nodes: list[IRNode | None] = []
        if graph_to_use:
            input_nodes = [graph_to_use.nodes.get(inp) for inp in inputs_list]
        in0_shape: list[int] = (
            self._get_shape(input_nodes[0]) if len(input_nodes) > 0 and input_nodes[0] else [1, 1, 1, 1, 1]
        )
        w_shape: list[int] = (
            self._get_shape(input_nodes[1]) if len(input_nodes) > 1 and input_nodes[1] else [1, 1, 1, 1, 1]
        )
        shape: list[int] = self._get_shape(node)

        if len(in0_shape) < 5:
            in0_shape = [1] * (5 - len(in0_shape)) + in0_shape
        if len(w_shape) < 5:
            w_shape = [1] * (5 - len(w_shape)) + w_shape
        if len(shape) < 5:
            shape = [1] * (5 - len(shape)) + shape

        attrs: dict[str, AttrType] = _node_attrs_dict(node)
        stride: int | tuple[int, ...] | list[int] = coerce_int_or_tuple(attrs.get("stride", [1, 1, 1]))
        if isinstance(stride, int):
            stride_d = stride_h = stride_w = stride
        elif len(stride) >= 3:
            stride_d, stride_h, stride_w = int(stride[0]), int(stride[1]), int(stride[2])
        else:
            stride_d = stride_h = stride_w = 1

        expr_args: dict[str, AttrType] = {
            "N": in0_shape[0],
            "C_in": in0_shape[1],
            "D": in0_shape[2],
            "H": in0_shape[3],
            "W": in0_shape[4],
            "C_out": w_shape[0],
            "KD": w_shape[2],
            "KH": w_shape[3],
            "KW": w_shape[4],
            "D_out": shape[2],
            "H_out": shape[3],
            "W_out": shape[4],
            "stride_d": stride_d,
            "stride_h": stride_h,
            "stride_w": stride_w,
            "clean_id": node.id.replace("-", "_"),
            "in0": inputs_list[0] if len(inputs_list) > 0 else "dummy",
            "in1": inputs_list[1] if len(inputs_list) > 1 else "dummy",
            "out_shape_str": "{" + ", ".join(map(str, shape)) + "}",
        }
        body: str = str(template["body"]).format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_MaxPool3D(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate MaxPool3D LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): Input `graph_to_use` argument (graph to use).
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | BackendArg] = get_cpp_template("maxpool3d")
        inputs_list: list[str] = _node_inputs_list(node)
        input_nodes: list[IRNode | None] = [graph_to_use.nodes.get(inp) for inp in inputs_list] if graph_to_use else []
        in0_shape: list[int] = (
            self._get_shape(input_nodes[0]) if len(input_nodes) > 0 and input_nodes[0] else [1, 1, 1, 1, 1]
        )
        shape: list[int] = self._get_shape(node)
        if len(in0_shape) < 5:
            in0_shape = [1] * (5 - len(in0_shape)) + in0_shape
        if len(shape) < 5:
            shape = [1] * (5 - len(shape)) + shape

        attrs: dict[str, AttrType] = _node_attrs_dict(node)
        ksize: int | tuple[int, ...] | list[int] = coerce_int_or_tuple(attrs.get("kernel_size", [2, 2, 2]))
        kd: int = int(ksize[0]) if isinstance(ksize, (list, tuple)) and len(ksize) > 0 else 2
        kh: int = int(ksize[1]) if isinstance(ksize, (list, tuple)) and len(ksize) > 1 else 2
        kw: int = int(ksize[2]) if isinstance(ksize, (list, tuple)) and len(ksize) > 2 else 2
        stride: int | tuple[int, ...] | list[int] = coerce_int_or_tuple(attrs.get("stride", [kd, kh, kw]))
        stride_d: int = int(stride[0]) if isinstance(stride, (list, tuple)) and len(stride) > 0 else kd
        stride_h: int = int(stride[1]) if isinstance(stride, (list, tuple)) and len(stride) > 1 else kh
        stride_w: int = int(stride[2]) if isinstance(stride, (list, tuple)) and len(stride) > 2 else kw

        expr_args: dict[str, AttrType] = {
            "N": in0_shape[0],
            "C": in0_shape[1],
            "D": in0_shape[2],
            "H": in0_shape[3],
            "W": in0_shape[4],
            "KD": kd,
            "KH": kh,
            "KW": kw,
            "D_out": shape[2],
            "H_out": shape[3],
            "W_out": shape[4],
            "stride_d": stride_d,
            "stride_h": stride_h,
            "stride_w": stride_w,
            "clean_id": node.id.replace("-", "_"),
            "in0": inputs_list[0] if len(inputs_list) > 0 else "dummy",
            "out_shape_str": "{" + ", ".join(map(str, shape)) + "}",
        }
        body: str = str(template["body"]).format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_AvgPool3D(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate AvgPool3D LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): Input `graph_to_use` argument (graph to use).
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | BackendArg] = get_cpp_template("avgpool3d")
        inputs_list: list[str] = _node_inputs_list(node)
        input_nodes: list[IRNode | None] = [graph_to_use.nodes.get(inp) for inp in inputs_list] if graph_to_use else []
        in0_shape: list[int] = (
            self._get_shape(input_nodes[0]) if len(input_nodes) > 0 and input_nodes[0] else [1, 1, 1, 1, 1]
        )
        shape: list[int] = self._get_shape(node)
        if len(in0_shape) < 5:
            in0_shape = [1] * (5 - len(in0_shape)) + in0_shape
        if len(shape) < 5:
            shape = [1] * (5 - len(shape)) + shape

        attrs: dict[str, AttrType] = _node_attrs_dict(node)
        ksize: int | tuple[int, ...] | list[int] = coerce_int_or_tuple(attrs.get("kernel_size", [2, 2, 2]))
        kd: int = int(ksize[0]) if isinstance(ksize, (list, tuple)) and len(ksize) > 0 else 2
        kh: int = int(ksize[1]) if isinstance(ksize, (list, tuple)) and len(ksize) > 1 else 2
        kw: int = int(ksize[2]) if isinstance(ksize, (list, tuple)) and len(ksize) > 2 else 2
        stride: int | tuple[int, ...] | list[int] = coerce_int_or_tuple(attrs.get("stride", [kd, kh, kw]))
        stride_d: int = int(stride[0]) if isinstance(stride, (list, tuple)) and len(stride) > 0 else kd
        stride_h: int = int(stride[1]) if isinstance(stride, (list, tuple)) and len(stride) > 1 else kh
        stride_w: int = int(stride[2]) if isinstance(stride, (list, tuple)) and len(stride) > 2 else kw

        expr_args: dict[str, AttrType] = {
            "N": in0_shape[0],
            "C": in0_shape[1],
            "D": in0_shape[2],
            "H": in0_shape[3],
            "W": in0_shape[4],
            "KD": kd,
            "KH": kh,
            "KW": kw,
            "D_out": shape[2],
            "H_out": shape[3],
            "W_out": shape[4],
            "stride_d": stride_d,
            "stride_h": stride_h,
            "stride_w": stride_w,
            "clean_id": node.id.replace("-", "_"),
            "in0": inputs_list[0] if len(inputs_list) > 0 else "dummy",
            "out_shape_str": "{" + ", ".join(map(str, shape)) + "}",
        }
        body: str = str(template["body"]).format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")
