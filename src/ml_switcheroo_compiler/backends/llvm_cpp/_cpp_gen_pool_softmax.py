"""2D pooling, GroupNorm, Softmax, and tree/warp reduction visitors for CppGenerator."""

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
from ml_switcheroo_compiler.backends.llvm_cpp._cpp_gen_norm_index import _CppGenNormIndex


class _CppGenPoolSoftmax(_CppGenNormIndex):
    """C++17 emission layer for 2D pooling, GroupNorm, Softmax, and parallel reductions."""

    def visit_MaxPool2D(self, node: IRNode, graph_to_use: IRGraph | None = None, is_global: bool = False) -> None:
        """Generate MaxPool2D LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): Input `graph_to_use` argument (graph to use).
            is_global (bool): Whether this is a global max pooling operation.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | BackendArg] = get_cpp_template("maxpool2d")
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
        if is_global:
            kh: int = in0_shape[2]
            kw: int = in0_shape[3]
            stride_h: int = 1
            stride_w: int = 1
        else:
            ksize: AttrType = attrs.get("kernel_size", [2, 2])
            kh = (
                int(ksize[0])
                if isinstance(ksize, (list, tuple)) and len(ksize) > 0 and isinstance(ksize[0], (int, float, str))
                else (int(ksize) if isinstance(ksize, int) else 2)
            )
            kw = (
                int(ksize[1])
                if isinstance(ksize, (list, tuple)) and len(ksize) > 1 and isinstance(ksize[1], (int, float, str))
                else kh
            )
            stride: AttrType = attrs.get("stride", [kh, kw])
            stride_h = (
                int(stride[0])
                if isinstance(stride, (list, tuple)) and len(stride) > 0 and isinstance(stride[0], (int, float, str))
                else (int(stride) if isinstance(stride, int) else kh)
            )
            stride_w = (
                int(stride[1])
                if isinstance(stride, (list, tuple)) and len(stride) > 1 and isinstance(stride[1], (int, float, str))
                else stride_h
            )

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
            "clean_id": node.id.replace("-", "_"),
            "in0": inputs_list[0] if len(inputs_list) > 0 else "dummy",
            "out_shape_str": "{" + ", ".join(map(str, shape)) + "}",
        }
        body: str = str(template["body"]).format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_AvgPool2D(self, node: IRNode, graph_to_use: IRGraph | None = None, is_global: bool = False) -> None:
        """Generate AvgPool2D LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): Input `graph_to_use` argument (graph to use).
            is_global (bool): Whether this is a global average pooling operation.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | BackendArg] = get_cpp_template("avgpool2d")
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
        if is_global:
            kh: int = in0_shape[2]
            kw: int = in0_shape[3]
            stride_h: int = 1
            stride_w: int = 1
        else:
            ksize: AttrType = attrs.get("kernel_size", [2, 2])
            kh = (
                int(ksize[0])
                if isinstance(ksize, (list, tuple)) and len(ksize) > 0 and isinstance(ksize[0], (int, float, str))
                else (int(ksize) if isinstance(ksize, int) else 2)
            )
            kw = (
                int(ksize[1])
                if isinstance(ksize, (list, tuple)) and len(ksize) > 1 and isinstance(ksize[1], (int, float, str))
                else kh
            )
            stride: AttrType = attrs.get("stride", [kh, kw])
            stride_h = (
                int(stride[0])
                if isinstance(stride, (list, tuple)) and len(stride) > 0 and isinstance(stride[0], (int, float, str))
                else (int(stride) if isinstance(stride, int) else kh)
            )
            stride_w = (
                int(stride[1])
                if isinstance(stride, (list, tuple)) and len(stride) > 1 and isinstance(stride[1], (int, float, str))
                else stride_h
            )

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
            "clean_id": node.id.replace("-", "_"),
            "in0": inputs_list[0] if len(inputs_list) > 0 else "dummy",
            "out_shape_str": "{" + ", ".join(map(str, shape)) + "}",
        }
        body: str = str(template["body"]).format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_GroupNorm(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate GroupNorm LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): Input `graph_to_use` argument (graph to use).
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | BackendArg] = get_cpp_template("groupnorm")
        inputs_list: list[str] = _node_inputs_list(node)
        has_gamma: str = "true" if len(inputs_list) > 1 else "false"
        has_beta: str = "true" if len(inputs_list) > 2 else "false"
        gamma_var: str = (
            inputs_list[1] if len(inputs_list) > 1 else (inputs_list[0] if len(inputs_list) > 0 else "dummy")
        )
        beta_var: str = (
            inputs_list[2] if len(inputs_list) > 2 else (inputs_list[0] if len(inputs_list) > 0 else "dummy")
        )
        shape: list[int] = self._get_shape(node)
        attrs: dict[str, AttrType] = _node_attrs_dict(node)
        raw_ng = attrs.get("num_groups", attrs.get("groups", 1))
        num_groups: int = int(raw_ng) if isinstance(raw_ng, (int, float, str)) else 1
        raw_eps = attrs.get("eps", attrs.get("epsilon", 1e-5))
        eps: float = float(raw_eps) if isinstance(raw_eps, (int, float, str)) else 1e-5

        N: int = shape[0] if len(shape) > 0 else 1
        C: int = shape[1] if len(shape) > 1 else 1
        spatial: int = 1
        for s in shape[2:]:
            spatial *= s
        if num_groups <= 0:
            num_groups = 1
        c_per_g: int = max(1, C // num_groups)

        expr_args: dict[str, AttrType] = {
            "N": N,
            "G": num_groups,
            "C_per_G": c_per_g,
            "spatial": spatial,
            "eps": f"{eps}f",
            "has_gamma": has_gamma,
            "has_beta": has_beta,
            "gamma": gamma_var,
            "beta": beta_var,
            "clean_id": node.id.replace("-", "_"),
            "in0": inputs_list[0] if len(inputs_list) > 0 else "dummy",
            "out_shape_str": "{" + ", ".join(map(str, shape)) + "}",
        }
        body: str = str(template["body"]).format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_Softmax(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate Softmax LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): Input `graph_to_use` argument (graph to use).
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | BackendArg] = get_cpp_template("softmax")
        inputs_list: list[str] = _node_inputs_list(node)
        shape: list[int] = self._get_shape(node)
        D: int = shape[-1] if shape else 1
        expr_args: dict[str, AttrType] = {
            "D": D,
            "clean_id": node.id.replace("-", "_"),
            "in0": inputs_list[0] if len(inputs_list) > 0 else "dummy",
            "out_shape_str": "{" + ", ".join(map(str, shape)) + "}",
        }
        body: str = str(template["body"]).format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_LogSoftmax(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate LogSoftmax LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): Input `graph_to_use` argument (graph to use).
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | BackendArg] = get_cpp_template("logsoftmax")
        inputs_list: list[str] = _node_inputs_list(node)
        shape: list[int] = self._get_shape(node)
        D: int = shape[-1] if shape else 1
        expr_args: dict[str, AttrType] = {
            "D": D,
            "clean_id": node.id.replace("-", "_"),
            "in0": inputs_list[0] if len(inputs_list) > 0 else "dummy",
            "out_shape_str": "{" + ", ".join(map(str, shape)) + "}",
        }
        body: str = str(template["body"]).format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_WarpShuffleReduce(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate WarpShuffleReduce LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): Input `graph_to_use` argument (graph to use).
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | BackendArg] = get_cpp_template("warp_shuffle_reduce")
        inputs_list: list[str] = _node_inputs_list(node)
        shape: list[int] = self._get_shape(node)
        expr_args: dict[str, AttrType] = {
            "clean_id": node.id.replace("-", "_"),
            "in0": inputs_list[0] if len(inputs_list) > 0 else "dummy",
            "out_shape_str": "{" + ", ".join(map(str, shape)) + "}",
        }
        body: str = str(template["body"]).format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_SharedMemoryTreeReduce(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate SharedMemoryTreeReduce LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): Input `graph_to_use` argument (graph to use).
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | BackendArg] = get_cpp_template("shared_memory_tree_reduce")
        inputs_list: list[str] = _node_inputs_list(node)
        shape: list[int] = self._get_shape(node)
        expr_args: dict[str, AttrType] = {
            "clean_id": node.id.replace("-", "_"),
            "in0": inputs_list[0] if len(inputs_list) > 0 else "dummy",
            "out_shape_str": "{" + ", ".join(map(str, shape)) + "}",
        }
        body: str = str(template["body"]).format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")
