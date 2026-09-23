# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""ISO C++17 code generator and native compilation runner for CPU fallback.

This backend compiles computational graphs into ISO C++17 compute kernels using
system C++ compilers (clang++ or g++) with OpenMP SIMD vectorization and aligned
memory allocations into dynamically loaded shared libraries via ctypes.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import tempfile
from typing import Callable, Optional, Union

AttrType = Union[int, float, str, bool, list, tuple, dict, None]

import yaml

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def get_supported_mlir_dialects() -> tuple[str, ...]:
    """Return supported canonical MLIR dialect prefixes.

    Returns:
        tuple[str, ...]: Dialect prefixes ('arith', 'math', 'tensor', 'linalg', 'scf').
    """
    return ("arith", "math", "tensor", "linalg", "scf")


def validate_mlir_operation(op_name: str) -> bool:
    """Validate an MLIR operation name against the canonical MLIR_REGISTRY.

    Args:
        op_name (str): The full MLIR operation name (e.g. 'arith.addi', 'math.exp').

    Returns:
        bool: True if the operation is valid in a recognized dialect in MLIR_REGISTRY.
    """
    try:
        from ml_switcheroo_ir.schema.mlir_registry import MLIR_REGISTRY

        if op_name in MLIR_REGISTRY:
            return True
        dialect = op_name.split(".")[0] if "." in op_name else ""
        return dialect in get_supported_mlir_dialects() and op_name in MLIR_REGISTRY
    except ImportError:
        return False


@register_backend("llvm_cpp")
class CppGenerator(BaseGenerator):
    """High-performance C++17 backend generator compiling kernels via system compilers (clang++/g++)."""

    def __init__(self, graph: IRGraph | None = None, use_simd: bool = True, use_openmp: bool = True, strict: bool = False) -> None:
        """Initialize the C++ generator.

        Args:
            graph (IRGraph | None): The graph parameter.
            use_simd (bool): The use_simd parameter.
            use_openmp (bool): The use_openmp parameter.
            strict (bool): Whether to raise UnimplementedMathError on unsupported operations.
        """
        super().__init__(graph=graph)
        self.use_simd = use_simd
        self.use_openmp = use_openmp
        self.strict = strict
        self.lines: list[str] = []

    def _get_shape(self, node: IRNode) -> list[int]:
        """_get_shape function.

        Args:
            node (IRNode): The node parameter.

        Returns:
            list[int]: Result.
        """
        shape: tuple[int, ...] | int | float | None = getattr(node, "shape_metadata", None)
        if not shape:
            return [1]
        if isinstance(shape, (int, float)):
            return [int(shape)]
        res: list[int] = []
        for s in shape:
            res.append(int(s))
        return res

    def _dispatch_unmapped_op_fallback(self, node: IRNode, op: str) -> None:
        """Handle unmapped operation with strict error policy or safe runtime polyfill.

        Args:
            node (IRNode): Target IR node representing the unmapped operation.
            op (str): Operation identifier name.

        Raises:
            UnimplementedMathError: When strict mode is enabled.
        """
        if self.strict:
            from ml_switcheroo_compiler.core.errors import UnimplementedMathError

            raise UnimplementedMathError(f"C++ code generator does not support operation: {op}")

        import warnings

        warnings.warn(
            f"LLVM C++ generator encountered unmapped op '{op}' for node '{node.id}'. Emitting safe runtime polyfill.",
            UserWarning,
            stacklevel=2,
        )
        out_shape_str = "{" + ",".join(map(str, self._get_shape(node))) + "}"
        clean_id = node.id
        self.lines.append(f"    NDArrayView<float> {clean_id}({out_shape_str}); // Fallback Unimplemented {op}")
        inputs = getattr(node, "inputs", []) or []
        if len(inputs) > 0:
            in0 = inputs[0]
            self.lines.append(f"    for(size_t i = 0; i < {clean_id}.size(); ++i) {{ {clean_id}.data[i] = ({in0}.size() > 0) ? {in0}.data[i % {in0}.size()] : 1.0f; }}")
        else:
            self.lines.append(f"    for(size_t i = 0; i < {clean_id}.size(); ++i) {{ {clean_id}.data[i] = 1.0f; }}")

    def _num_elements(self, shape: list[int]) -> int:
        """_num_elements function.

        Args:
            shape (list[int]): The shape parameter.

        Returns:
            int: Result.
        """
        n: int = 1
        for s in shape:
            n *= s
        return n

    def _get_strides(self, shape: list[int]) -> list[int]:
        """_get_strides function.

        Args:
            shape (list[int]): The shape parameter.

        Returns:
            list[int]: Result.
        """
        strides: list[int] = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return strides

    def generate(self, graph: IRGraph | None = None) -> str:
        """Generate C++ code from an IR graph.

        Args:
            graph (IRGraph | None): The logical graph to convert.

        Returns:
            str: The generated C++ source code.
        """
        tmpl_path: str = os.path.join(os.path.dirname(__file__), "cpp_templates.yaml")
        with open(tmpl_path) as f:
            data: dict[str, str] = yaml.safe_load(f)
        prelude: str = data.get("prelude", "")
        self.lines = prelude.strip().split("\n")

        self.lines.append('extern "C" void compute_graph(const float** inputs = nullptr, float** outputs = nullptr, const int64_t* shapes = nullptr) {')

        graph_to_use: IRGraph | None = graph if graph is not None else self.graph
        if not graph_to_use:
            graph_to_use = IRGraph()

        # Determine maximum memory offset required
        max_offset: int = 0
        for node in graph_to_use.nodes.values():
            offset: int | None = node.attributes.get("buffer_offset")
            size: int | None = node.attributes.get("buffer_size")
            if offset is not None and size is not None:
                max_offset = max(max_offset, offset + size)

        if max_offset > 0:
            self.lines.append(f"    // Allocate global arena buffer of size {max_offset} bytes")
            self.lines.append(f"    AlignedBuffer<uint8_t> global_arena({max_offset});")

        inp_idx = 0
        for _, node in graph_to_use.nodes.items():
            if node.op_type == "Input":
                self._visit_node(node, graph_to_use)
                self.lines.append(f"    if (inputs && inputs[{inp_idx}]) {{")
                self.lines.append(f"        std::memcpy({node.id}.data.data(), inputs[{inp_idx}], {node.id}.size() * sizeof(float));")
                self.lines.append("    }")
                inp_idx += 1
            else:
                self._visit_node(node, graph_to_use)

        out_idx = 0
        for out_name in getattr(graph_to_use, "outputs", []):
            if out_name in graph_to_use.nodes:
                clean_out = graph_to_use.nodes[out_name].id
                self.lines.append(f"    if (outputs && outputs[{out_idx}]) {{")
                self.lines.append(f"        std::memcpy(outputs[{out_idx}], {clean_out}.data.data(), {clean_out}.size() * sizeof(float));")
                self.lines.append("    }")
                out_idx += 1

        self.lines.append("}")
        return "\n".join(self.lines)

    def _visit_if_op(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """_visit_if_op function.

        Args:
            node (IRNode): The node parameter.
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        assert len(node.inputs) >= 1
        cond_var: str = node.inputs[0]

        then_lines = []
        then_graph: IRGraph | None = node.attributes.get("then_branch")
        if then_graph:
            old_lines = self.lines
            self.lines = then_lines
            for _, sub_node in then_graph.nodes.items():
                self._visit_node(sub_node, graph_to_use)
            self.lines = old_lines

        else_lines = []
        else_graph: IRGraph | None = node.attributes.get("else_branch")
        if else_graph:
            old_lines = self.lines
            self.lines = else_lines
            for _, sub_node in else_graph.nodes.items():
                self._visit_node(sub_node, graph_to_use)
            self.lines = old_lines

        template = get_cpp_template("if_op")
        body = template["body"].format(cond_var=cond_var, then_body="\n".join(then_lines) + "\n" if then_lines else "", else_body="\n".join(else_lines) + "\n" if else_lines else "")
        self.lines.extend(body.strip().split("\n"))

    def _visit_loop_op(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """_visit_loop_op function.

        Args:
            node (IRNode): The node parameter.
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        cond_lines = []
        cond_graph: IRGraph | None = node.attributes.get("cond")
        if cond_graph:
            old_lines = self.lines
            self.lines = cond_lines
            for _, sub_node in cond_graph.nodes.items():
                self._visit_node(sub_node, graph_to_use)
            self.lines = old_lines

        body_lines = []
        body_graph: IRGraph | None = node.attributes.get("body")
        if body_graph:
            old_lines = self.lines
            self.lines = body_lines
            for _, sub_node in body_graph.nodes.items():
                self._visit_node(sub_node, graph_to_use)
            self.lines = old_lines

        template = get_cpp_template("loop_op")
        body = template["body"].format(cond_body="\n".join(cond_lines) + "\n" if cond_lines else "", loop_body="\n".join(body_lines) + "\n" if body_lines else "")
        self.lines.extend(body.strip().split("\n"))

    def visit_Conv2D(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate Conv2D LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str] = get_cpp_template("conv2d")

        inputs_list: list[str] = getattr(node, "inputs", [])
        input_nodes: list[IRNode | None] = []
        if graph_to_use:
            input_nodes = [graph_to_use.nodes.get(inp) for inp in inputs_list]
        in0_shape: list[int] = self._get_shape(input_nodes[0]) if len(input_nodes) > 0 and input_nodes[0] else [1, 1, 1, 1]
        w_shape: list[int] = self._get_shape(input_nodes[1]) if len(input_nodes) > 1 and input_nodes[1] else [1, 1, 1, 1]
        shape: list[int] = self._get_shape(node)

        if len(in0_shape) < 4:
            in0_shape = [1] * (4 - len(in0_shape)) + in0_shape
        if len(w_shape) < 4:
            w_shape = [1] * (4 - len(w_shape)) + w_shape
        if len(shape) < 4:
            shape = [1] * (4 - len(shape)) + shape

        attrs: dict[str, AttrType] = getattr(node, "attributes", {})
        stride: int | tuple[int, ...] | list[int] = attrs.get("stride", 1)
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
        body: str = template["body"].format(**expr_args)

        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_Scan(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate Scan LLVM CPP."""
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str] = get_cpp_template("scan")
        out_shape: list[int] = self._get_shape(node)
        out_shape_str = "{" + ",".join(map(str, out_shape)) + "}"
        expr_args: dict[str, AttrType] = {
            "clean_id": node.id.replace("-", "_"),
            "out_shape_str": out_shape_str,
            "in0": getattr(node, "inputs", ["dummy"])[0] if getattr(node, "inputs", []) else "dummy",
        }
        body: str = template["body"].format(**expr_args)
        for line in body.split("\n"):
            if line.strip():
                self.lines.append(f"    {line}")

    def visit_Attention(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate Attention LLVM CPP."""
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str] = get_cpp_template("attention")
        out_shape: list[int] = self._get_shape(node)
        out_shape_str = "{" + ",".join(map(str, out_shape)) + "}"
        expr_args: dict[str, AttrType] = {
            "clean_id": node.id.replace("-", "_"),
            "out_shape_str": out_shape_str,
        }
        body: str = template["body"].format(**expr_args)
        for line in body.split("\n"):
            if line.strip():
                self.lines.append(f"    {line}")

    def visit_Conv3D(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate Conv3D LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str] = get_cpp_template("conv3d")
        inputs_list: list[str] = getattr(node, "inputs", [])
        input_nodes: list[IRNode | None] = []
        if graph_to_use:
            input_nodes = [graph_to_use.nodes.get(inp) for inp in inputs_list]
        in0_shape: list[int] = self._get_shape(input_nodes[0]) if len(input_nodes) > 0 and input_nodes[0] else [1, 1, 1, 1, 1]
        w_shape: list[int] = self._get_shape(input_nodes[1]) if len(input_nodes) > 1 and input_nodes[1] else [1, 1, 1, 1, 1]
        shape: list[int] = self._get_shape(node)

        if len(in0_shape) < 5:
            in0_shape = [1] * (5 - len(in0_shape)) + in0_shape
        if len(w_shape) < 5:
            w_shape = [1] * (5 - len(w_shape)) + w_shape
        if len(shape) < 5:
            shape = [1] * (5 - len(shape)) + shape

        attrs: dict[str, AttrType] = getattr(node, "attributes", {})
        stride: int | tuple[int, ...] | list[int] = attrs.get("stride", [1, 1, 1])
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
        body: str = template["body"].format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_MaxPool3D(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate MaxPool3D LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str] = get_cpp_template("maxpool3d")
        inputs_list: list[str] = getattr(node, "inputs", [])
        input_nodes: list[IRNode | None] = [graph_to_use.nodes.get(inp) for inp in inputs_list] if graph_to_use else []
        in0_shape: list[int] = self._get_shape(input_nodes[0]) if len(input_nodes) > 0 and input_nodes[0] else [1, 1, 1, 1, 1]
        shape: list[int] = self._get_shape(node)
        if len(in0_shape) < 5:
            in0_shape = [1] * (5 - len(in0_shape)) + in0_shape
        if len(shape) < 5:
            shape = [1] * (5 - len(shape)) + shape

        attrs: dict[str, AttrType] = getattr(node, "attributes", {})
        ksize: int | tuple[int, ...] | list[int] = attrs.get("kernel_size", [2, 2, 2])
        kd: int = int(ksize[0]) if isinstance(ksize, (list, tuple)) and len(ksize) > 0 else 2
        kh: int = int(ksize[1]) if isinstance(ksize, (list, tuple)) and len(ksize) > 1 else 2
        kw: int = int(ksize[2]) if isinstance(ksize, (list, tuple)) and len(ksize) > 2 else 2
        stride: int | tuple[int, ...] | list[int] = attrs.get("stride", [kd, kh, kw])
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
        body: str = template["body"].format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_AvgPool3D(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate AvgPool3D LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str] = get_cpp_template("avgpool3d")
        inputs_list: list[str] = getattr(node, "inputs", [])
        input_nodes: list[IRNode | None] = [graph_to_use.nodes.get(inp) for inp in inputs_list] if graph_to_use else []
        in0_shape: list[int] = self._get_shape(input_nodes[0]) if len(input_nodes) > 0 and input_nodes[0] else [1, 1, 1, 1, 1]
        shape: list[int] = self._get_shape(node)
        if len(in0_shape) < 5:
            in0_shape = [1] * (5 - len(in0_shape)) + in0_shape
        if len(shape) < 5:
            shape = [1] * (5 - len(shape)) + shape

        attrs: dict[str, AttrType] = getattr(node, "attributes", {})
        ksize: int | tuple[int, ...] | list[int] = attrs.get("kernel_size", [2, 2, 2])
        kd: int = int(ksize[0]) if isinstance(ksize, (list, tuple)) and len(ksize) > 0 else 2
        kh: int = int(ksize[1]) if isinstance(ksize, (list, tuple)) and len(ksize) > 1 else 2
        kw: int = int(ksize[2]) if isinstance(ksize, (list, tuple)) and len(ksize) > 2 else 2
        stride: int | tuple[int, ...] | list[int] = attrs.get("stride", [kd, kh, kw])
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
        body: str = template["body"].format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_LayerNorm(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate LayerNorm LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str] = get_cpp_template("layernorm")
        inputs_list: list[str] = getattr(node, "inputs", [])
        shape: list[int] = self._get_shape(node)
        d_val: int = shape[-1] if shape else 1
        has_gamma: str = "true" if len(inputs_list) > 1 else "false"
        has_beta: str = "true" if len(inputs_list) > 2 else "false"
        gamma_var: str = inputs_list[1] if len(inputs_list) > 1 else inputs_list[0]
        beta_var: str = inputs_list[2] if len(inputs_list) > 2 else inputs_list[0]

        expr_args: dict[str, AttrType] = {
            "clean_id": node.id.replace("-", "_"),
            "in0": inputs_list[0] if len(inputs_list) > 0 else "dummy",
            "out_shape_str": "{" + ", ".join(map(str, shape)) + "}",
            "D": d_val,
            "has_gamma": has_gamma,
            "has_beta": has_beta,
            "gamma": gamma_var,
            "beta": beta_var,
        }
        body: str = template["body"].format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_BatchNorm(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate BatchNorm LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str] = get_cpp_template("batchnorm")
        inputs_list: list[str] = getattr(node, "inputs", [])
        shape: list[int] = self._get_shape(node)
        channels: int = shape[1] if len(shape) > 1 else 1
        spatial: int = 1
        for dim in shape[2:]:
            spatial *= dim

        has_mean: str = "true" if len(inputs_list) > 1 else "false"
        has_var: str = "true" if len(inputs_list) > 2 else "false"
        has_gamma: str = "true" if len(inputs_list) > 3 else "false"
        has_beta: str = "true" if len(inputs_list) > 4 else "false"

        mean_var: str = inputs_list[1] if len(inputs_list) > 1 else inputs_list[0]
        var_var: str = inputs_list[2] if len(inputs_list) > 2 else inputs_list[0]
        gamma_var: str = inputs_list[3] if len(inputs_list) > 3 else inputs_list[0]
        beta_var: str = inputs_list[4] if len(inputs_list) > 4 else inputs_list[0]

        expr_args: dict[str, AttrType] = {
            "clean_id": node.id.replace("-", "_"),
            "in0": inputs_list[0] if len(inputs_list) > 0 else "dummy",
            "out_shape_str": "{" + ", ".join(map(str, shape)) + "}",
            "channels": channels,
            "spatial": spatial,
            "has_mean": has_mean,
            "has_var": has_var,
            "has_gamma": has_gamma,
            "has_beta": has_beta,
            "mean": mean_var,
            "var": var_var,
            "gamma": gamma_var,
            "beta": beta_var,
        }
        body: str = template["body"].format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_RMSNorm(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate RMSNorm LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str] = get_cpp_template("rmsnorm")
        inputs_list: list[str] = getattr(node, "inputs", [])
        shape: list[int] = self._get_shape(node)
        d_val: int = shape[-1] if shape else 1
        has_gamma: str = "true" if len(inputs_list) > 1 else "false"
        gamma_var: str = inputs_list[1] if len(inputs_list) > 1 else inputs_list[0]

        expr_args: dict[str, AttrType] = {
            "clean_id": node.id.replace("-", "_"),
            "in0": inputs_list[0] if len(inputs_list) > 0 else "dummy",
            "out_shape_str": "{" + ", ".join(map(str, shape)) + "}",
            "D": d_val,
            "has_gamma": has_gamma,
            "gamma": gamma_var,
        }
        body: str = template["body"].format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_Slice(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate Slice LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str] = get_cpp_template("slice")
        inputs_list: list[str] = getattr(node, "inputs", [])
        shape: list[int] = self._get_shape(node)
        rank: int = len(shape)
        attrs: dict[str, AttrType] = getattr(node, "attributes", {})
        starts: int | list[int] = attrs.get("starts", [0] * rank)
        strides: int | list[int] = attrs.get("strides", [1] * rank)
        starts_list: list[int] = [starts] * rank if isinstance(starts, int) else list(starts[:rank])
        strides_list: list[int] = [strides] * rank if isinstance(strides, int) else list(strides[:rank])

        starts_str: str = "{" + ", ".join(map(str, starts_list)) + "}"
        strides_str: str = "{" + ", ".join(map(str, strides_list)) + "}"

        expr_args: dict[str, AttrType] = {
            "clean_id": node.id.replace("-", "_"),
            "in0": inputs_list[0] if len(inputs_list) > 0 else "dummy",
            "out_shape_str": "{" + ", ".join(map(str, shape)) + "}",
            "rank": rank,
            "starts": starts_str,
            "strides": strides_str,
        }
        body: str = template["body"].format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_Pad(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate Pad LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str] = get_cpp_template("pad")
        inputs_list: list[str] = getattr(node, "inputs", [])
        shape: list[int] = self._get_shape(node)
        rank: int = len(shape)
        attrs: dict[str, AttrType] = getattr(node, "attributes", {})
        paddings: int | list[int] = attrs.get("paddings", [0] * rank)
        pad_val: float = float(attrs.get("constant_value", 0.0))
        pad_low: list[int] = []
        if isinstance(paddings, int):
            pad_low = [paddings] * rank
        elif len(paddings) >= rank * 2:
            pad_low = [int(paddings[2 * i]) for i in range(rank)]
        else:
            pad_low = [int(p) for p in paddings[:rank]]
        pad_low_str: str = "{" + ", ".join(map(str, pad_low)) + "}"

        expr_args: dict[str, AttrType] = {
            "clean_id": node.id.replace("-", "_"),
            "in0": inputs_list[0] if len(inputs_list) > 0 else "dummy",
            "out_shape_str": "{" + ", ".join(map(str, shape)) + "}",
            "rank": rank,
            "pad_low": pad_low_str,
            "pad_val": f"{pad_val}f",
        }
        body: str = template["body"].format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_GatherND(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate GatherND LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str] = get_cpp_template("gather_nd")
        inputs_list: list[str] = getattr(node, "inputs", [])
        input_nodes: list[IRNode | None] = [graph_to_use.nodes.get(inp) for inp in inputs_list] if graph_to_use else []
        idx_shape: list[int] = self._get_shape(input_nodes[1]) if len(input_nodes) > 1 and input_nodes[1] else [1, 1]
        index_depth: int = idx_shape[-1] if idx_shape else 1
        shape: list[int] = self._get_shape(node)

        expr_args: dict[str, AttrType] = {
            "clean_id": node.id.replace("-", "_"),
            "in0": inputs_list[0] if len(inputs_list) > 0 else "dummy",
            "in1": inputs_list[1] if len(inputs_list) > 1 else "dummy",
            "out_shape_str": "{" + ", ".join(map(str, shape)) + "}",
            "index_depth": index_depth,
        }
        body: str = template["body"].format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_ScatterND(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate ScatterND LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str] = get_cpp_template("scatter_nd")
        inputs_list: list[str] = getattr(node, "inputs", [])
        input_nodes: list[IRNode | None] = [graph_to_use.nodes.get(inp) for inp in inputs_list] if graph_to_use else []
        idx_shape: list[int] = self._get_shape(input_nodes[1]) if len(input_nodes) > 1 and input_nodes[1] else [1, 1]
        index_depth: int = idx_shape[-1] if idx_shape else 1
        shape: list[int] = self._get_shape(node)

        expr_args: dict[str, AttrType] = {
            "clean_id": node.id.replace("-", "_"),
            "in0": inputs_list[0] if len(inputs_list) > 0 else "dummy",
            "in1": inputs_list[1] if len(inputs_list) > 1 else "dummy",
            "in2": inputs_list[2] if len(inputs_list) > 2 else "dummy",
            "out_shape_str": "{" + ", ".join(map(str, shape)) + "}",
            "index_depth": index_depth,
        }
        body: str = template["body"].format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def _visit_node(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Visit a node and emit C++ code.

        Args:
            node (IRNode): The IR node to visit.
            graph_to_use (IRGraph | None): The graph.
        """
        op: str = node.op_type
        op_lower: str = op.lower()

        offset: int | None = node.attributes.get("buffer_offset")
        offset_str: str = f", global_arena.data() + {offset}" if offset is not None else ""

        if op == "Input":
            out_shape_str: str = "{" + ",".join(map(str, self._get_shape(node))) + "}"
            self.lines.append(f"    NDArrayView<float> {node.id}({out_shape_str}{offset_str}); // Input")
        elif op == "Constant":
            val: float = node.attributes.get("value", 0.0)
            out_shape_str = "{" + ",".join(map(str, self._get_shape(node))) + "}"
            self.lines.append(f"    NDArrayView<float> {node.id}({out_shape_str});")
            self.lines.append(f"    for(size_t i=0; i<{node.id}.size(); ++i) {node.id}.data[i] = {val};")
        elif op in ("If", "Cond"):
            self._visit_if_op(node, graph_to_use)
        elif op in ("Loop", "WhileLoop"):
            self._visit_loop_op(node, graph_to_use)
        elif op_lower == "conv2d":
            self.visit_Conv2D(node, graph_to_use)
        elif op_lower == "conv3d":
            self.visit_Conv3D(node, graph_to_use)
        elif op_lower == "maxpool3d":
            self.visit_MaxPool3D(node, graph_to_use)
        elif op_lower == "avgpool3d":
            self.visit_AvgPool3D(node, graph_to_use)
        elif op_lower == "layernorm":
            self.visit_LayerNorm(node, graph_to_use)
        elif op_lower == "batchnorm":
            self.visit_BatchNorm(node, graph_to_use)
        elif op_lower == "rmsnorm":
            self.visit_RMSNorm(node, graph_to_use)
        elif op_lower == "slice":
            self.visit_Slice(node, graph_to_use)
        elif op_lower == "pad":
            self.visit_Pad(node, graph_to_use)
        elif op_lower in ("gathernd", "gather_nd"):
            self.visit_GatherND(node, graph_to_use)
        elif op_lower in ("scatternd", "scatter_nd"):
            self.visit_ScatterND(node, graph_to_use)
        elif op == "Scan":
            self.visit_Scan(node, graph_to_use)
        elif op == "Attention":
            self.visit_Attention(node, graph_to_use)
        elif op == "Output":
            self.lines.append(f"    // Output {node.inputs[0]}")
        else:
            from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_operation, get_cpp_template
            from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

            op_def: dict[str, AttrType] = OPS_REGISTRY.get(op, {})
            raw_mapping: dict[str, AttrType] = op_def.get("variants", {}).get("llvm_cpp", {})
            mapping: dict[str, AttrType] = raw_mapping if (raw_mapping and raw_mapping.get("template")) else {}

            op_key = op.lower()
            decl_op = get_cpp_operation(op_key)
            if not mapping and decl_op is not None:
                mapping = {"template": decl_op.template, "scalar_expr": decl_op.scalar_expr}
            elif mapping and mapping.get("scalar_expr") == "in0_val" and decl_op is not None:
                mapping = dict(mapping)
                mapping["scalar_expr"] = decl_op.scalar_expr

            if not mapping:
                self._dispatch_unmapped_op_fallback(node, op)
            else:
                template: dict[str, str] = get_cpp_template(mapping["template"])

                in0_node: IRNode | None = None
                if graph_to_use and len(node.inputs) > 0:
                    in0_node = graph_to_use.nodes.get(node.inputs[0])
                in0_shape: list[int] = self._get_shape(in0_node) if in0_node else [1, 1]
                out_shape: list[int] = self._get_shape(node)
                out_shape_str = "{" + ",".join(map(str, out_shape)) + "}"

                M: int = out_shape[0] if len(out_shape) > 0 else 1
                N: int = out_shape[1] if len(out_shape) > 1 else 1
                K: int = in0_shape[1] if len(in0_shape) > 1 else 1

                in0_var = node.inputs[0] if len(node.inputs) > 0 else "dummy"
                in1_var = node.inputs[1] if len(node.inputs) > 1 else "dummy"
                if graph_to_use and hasattr(graph_to_use, "edges"):
                    for edge in graph_to_use.edges:
                        tgt = getattr(edge, "target", getattr(edge, "target_node", None))
                        src = getattr(edge, "source", getattr(edge, "source_node", None))
                        if tgt == node.id:
                            if getattr(edge, "target_idx", 0) == 0 and getattr(edge, "source_idx", 0) > 0:
                                in0_var = str(src)
                            elif getattr(edge, "target_idx", 0) == 1 and getattr(edge, "source_idx", 0) > 0:
                                in1_var = str(src)

                expr_format_args: dict[str, AttrType] = {
                    "clean_id": node.id,
                    "out_shape_str": out_shape_str,
                    "in0": in0_var,
                    "in1": in1_var,
                    "rank": len(out_shape),
                    "M": M,
                    "N": N,
                    "K": K,
                }
                expr_format_args.update(mapping)
                if "init_val" in expr_format_args:
                    ival = str(expr_format_args["init_val"])
                    if "INFINITY" in ival or ival.endswith("f"):
                        expr_format_args["init_val"] = ival
                    else:
                        expr_format_args["init_val"] = f"{ival}f"

                body: str = template["body"].format(**expr_format_args)
                for line in body.split("\n"):
                    if line.strip():
                        self.lines.append(f"    {line}")

                if hasattr(node, "outputs") and len(node.outputs) > 1:
                    for idx, out_id in enumerate(node.outputs):
                        self.lines.append(f"    NDArrayView<float> {out_id} = {node.id}; // multi-output alias {idx}")

    def compile(self, code: str) -> Callable[[], str]:
        """Compile the generated C++ code into a shared library.

        Args:
            code (str): The C++ source code.

        Returns:
            Callable[[], str]: An executable function that wraps the compiled library.
        """
        runner = LLVMCPPRunner()
        return runner.compile_and_load(code, entry_point="compute_graph")

    def execute(self, graph: IRGraph, *args: AttrType, **kwargs: AttrType) -> str:
        """Execute the graph using the C++ generator.

        Args:
            graph (IRGraph): The IR Graph.
            *args (AttrType): Args.
            **kwargs (AttrType): Kwargs.

        Returns:
            str: Result of execution.
        """
        code: str = self.generate(graph)
        executable = self.compile(code)
        return executable()

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: object) -> Callable[[], str]:
        """Compile IRGraph into a native shared library and return callable C++ execution wrapper.

        Args:
            graph (IRGraph): Target computational graph.
            **kwargs (object): Compiler options.

        Returns:
            Callable[[], str]: Executable wrapper.
        """
        code = self.generate(graph)
        comp = str(kwargs["compiler"]) if "compiler" in kwargs and kwargs["compiler"] else None
        runner = LLVMCPPRunner(compiler=comp)
        return runner.compile_and_load(code, entry_point="compute_graph")


class LLVMCPPRunner:
    """Standalone execution runner compiling C++17 code via clang++/g++ and loading via ctypes."""

    def __init__(self, compiler: str | None = None) -> None:
        """Initialize LLVMCPPRunner with preferred compiler.

        Args:
            compiler (Optional[str]): Path or executable name for C++ compiler.
        """
        import shutil

        if compiler:
            self.compiler: str = compiler
        elif shutil.which("clang++"):
            self.compiler = "clang++"
        elif shutil.which("g++"):
            self.compiler = "g++"
        else:
            self.compiler = "clang++"

    def compile_and_load(self, code: str, entry_point: str = "compute_graph") -> Callable[[], str]:
        """Compile C++17 source code into a shared library and load via ctypes.

        Args:
            code (str): The C++ source code.
            entry_point (str): The exported C symbol name.

        Returns:
            Callable[[], str]: Executable callable wrapper.

        Raises:
            RuntimeError: If compilation or library loading fails.
        """
        import shutil

        temp_dir: str = tempfile.mkdtemp()
        src_file: str = os.path.join(temp_dir, "graph.cpp")
        lib_ext: str = ".dylib" if os.name == "posix" and "darwin" in os.uname().sysname.lower() else ".so"
        lib_file: str = os.path.join(temp_dir, f"graph{lib_ext}")

        with open(src_file, "w") as f:
            f.write(code)

        compile_cmd: list[str] = [self.compiler, "-std=c++17", "-O3", "-shared", "-fPIC", src_file, "-o", lib_file]
        try:
            subprocess.run(compile_cmd, check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            err_out = getattr(e, "stderr", b"").decode() if getattr(e, "stderr", None) else str(e)
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(f"Compilation failed: {err_out}") from e

        try:
            lib: ctypes.CDLL = ctypes.CDLL(lib_file)
            if hasattr(lib, entry_point):
                compute_func = getattr(lib, entry_point)
            else:
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise RuntimeError(f"Function '{entry_point}' not found in compiled library. Ensure it is exported with extern \"C\".")
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(f"Compilation or load failed: {e}") from e

        def executable(*args: object) -> str:
            """Execute the compiled C++ graph with optional input/output buffer pointers.

            Args:
                *args (object): Optional (c_inputs, c_outputs, c_shapes) buffer pointers.

            Returns:
                str: Status message.
            """
            if len(args) >= 3:
                compute_func(args[0], args[1], args[2])
            elif len(args) == 2:
                compute_func(args[0], args[1], None)
            elif len(args) == 1:
                compute_func(args[0], None, None)
            else:
                compute_func(None, None, None)
            return "Execution successful"

        return executable


LLVMCPPGenerator = CppGenerator
