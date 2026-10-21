# ruff: noqa: E402
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
import typing
from collections.abc import Callable  # noqa: F401

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, as_float, as_int_or_seq
from ml_switcheroo_compiler.ir.core import as_graph, as_opt_graph

AttrType = typing.Union[int, float, str, bool, list["AttrType"], tuple["AttrType", ...], dict[str, "AttrType"], "None"]

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
        super().__init__(graph=as_graph(graph))
        self.use_simd = use_simd
        self.use_openmp = use_openmp
        self.strict = strict
        self.lines: list[str] = []

    def _get_shape(self, node: IRNode) -> list[int]:
        """Execute get shape for compiler processing.

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
            self.lines.append("    #pragma omp simd")
            self.lines.append(f"    for(size_t i = 0; i < {clean_id}.size(); ++i) {{ {clean_id}.data[i] = ({in0}.size() > 0) ? {in0}.data[i % {in0}.size()] : 0.0f; }}")
        else:
            self.lines.append("    #pragma omp simd")
            self.lines.append(f"    for(size_t i = 0; i < {clean_id}.size(); ++i) {{ {clean_id}.data[i] = 0.0f; }}")

    def _num_elements(self, shape: list[int]) -> int:
        """Execute num elements for compiler processing.

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
        """Execute get strides for compiler processing.

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
            raw_offset = node.attributes.get("buffer_offset")
            raw_size = node.attributes.get("buffer_size")
            offset: int | None = int(raw_offset) if isinstance(raw_offset, (int, float)) else None
            size: int | None = int(raw_size) if isinstance(raw_size, (int, float)) else None
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
        """Execute visit if op for compiler processing.

        Args:
            node (IRNode): The node parameter.
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        assert len(node.inputs) >= 1
        cond_var: str = node.inputs[0]

        then_lines: list[str] = []
        then_graph: IRGraph | None = as_opt_graph(node.attributes.get("then_branch"))
        if then_graph:
            old_lines = self.lines
            self.lines = then_lines
            for _, sub_node in then_graph.nodes.items():
                self._visit_node(sub_node, graph_to_use)
            self.lines = old_lines

        else_lines: list[str] = []
        else_graph: IRGraph | None = as_opt_graph(node.attributes.get("else_branch"))
        if else_graph:
            old_lines = self.lines
            self.lines = else_lines
            for _, sub_node in else_graph.nodes.items():
                self._visit_node(sub_node, graph_to_use)
            self.lines = old_lines

        template = get_cpp_template("if_op")
        body = getattr(template["body"], "format")(cond_var=cond_var, then_body="\n".join(then_lines) + "\n" if then_lines else "", else_body="\n".join(else_lines) + "\n" if else_lines else "")
        self.lines.extend(body.strip().split("\n"))

    def _collect_subgraph_lines(self, subgraph: IRGraph | None, graph_to_use: IRGraph | None) -> list[str]:
        """Collect emitted C++ lines for a subgraph branch.

        Args:
            subgraph (IRGraph | None): The branch subgraph to traverse.
            graph_to_use (IRGraph | None): The enclosing computational graph.

        Returns:
            list[str]: Collected C++ source lines for the subgraph.
        """
        branch_lines: list[str] = []
        if subgraph:
            old_lines = self.lines
            self.lines = branch_lines
            for _, sub_node in subgraph.nodes.items():
                self._visit_node(sub_node, graph_to_use)
            self.lines = old_lines
        return branch_lines

    def _emit_loop_annotations(self, attrs: dict[str, AttrType]) -> None:
        """Emit loop option comments and OpenMP pragma directives.

        Args:
            attrs (dict[str, AttrType]): Loop node attributes dictionary.
        """
        max_iters = attrs.get("maximum_iterations") or attrs.get("max_iters")
        parallel_iters = attrs.get("parallel_iterations")
        swap_memory = attrs.get("swap_memory")
        shape_invariants = attrs.get("shape_invariants")
        annotations: list[str] = []
        for key, val in (
            ("parallel_iterations", parallel_iters),
            ("swap_memory", swap_memory),
            ("maximum_iterations", max_iters),
            ("shape_invariants", shape_invariants),
        ):
            if val is not None:
                annotations.append(f"{key}={val}")
        if annotations:
            self.lines.append(f"    // Loop annotations: {', '.join(annotations)}")
        if parallel_iters is not None:
            self.lines.append(f"    #pragma omp parallel for num_threads({parallel_iters})")

    def _visit_loop_op(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate Loop or WhileLoop C++ code with dynamic loop options propagation.

        Args:
            node (IRNode): The loop IR node.
            graph_to_use (IRGraph | None): The computational graph to reference.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        attrs: dict[str, AttrType] = getattr(node, "attributes", {})
        self._emit_loop_annotations(attrs)

        raw_cond: EagerValue = attrs.get("cond")
        cond_graph: IRGraph | None = raw_cond if isinstance(raw_cond, IRGraph) else None
        cond_lines = self._collect_subgraph_lines(cond_graph, graph_to_use)

        raw_body: EagerValue = attrs.get("body")
        body_graph: IRGraph | None = raw_body if isinstance(raw_body, IRGraph) else None
        body_lines = self._collect_subgraph_lines(body_graph, graph_to_use)

        template = get_cpp_template("loop_op")
        body = getattr(template["body"], "format")(
            cond_body="\n".join(cond_lines) + "\n" if cond_lines else "",
            loop_body="\n".join(body_lines) + "\n" if body_lines else "",
        )
        self.lines.extend(body.strip().split("\n"))

    def visit_Conv2D(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate Conv2D LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | EagerValue] = get_cpp_template("conv2d")

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
        stride: int | tuple[int, ...] | list[int] = as_int_or_seq(attrs.get("stride", 1))
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
            node: The node parameter.
            graph_to_use: The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | EagerValue] = get_cpp_template("scan")
        out_shape: list[int] = self._get_shape(node)
        out_shape_str = "{" + ",".join(map(str, out_shape)) + "}"
        expr_args: dict[str, AttrType] = {
            "clean_id": node.id.replace("-", "_"),
            "out_shape_str": out_shape_str,
            "in0": getattr(node, "inputs", ["dummy"])[0] if getattr(node, "inputs", []) else "dummy",
        }
        body: str = str(template["body"]).format(**expr_args)
        for line in body.split("\n"):
            if line.strip():
                self.lines.append(f"    {line}")

    def visit_Attention(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate Attention LLVM CPP.

        Args:
            node: The node parameter.
            graph_to_use: The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | EagerValue] = get_cpp_template("attention")
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
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | EagerValue] = get_cpp_template("conv3d")
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
        stride: int | tuple[int, ...] | list[int] = as_int_or_seq(attrs.get("stride", [1, 1, 1]))
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
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | EagerValue] = get_cpp_template("maxpool3d")
        inputs_list: list[str] = getattr(node, "inputs", [])
        input_nodes: list[IRNode | None] = [graph_to_use.nodes.get(inp) for inp in inputs_list] if graph_to_use else []
        in0_shape: list[int] = self._get_shape(input_nodes[0]) if len(input_nodes) > 0 and input_nodes[0] else [1, 1, 1, 1, 1]
        shape: list[int] = self._get_shape(node)
        if len(in0_shape) < 5:
            in0_shape = [1] * (5 - len(in0_shape)) + in0_shape
        if len(shape) < 5:
            shape = [1] * (5 - len(shape)) + shape

        attrs: dict[str, AttrType] = getattr(node, "attributes", {})
        ksize: int | tuple[int, ...] | list[int] = as_int_or_seq(attrs.get("kernel_size", [2, 2, 2]))
        kd: int = int(ksize[0]) if isinstance(ksize, (list, tuple)) and len(ksize) > 0 else 2
        kh: int = int(ksize[1]) if isinstance(ksize, (list, tuple)) and len(ksize) > 1 else 2
        kw: int = int(ksize[2]) if isinstance(ksize, (list, tuple)) and len(ksize) > 2 else 2
        stride: int | tuple[int, ...] | list[int] = as_int_or_seq(attrs.get("stride", [kd, kh, kw]))
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
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | EagerValue] = get_cpp_template("avgpool3d")
        inputs_list: list[str] = getattr(node, "inputs", [])
        input_nodes: list[IRNode | None] = [graph_to_use.nodes.get(inp) for inp in inputs_list] if graph_to_use else []
        in0_shape: list[int] = self._get_shape(input_nodes[0]) if len(input_nodes) > 0 and input_nodes[0] else [1, 1, 1, 1, 1]
        shape: list[int] = self._get_shape(node)
        if len(in0_shape) < 5:
            in0_shape = [1] * (5 - len(in0_shape)) + in0_shape
        if len(shape) < 5:
            shape = [1] * (5 - len(shape)) + shape

        attrs: dict[str, AttrType] = getattr(node, "attributes", {})
        ksize: int | tuple[int, ...] | list[int] = as_int_or_seq(attrs.get("kernel_size", [2, 2, 2]))
        kd: int = int(ksize[0]) if isinstance(ksize, (list, tuple)) and len(ksize) > 0 else 2
        kh: int = int(ksize[1]) if isinstance(ksize, (list, tuple)) and len(ksize) > 1 else 2
        kw: int = int(ksize[2]) if isinstance(ksize, (list, tuple)) and len(ksize) > 2 else 2
        stride: int | tuple[int, ...] | list[int] = as_int_or_seq(attrs.get("stride", [kd, kh, kw]))
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

    def visit_LayerNorm(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate LayerNorm LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | EagerValue] = get_cpp_template("layernorm")
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
        body: str = str(template["body"]).format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_BatchNorm(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate BatchNorm LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | EagerValue] = get_cpp_template("batchnorm")
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
        body: str = str(template["body"]).format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_RMSNorm(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate RMSNorm LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | EagerValue] = get_cpp_template("rmsnorm")
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
        body: str = str(template["body"]).format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_Slice(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate Slice LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | EagerValue] = get_cpp_template("slice")
        inputs_list: list[str] = getattr(node, "inputs", [])
        shape: list[int] = self._get_shape(node)
        rank: int = len(shape)
        attrs: dict[str, AttrType] = getattr(node, "attributes", {})
        starts: int | tuple[int, ...] = as_int_or_seq(attrs.get("starts", [0] * rank))
        strides: int | tuple[int, ...] = as_int_or_seq(attrs.get("strides", [1] * rank))
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
        body: str = str(template["body"]).format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_Pad(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate Pad LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | EagerValue] = get_cpp_template("pad")
        inputs_list: list[str] = getattr(node, "inputs", [])
        shape: list[int] = self._get_shape(node)
        rank: int = len(shape)
        attrs: dict[str, AttrType] = getattr(node, "attributes", {})
        paddings: int | tuple[int, ...] = as_int_or_seq(attrs.get("paddings", [0] * rank))
        pad_val: float = float(as_float(attrs.get("constant_value", 0.0)))
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
        body: str = str(template["body"]).format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_GatherND(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate GatherND LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | EagerValue] = get_cpp_template("gather_nd")
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
        body: str = str(template["body"]).format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_ScatterND(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate ScatterND LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | EagerValue] = get_cpp_template("scatter_nd")
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
        body: str = str(template["body"]).format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_MaxPool2D(self, node: IRNode, graph_to_use: IRGraph | None = None, is_global: bool = False) -> None:
        """Generate MaxPool2D LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): The graph_to_use parameter.
            is_global (bool): Whether this is a global max pooling operation.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | EagerValue] = get_cpp_template("maxpool2d")
        inputs_list: list[str] = getattr(node, "inputs", [])
        input_nodes: list[IRNode | None] = [graph_to_use.nodes.get(inp) for inp in inputs_list] if graph_to_use else []
        in0_shape: list[int] = self._get_shape(input_nodes[0]) if len(input_nodes) > 0 and input_nodes[0] else [1, 1, 1, 1]
        shape: list[int] = self._get_shape(node)
        if len(in0_shape) < 4:
            in0_shape = [1] * (4 - len(in0_shape)) + in0_shape
        if len(shape) < 4:
            shape = [1] * (4 - len(shape)) + shape

        attrs: dict[str, AttrType] = getattr(node, "attributes", {})
        if is_global:
            kh: int = in0_shape[2]
            kw: int = in0_shape[3]
            stride_h: int = 1
            stride_w: int = 1
        else:
            ksize: AttrType = attrs.get("kernel_size", [2, 2])
            kh = int(ksize[0]) if isinstance(ksize, (list, tuple)) and len(ksize) > 0 and isinstance(ksize[0], (int, float, str)) else (int(ksize) if isinstance(ksize, int) else 2)
            kw = int(ksize[1]) if isinstance(ksize, (list, tuple)) and len(ksize) > 1 and isinstance(ksize[1], (int, float, str)) else kh
            stride: AttrType = attrs.get("stride", [kh, kw])
            stride_h = int(stride[0]) if isinstance(stride, (list, tuple)) and len(stride) > 0 and isinstance(stride[0], (int, float, str)) else (int(stride) if isinstance(stride, int) else kh)
            stride_w = int(stride[1]) if isinstance(stride, (list, tuple)) and len(stride) > 1 and isinstance(stride[1], (int, float, str)) else stride_h

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
            graph_to_use (IRGraph | None): The graph_to_use parameter.
            is_global (bool): Whether this is a global average pooling operation.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | EagerValue] = get_cpp_template("avgpool2d")
        inputs_list: list[str] = getattr(node, "inputs", [])
        input_nodes: list[IRNode | None] = [graph_to_use.nodes.get(inp) for inp in inputs_list] if graph_to_use else []
        in0_shape: list[int] = self._get_shape(input_nodes[0]) if len(input_nodes) > 0 and input_nodes[0] else [1, 1, 1, 1]
        shape: list[int] = self._get_shape(node)
        if len(in0_shape) < 4:
            in0_shape = [1] * (4 - len(in0_shape)) + in0_shape
        if len(shape) < 4:
            shape = [1] * (4 - len(shape)) + shape

        attrs: dict[str, AttrType] = getattr(node, "attributes", {})
        if is_global:
            kh: int = in0_shape[2]
            kw: int = in0_shape[3]
            stride_h: int = 1
            stride_w: int = 1
        else:
            ksize: AttrType = attrs.get("kernel_size", [2, 2])
            kh = int(ksize[0]) if isinstance(ksize, (list, tuple)) and len(ksize) > 0 and isinstance(ksize[0], (int, float, str)) else (int(ksize) if isinstance(ksize, int) else 2)
            kw = int(ksize[1]) if isinstance(ksize, (list, tuple)) and len(ksize) > 1 and isinstance(ksize[1], (int, float, str)) else kh
            stride: AttrType = attrs.get("stride", [kh, kw])
            stride_h = int(stride[0]) if isinstance(stride, (list, tuple)) and len(stride) > 0 and isinstance(stride[0], (int, float, str)) else (int(stride) if isinstance(stride, int) else kh)
            stride_w = int(stride[1]) if isinstance(stride, (list, tuple)) and len(stride) > 1 and isinstance(stride[1], (int, float, str)) else stride_h

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
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | EagerValue] = get_cpp_template("groupnorm")
        inputs_list: list[str] = getattr(node, "inputs", [])
        has_gamma: str = "true" if len(inputs_list) > 1 else "false"
        has_beta: str = "true" if len(inputs_list) > 2 else "false"
        gamma_var: str = inputs_list[1] if len(inputs_list) > 1 else (inputs_list[0] if len(inputs_list) > 0 else "dummy")
        beta_var: str = inputs_list[2] if len(inputs_list) > 2 else (inputs_list[0] if len(inputs_list) > 0 else "dummy")
        shape: list[int] = self._get_shape(node)
        attrs: dict[str, AttrType] = getattr(node, "attributes", {})
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
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | EagerValue] = get_cpp_template("softmax")
        inputs_list: list[str] = getattr(node, "inputs", [])
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
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | EagerValue] = get_cpp_template("logsoftmax")
        inputs_list: list[str] = getattr(node, "inputs", [])
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
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | EagerValue] = get_cpp_template("warp_shuffle_reduce")
        inputs_list: list[str] = getattr(node, "inputs", [])
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
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | EagerValue] = get_cpp_template("shared_memory_tree_reduce")
        inputs_list: list[str] = getattr(node, "inputs", [])
        shape: list[int] = self._get_shape(node)
        expr_args: dict[str, AttrType] = {
            "clean_id": node.id.replace("-", "_"),
            "in0": inputs_list[0] if len(inputs_list) > 0 else "dummy",
            "out_shape_str": "{" + ", ".join(map(str, shape)) + "}",
        }
        body: str = str(template["body"]).format(**expr_args)
        for line in body.split("\n"):
            self.lines.append(f"    {line}")

    def visit_TiledMatMul2D(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Generate TiledMatMul2D LLVM CPP.

        Args:
            node (IRNode): The IRNode.
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | EagerValue] = get_cpp_template("tiled_matmul_2d")
        inputs_list: list[str] = getattr(node, "inputs", [])
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
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | EagerValue] = get_cpp_template("conv2d_shared_halo")
        inputs_list: list[str] = getattr(node, "inputs", [])
        input_nodes: list[IRNode | None] = [graph_to_use.nodes.get(inp) for inp in inputs_list] if graph_to_use else []
        in0_shape: list[int] = self._get_shape(input_nodes[0]) if len(input_nodes) > 0 and input_nodes[0] else [1, 1, 1, 1]
        w_shape: list[int] = self._get_shape(input_nodes[1]) if len(input_nodes) > 1 and input_nodes[1] else ([1, 1, 1, 1] if len(in0_shape) >= 4 else [1, 1, 3, 3])
        shape: list[int] = self._get_shape(node)
        if len(in0_shape) < 4:
            in0_shape = [1] * (4 - len(in0_shape)) + in0_shape
        if len(w_shape) < 4:
            w_shape = [1] * (4 - len(w_shape)) + w_shape
        if len(shape) < 4:
            shape = [1] * (4 - len(shape)) + shape

        attrs: dict[str, AttrType] = getattr(node, "attributes", {})
        stride: AttrType = attrs.get("stride", 1)
        stride_h: int = int(stride[0]) if isinstance(stride, (tuple, list)) and isinstance(stride[0], (int, float, str)) else (int(stride) if isinstance(stride, (int, float, str)) else 1)
        stride_w: int = int(stride[1]) if isinstance(stride, (tuple, list)) and len(stride) > 1 and isinstance(stride[1], (int, float, str)) else stride_h

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
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | EagerValue] = get_cpp_template("maxpool2d_dynamic")
        inputs_list: list[str] = getattr(node, "inputs", [])
        input_nodes: list[IRNode | None] = [graph_to_use.nodes.get(inp) for inp in inputs_list] if graph_to_use else []
        in0_shape: list[int] = self._get_shape(input_nodes[0]) if len(input_nodes) > 0 and input_nodes[0] else [1, 1, 1, 1]
        shape: list[int] = self._get_shape(node)
        if len(in0_shape) < 4:
            in0_shape = [1] * (4 - len(in0_shape)) + in0_shape
        if len(shape) < 4:
            shape = [1] * (4 - len(shape)) + shape

        attrs: dict[str, AttrType] = getattr(node, "attributes", {})
        ksize: AttrType = attrs.get("kernel_size", [2, 2])
        kh: int = int(ksize[0]) if isinstance(ksize, (list, tuple)) and len(ksize) > 0 and isinstance(ksize[0], (int, float, str)) else (int(ksize) if isinstance(ksize, int) else 2)
        kw: int = int(ksize[1]) if isinstance(ksize, (list, tuple)) and len(ksize) > 1 and isinstance(ksize[1], (int, float, str)) else kh
        stride: AttrType = attrs.get("stride", [kh, kw])
        stride_h: int = int(stride[0]) if isinstance(stride, (list, tuple)) and len(stride) > 0 and isinstance(stride[0], (int, float, str)) else (int(stride) if isinstance(stride, int) else kh)
        stride_w: int = int(stride[1]) if isinstance(stride, (list, tuple)) and len(stride) > 1 and isinstance(stride[1], (int, float, str)) else stride_h
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
            graph_to_use (IRGraph | None): The graph_to_use parameter.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | EagerValue] = get_cpp_template("avgpool2d_dynamic")
        inputs_list: list[str] = getattr(node, "inputs", [])
        input_nodes: list[IRNode | None] = [graph_to_use.nodes.get(inp) for inp in inputs_list] if graph_to_use else []
        in0_shape: list[int] = self._get_shape(input_nodes[0]) if len(input_nodes) > 0 and input_nodes[0] else [1, 1, 1, 1]
        shape: list[int] = self._get_shape(node)
        if len(in0_shape) < 4:
            in0_shape = [1] * (4 - len(in0_shape)) + in0_shape
        if len(shape) < 4:
            shape = [1] * (4 - len(shape)) + shape

        attrs: dict[str, AttrType] = getattr(node, "attributes", {})
        ksize: AttrType = attrs.get("kernel_size", [2, 2])
        kh: int = int(ksize[0]) if isinstance(ksize, (list, tuple)) and len(ksize) > 0 and isinstance(ksize[0], (int, float, str)) else (int(ksize) if isinstance(ksize, int) else 2)
        kw: int = int(ksize[1]) if isinstance(ksize, (list, tuple)) and len(ksize) > 1 and isinstance(ksize[1], (int, float, str)) else kh
        stride: AttrType = attrs.get("stride", [kh, kw])
        stride_h: int = int(stride[0]) if isinstance(stride, (list, tuple)) and len(stride) > 0 and isinstance(stride[0], (int, float, str)) else (int(stride) if isinstance(stride, int) else kh)
        stride_w: int = int(stride[1]) if isinstance(stride, (list, tuple)) and len(stride) > 1 and isinstance(stride[1], (int, float, str)) else stride_h
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

    def _dispatch_named_op(self, node: IRNode, graph_to_use: IRGraph | None) -> bool:
        """Dispatch known specialized C++ visitor methods by operation name.

        Args:
            node (IRNode): The IR node to visit.
            graph_to_use (IRGraph | None): The computational graph context.

        Returns:
            bool: True if the node was handled by a specialized visitor, False otherwise.
        """
        op: str = node.op_type
        op_lower: str = op.lower()
        exact_dispatch: dict[str, Callable[[IRNode, IRGraph | None], None]] = {
            "If": self._visit_if_op,
            "Cond": self._visit_if_op,
            "Loop": self._visit_loop_op,
            "WhileLoop": self._visit_loop_op,
            "Scan": self.visit_Scan,
            "Attention": self.visit_Attention,
        }
        if op in exact_dispatch:
            exact_dispatch[op](node, graph_to_use)
            return True

        if op_lower in ("globalmaxpool2d", "global_maxpool2d"):
            self.visit_MaxPool2D(node, graph_to_use, is_global=True)
            return True
        if op_lower in ("globalavgpool2d", "global_avgpool2d"):
            self.visit_AvgPool2D(node, graph_to_use, is_global=True)
            return True

        lower_dispatch: dict[str, Callable[[IRNode, IRGraph | None], None]] = {
            "conv2d": self.visit_Conv2D,
            "conv3d": self.visit_Conv3D,
            "maxpool2d": self.visit_MaxPool2D,
            "avgpool2d": self.visit_AvgPool2D,
            "groupnorm": self.visit_GroupNorm,
            "softmax": self.visit_Softmax,
            "logsoftmax": self.visit_LogSoftmax,
            "warp_shuffle_reduce": self.visit_WarpShuffleReduce,
            "shared_memory_tree_reduce": self.visit_SharedMemoryTreeReduce,
            "tiled_matmul_2d": self.visit_TiledMatMul2D,
            "conv2d_shared_halo": self.visit_Conv2DSharedHalo,
            "maxpool2d_dynamic": self.visit_MaxPool2DDynamic,
            "avgpool2d_dynamic": self.visit_AvgPool2DDynamic,
            "maxpool3d": self.visit_MaxPool3D,
            "avgpool3d": self.visit_AvgPool3D,
            "layernorm": self.visit_LayerNorm,
            "batchnorm": self.visit_BatchNorm,
            "rmsnorm": self.visit_RMSNorm,
            "slice": self.visit_Slice,
            "pad": self.visit_Pad,
            "gathernd": self.visit_GatherND,
            "gather_nd": self.visit_GatherND,
            "scatternd": self.visit_ScatterND,
            "scatter_nd": self.visit_ScatterND,
        }
        if op_lower in lower_dispatch:
            lower_dispatch[op_lower](node, graph_to_use)
            return True
        return False

    def _resolve_cpp_op_mapping(self, node: IRNode, op: str) -> dict[str, AttrType]:
        """Resolve the C++ template mapping dictionary for a generic operation.

        Args:
            node (IRNode): The IR node being compiled.
            op (str): Operation type string.

        Returns:
            dict[str, AttrType]: Resolved template mapping dictionary (empty if unmapped).
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_operation
        from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

        op_def = OPS_REGISTRY.get(op, {})
        raw_variants = op_def.get("variants")
        variants = raw_variants if isinstance(raw_variants, dict) else {}
        raw_llvm = variants.get("llvm_cpp")
        raw_mapping = raw_llvm if isinstance(raw_llvm, dict) else {}
        mapping: dict[str, AttrType] = dict(raw_mapping) if (raw_mapping and raw_mapping.get("template")) else {}

        op_key = op.lower()
        num_inputs = len(getattr(node, "inputs", []) or [])
        is_registered = op in OPS_REGISTRY or op_key in OPS_REGISTRY
        decl_op = get_cpp_operation(op_key, num_inputs=num_inputs, allow_synth=not self.strict and is_registered)
        if decl_op is None:
            return mapping

        if not mapping:
            mapping = {"template": decl_op.template, "scalar_expr": decl_op.scalar_expr}
        elif mapping.get("scalar_expr") == "in0_val":
            mapping["scalar_expr"] = decl_op.scalar_expr

        if getattr(decl_op, "init_val", None) is not None and "init_val" not in mapping:
            mapping["init_val"] = decl_op.init_val
        if getattr(decl_op, "final_combine", None) is not None and "final_combine" not in mapping:
            mapping["final_combine"] = decl_op.final_combine
        return mapping

    def _resolve_node_input_vars(self, node: IRNode, graph_to_use: IRGraph | None) -> tuple[str, str, str]:
        """Resolve input variable identifiers for template substitution, accounting for multi-output edges.

        Args:
            node (IRNode): The target IR node.
            graph_to_use (IRGraph | None): The computational graph containing edges.

        Returns:
            tuple[str, str, str]: Resolved variable names (in0_var, in1_var, in2_var).
        """
        in0_var = node.inputs[0] if len(node.inputs) > 0 else "dummy"
        in1_var = node.inputs[1] if len(node.inputs) > 1 else in0_var
        in2_var = node.inputs[2] if len(node.inputs) > 2 else in0_var
        if graph_to_use and hasattr(graph_to_use, "edges"):
            for edge in graph_to_use.edges:
                tgt = getattr(edge, "target", getattr(edge, "target_node", None))
                src = getattr(edge, "source", getattr(edge, "source_node", None))
                if tgt == node.id and getattr(edge, "source_idx", 0) > 0:
                    t_idx = getattr(edge, "target_idx", 0)
                    if t_idx == 0:
                        in0_var = str(src)
                    elif t_idx == 1:
                        in1_var = str(src)
                    else:
                        in2_var = str(src)
        return in0_var, in1_var, in2_var

    def _emit_mapped_cpp_template(
        self,
        node: IRNode,
        graph_to_use: IRGraph | None,
        mapping: dict[str, AttrType],
    ) -> None:
        """Format and emit C++ template lines for a mapped generic operation.

        Args:
            node (IRNode): The IR node to emit.
            graph_to_use (IRGraph | None): The computational graph context.
            mapping (dict[str, AttrType]): Resolved template configuration.
        """
        from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template

        template: dict[str, str | EagerValue] = get_cpp_template(str(mapping["template"]))
        in0_node: IRNode | None = graph_to_use.nodes.get(node.inputs[0]) if (graph_to_use and len(node.inputs) > 0) else None
        in0_shape: list[int] = self._get_shape(in0_node) if in0_node else [1, 1]
        out_shape: list[int] = self._get_shape(node)
        out_shape_str = "{" + ",".join(map(str, out_shape)) + "}"

        in0_var, in1_var, in2_var = self._resolve_node_input_vars(node, graph_to_use)
        expr_format_args: dict[str, AttrType] = {
            "clean_id": node.id,
            "out_shape_str": out_shape_str,
            "in0": in0_var,
            "in1": in1_var,
            "in2": in2_var,
            "rank": len(out_shape),
            "M": out_shape[0] if len(out_shape) > 0 else 1,
            "N": out_shape[1] if len(out_shape) > 1 else 1,
            "K": in0_shape[1] if len(in0_shape) > 1 else 1,
        }
        expr_format_args.update(mapping)
        if mapping.get("template") == "reduction":
            expr_format_args.setdefault("init_val", "0.0f")
            expr_format_args.setdefault("final_combine", "res")
        if "init_val" in expr_format_args:
            ival = str(expr_format_args["init_val"])
            expr_format_args["init_val"] = ival if ("INFINITY" in ival or ival.endswith("f")) else f"{ival}f"

        body: str = str(template["body"]).format(**expr_format_args)
        for line in body.split("\n"):
            if line.strip():
                self.lines.append(f"    {line}")

        if hasattr(node, "outputs") and len(node.outputs) > 1:
            for idx, out_id in enumerate(node.outputs):
                self.lines.append(f"    NDArrayView<float> {out_id} = {node.id}; // multi-output alias {idx}")

    def _visit_node(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Visit a node and emit C++ code.

        Args:
            node (IRNode): The IR node to visit.
            graph_to_use (IRGraph | None): The graph.
        """
        op: str = node.op_type

        if op == "Input":
            raw_offset = node.attributes.get("buffer_offset")
            offset: int | None = int(raw_offset) if isinstance(raw_offset, (int, float, str)) else None
            offset_str: str = f", global_arena.data() + {offset}" if offset is not None else ""
            out_shape_str: str = "{" + ",".join(map(str, self._get_shape(node))) + "}"
            self.lines.append(f"    NDArrayView<float> {node.id}({out_shape_str}{offset_str}); // Input")
            return
        if op == "Constant":
            raw_val = node.attributes.get("value", 0.0)
            val: float = float(raw_val) if isinstance(raw_val, (int, float, str)) else 0.0
            out_shape_str = "{" + ",".join(map(str, self._get_shape(node))) + "}"
            self.lines.append(f"    NDArrayView<float> {node.id}({out_shape_str});")
            self.lines.append(f"    for(size_t i=0; i<{node.id}.size(); ++i) {node.id}.data[i] = {val};")
            return
        if op == "Output":
            self.lines.append(f"    // Output {node.inputs[0]}")
            return
        if self._dispatch_named_op(node, graph_to_use):
            return

        mapping = self._resolve_cpp_op_mapping(node, op)
        if not mapping:
            self._dispatch_unmapped_op_fallback(node, op)
        else:
            self._emit_mapped_cpp_template(node, graph_to_use, mapping)

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

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: EagerValue) -> Callable[[], str]:
        """Compile IRGraph into a native shared library and return callable C++ execution wrapper.

        Args:
            graph (IRGraph): Target computational graph.
            **kwargs (EagerValue): Compiler options.

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

        def executable(*args: EagerValue) -> str:
            """Execute the compiled C++ graph with optional input/output buffer pointers.

            Args:
                *args (EagerValue): Optional (c_inputs, c_outputs, c_shapes) buffer pointers.

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
