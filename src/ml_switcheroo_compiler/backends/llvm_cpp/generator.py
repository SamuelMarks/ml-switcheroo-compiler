# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""LLVM / C++ code generator for CPU fallback."""

from __future__ import annotations

import ctypes
import os
import subprocess
import tempfile
from typing import Callable, Union

AttrType = Union[int, float, str, bool, list, tuple, dict, None]

import yaml

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


@register_backend("llvm_cpp")
class CppGenerator(BaseGenerator):
    """C++ backend generator."""

    def __init__(self, graph: IRGraph | None = None, use_simd: bool = True, use_openmp: bool = True) -> None:
        """Initialize the C++ generator.

        Args:
            graph (IRGraph | None): The graph parameter.
            use_simd (bool): The use_simd parameter.
            use_openmp (bool): The use_openmp parameter.
        """
        super().__init__(graph=graph)
        self.use_simd = use_simd
        self.use_openmp = use_openmp
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

        self.lines.append('extern "C" void compute_graph() {')

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
            self.lines.append(f"    std::vector<uint8_t> global_arena({max_offset}, 0);")

        for _, node in graph_to_use.nodes.items():
            self._visit_node(node, graph_to_use)

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

    def _visit_node(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Visit a node and emit C++ code.

        Args:
            node (IRNode): The IR node to visit.
            graph_to_use (IRGraph | None): The graph.
        """
        op: str = node.op_type

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
        elif op == "Conv2D":
            self.visit_Conv2D(node, graph_to_use)
        elif op == "Scan":
            self.visit_Scan(node, graph_to_use)
        elif op == "Attention":
            self.visit_Attention(node, graph_to_use)
        elif op == "Output":
            self.lines.append(f"    // Output {node.inputs[0]}")
        else:
            from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import get_cpp_template
            from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

            op_def: dict[str, AttrType] = OPS_REGISTRY.get(op, {})
            mapping: dict[str, AttrType] = op_def.get("variants", {}).get("llvm_cpp", {})

            if not mapping:
                out_shape_str = "{" + ",".join(map(str, self._get_shape(node))) + "}"
                self.lines.append(f"    NDArrayView<float> {node.id}({out_shape_str}); // Fallback Unimplemented {op}")
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

                expr_format_args: dict[str, AttrType] = {
                    "clean_id": node.id,
                    "out_shape_str": out_shape_str,
                    "in0": node.inputs[0] if len(node.inputs) > 0 else "dummy",
                    "in1": node.inputs[1] if len(node.inputs) > 1 else "dummy",
                    "rank": len(out_shape),
                    "M": M,
                    "N": N,
                    "K": K,
                }
                expr_format_args.update(mapping)

                body: str = template["body"].format(**expr_format_args)
                for line in body.split("\n"):
                    if line.strip():
                        self.lines.append(f"    {line}")

    def compile(self, code: str) -> Callable[[], str]:
        """Compile the generated C++ code into a shared library.

        Args:
            code (str): The C++ source code.

        Returns:
            Callable[[], str]: An executable function that wraps the compiled library.
        """
        # Create a temporary directory for compilation
        temp_dir: str = tempfile.mkdtemp()
        src_file: str = os.path.join(temp_dir, "graph.cpp")
        lib_ext: str = ".dylib" if os.name == "posix" and "darwin" in os.uname().sysname.lower() else ".so"
        lib_file: str = os.path.join(temp_dir, f"graph{lib_ext}")

        with open(src_file, "w") as f:
            f.write(code)

        # Compile using clang++ (or g++)
        compile_cmd: list[str] = ["clang++", "-O3", "-shared", "-fPIC", src_file, "-o", lib_file]

        try:
            subprocess.run(compile_cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            err_out = e.stderr.decode() if e.stderr else ""
            raise RuntimeError(f"Compilation failed: {err_out}") from e
        try:
            lib: ctypes.CDLL = ctypes.CDLL(lib_file)
            # Find the compute_graph function
            if hasattr(lib, "compute_graph"):
                compute_func = lib.compute_graph
            else:
                raise RuntimeError("Function 'compute_graph' not found in compiled library. Ensure it is exported with extern \"C\".")

        except Exception as e:
            raise RuntimeError(f"Compilation or load failed: {e}") from e

        def executable() -> str:
            """Executable function.

            Returns:
                str: Result.
            """
            compute_func()
            return "Execution successful"

        return executable

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
