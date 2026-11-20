"""ISO C++17 code generator and native compilation runner for CPU fallback.

This backend compiles computational graphs into ISO C++17 compute kernels using
system C++ compilers (clang++ or g++) with OpenMP SIMD vectorization and aligned
memory allocations into dynamically loaded shared libraries via ctypes.
"""

from __future__ import annotations

import ctypes as ctypes
import os as os
import subprocess as subprocess
import tempfile as tempfile
from collections.abc import Callable, Sequence  # noqa: F401

import yaml
from ml_switcheroo_ir import IRGraph, IRNode

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
)
from ml_switcheroo_compiler.backends.llvm_cpp._cpp_gen_base import (
    AttrType as AttrType,
)
from ml_switcheroo_compiler.backends.llvm_cpp._cpp_gen_base import (
    _as_attr_d1 as _as_attr_d1,
)
from ml_switcheroo_compiler.backends.llvm_cpp._cpp_gen_base import (
    _as_attr_leaf as _as_attr_leaf,
)
from ml_switcheroo_compiler.backends.llvm_cpp._cpp_gen_base import (
    _as_attr_val as _as_attr_val,
)
from ml_switcheroo_compiler.backends.llvm_cpp._cpp_gen_base import (
    _AttrD1 as _AttrD1,
)
from ml_switcheroo_compiler.backends.llvm_cpp._cpp_gen_base import (
    _AttrLeaf as _AttrLeaf,
)
from ml_switcheroo_compiler.backends.llvm_cpp._cpp_gen_base import (
    _CppGenBase as _CppGenBase,
)
from ml_switcheroo_compiler.backends.llvm_cpp._cpp_gen_base import (
    _node_attrs_dict as _node_attrs_dict,
)
from ml_switcheroo_compiler.backends.llvm_cpp._cpp_gen_base import (
    _node_inputs_list as _node_inputs_list,
)
from ml_switcheroo_compiler.backends.llvm_cpp._cpp_gen_base import (
    get_supported_mlir_dialects as get_supported_mlir_dialects,
)
from ml_switcheroo_compiler.backends.llvm_cpp._cpp_gen_base import (
    validate_mlir_operation as validate_mlir_operation,
)
from ml_switcheroo_compiler.backends.llvm_cpp._cpp_gen_dispatch import _CppGenDispatch as _CppGenDispatch
from ml_switcheroo_compiler.backends.llvm_cpp._cpp_runner import LLVMCPPRunner as LLVMCPPRunner
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module
from ml_switcheroo_compiler.backends.registry import register_backend
from ml_switcheroo_compiler.ir.core import as_opt_graph


@register_backend("llvm_cpp")
class CppGenerator(_CppGenDispatch):
    """High-performance C++17 backend generator compiling kernels via system compilers (clang++/g++)."""

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

        self.lines.append(
            'extern "C" void compute_graph(const float** inputs = nullptr, float** outputs = nullptr, '
            "const int64_t* shapes = nullptr) {"
        )

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
                self.lines.append(
                    f"        std::memcpy({node.id}.data.data(), inputs[{inp_idx}], {node.id}.size() * sizeof(float));"
                )
                self.lines.append("    }")
                inp_idx += 1
            else:
                self._visit_node(node, graph_to_use)

        out_idx = 0
        raw_outs = adapt_backend_module(graph_to_use).get_symbol("outputs", [])
        for out_name_raw in raw_outs if isinstance(raw_outs, (list, tuple)) else []:
            out_name = str(out_name_raw)
            if out_name in graph_to_use.nodes:
                clean_out = graph_to_use.nodes[out_name].id
                self.lines.append(f"    if (outputs && outputs[{out_idx}]) {{")
                self.lines.append(
                    f"        std::memcpy(outputs[{out_idx}], {clean_out}.data.data(), {clean_out}.size() * "
                    "sizeof(float));"
                )
                self.lines.append("    }")
                out_idx += 1

        self.lines.append("}")
        return "\n".join(self.lines)

    def _visit_if_op(self, node: IRNode, graph_to_use: IRGraph | None = None) -> None:
        """Execute visit if op for compiler processing.

        Args:
            node (IRNode): Input `node` argument (node).
            graph_to_use (IRGraph | None): Input `graph_to_use` argument (graph to use).
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
        body = str(template["body"]).format(
            cond_var=cond_var,
            then_body="\n".join(then_lines) + "\n" if then_lines else "",
            else_body="\n".join(else_lines) + "\n" if else_lines else "",
        )
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

        attrs: dict[str, AttrType] = _node_attrs_dict(node)
        self._emit_loop_annotations(attrs)

        raw_attrs_e = adapt_backend_module(node).get_symbol("attributes", {})
        raw_cond: BackendArg = raw_attrs_e.get("cond") if isinstance(raw_attrs_e, dict) else None
        cond_graph: IRGraph | None = raw_cond if isinstance(raw_cond, IRGraph) else None
        cond_lines = self._collect_subgraph_lines(cond_graph, graph_to_use)

        raw_body: BackendArg = raw_attrs_e.get("body") if isinstance(raw_attrs_e, dict) else None
        body_graph: IRGraph | None = raw_body if isinstance(raw_body, IRGraph) else None
        body_lines = self._collect_subgraph_lines(body_graph, graph_to_use)

        template = get_cpp_template("loop_op")
        body = str(template["body"]).format(
            cond_body="\n".join(cond_lines) + "\n" if cond_lines else "",
            loop_body="\n".join(body_lines) + "\n" if body_lines else "",
        )
        self.lines.extend(body.strip().split("\n"))

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

    def _compile_aot_impl(self, graph: IRGraph, **kwargs: BackendArg) -> Callable[[], str]:
        """Compile IRGraph into a native shared library and return callable C++ execution wrapper.

        Args:
            graph (IRGraph): Target computational graph.
            **kwargs (BackendArg): Compiler options.

        Returns:
            Callable[[], str]: Executable wrapper.
        """
        code = self.generate(graph)
        comp = str(kwargs["compiler"]) if "compiler" in kwargs and kwargs["compiler"] else None
        runner = LLVMCPPRunner(compiler=comp)
        return runner.compile_and_load(code, entry_point="compute_graph")


LLVMCPPGenerator = CppGenerator
