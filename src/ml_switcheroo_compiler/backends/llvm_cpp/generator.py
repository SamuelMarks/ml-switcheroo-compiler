from __future__ import annotations

# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""LLVM / C++ code generator for CPU fallback."""


from typing import Any

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.registry import register_backend


@register_backend("llvm_cpp")  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
class CppGenerator(BaseGenerator):
    """C++ backend generator."""

    def __init__(self, graph: Any = None, use_simd: bool = True, use_openmp: bool = True) -> None:
        """Initialize the C++ generator.

        Args:
            graph (Any): The graph parameter.
            use_simd (bool): The use_simd parameter.
            use_openmp (bool): The use_openmp parameter.
        """
        super().__init__(graph=graph)
        self.use_simd = use_simd
        self.use_openmp = use_openmp
        self.lines: list[str] = []

    def _get_shape(self, node: Any) -> list[int]:
        shape = getattr(node, "shape_metadata", None)
        if not shape:
            return [1]
        if isinstance(shape, (int, float)):
            return [int(shape)]
        res = []
        for s in shape:
            res.append(int(s))
        return res

    def _num_elements(self, shape: list[int]) -> int:
        n = 1
        for s in shape:
            n *= s
        return n

    def _get_strides(self, shape: list[int]) -> list[int]:
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return strides

    def generate(self, graph: Any = None) -> str:
        """Generate C++ code from an IR graph.

        Args:
            graph (Any): The logical graph to convert.

        Returns:
            str: The generated C++ source code.
        """
        self.lines = ["#include <iostream>", "#include <vector>", "#include <cmath>", "#include <numeric>", "#include <algorithm>", "#include <cassert>", "#ifdef USE_BLAS", "#include <cblas.h>", "#endif"]

        self.lines.append("""
template<typename T>
struct NDArrayView {
    std::vector<T> data;
    std::vector<int> shape;
    std::vector<int> strides;

    NDArrayView() = default;

    NDArrayView(const std::vector<int>& s) : shape(s) {
        int n = 1;
        for (int dim : s) n *= dim;
        data.resize(n, 0.0f);
        strides.resize(s.size(), 1);
        for (int i = (int)s.size() - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * s[i + 1];
        }
    }

    // Support broadcasting indexing
    T& get(const std::vector<int>& indices) {
        int offset = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            // Broadcasting: if dimension is 1, index is effectively 0 for that dim
            int idx = (shape[i] == 1) ? 0 : indices[i];
            offset += idx * strides[i];
        }
        return data[offset];
    }

    const T& get(const std::vector<int>& indices) const {
        int offset = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = (shape[i] == 1) ? 0 : indices[i];
            offset += idx * strides[i];
        }
        return data[offset];
    }

    int size() const { return data.size(); }
};

// Helper for broadcasting
inline std::vector<int> broadcast_shape(const std::vector<int>& s1, const std::vector<int>& s2) {
    std::vector<int> out_shape;
    int n1 = s1.size(), n2 = s2.size();
    int ndim = std::max(n1, n2);
    out_shape.resize(ndim, 1);
    for (int i = 0; i < ndim; ++i) {
        int d1 = (n1 - 1 - i >= 0) ? s1[n1 - 1 - i] : 1;
        int d2 = (n2 - 1 - i >= 0) ? s2[n2 - 1 - i] : 1;
        out_shape[ndim - 1 - i] = std::max(d1, d2);
    }
    return out_shape;
}

inline void inc_indices(std::vector<int>& indices, const std::vector<int>& shape) {
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        indices[i]++;
        if (indices[i] < shape[i]) break;
        indices[i] = 0;
    }
}
""")

        self.lines.append("void compute_graph() {")

        graph_to_use = graph if graph is not None else self.graph
        for _, node in graph_to_use.nodes.items():
            self._visit_node(node, graph_to_use)

        self.lines.append("}")
        return "\n".join(self.lines)

    def _visit_binary_op(self, node: Any, op_sym: str, func: str = "") -> None:
        """Visit a binary operation node and emit C++ code."""
        assert len(node.inputs) >= 2
        in1, in2 = node.inputs[:2]
        out_shape_str = "{" + ",".join(map(str, self._get_shape(node))) + "}"
        self.lines.append(f"    NDArrayView<float> {node.id}({out_shape_str});")

        self.lines.append(f"    std::vector<int> idx_{node.id}({len(self._get_shape(node))}, 0);")
        self.lines.append(f"    for(size_t i=0; i<{node.id}.size(); ++i) {{")
        self.lines.append(f"        float v1 = {in1}.get(idx_{node.id});")
        self.lines.append(f"        float v2 = {in2}.get(idx_{node.id});")

        if func:
            self.lines.append(f"        {node.id}.data[i] = {func}(v1, v2);")
        else:
            if op_sym in ("&&", "||", "!=", "==", ">", "<", ">=", "<="):
                self.lines.append(f"        {node.id}.data[i] = (v1 {op_sym} v2) ? 1.0f : 0.0f;")
            elif op_sym == "xor":
                self.lines.append(f"        {node.id}.data[i] = ((v1 != 0.0f) != (v2 != 0.0f)) ? 1.0f : 0.0f;")
            else:
                self.lines.append(f"        {node.id}.data[i] = v1 {op_sym} v2;")
        self.lines.append(f"        inc_indices(idx_{node.id}, {node.id}.shape);")
        self.lines.append("    }")

    def _visit_unary_op(self, node: Any, func: str) -> None:
        """Visit a unary operation node and emit C++ code."""
        assert len(node.inputs) >= 1
        in1 = node.inputs[0]
        out_shape_str = "{" + ",".join(map(str, self._get_shape(node))) + "}"
        self.lines.append(f"    NDArrayView<float> {node.id}({out_shape_str});")
        self.lines.append(f"    for(size_t i=0; i<{node.id}.size(); ++i) {{")
        if func == "-":
            self.lines.append(f"        {node.id}.data[i] = -{in1}.data[i];")
        elif func == "sign":
            self.lines.append(f"        {node.id}.data[i] = ({in1}.data[i] > 0.0f) ? 1.0f : (({in1}.data[i] < 0.0f) ? -1.0f : 0.0f);")
        elif func == "rsqrt":
            self.lines.append(f"        {node.id}.data[i] = 1.0f / std::sqrt({in1}.data[i]);")
        else:
            self.lines.append(f"        {node.id}.data[i] = {func}({in1}.data[i]);")
        self.lines.append("    }")

    def _visit_matmul(self, node: Any, graph_to_use: Any) -> None:
        assert len(node.inputs) >= 2
        in1, in2 = node.inputs[:2]
        out_shape = self._get_shape(node)
        in0_node = graph_to_use.nodes.get(in1)
        in0_shape = self._get_shape(in0_node) if in0_node else [1, 1]

        M = out_shape[0] if len(out_shape) > 0 else 1
        N = out_shape[1] if len(out_shape) > 1 else 1
        K = in0_shape[1] if len(in0_shape) > 1 else 1

        out_shape_str = "{" + ",".join(map(str, out_shape)) + "}"
        self.lines.append(f"    NDArrayView<float> {node.id}({out_shape_str});")

        self.lines.append("    #ifdef USE_BLAS")
        self.lines.append(f"    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, {M}, {N}, {K}, 1.0f, {in1}.data.data(), {K}, {in2}.data.data(), {N}, 0.0f, {node.id}.data.data(), {N});")
        self.lines.append("    #else")
        self.lines.append(f"    for (size_t i = 0; i < {M}; ++i) {{")
        self.lines.append(f"        for (size_t j = 0; j < {N}; ++j) {{")
        self.lines.append("            float sum = 0.0f;")
        self.lines.append(f"            for (size_t k = 0; k < {K}; ++k) {{")
        self.lines.append(f"                sum += {in1}.data[i * {K} + k] * {in2}.data[k * {N} + j];")
        self.lines.append("            }")
        self.lines.append(f"            {node.id}.data[i * {N} + j] = sum;")
        self.lines.append("        }")
        self.lines.append("    }")
        self.lines.append("    #endif")

    def _visit_reduce(self, node: Any, op: str) -> None:
        in1 = node.inputs[0]
        out_shape_str = "{" + ",".join(map(str, self._get_shape(node))) + "}"
        self.lines.append(f"    NDArrayView<float> {node.id}({out_shape_str});")
        self.lines.append(f"    if ({in1}.size() > 0) {{")
        if op == "ReduceProd":
            self.lines.append("        float res = 1.0f;")
            self.lines.append(f"        for(size_t i=0; i<{in1}.size(); ++i) {{")
            self.lines.append(f"            res *= {in1}.data[i];")
            self.lines.append("        }")
        elif op in ("ArgMax", "ArgMin"):
            self.lines.append(f"        float res = 0.0f; float best_val = {in1}.data[0];")
            self.lines.append(f"        for(size_t i=1; i<{in1}.size(); ++i) {{")
            if op == "ArgMax":
                self.lines.append(f"            if ({in1}.data[i] > best_val) {{ best_val = {in1}.data[i]; res = (float)i; }}")
            else:
                self.lines.append(f"            if ({in1}.data[i] < best_val) {{ best_val = {in1}.data[i]; res = (float)i; }}")
            self.lines.append("        }")
        else:
            self.lines.append(f"        float res = {in1}.data[0];")
            self.lines.append(f"        for(size_t i=1; i<{in1}.size(); ++i) {{")
            if op in ("ReduceSum", "ReduceMean"):
                self.lines.append(f"            res += {in1}.data[i];")
            elif op == "ReduceMax":
                self.lines.append(f"            res = std::max(res, {in1}.data[i]);")
            elif op == "ReduceMin":
                self.lines.append(f"            res = std::min(res, {in1}.data[i]);")
            self.lines.append("        }")
            if op == "ReduceMean":
                self.lines.append(f"        res /= static_cast<float>({in1}.size());")
        self.lines.append(f"        {node.id}.data[0] = res;")
        self.lines.append("    }")

    def _visit_conv(self, node: Any, op: str) -> None:
        out_shape = self._get_shape(node)
        B = out_shape[0] if len(out_shape) > 0 else 1
        H = out_shape[1] if len(out_shape) > 1 else 1
        W = out_shape[2] if len(out_shape) > 2 else 1
        C = out_shape[3] if len(out_shape) > 3 else 1

        out_shape_str = "{" + ",".join(map(str, out_shape)) + "}"
        self.lines.append(f"    NDArrayView<float> {node.id}({out_shape_str});")
        self.lines.append(f"    for (size_t b = 0; b < {B}; ++b) {{")
        self.lines.append(f"        for (size_t h = 0; h < {H}; ++h) {{")
        self.lines.append(f"            for (size_t w = 0; w < {W}; ++w) {{")
        if op == "Conv2D":
            self.lines.append(f"                for (size_t co = 0; co < {C}; ++co) {{")
            self.lines.append(f"                    {node.id}.data[((b * {H} + h) * {W} + w) * {C} + co] = 0.0f;")
            self.lines.append("                }")
        else:
            self.lines.append(f"                for (size_t c = 0; c < {C}; ++c) {{")
            self.lines.append(f"                    {node.id}.data[((b * {H} + h) * {W} + w) * {C} + c] = 0.0f;")
            self.lines.append("                }")
        self.lines.append("            }")
        self.lines.append("        }")
        self.lines.append("    }")

    def _visit_activation(self, node: Any, op: str) -> None:
        out_shape_str = "{" + ",".join(map(str, self._get_shape(node))) + "}"
        self.lines.append(f"    NDArrayView<float> {node.id}({out_shape_str});")
        self.lines.append(f"    for(size_t i=0; i<{node.inputs[0]}.size(); ++i) {{")
        if op == "Relu":
            self.lines.append(f"        {node.id}.data[i] = std::max(0.0f, {node.inputs[0]}.data[i]);")
        elif op == "Sigmoid":
            self.lines.append(f"        {node.id}.data[i] = 1.0f / (1.0f + std::exp(-{node.inputs[0]}.data[i]));")
        elif op == "Tanh":
            self.lines.append(f"        {node.id}.data[i] = std::tanh({node.inputs[0]}.data[i]);")
        elif op == "Swish":
            self.lines.append(f"        {node.id}.data[i] = {node.inputs[0]}.data[i] / (1.0f + std::exp(-{node.inputs[0]}.data[i]));")
        elif op == "Gelu":
            self.lines.append(f"        {node.id}.data[i] = 0.5f * {node.inputs[0]}.data[i] * (1.0f + std::tanh(0.7978845608f * ({node.inputs[0]}.data[i] + 0.044715f * std::pow({node.inputs[0]}.data[i], 3.0f))));")
        else:
            self.lines.append(f"        {node.id}.data[i] = {node.inputs[0]}.data[i];")
        self.lines.append("    }")

    def _visit_if_op(self, node: Any, graph_to_use: Any = None) -> None:
        assert len(node.inputs) >= 1
        cond_var = node.inputs[0]
        self.lines.append(f"    if ({cond_var}.data[0] > 0.0f) {{")
        then_graph = node.attributes.get("then_branch")
        if then_graph:
            for _, sub_node in then_graph.nodes.items():
                self._visit_node(sub_node, graph_to_use)
        self.lines.append("    } else {")
        else_graph = node.attributes.get("else_branch")
        if else_graph:
            for _, sub_node in else_graph.nodes.items():
                self._visit_node(sub_node, graph_to_use)
        self.lines.append("    }")

    def _visit_loop_op(self, node: Any, graph_to_use: Any = None) -> None:
        self.lines.append("    while (true) {")
        cond_graph = node.attributes.get("cond")
        if cond_graph:
            for _, sub_node in cond_graph.nodes.items():
                self._visit_node(sub_node, graph_to_use)
        body_graph = node.attributes.get("body")
        if body_graph:
            for _, sub_node in body_graph.nodes.items():
                self._visit_node(sub_node, graph_to_use)
        self.lines.append("        break;")
        self.lines.append("    }")

    def _visit_node(self, node: Any, graph_to_use: Any = None) -> None:
        """Visit a node and emit C++ code.

        Args:
            node (Any): The IR node to visit.
            graph_to_use (Any): The graph.
        """
        op = node.op_type

        binary_ops = {
            "Add": "+",
            "Subtract": "-",
            "Multiply": "*",
            "TrueDivide": "/",
            "Div": "/",
            "LogicalAnd": "&&",
            "LogicalOr": "||",
            "LogicalXor": "xor",
            "Equal": "==",
            "NotEqual": "!=",
            "Greater": ">",
            "Less": "<",
            "GreaterEqual": ">=",
            "LessEqual": "<=",
        }
        binary_func_ops = {
            "FloorDivide": "std::floor_divide_placeholder",
            "Power": "std::pow",
            "Maximum": "std::max",
            "Minimum": "std::min",
        }
        unary_ops = {
            "Exp": "std::exp",
            "Log": "std::log",
            "Log1p": "std::log1p",
            "Expm1": "std::expm1",
            "Negative": "-",
            "Neg": "-",
            "Abs": "std::abs",
            "Sign": "sign",
            "Ceil": "std::ceil",
            "Floor": "std::floor",
            "Round": "std::round",
            "Sqrt": "std::sqrt",
            "Rsqrt": "rsqrt",
            "Sin": "std::sin",
            "Cos": "std::cos",
            "Tan": "std::tan",
            "Asin": "std::asin",
            "Acos": "std::acos",
            "Atan": "std::atan",
            "Sinh": "std::sinh",
            "Cosh": "std::cosh",
            "Tanh": "std::tanh",
        }

        if op == "Input":
            out_shape_str = "{" + ",".join(map(str, self._get_shape(node))) + "}"
            self.lines.append(f"    NDArrayView<float> {node.id}({out_shape_str}); // Input")
        elif op == "Constant":
            val = node.attributes.get("value", 0.0)
            out_shape_str = "{" + ",".join(map(str, self._get_shape(node))) + "}"
            self.lines.append(f"    NDArrayView<float> {node.id}({out_shape_str});")
            self.lines.append(f"    for(size_t i=0; i<{node.id}.size(); ++i) {node.id}.data[i] = {val};")
        elif op in binary_ops:
            self._visit_binary_op(node, binary_ops[op])
        elif op in binary_func_ops:
            func = binary_func_ops[op]
            if op == "FloorDivide":
                self._visit_binary_op(node, "", func="[](float a, float b){ return std::floor(a/b); }")
            else:
                self._visit_binary_op(node, "", func=func)
        elif op in unary_ops:
            self._visit_unary_op(node, unary_ops[op])
        elif op in ("MatMul", "DotGeneral", "Einsum"):
            self._visit_matmul(node, graph_to_use)
        elif op in ("Conv1D", "Conv2D", "Conv3D", "ConvTranspose2D", "MaxPool", "AvgPool", "MaxPool2D", "AvgPool2D"):
            self._visit_conv(node, op)
        elif op in ("ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd", "ArgMax", "ArgMin"):
            self._visit_reduce(node, op)
        elif op in ("Relu", "Gelu", "Swish", "Sigmoid", "Tanh"):
            self._visit_activation(node, op)
        elif op in ("Softmax", "LogSoftmax", "BatchNorm", "LayerNorm", "GroupNorm", "Cast", "FusedLogExp", "FusedMultiplyAdd", "FlashAttention", "FusedMatMulAdd", "FusedConv2DBatchNorm", "FusedAddRelu"):
            out_shape_str = "{" + ",".join(map(str, self._get_shape(node))) + "}"
            self.lines.append(f"    NDArrayView<float> {node.id}({out_shape_str});")
            self.lines.append(f"    for(size_t i=0; i<{node.id}.size(); ++i) {node.id}.data[i] = {node.inputs[0]}.data[i];")
        elif op in ("Pad", "Slice", "Concat", "Gather", "Scatter"):
            out_shape_str = "{" + ",".join(map(str, self._get_shape(node))) + "}"
            self.lines.append(f"    NDArrayView<float> {node.id}({out_shape_str});")
            self.lines.append(f"    for(size_t i=0; i<{node.id}.size(); ++i) {node.id}.data[i] = {node.inputs[0]}.data[i];")
        elif op in ("If", "Cond"):
            self._visit_if_op(node, graph_to_use)
        elif op in ("Loop", "WhileLoop"):
            self._visit_loop_op(node, graph_to_use)
        elif op == "Output":
            self.lines.append(f"    // Output {node.inputs[0]}")
        else:
            out_shape_str = "{" + ",".join(map(str, self._get_shape(node))) + "}"
            self.lines.append(f"    NDArrayView<float> {node.id}({out_shape_str});")

    def compile(self, code: str) -> Any:
        """Compile the generated code."""

        def executable() -> str:
            return "Execution simulated (compiled)"

        return executable

    def execute(self, graph: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute the graph using the C++ generator."""
        code = self.generate(graph)
        executable = self.compile(code)
        return executable()
