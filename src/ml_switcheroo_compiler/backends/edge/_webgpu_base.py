"""Node attribute helpers, WGSL type mapping, bindings, and coordinate offset generation."""

from __future__ import annotations

import typing  # noqa: F401

from ml_switcheroo_ir import IRGraph, IRNode

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_float,
    coerce_int,
    coerce_int_list,
    coerce_str,
)
from ml_switcheroo_compiler.backends.edge.wgsl_ast import WGSLEmitter
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module
from ml_switcheroo_compiler.backends.visitor import CodeGeneratorVisitor


def _node_attrs_dict(node: BackendArg) -> dict[str, BackendArg]:
    """Return attributes dictionary on `node`.

    Args:
        node (BackendArg): Target node.

    Returns:
        dict[str, BackendArg]: Attributes mapping.
    """
    raw = adapt_backend_module(node).get_symbol("attributes", {})
    return {str(k): v for k, v in raw.items()} if isinstance(raw, dict) else {}


def _node_inputs_list(node: BackendArg) -> list[str]:
    """Return string list of input IDs on `node`.

    Args:
        node (BackendArg): Target node.

    Returns:
        list[str]: Node input IDs.
    """
    raw = adapt_backend_module(node).get_symbol("inputs", [])
    return [str(x) for x in raw] if isinstance(raw, (list, tuple)) else []


def _as_wgsl_visit_result(val: BackendArg) -> tuple[list[str], str, str, str]:
    """Narrow visitor return value to `(list[str], str, str, str)`.

    Args:
        val (BackendArg): Raw visitor return value.

    Returns:
        tuple[list[str], str, str, str]: Typed WGSL lines and dispatch dimensions.
    """
    assert isinstance(val, tuple) and len(val) == 4
    raw_lines = val[0]
    assert isinstance(raw_lines, list)
    return [str(x) for x in raw_lines], str(val[1]), str(val[2]), str(val[3])


class _WebGPUGenBase(BaseGenerator):
    """Base WebGPU WGSL generator layer managing bindings, strides, and offset math."""

    def __init__(self, graph: IRGraph, delegates: list[CodeGeneratorVisitor] | None = None) -> None:
        """Initialize WebGPUCodeGenerator.

        Args:
            graph (IRGraph): The IR graph to process.
            delegates (list[CodeGeneratorVisitor] | None): Visitor delegates.
        """
        super().__init__(graph, delegates)
        self.var_map: dict[str, str] = {}
        self.body_lines: list[str] = []
        self.emitter = WGSLEmitter()

    def _map_type(self, dtype: str) -> str:
        """Map data type to WGSL primitive.

        Args:
            dtype (str): The data type.

        Returns:
            str: WGSL primitive type representation.
        """
        lowered: str = str(dtype).lower()
        if lowered == "float64":
            import warnings

            warnings.warn(
                "WebGPU WGSL lacks native float64 support; using emulated double-precision vec2<f32> representation.",
                UserWarning,
                stacklevel=2,
            )
            return "vec2<f32>"
        return {
            "float32": "f32",
            "int32": "i32",
            "bool": "bool",
        }.get(lowered, "f32")

    def _get_binding_for_input(self, input_idx: int) -> int:
        """Get the WebGPU binding index for an input tensor.

        Args:
            input_idx (int): Zero-based index of the input.

        Returns:
            int: Binding index in bind group 0.
        """
        if input_idx < 3:
            return input_idx
        return input_idx + 1

    def _get_binding_for_output(self, output_idx: int) -> int:
        """Get the WebGPU binding index for an output tensor.

        Args:
            output_idx (int): Zero-based index of the output.

        Returns:
            int: Binding index in bind group 0.
        """
        if output_idx == 0:
            return 3
        return 5 + output_idx

    def _generate_bind_group_declarations(self, max_inputs: int, max_outputs: int = 1) -> str:
        """Generate dynamic WGSL bind group declarations for storage buffers.

        Args:
            max_inputs (int): Maximum input count needed across all graph nodes.
            max_outputs (int): Maximum output count needed across all graph nodes.

        Returns:
            str: Dynamic WGSL declarations for storage buffer bind group 0.
        """
        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import generate_dynamic_bindgroup_declarations

        return generate_dynamic_bindgroup_declarations(num_inputs=max_inputs, num_outputs=max_outputs)

    def _get_shape_and_strides(self, node: IRNode | None) -> tuple[list[int], list[int]]:
        """Get the shape and contiguous strides of an IR node.

        Args:
            node (IRNode | None): The IR node to analyze.

        Returns:
            Tuple[List[int], List[int]]: The shape as a list of dimensions and the corresponding strides.
        """
        shape_meta = adapt_backend_module(node).get_symbol("shape_metadata", None)
        if shape_meta is None:
            return [], []

        if isinstance(shape_meta, (int, float)):
            shape: list[int] = [int(shape_meta)]
        else:
            shape = [int(s) for s in coerce_int_list(shape_meta)]

        if not shape:
            return [], []

        # Check if custom memory strides are explicitly specified in attributes
        custom_strides = _node_attrs_dict(node).get("strides") if node else None
        if (
            custom_strides is not None
            and isinstance(custom_strides, (list, tuple))
            and len(custom_strides) == len(shape)
        ):
            return shape, [int(coerce_int(st)) for st in custom_strides]

        strides: list[int] = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return shape, strides

    def _num_elements(self, shape: list[int]) -> int:
        """Execute num elements for compiler processing.

        Args:
        self (OpValue): Input `self` argument (self).
        shape (list[int]): Input `shape` argument (shape).

        Returns:
        int: Computed `int` output.
        """
        n: int = 1
        for s in shape:
            n *= s
        return n

    def generic_visit(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Process a node and return its generated WGSL variable name.

        Args:
            node (IRNode): The IR node.
            input_vars (List[str]): Names of the input variables.
            **kwargs (BackendArg): Additional attributes.

        Returns:
            str: Variable name of the evaluated node.
        """
        return coerce_str(adapt_backend_module(node).get_symbol("id", ""))

    def _gen_offset_computation(
        self, idx_var: str, shape: list[int], strides: list[int], out_var: str, base_offset_words: int = 0
    ) -> list[str]:
        """Generate N-dimensional coordinate-to-offset resolution.

        Args:
            idx_var: Input `idx_var` argument (idx var).
            shape: Input `shape` argument (shape).
            strides: Input `strides` argument (strides).
            out_var: Input `out_var` argument (out var).
            base_offset_words: Input `base_offset_words` argument (base offset words).

        Returns:
            list[str]: Computed result.
        """
        if not shape:
            return [f"let {out_var}: u32 = {base_offset_words}u;"]

        nodes: list[str] = []
        nodes.append(f"var {out_var}_offset: u32 = 0u;")
        nodes.append(f"var {out_var}_remaining: u32 = {idx_var};")
        for i in range(len(shape) - 1, -1, -1):
            nodes.append(f"let {out_var}_d{i}: u32 = {out_var}_remaining % {shape[i]}u;")
            nodes.append(f"{out_var}_remaining = {out_var}_remaining / {shape[i]}u;")
            nodes.append(f"{out_var}_offset = {out_var}_offset + {out_var}_d{i} * {strides[i]}u;")
        nodes.append(f"let {out_var}: u32 = {out_var}_offset + {base_offset_words}u;")
        return nodes

    def _resolve_wgsl_op_name(self, op_type: str) -> str:
        """Resolve an IR operation type name to a canonical WGSL op name.

        Args:
            op_type (str): Raw operation type from the IR node.

        Returns:
            str: Canonical operation name.
        """
        from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

        if op_type in OPS_REGISTRY:
            return op_type
        prefix_map: dict[str, str] = {
            "add": "Add",
            "sub": "Sub",
            "mul": "Mul",
            "div": "Div",
            "m": "MatMul",
            "matmul": "MatMul",
            "max": "MaxPool2D",
            "maxpool": "MaxPool2D",
            "maxpool2d": "MaxPool2D",
            "avg": "AvgPool2D",
            "avgpool": "AvgPool2D",
            "avgpool2d": "AvgPool2D",
            "conv": "Conv2D",
            "conv2d": "Conv2D",
            "in": "Input",
            "input": "Input",
            "out": "Output",
            "output": "Output",
            "linear": "Linear",
        }
        clean_prefix: str = op_type.split("_")[0].lower() if "_" in op_type else op_type.lower()
        return prefix_map.get(clean_prefix, op_type)

    def _resolve_wgsl_mapping(self, node: IRNode, op_type: str, resolved_op: str) -> dict[str, str]:
        """Resolve the WGSL template mapping dictionary for an operation node.

        Args:
            node (IRNode): The IR node being lowered.
            op_type (str): Original IR operation name.
            resolved_op (str): Canonical resolved operation name.

        Returns:
            dict[str, str]: WGSL template mapping configuration.

        Raises:
            CompilationError: If the operation cannot be mapped to a valid WGSL kernel.
        """
        from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import get_wgsl_op_mapping
        from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

        raw_decl = get_wgsl_op_mapping(resolved_op) or get_wgsl_op_mapping(op_type) or {}
        decl_mapping: dict[str, str] = {str(k): str(v) for k, v in raw_decl.items()}
        if decl_mapping:
            mapping: dict[str, str] = decl_mapping
            if resolved_op == "Constant":
                const_val: float = float(coerce_float(_node_attrs_dict(node).get("value", 1.0)))
                mapping["expr"] = mapping.get("expr", "{const_val}f").replace("{const_val}", str(const_val))
        else:
            op_def = OPS_REGISTRY.get(resolved_op, OPS_REGISTRY.get(op_type, {}))
            mapping = dict(op_def.get("variants", {}).get("edge_wgsl", {}))

        if not mapping:
            from ml_switcheroo_compiler.core.errors import CompilationError

            raise CompilationError(f"Operation '{op_type}' cannot be mapped to a valid WGSL compute kernel.")
        return mapping

    def _emit_node_offset_code(
        self,
        target_node: IRNode | None,
        shape: list[int],
        strides: list[int],
        out_var: str,
    ) -> str:
        """Emit WGSL coordinate-to-offset code for a given tensor node.

        Args:
            target_node (IRNode | None): Node whose buffer offset and shape are used.
            shape (list[int]): Tensor shape dimensions.
            strides (list[int]): Tensor memory strides.
            out_var (str): Output WGSL offset variable name.

        Returns:
            str: Emitted WGSL statements for computing the linear word offset.
        """
        offset_words: int = (
            int(coerce_int(_node_attrs_dict(target_node).get("buffer_offset", 0))) // 4 if target_node else 0
        )
        stmts: list[str] = self._gen_offset_computation("idx", shape, strides, out_var, offset_words)
        return "\n".join([self.emitter.emit(n) for n in stmts])
