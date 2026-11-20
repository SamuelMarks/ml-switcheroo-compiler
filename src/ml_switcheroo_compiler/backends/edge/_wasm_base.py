"""Attribute types, memory alignment, striding, and shape helpers for WasmCodeGenerator."""

from __future__ import annotations

import os
import typing
from collections.abc import Sequence

import yaml
from ml_switcheroo_ir import IRGraph, IRNode

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_int,
    coerce_int_list,
    coerce_operand_list,
    coerce_str,
)
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module

_WasmLeaf = typing.Union[IRNode, IRGraph, str, int, float, bool, None]  # noqa: UP007
_WasmD1 = typing.Union[_WasmLeaf, dict[str, _WasmLeaf], list[_WasmLeaf], tuple[_WasmLeaf, ...]]  # noqa: UP007
WasmAttrType = typing.Union[_WasmD1, dict[str, _WasmD1], list[_WasmD1], tuple[_WasmD1, ...]]  # noqa: UP007


def _node_attrs(node: BackendArg) -> dict[str, WasmAttrType]:
    """Return node attributes dictionary.

    Args:
        node (BackendArg): Target IR node.

    Returns:
        dict[str, WasmAttrType]: Node attributes mapping.
    """
    raw = adapt_backend_module(node).get_symbol("attributes", {})
    return {str(k): _as_wasm_attr_val(v) for k, v in raw.items()} if isinstance(raw, dict) else {}


def _as_wasm_leaf(val: BackendArg) -> _WasmLeaf:
    """Convert or validate a scalar or IR leaf attribute value.

    Args:
        val (BackendArg): Raw leaf attribute value.

    Returns:
        _WasmLeaf: Validated scalar, IR object, or None.
    """
    return val if val is None or isinstance(val, (IRNode, IRGraph, str, int, float, bool)) else str(val)


def _as_wasm_d1(val: BackendArg) -> _WasmD1:
    """Convert a depth-1 WASM attribute value without unsound TypeGuards.

    Args:
        val (BackendArg): Raw depth-1 attribute value.

    Returns:
        _WasmD1: Validated depth-1 attribute value.
    """
    return [_as_wasm_leaf(x) for x in val] if isinstance(val, (list, tuple)) else _as_wasm_leaf(val)


def _as_wasm_attr_val(val: BackendArg) -> WasmAttrType:
    """Convert an `BackendArg` attribute to `WasmAttrType`.

    Args:
        val (BackendArg): Candidate attribute value.

    Returns:
        WasmAttrType: Validated attribute value.
    """
    if isinstance(val, (list, tuple)):
        return [_as_wasm_d1(x) for x in val]
    if isinstance(val, dict):
        return {str(k): _as_wasm_d1(v) for k, v in val.items()}
    return _as_wasm_d1(val)


def _str_list_attr(obj: BackendArg, name: str) -> list[str]:
    """Extract a list of strings from an attribute on `obj`.

    Args:
        obj (BackendArg): Target node or graph.
        name (str): Attribute name.

    Returns:
        list[str]: Extracted string list.
    """
    raw = adapt_backend_module(obj).get_symbol(name, [])
    return [str(x) for x in coerce_operand_list(raw)]


class _WasmGenBase(BaseGenerator):
    """Base WASM-SIMD C++ generator layer holding memory/shape helpers and fallback."""

    def __init__(self, graph: IRGraph, delegates: WasmAttrType = None) -> None:
        """Initialize WasmCodeGenerator.

        Args:
            graph (IRGraph): The IR graph to process.
            delegates (WasmAttrType, optional): Visitor delegates.
        """
        from ml_switcheroo_compiler.transforms.passes.strip_offline_nodes import (
            OFFLINE_DIAGNOSTIC_OPS,
            strip_offline_diagnostic_nodes_pass,
        )

        sanitized_graph = strip_offline_diagnostic_nodes_pass(graph)
        super().__init__(sanitized_graph, delegates)
        self.sorted_nodes = [
            n
            for n in self.sorted_nodes
            if coerce_str(adapt_backend_module(n).get_symbol("op_type", "")) not in OFFLINE_DIAGNOSTIC_OPS
        ]
        self.var_map: dict[str, str] = {}
        self.is_simd: bool = False

    def _allocate_aligned_memory(self, size_bytes: int, alignment: int = 16) -> str:
        """Generate C++ aligned memory allocation statement.

        Args:
            size_bytes (int): Number of bytes to allocate.
            alignment (int): Byte alignment boundary (default 16 for SIMD).

        Returns:
            str: C++ `std::aligned_alloc` invocation string.
        """
        return f"std::aligned_alloc({alignment}, {size_bytes});"

    def _generate_striding_logic(self, shape: list[int]) -> tuple[list[int], str]:
        """Compute contiguous row-major strides and linear index C++ expression for `shape`.

        Args:
            shape (list[int]): Tensor dimension sizes.

        Returns:
            tuple[list[int], str]: Tuple of `(strides_list, c_index_expression)`.
        """
        if not shape:
            return [], "0"
        strides: list[int] = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]

        c_code: str = " + ".join(
            [f"((idx / {s}) % {d}) * {s}" if s > 1 else f"(idx % {d}) * {s}" for d, s in zip(shape, strides)]
        )
        return strides, c_code

    def _map_type(self, dtype: str) -> str:
        """Map an IR data type string to its C++ primitive scalar type.

        Args:
            dtype (str): IR data type string.

        Returns:
            str: Corresponding C++ scalar type name.
        """
        return {
            "float32": "float",
            "float64": "double",
            "int32": "int",
            "bool": "bool",
        }.get(str(dtype).lower(), "float")

    def _num_elements(self, shape: Sequence[int] | int | float) -> int:
        """Compute the total number of scalar elements for `shape`.

        Args:
            shape (Sequence[int] | int | float): Tensor dimension sizes.

        Returns:
            int: Product of dimensions.
        """
        if isinstance(shape, (int, float)):
            return 1
        n: int = 1
        for s in shape:
            n *= s
        return n

    def get_helper_functions(self) -> list[str]:
        """Return C++ helper functions/macros for missing WASM math.

        Returns:
            list[str]: Lines of C++ macros/functions.
        """
        helpers: list[str] = ["// --- Scalar Fallback Helpers ---"]

        from ml_switcheroo_compiler.backends.edge.wasm_simd.config_models import WasmIntrinsicsConfig

        yaml_path: str = os.path.join(os.path.dirname(__file__), "wasm_simd", "intrinsics.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                data = WasmIntrinsicsConfig(**yaml.safe_load(f)).model_dump()

            # Load scalar helpers
            scalars: dict[str, str] = data.get("scalars") or {}
            for name, body in scalars.items():
                helpers.append(f"inline float _scalar_{name.lower()}(float a, float b) {{ {body} }}")

            helpers.append("// --- SIMD Fast Math Approximations from YAML ---")

            intrinsics: dict[str, dict[str, str]] = data.get("intrinsics", {})
            for _op, intrinsic_data in intrinsics.items():
                if intrinsic_data.get("macro_name") and intrinsic_data.get("simd_expr"):
                    is_binary: bool = "b" in intrinsic_data["simd_expr"]
                    arg_sig: str = "v128_t a, v128_t b" if is_binary else "v128_t x"
                    helpers.append(f"inline v128_t {intrinsic_data['macro_name']}({arg_sig}) {{")
                    for line in intrinsic_data["simd_expr"].split("\n"):
                        if line.strip():
                            helpers.append(f"    {line}")
                    helpers.append("}")

        from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_cpp_helpers

        helpers.extend(get_cpp_helpers())
        return helpers

    def generic_visit(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> str:
        """Process a node and return its generated C++ variable name.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (BackendArg): Additional attributes.

        Returns:
            str: Variable name of the evaluated node.
        """
        del input_vars, kwargs
        return coerce_str(adapt_backend_module(node).get_symbol("id", ""))

    @staticmethod
    def _normalize_ndim_shape(raw_shape: BackendArg | list[int], target_ndim: int) -> list[int]:
        """Normalize a raw shape metadata value into a fixed-rank integer list.

        Args:
            raw_shape (BackendArg | list[int]): Raw shape metadata (list, tuple, scalar, or None).
            target_ndim (int): Desired number of dimensions (e.g. 4 for 2D conv, 5 for 3D conv).

        Returns:
            list[int]: Normalized shape list of length at least target_ndim.
        """
        if not raw_shape:
            res = [1] * target_ndim
        elif isinstance(raw_shape, (int, float)):
            res = [1] * (target_ndim - 1) + [int(raw_shape)]
        else:
            res = [int(x) for x in coerce_int_list(raw_shape)]
        if len(res) < target_ndim:
            res = [1] * (target_ndim - len(res)) + res
        return res

    @staticmethod
    def _parse_conv3d_spatial_triplet(val: WasmAttrType, default_val: int) -> tuple[int, int, int]:
        """Parse a 3D spatial parameter (stride or dilation) into (d, h, w) integers.

        Args:
            val (WasmAttrType): Attribute value (int, float, tuple, or list).
            default_val (int): Default value when sequence is empty or non-numeric.

        Returns:
            tuple[int, int, int]: Parsed (depth, height, width) triplet.
        """
        if isinstance(val, (tuple, list)):
            if len(val) == 3:
                return int(coerce_int(val[0])), int(coerce_int(val[1])), int(coerce_int(val[2]))
            if len(val) == 2:
                return default_val, int(coerce_int(val[0])), int(coerce_int(val[1]))
            s_val = int(coerce_int(val[0])) if val else default_val
            return s_val, s_val, s_val
        scalar = int(val) if isinstance(val, (int, float)) else default_val
        return scalar, scalar, scalar

    @classmethod
    def _parse_conv3d_padding(
        cls,
        pad: WasmAttrType,
        out_shape: list[int],
        in_shape: list[int],
        w_shape: list[int],
        strides: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        """Parse 3D convolution padding attribute including 'SAME' mode.

        Args:
            pad (WasmAttrType): Raw padding attribute value.
            out_shape (list[int]): Normalized 5D output shape.
            in_shape (list[int]): Normalized 5D input shape.
            w_shape (list[int]): Normalized 5D weight shape.
            strides (tuple[int, int, int]): Parsed (stride_d, stride_h, stride_w).

        Returns:
            tuple[int, int, int]: Computed (pad_d, pad_h, pad_w) triplet.
        """
        if isinstance(pad, str) and pad.upper() == "SAME":
            sd, sh, sw = strides
            pad_d = max(0, (out_shape[2] - 1) * sd + w_shape[2] - in_shape[2]) // 2
            pad_h = max(0, (out_shape[3] - 1) * sh + w_shape[3] - in_shape[3]) // 2
            pad_w = max(0, (out_shape[4] - 1) * sw + w_shape[4] - in_shape[4]) // 2
            return pad_d, pad_h, pad_w
        return cls._parse_conv3d_spatial_triplet(pad, 0)

    def _resolve_node_input_shapes(self, node: IRNode, target_ndim: int) -> tuple[list[int], list[int]]:
        """Resolve and normalize the first two input shapes of a node.

        Args:
            node (IRNode): The IR node whose inputs are inspected.
            target_ndim (int): Target rank for normalization.

        Returns:
            tuple[list[int], list[int]]: Normalized (in0_shape, in1_shape) lists.
        """
        inputs_list: list[str] = _str_list_attr(node, "inputs")
        input_nodes = [
            next((n for n in self.sorted_nodes if adapt_backend_module(n).get_symbol("id", None) == inp), None)
            for inp in inputs_list
        ]
        raw_in0: BackendArg = (
            adapt_backend_module(input_nodes[0]).get_symbol("shape_metadata", None)
            if len(input_nodes) > 0 and input_nodes[0]
            else None
        )
        raw_in1: BackendArg = (
            adapt_backend_module(input_nodes[1]).get_symbol("shape_metadata", None)
            if len(input_nodes) > 1 and input_nodes[1]
            else None
        )
        return self._normalize_ndim_shape(raw_in0, target_ndim), self._normalize_ndim_shape(raw_in1, target_ndim)
