"""Attribute narrowing, MLIR dialect validation, and base state for CppGenerator."""

from __future__ import annotations

import typing
from collections.abc import Callable, Sequence  # noqa: F401

from ml_switcheroo_ir import IRGraph, IRNode

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_int_list,
    coerce_operand_list,
)
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module

_AttrLeaf = typing.Union[int, float, str, bool, None]  # noqa: UP007
_AttrD1 = typing.Union[_AttrLeaf, Sequence[_AttrLeaf], dict[str, _AttrLeaf]]  # noqa: UP007
AttrType = typing.Union[_AttrD1, Sequence[_AttrD1], dict[str, _AttrD1]]  # noqa: UP007


def _as_attr_leaf(val: BackendArg) -> _AttrLeaf:
    """Convert or validate a primitive leaf attribute value.

    Args:
        val (BackendArg): Raw leaf attribute value.

    Returns:
        _AttrLeaf: Validated primitive scalar or None.
    """
    return val if val is None or isinstance(val, (int, float, str, bool)) else str(val)


def _as_attr_d1(val: BackendArg) -> _AttrD1:
    """Convert a depth-1 C++ generator attribute value.

    Args:
        val (BackendArg): Raw depth-1 attribute value.

    Returns:
        _AttrD1: Validated depth-1 attribute value.
    """
    if isinstance(val, tuple):
        return tuple(_as_attr_leaf(x) for x in val)
    return [_as_attr_leaf(x) for x in val] if isinstance(val, list) else _as_attr_leaf(val)


def _as_attr_val(val: BackendArg) -> AttrType:
    """Convert an `BackendArg` attribute to `AttrType`.

    Args:
        val (BackendArg): Candidate attribute value.

    Returns:
        AttrType: Validated attribute value.
    """
    if isinstance(val, (list, tuple)):
        return [_as_attr_d1(x) for x in val]
    return _as_attr_d1(val)


def _node_inputs_list(node: BackendArg) -> list[str]:
    """Return string list of input IDs on `node`.

    Args:
        node (BackendArg): Target node.

    Returns:
        list[str]: Node input IDs.
    """
    raw = adapt_backend_module(node).get_symbol("inputs", [])
    return [str(x) for x in coerce_operand_list(raw)]


def _node_attrs_dict(node: BackendArg) -> dict[str, AttrType]:
    """Return attributes dictionary on `node`.

    Args:
        node (BackendArg): Target node.

    Returns:
        dict[str, AttrType]: Attributes mapping.
    """
    raw = adapt_backend_module(node).get_symbol("attributes", {})
    return {str(k): _as_attr_val(v) for k, v in raw.items()} if isinstance(raw, dict) else {}


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
        import ml_switcheroo_compiler.backends.llvm_cpp.generator as _cpp_mod

        dialect = op_name.split(".")[0] if "." in op_name else ""
        return dialect in _cpp_mod.get_supported_mlir_dialects() and op_name in MLIR_REGISTRY
    except ImportError:
        return False


class _CppGenBase(BaseGenerator):
    """Base C++17 generator holding configuration flags, shape helpers, and unmapped fallback."""

    def __init__(
        self, graph: IRGraph | None = None, use_simd: bool = True, use_openmp: bool = True, strict: bool = False
    ) -> None:
        """Initialize the C++ generator.

        Args:
            graph (IRGraph | None): Input `graph` argument (graph).
            use_simd (bool): Input `use_simd` argument (use simd).
            use_openmp (bool): Input `use_openmp` argument (use openmp).
            strict (bool): Whether to raise UnimplementedMathError on unsupported operations.
        """
        super().__init__(graph=graph if isinstance(graph, IRGraph) else IRGraph())
        self.use_simd = use_simd
        self.use_openmp = use_openmp
        self.strict = strict
        self.lines: list[str] = []

    def _get_shape(self, node: IRNode) -> list[int]:
        """Execute get shape for compiler processing.

        Args:
            node (IRNode): Input `node` argument (node).

        Returns:
            list[int]: Computed `list[int]` output.
        """
        shape = adapt_backend_module(node).get_symbol("shape_metadata", None)
        if not shape:
            return [1]
        if isinstance(shape, (int, float)):
            return [int(shape)]
        return [int(s) for s in coerce_int_list(shape)]

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
        inputs = adapt_backend_module(node).get_symbol("inputs", []) or []
        if len(inputs) > 0:
            in0 = inputs[0]
            self.lines.append("    #pragma omp simd")
            self.lines.append(
                f"    for(size_t i = 0; i < {clean_id}.size(); ++i) {{ {clean_id}.data[i] = ({in0}.size() > "
                f"0) ? {in0}.data[i % {in0}.size()] : 0.0f; }}"
            )
        else:
            self.lines.append("    #pragma omp simd")
            self.lines.append(f"    for(size_t i = 0; i < {clean_id}.size(); ++i) {{ {clean_id}.data[i] = 0.0f; }}")

    def _num_elements(self, shape: list[int]) -> int:
        """Execute num elements for compiler processing.

        Args:
            shape (list[int]): Input `shape` argument (shape).

        Returns:
            int: Computed `int` output.
        """
        n: int = 1
        for s in shape:
            n *= s
        return n

    def _get_strides(self, shape: list[int]) -> list[int]:
        """Execute get strides for compiler processing.

        Args:
            shape (list[int]): Input `shape` argument (shape).

        Returns:
            list[int]: Computed `list[int]` output.
        """
        strides: list[int] = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return strides
