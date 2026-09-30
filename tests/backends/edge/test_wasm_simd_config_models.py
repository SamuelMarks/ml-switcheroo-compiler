"""Tests for test_wasm_simd_config_models."""

from __future__ import annotations

from ml_switcheroo_ir import LogicalGraph, LogicalNode

import ml_switcheroo_compiler.backends.edge.wasm_simd.config_models as wasm_simd_cfg
import ml_switcheroo_compiler.backends.generator_mixins as gen_mixins


class _MockDummyGenerator(gen_mixins.GeneratorLifecycleMixin):
    """Mock generator class using GeneratorLifecycleMixin."""

    def __init__(self, graph: LogicalGraph) -> None:
        """Initialize mock generator.

        Args:
            graph (LogicalGraph): Logical graph to simulate.
        """
        self.graph = graph
        self.header = "# Header comment"
        self.code: list[str] = []
        self.indent_level: int = 0
        self.sorted_nodes: list[LogicalNode] = list(graph.nodes.values())
        self.input_idx: int = 0
        self._output_returns: list[str] = []
        self.var_names: dict[str, str] = {"in1": "args[0]"}

    def assign_var_name(self, node_id: str) -> str:
        """Assign variable name.

        Args:
            node_id (str): Node id.

        Returns:
            str: Assigned variable name.
        """
        name = f"var_{node_id}"
        self.var_names[node_id] = name
        return name

    def visit(self, node: LogicalNode, input_vars: list[str]) -> str:
        """Generate code for node.

        Args:
            node (LogicalNode): The node.
            input_vars (list[str]): Input variables.

        Returns:
            str: Code string.
        """
        return f"Identity({', '.join(input_vars)})"

    def _emit_body_return(self, returns: list[str]) -> None:
        """Emit return statement for body.

        Args:
            returns (list[str]): Return variables.
        """
        self.add_line(f"return {', '.join(returns) if returns else 'None'}")

    def add_line(self, line: str) -> None:
        """Add line to code buffer.

        Args:
            line (str): Line string.
        """
        self.code.append("    " * self.indent_level + line)


class _MockCustomModule:
    """Mock backend module exposing zeros, array, asarray, and item."""

    def zeros(self, shape: tuple[int, ...]) -> tuple[str, tuple[int, ...]]:
        """Mock zeros.

        Args:
            shape (tuple[int, ...]): Target shape.

        Returns:
            tuple[str, tuple[int, ...]]: Descriptor.
        """
        return ("custom_zeros", shape)

    def array(self, data: object, dtype: object | None = None) -> tuple[str, object, object | None]:
        """Mock array.

        Args:
            data (object): Input data.
            dtype (object | None): Data type.

        Returns:
            tuple[str, object, object | None]: Descriptor.
        """
        return ("custom_array", data, dtype)

    def asarray(self, data: object) -> tuple[str, object]:
        """Mock asarray.

        Args:
            data (object): Input data.

        Returns:
            tuple[str, object]: Descriptor.
        """
        return ("custom_asarray", data)

    def item(self, data: object) -> float:
        """Mock item.

        Args:
            data (object): Input object.

        Returns:
            float: 99.9.
        """
        return 99.9


class _MockEagerBackendWithModule(gen_mixins.EagerExecutionMixin):
    """Eager backend class providing custom module."""

    @classmethod
    def get_module(cls: type) -> _MockCustomModule:
        """Return custom backend module.

        Returns:
            _MockCustomModule: Mock module.
        """
        return _MockCustomModule()


class _MockEagerBackendDefault(gen_mixins.EagerExecutionMixin):
    """Eager backend class using default numpy module."""

    pass


class _MockScalarHolder:
    """Mock scalar container providing item()."""

    def item(self) -> float:
        """Return scalar value.

        Returns:
            float: Scalar value.
        """
        return 33.3


def test_wasm_simd_config_models() -> None:
    """Verify all Pydantic models in backends/edge/wasm_simd/config_models.py."""
    mem_align = wasm_simd_cfg.MemoryAlignmentConfig(
        alignment_bytes=32,
        requires_aligned_load=False,
        peeling_loop_required=False,
    )
    assert mem_align.alignment_bytes == 32
    assert mem_align.requires_aligned_load is False
    assert mem_align.peeling_loop_required is False

    lane_cfg = wasm_simd_cfg.LaneArithmeticConfig(
        lanes=8,
        element_type="f32",
        simd_intrinsic="wasm_v128_load",
        lane_mask="0x0f",
    )
    assert lane_cfg.lanes == 8
    assert lane_cfg.element_type == "f32"
    assert lane_cfg.simd_intrinsic == "wasm_v128_load"
    assert lane_cfg.lane_mask == "0x0f"

    tmpl = wasm_simd_cfg.WasmTemplateConfig(
        simd_unroll_factor=4,
        body="v128_t acc = ...",
        peel_loop="while (n--) ...",
        global_code="static void helper() {}",
        alignment=mem_align,
        lane_arithmetic=lane_cfg,
    )
    assert tmpl.simd_unroll_factor == 4
    assert tmpl.body == "v128_t acc = ..."
    assert tmpl.peel_loop == "while (n--) ..."
    assert tmpl.global_code == "static void helper() {}"
    assert tmpl.alignment == mem_align
    assert tmpl.lane_arithmetic == lane_cfg

    templates_cfg = wasm_simd_cfg.WasmTemplatesConfig(
        templates={"add_kernel": tmpl},
        js_orchestration={"init": "WebAssembly.instantiate()"},
        cpp_helpers=["helper.h"],
    )
    assert "add_kernel" in templates_cfg.templates
    assert "init" in templates_cfg.js_orchestration
    assert templates_cfg.cpp_helpers == ["helper.h"]

    intrinsic = wasm_simd_cfg.WasmIntrinsicConfig(
        macro_name="WASM_ADD_F32",
        simd_expr="wasm_f32x4_add(a, b)",
        scalar_fallback="a + b",
        alignment=mem_align,
        lane_arithmetic=lane_cfg,
    )
    assert intrinsic.macro_name == "WASM_ADD_F32"
    assert intrinsic.simd_expr == "wasm_f32x4_add(a, b)"
    assert intrinsic.scalar_fallback == "a + b"

    intrinsics_cfg = wasm_simd_cfg.WasmIntrinsicsConfig(
        intrinsics={"Add": intrinsic},
        scalars={"Add": "add_scalar"},
    )
    assert "Add" in intrinsics_cfg.intrinsics
    assert intrinsics_cfg.scalars == {"Add": "add_scalar"}
