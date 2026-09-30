"""Tests for test_formats_config_models."""

from __future__ import annotations

from ml_switcheroo_ir import LogicalGraph, LogicalNode

import ml_switcheroo_compiler.backends.generator_mixins as gen_mixins
import ml_switcheroo_compiler.serialization.formats.config_models as ser_cfg


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


def test_serialization_formats_config_models() -> None:
    """Verify Pydantic models in serialization/formats/config_models.py."""
    spec = ser_cfg.FormatSpecConfig(dtype_map={"F32": "float32", "I32": "int32"})
    assert spec.dtype_map == {"F32": "float32", "I32": "int32"}

    schema = ser_cfg.SerializationSchemaConfig(safetensors=spec)
    assert schema.safetensors == spec
