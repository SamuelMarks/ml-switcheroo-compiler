"""Tests for test_generator_mixins_lifecycle."""

from __future__ import annotations

from ml_switcheroo_ir import LogicalGraph, LogicalNode

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


def test_generator_mixins_lifecycle_and_eager() -> None:
    """Verify GeneratorLifecycleMixin and EagerExecutionMixin in backends/generator_mixins.py."""
    # Graph & generator lifecycle
    graph = LogicalGraph(name="test_graph", outputs=["out"])
    node = LogicalNode(id="n1", op_type="Identity", inputs=["in1"], outputs=["out"])
    graph.nodes["n1"] = node

    gen = _MockDummyGenerator(graph)
    assert gen._generate_file_header() == ["# Header comment"]
    assert gen._resolve_imports() == []
    assert gen._generate_return_block() is None

    code_str = gen.generate()
    assert "# Header comment" in code_str
    assert "def apply_model" in code_str
    assert "Identity" in code_str

    # Eager execution mixin
    assert _MockEagerBackendDefault.execute_op("Add", 1, 2) is None

    # Eager zeros, array, asarray, item with custom module
    assert _MockEagerBackendWithModule.zeros((2, 3)) == ("custom_zeros", (2, 3))
    assert _MockEagerBackendWithModule.array([1, 2], dtype="float32") == ("custom_array", [1, 2], "float32")
    assert _MockEagerBackendWithModule.asarray([3, 4]) == ("custom_asarray", [3, 4])
    assert _MockEagerBackendWithModule.item(_MockScalarHolder()) == 33.3

    # Eager zeros, array, asarray, item with default numpy fallback
    z_def = _MockEagerBackendDefault.zeros((2, 2))
    assert z_def.shape == (2, 2)

    arr_def = _MockEagerBackendDefault.array([5, 6])
    assert arr_def.tolist() == [5, 6]

    as_arr_def = _MockEagerBackendDefault.asarray([7.5, 8.5])
    assert as_arr_def.tolist() == [7.5, 8.5]

    item_val = _MockEagerBackendDefault.item(_MockScalarHolder())
    assert item_val == 33.3
