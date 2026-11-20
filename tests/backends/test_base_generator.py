"""Tests for test_base_generator."""

from __future__ import annotations

import typing

import pytest
from ml_switcheroo_ir import IRGraph, IRNode, LogicalGraph

from ml_switcheroo_compiler.backends.base_generator import (
    BaseGenerator,
    ClassBasedGenerator,
    CodeGeneratorVisitor,
    EmitUtilsMixin,
    FormatterProxyMixin,
    PythonStringGenerator,
)
from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from tests.testing_utils import (
    EagerBoundCallable,
    coerce_str,
    require_callable,
)

if typing.TYPE_CHECKING:
    from unittest.mock import MagicMock


class DummyFormatter:
    """Test class DummyFormatter."""

    def __init__(self) -> None:
        """Test function __init__."""
        self.var_names: dict[str, BackendArg] = {}
        self.code: list[str] = []
        self.header = ""
        self.indent_level = 0

    def get_indent(self) -> str:
        """Test function get_indent.

        Returns:
            str: Computed output value from `get_indent`.
        """
        return "  " * self.indent_level

    def add_line(self, line: str) -> None:
        """Test function add_line.

        Args:
            line: Test parameter `line`.
        """
        self.code.append(self.get_indent() + line)

    def assign_var_name(self, node_id: str, prefix: str = "tensor") -> str:
        """Test function assign_var_name.

        Args:
            node_id: Test parameter `node_id`.
            prefix: Test parameter `prefix`.

        Returns:
            str: Computed output value from `assign_var_name`.
        """
        return f"{prefix}_foo"


class DummyGenerator(BaseGenerator):
    """Test class DummyGenerator."""

    def get_fallback_prefix(self) -> str:
        """Test function get_fallback_prefix.

        Returns:
            str: Computed output value from `get_fallback_prefix`.
        """
        return "dummy"


def test_formatter_proxy_mixin() -> None:
    """Test function test_formatter_proxy_mixin."""
    proxy = FormatterProxyMixin()
    pytest.MonkeyPatch().setattr(proxy, "formatter", DummyFormatter(), raising=False)
    proxy.var_names = {"a": "b"}
    assert proxy.var_names == {"a": "b"}
    proxy.header = "head"
    assert proxy.header == "head"
    assert proxy.code == []
    assert proxy.get_indent() == ""
    proxy.add_line("test")
    assert proxy.code == ["test"]
    assert proxy.assign_var_name("1") == "tensor_foo"


class DummyEmitUtils(EmitUtilsMixin):
    """Test class DummyEmitUtils."""

    def __init__(self) -> None:
        """Test function __init__."""
        self.code: list[str] = []

    def add_line(self, line: str) -> None:
        """Test function add_line.

        Args:
            line (str): Test parameter `line`.
        """
        self.code.append(line)


def test_emit_utils_mixin() -> None:
    """Test function test_emit_utils_mixin."""
    eu = DummyEmitUtils()
    eu._emit_constant_assignment("c", "42")
    assert eu.code == ["c = 42"]

    eu.code = []
    eu._emit_body_return(["a"])
    assert eu.code == ["return a"]

    eu.code = []
    eu._emit_body_return(["a", "b"])
    assert eu.code == ["return (a, b)"]

    eu.code = []
    eu._emit_body_return([])
    assert eu.code == ["return None"]

    graph = IRGraph()
    pytest.MonkeyPatch().setattr(eu, "graph", graph, raising=False)
    pytest.MonkeyPatch().setattr(eu, "formatter", DummyFormatter(), raising=False)
    pytest.MonkeyPatch().setattr(graph, "outputs", ["out1"], raising=False)
    eu.code = []
    eu._emit_body_return([])
    assert eu.code == ["return out1"]

    pytest.MonkeyPatch().setattr(graph, "outputs", ["out1", "out2"], raising=False)
    eu.code = []
    eu._emit_body_return([])
    assert eu.code == ["return (out1, out2)"]


def test_base_generator_methods() -> None:
    """Test function test_base_generator_methods."""
    graph = IRGraph()
    gen = DummyGenerator(graph)
    assert gen.get_language() == "python"
    node = IRNode("id", "op", attributes={"value": [1, 2, 3]})
    assert gen.emit_constant(node) == "[1, 2, 3]"

    assert gen.get_fallback_prefix() == "dummy"
    assert gen.get_fallback_axis_kwarg() == "axis"
    assert gen.get_fallback_keepdims_kwarg() == "keepdims"

    # kwargs mapping substitution
    # op_type in ops_map
    def get_ops_map(kwargs: BackendArg) -> BackendArg:
        """Test function get_ops_map.

        Args:
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `get_ops_map`.
        """
        return {"TestOp": "test_prefix.testop({0}, {1}, kw={kw})"}

    pytest.MonkeyPatch().setattr(gen, "get_ops_map", get_ops_map, raising=False)
    res = gen.generic_visit(IRNode("id", "TestOp"), ["a", "b"], kw=42)
    assert res == "test_prefix.testop(a, b, kw=42)"

    # kwargs replacement where kwargs left over
    def get_ops_map2(kwargs: BackendArg) -> BackendArg:
        """Test function get_ops_map2.

        Args:
            kwargs: Test parameter `kwargs`.

        Returns:
            BackendArg: Computed output value from `get_ops_map2`.
        """
        return {"TestOp2": "test_prefix.testop({0}, {1}, kw={kw})"}

    pytest.MonkeyPatch().setattr(gen, "get_ops_map", get_ops_map2, raising=False)
    res2 = gen.generic_visit(IRNode("id", "TestOp2"), ["a", "b"], kw=42, unrelated="{not_fmt}")
    # unrelated should be cleaned up by regex
    assert "test_prefix.testop" in coerce_str(res2)

    gen.code = []
    gen._emit_input_assignment("v1", IRNode("i", "Input"), "args", 0)
    assert gen.code[0] == "v1 = args[0]"

    gen._emit_output_assignment(IRNode("id", "op"), [], "ret1")
    assert gen._output_returns == ["ret1"]
    gen._emit_output_assignment(IRNode("id2", "op2"), [], "ret2")
    assert gen._output_returns == ["ret1", "ret2"]


class DummyPyGen(PythonStringGenerator):
    """Test class DummyPyGen."""

    _import_header = "import dummy"

    def _generate_body(self, input_prefix: str = "args") -> None:
        """Test function _generate_body.

        Args:
            input_prefix: Test parameter `input_prefix`.
        """
        pass


def test_python_string_generator() -> None:
    """Test function test_python_string_generator."""
    gen = DummyPyGen(IRGraph())
    code = gen.generate()
    assert "import dummy" in code

    gen2 = DummyPyGen(IRGraph())
    pytest.MonkeyPatch().setattr(gen2, "_import_header", ["import a", "import b"], raising=False)
    assert "import a\nimport b" in gen2.generate()

    gen3 = DummyPyGen(IRGraph())
    pytest.MonkeyPatch().setattr(gen3, "_import_header", 42, raising=False)  # something else to test branch
    gen3.generate()


class DummyClassGen(ClassBasedGenerator):
    """Test class DummyClassGen."""

    def _generate_body(self, input_prefix: str = "args") -> None:
        """Test function _generate_body.

        Args:
            input_prefix: Test parameter `input_prefix`.
        """
        pass


def test_class_based_generator() -> None:
    """Test function test_class_based_generator."""
    gen = DummyClassGen(IRGraph())
    assert gen._get_prefix_code() == []
    assert gen._emit_init_body() is False


class _TestCodeGeneratorVisitorMissingLinesState(typing.TypedDict, total=False):
    """Typed shared state for `_test_code_generator_visitor_missing_lines` split helpers."""

    BaseGenerator: type[BaseGenerator]
    CodeGeneratorVisitor: type[CodeGeneratorVisitor]
    IRGraph: type[IRGraph]
    IRNode: type[IRNode]
    MockGenerator: type[BaseGenerator]
    gen: BaseGenerator
    node: IRNode
    visitor: CodeGeneratorVisitor


def _test_code_generator_visitor_missing_lines_part_1(_state: _TestCodeGeneratorVisitorMissingLinesState) -> None:
    """Execute part 1 of test_code_generator_visitor_missing_lines.

    Args:
        _state (_TestCodeGeneratorVisitorMissingLinesState): Shared test state dictionary.
    """
    from ml_switcheroo_ir import IRGraph as _IRGraphCls
    from ml_switcheroo_ir import IRNode as _IRNodeCls

    from ml_switcheroo_compiler.backends.base_generator import CodeGeneratorVisitor as _VisitorCls

    _state.update({"CodeGeneratorVisitor": _VisitorCls, "IRGraph": _IRGraphCls, "IRNode": _IRNodeCls})


def _test_code_generator_visitor_missing_lines_part_2(_state: _TestCodeGeneratorVisitorMissingLinesState) -> None:
    """Execute part 2 of test_code_generator_visitor_missing_lines.

    Args:
        _state (_TestCodeGeneratorVisitorMissingLinesState): Shared test state dictionary.
    """
    IRGraphCls, IRNodeCls = _state["IRGraph"], _state["IRNode"]

    class MockGenerator(BaseGenerator):
        """Test class MockGenerator."""

        def __init__(self, graph: IRGraph | None = None) -> None:
            """Test function __init__.

            Args:
                graph (IRGraph | None): Optional computation graph.
            """
            from tests.testing_utils import MockGraphStub

            super().__init__(MockGraphStub() if graph is not None else MockGraphStub())
            in_node = IRNodeCls("in1", "Input", "in1")
            out_node = IRNodeCls("out1", "Output", "out1")
            pytest.MonkeyPatch().setattr(self.graph, "inputs", [in_node], raising=False)
            pytest.MonkeyPatch().setattr(self.graph, "outputs", [out_node], raising=False)
            self.sorted_nodes = [in_node, out_node]
            self.var_names = {}

        def assign_var_name(self, node_id: str, prefix: str = "tensor") -> str:
            """Test function assign_var_name.

            Args:
                node_id (str): Test parameter `node_id`.
                prefix (str): Test parameter `prefix`.

            Returns:
                str: Computed output value from `assign_var_name`.
            """
            return "var_" + node_id

        def generic_visit(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> str:
            """Test function generic_visit.

            Args:
                node (IRNode): Test parameter `node`.
                input_vars (list[str]): Test parameter `input_vars`.
                **kwargs (BackendArg): Test parameter `kwargs`.

            Returns:
                str: Computed output value from `generic_visit`.
            """
            return f"generic({node.op_type})"

        def _emit_input_assignment(self, var_name: str, node: IRNode, input_prefix: str, input_idx: int) -> None:
            """Test function _emit_input_assignment.

            Args:
                var_name (str): Test parameter `var_name`.
                node (IRNode): Test parameter `node`.
                input_prefix (str): Test parameter `input_prefix`.
                input_idx (int): Test parameter `input_idx`.
            """
            self.code.append(f"{var_name} = {input_prefix}[{input_idx}]")

        def _emit_output_assignment(self, node: IRNode, input_vars: list[str], returns: str) -> None:
            """Test function _emit_output_assignment.

            Args:
                node (IRNode): Test parameter `node`.
                input_vars (list[str]): Test parameter `input_vars`.
                returns (str): Test parameter `returns`.
            """
            self.code.append(f"ret {returns}")

        def get_ops_map(self, kwargs: dict[str, BackendArg] | None = None) -> dict[str, str]:
            """Test function get_ops_map.

            Args:
                kwargs (dict[str, BackendArg] | None): Test parameter `kwargs`.

            Returns:
                dict[str, str]: Computed output value from `get_ops_map`.
            """
            return {}

        def _emit_body_return(self, returns: list[str]) -> None:
            """Test function _emit_body_return.

            Args:
                returns (list[str]): Test parameter `returns`.
            """
            pass

        def add_line(self, line: str) -> None:
            """Test function add_line.

            Args:
                line (str): Test parameter `line`.
            """
            pass

        def _emit_constant_assignment(self, var_name: str, val_repr: str) -> None:
            """Test function _emit_constant_assignment.

            Args:
                var_name (str): Test parameter `var_name`.
                val_repr (str): Test parameter `val_repr`.
            """
            pass

    _state.update({"MockGenerator": MockGenerator})
    _state.update({"IRGraph": IRGraphCls, "IRNode": IRNodeCls})


def _test_code_generator_visitor_missing_lines_part_3(_state: _TestCodeGeneratorVisitorMissingLinesState) -> None:
    """Execute part 3 of test_code_generator_visitor_missing_lines.

    Args:
        _state (_TestCodeGeneratorVisitorMissingLinesState): Shared test state dictionary.
    """
    VisitorCls, IRGraphCls, IRNodeCls, MockGenerator = (
        _state["CodeGeneratorVisitor"],
        _state["IRGraph"],
        _state["IRNode"],
        _state["MockGenerator"],
    )
    gen = MockGenerator(IRGraphCls())
    visitor = VisitorCls(gen)
    visitor.generate_body("args")
    assert "var_in1 = args[0]" in gen.code

    # Test visit method routing
    node = IRNodeCls("node1", "TestOp")

    class MockSpecificVisitor:
        """Test class MockSpecificVisitor."""

        def visit_TestOp(self, node: BackendArg, inputs: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function visit_TestOp.

            Args:
                node (BackendArg): Test parameter `node`.
                inputs (BackendArg): Test parameter `inputs`.
                **kwargs (BackendArg): Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `visit_TestOp`.
            """
            return "visited_TestOp"

    pytest.MonkeyPatch().setattr(gen, "visitors", [MockSpecificVisitor()], raising=False)

    # Test visit method routing
    node = IRNodeCls("node1", "TestOp")

    class MockSpecificVisitor2:
        """Test class MockSpecificVisitor2."""

        def visit_TestOp(self, node: BackendArg, inputs: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function visit_TestOp.

            Args:
                node (BackendArg): Test parameter `node`.
                inputs (BackendArg): Test parameter `inputs`.
                **kwargs (BackendArg): Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `visit_TestOp`.
            """
            return "visited_TestOp"

    pytest.MonkeyPatch().setattr(gen, "visitors", [MockSpecificVisitor2()], raising=False)
    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator as _BaseGenCls

    res = _BaseGenCls.visit(gen, node, [])
    assert res == "visited_TestOp"
    _state.update({"BaseGenerator": _BaseGenCls, "gen": gen, "node": node})
    _state.update({"visitor": visitor})
    _state.update({"CodeGeneratorVisitor": VisitorCls, "IRNode": IRNodeCls, "MockGenerator": MockGenerator})


def _test_code_generator_visitor_missing_lines_part_4(_state: _TestCodeGeneratorVisitorMissingLinesState) -> None:
    """Execute part 4 of test_code_generator_visitor_missing_lines.

    Args:
        _state (_TestCodeGeneratorVisitorMissingLinesState): Shared test state dictionary.
    """
    BaseGenCls, IRNodeCls, gen, node = _state["BaseGenerator"], _state["IRNode"], _state["gen"], _state["node"]
    visitor = _state["visitor"]

    class MockSpecificVisitor2:
        """Test class MockSpecificVisitor2."""

        def visit_OtherOp(self, node: BackendArg, inputs: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function visit_OtherOp.

            Args:
                node (BackendArg): Test parameter `node`.
                inputs (BackendArg): Test parameter `inputs`.
                **kwargs (BackendArg): Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `visit_OtherOp`.
            """
            return "visited_OtherOp"

    pytest.MonkeyPatch().setattr(gen, "visitors", [MockSpecificVisitor2()], raising=False)
    res2 = BaseGenCls.visit(gen, node, [])
    assert "generic" in res2

    # Test fallback

    node2 = IRNodeCls("node2", "UnknownOp")
    assert isinstance(visitor.generator, BaseGenCls)
    res2 = visitor.generator.generic_visit(node2, [])


def test_code_generator_visitor_missing_lines() -> None:
    """Test function test_code_generator_visitor_missing_lines."""
    _state: _TestCodeGeneratorVisitorMissingLinesState = {}
    _test_code_generator_visitor_missing_lines_part_1(_state)
    _test_code_generator_visitor_missing_lines_part_2(_state)
    _test_code_generator_visitor_missing_lines_part_3(_state)
    _test_code_generator_visitor_missing_lines_part_4(_state)
    assert len(_state) > 0


def test_base_generator_maps() -> None:
    """Test function test_base_generator_maps."""
    from ml_switcheroo_ir import IRGraph

    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator

    class MinimalGen(BaseGenerator):
        """Test class MinimalGen."""

        def get_fallback_prefix(self) -> str:
            """Test function get_fallback_prefix.

            Returns:
                str: Computed output value from `get_fallback_prefix`.
            """
            return "m"

        def get_ops_map(self, *args: BackendArg, **kwargs: BackendArg) -> dict[str, str]:
            """Test function get_ops_map.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                dict[str, str]: Computed output value from `get_ops_map`.
            """
            ops = super().get_ops_map(kwargs)
            ops["Add"] = "m.add"
            return ops

    gen = MinimalGen(IRGraph())
    ops = gen.get_ops_map({})
    assert ops["Add"] == "m.add"


def test_class_based_generator_full() -> None:
    """Test function test_class_based_generator_full."""
    from ml_switcheroo_ir import IRGraph

    from ml_switcheroo_compiler.backends.base_generator import ClassBasedGenerator

    class MyClassGen(ClassBasedGenerator):
        """Test class MyClassGen."""

        _base_class_name = "BaseClass"
        _forward_method_name = "forward"

        def _generate_body(self, input_prefix: str = "args") -> None:
            """Test function _generate_body.

            Args:
                input_prefix: Test parameter `input_prefix`.
            """
            self.add_line("self.generated = True")

        def _emit_init_body(self) -> bool:
            """Test function _emit_init_body.

            Returns:
                bool: Computed output value from `_emit_init_body`.
            """
            self.add_line("self.init = True")
            return True

        def get_fallback_prefix(self) -> str:
            """Test function get_fallback_prefix.

            Returns:
                str: Computed output value from `get_fallback_prefix`.
            """
            return "my"

        def get_language(self) -> str:
            """Test function get_language.

            Returns:
                str: Computed output value from `get_language`.
            """
            return "python"

    gen = MyClassGen(IRGraph())
    code = gen.generate()
    assert "class CompiledModel(BaseClass):" in code
    assert "super().__init__()" in code
    assert "self.init = True" in code
    assert "def forward(self, *args, **kwargs):" in code
    assert "self.generated = True" in code

    class MyClassGenNoBase(ClassBasedGenerator):
        """Test class MyClassGenNoBase."""

        _base_class_name = coerce_str(None)

        def _generate_body(self, input_prefix: str = "args") -> None:
            """Test function _generate_body.

            Args:
                input_prefix: Test parameter `input_prefix`.
            """
            pass

        def _emit_init_body(self) -> bool:
            """Test function _emit_init_body.

            Returns:
                bool: Computed output value from `_emit_init_body`.
            """
            return False

        def get_fallback_prefix(self) -> str:
            """Test function get_fallback_prefix.

            Returns:
                str: Computed output value from `get_fallback_prefix`.
            """
            return "my"

        def get_language(self) -> str:
            """Test function get_language.

            Returns:
                str: Computed output value from `get_language`.
            """
            return "python"

    gen2 = MyClassGenNoBase(IRGraph())
    code2 = gen2.generate()
    assert "class CompiledModel:" in code2
    assert "pass" in code2


def test_ir_graph_walker() -> None:
    """Test function test_ir_graph_walker."""
    from ml_switcheroo_ir import IRGraph

    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator

    class TrackedGenerator(BaseGenerator):
        """Generator subclass tracking walk invocation."""

        called: str | None = None

    class DummyVisitor:
        """Test class DummyVisitor."""

        def __init__(self, g: TrackedGenerator) -> None:
            """Test function __init__.

            Args:
                g (TrackedGenerator): Test parameter `g`.
            """
            self.g = g

        def generate_body(self, p: str = "args") -> None:
            """Test function generate_body.

            Args:
                p (str): Test parameter `p`.
            """
            self.g.called = p

    gen = TrackedGenerator(IRGraph())
    import sys

    try:
        pytest.MonkeyPatch().setattr(
            sys.modules["ml_switcheroo_compiler.backends.base_generator"],
            "CodeGeneratorVisitor",
            DummyVisitor,
            raising=False,
        )
        import ml_switcheroo_compiler.backends.base_generator as bg

        pytest.MonkeyPatch().setattr(bg, "CodeGeneratorVisitor", DummyVisitor, raising=False)
        walker = bg.IRGraphWalker(gen)
        walker.walk("args")
        assert gen.called == "args"
        assert walker.generator is gen
    finally:
        from ml_switcheroo_compiler.backends.visitor import CodeGeneratorVisitor

        pytest.MonkeyPatch().setattr(
            sys.modules["ml_switcheroo_compiler.backends.base_generator"],
            "CodeGeneratorVisitor",
            CodeGeneratorVisitor,
            raising=False,
        )


def test_ast_walker_missing_lines() -> None:
    """Test function test_ast_walker_missing_lines."""
    from ml_switcheroo_ir import IRGraph

    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator

    gen = BaseGenerator(IRGraph())
    # patch out the visitor call to prevent error
    gen.sorted_nodes = []
    from ml_switcheroo_compiler.backends.visitor import CodeGeneratorVisitor

    class DummyVisitor:
        """Test class DummyVisitor."""

        def __init__(self, g: BackendArg) -> None:
            """Test function __init__.

            Args:
                g: Test parameter `g`.
            """
            self.g = g

        def generate_body(self, p: BackendArg) -> None:
            """Test function generate_body.

            Args:
                p: Test parameter `p`.
            """
            pass

    import sys

    try:
        pytest.MonkeyPatch().setattr(
            sys.modules["ml_switcheroo_compiler.backends.base_generator"],
            "CodeGeneratorVisitor",
            DummyVisitor,
            raising=False,
        )
        import ml_switcheroo_compiler.backends.base_generator as bg

        pytest.MonkeyPatch().setattr(bg, "CodeGeneratorVisitor", DummyVisitor, raising=False)
        walker = bg.IRGraphWalker(gen)
        walker.walk("args")
        assert walker.generator is gen
    finally:
        pytest.MonkeyPatch().setattr(
            sys.modules["ml_switcheroo_compiler.backends.base_generator"],
            "CodeGeneratorVisitor",
            CodeGeneratorVisitor,
            raising=False,
        )


def test_class_based_generator_has_base_no_params() -> None:
    """Test function test_class_based_generator_has_base_no_params."""
    from ml_switcheroo_ir import IRGraph

    from ml_switcheroo_compiler.backends.base_generator import ClassBasedGenerator

    class Gen3(ClassBasedGenerator):
        """Test class Gen3."""

        _base_class_name = "Base3"

        def _generate_body(self, input_prefix: str = "args") -> None:
            """Test function _generate_body.

            Args:
                input_prefix: Test parameter `input_prefix`.
            """
            pass

        def _emit_init_body(self) -> bool:
            """Test function _emit_init_body.

            Returns:
                bool: Computed output value from `_emit_init_body`.
            """
            return False

        def get_fallback_prefix(self) -> str:
            """Test function get_fallback_prefix.

            Returns:
                str: Computed output value from `get_fallback_prefix`.
            """
            return "numpy"

        def get_language(self) -> str:
            """Test function get_language.

            Returns:
                str: Computed output value from `get_language`.
            """
            return "python"

    gen = Gen3(IRGraph())
    code = gen.generate()
    assert "super().__init__()" in code
    assert "pass" in code


def test_class_based_generator_has_base_no_params_not_python() -> None:
    """Test function test_class_based_generator_has_base_no_params_not_python."""
    from ml_switcheroo_ir import IRGraph

    from ml_switcheroo_compiler.backends.base_generator import ClassBasedGenerator

    class Gen4(ClassBasedGenerator):
        """Test class Gen4."""

        _base_class_name = "Base3"

        def _generate_body(self, input_prefix: str = "args") -> None:
            """Test function _generate_body.

            Args:
                input_prefix: Test parameter `input_prefix`.
            """
            pass

        def _emit_init_body(self) -> bool:
            """Test function _emit_init_body.

            Returns:
                bool: Computed output value from `_emit_init_body`.
            """
            return False

        def get_fallback_prefix(self) -> str:
            """Test function get_fallback_prefix.

            Returns:
                str: Computed output value from `get_fallback_prefix`.
            """
            return "numpy"

        def get_language(self) -> str:
            """Test function get_language.

            Returns:
                str: Computed output value from `get_language`.
            """
            return "cpp"

    gen = Gen4(IRGraph())
    code = gen.generate()
    assert "super().__init__()" in code


def test_ast_walker_proper() -> None:
    """Test function test_ast_walker_proper."""
    from ml_switcheroo_ir import IRNode

    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator, IRGraphWalker
    from tests.testing_utils import MockGraphStub

    g = MockGraphStub()
    n = IRNode("in", "Input")
    pytest.MonkeyPatch().setattr(g, "inputs", [n], raising=False)
    g.outputs = []

    class MyGen(BaseGenerator):
        """Test class MyGen."""

        def get_fallback_prefix(self) -> str:
            """Test function get_fallback_prefix.

            Returns:
                str: Computed output value from `get_fallback_prefix`.
            """
            return "numpy"

        def _emit_input_assignment(self, v: BackendArg, n: BackendArg, p: BackendArg, i: BackendArg) -> None:
            """Test function _emit_input_assignment.

            Args:
                v: Test parameter `v`.
                n: Test parameter `n`.
                p: Test parameter `p`.
                i: Test parameter `i`.
            """
            pass

        def _emit_output_assignment(self, n: BackendArg, i: BackendArg, v: BackendArg) -> None:
            """Test function _emit_output_assignment.

            Args:
                n: Test parameter `n`.
                i: Test parameter `i`.
                v: Test parameter `v`.
            """
            pass

        def get_ops_map(self, *args: BackendArg, **kwargs: BackendArg) -> dict[str, str]:
            """Test function get_ops_map.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                dict[str, str]: Computed output value from `get_ops_map`.
            """
            return {}

    gen = MyGen(g)
    w = IRGraphWalker(gen)
    w.walk("args")
    assert gen.get_fallback_prefix() == "numpy" and gen.get_ops_map() == {}


def test_class_based_generator_has_base_no_params_not_python_2() -> None:
    """Test function test_class_based_generator_has_base_no_params_not_python_2."""
    from ml_switcheroo_ir import IRGraph

    from ml_switcheroo_compiler.backends.base_generator import ClassBasedGenerator

    class Gen4(ClassBasedGenerator):
        """Test class Gen4."""

        _base_class_name = "Base3"

        def _generate_body(self, input_prefix: str = "args") -> None:
            """Test function _generate_body.

            Args:
                input_prefix: Test parameter `input_prefix`.
            """
            pass

        def _emit_init_body(self) -> bool:
            """Test function _emit_init_body.

            Returns:
                bool: Computed output value from `_emit_init_body`.
            """
            return False

        def get_fallback_prefix(self) -> str:
            """Test function get_fallback_prefix.

            Returns:
                str: Computed output value from `get_fallback_prefix`.
            """
            return "numpy"

        def get_language(self) -> str:
            """Test function get_language.

            Returns:
                str: Computed output value from `get_language`.
            """
            return "js"

    gen = Gen4(IRGraph())
    code = gen.generate()
    assert "super().__init__()" in code


def test_class_based_generator_has_base_no_params_is_python() -> None:
    """Test function test_class_based_generator_has_base_no_params_is_python."""
    from ml_switcheroo_ir import IRGraph

    from ml_switcheroo_compiler.backends.base_generator import ClassBasedGenerator

    class Gen5(ClassBasedGenerator):
        """Test class Gen5."""

        _base_class_name = "Base3"

        def _generate_body(self, input_prefix: str = "args") -> None:
            """Test function _generate_body.

            Args:
                input_prefix: Test parameter `input_prefix`.
            """
            pass

        def _emit_init_body(self) -> bool:
            """Test function _emit_init_body.

            Returns:
                bool: Computed output value from `_emit_init_body`.
            """
            return False

        def get_fallback_prefix(self) -> str:
            """Test function get_fallback_prefix.

            Returns:
                str: Computed output value from `get_fallback_prefix`.
            """
            return "numpy"

        def get_language(self) -> str:
            """Test function get_language.

            Returns:
                str: Computed output value from `get_language`.
            """
            return "python"

    gen = Gen5(IRGraph())
    code = gen.generate()
    assert "super().__init__()" in code
    assert "pass" in code


def test_class_based_generator_has_base_no_params_hit_457() -> None:
    """Test function test_class_based_generator_has_base_no_params_hit_457."""
    from ml_switcheroo_ir import IRGraph

    from ml_switcheroo_compiler.backends.base_generator import ClassBasedGenerator

    class Gen6(ClassBasedGenerator):
        """Test class Gen6."""

        def __init__(self, graph: IRGraph) -> None:
            """Test function __init__.

            Args:
                graph (IRGraph): Test parameter `graph`.
            """
            super().__init__(graph)
            self._base_class_name = "BaseClass"

        def _generate_body(self, input_prefix: str = "args") -> None:
            """Test function _generate_body.

            Args:
                input_prefix: Test parameter `input_prefix`.
            """
            pass

        def _emit_init_body(self) -> bool:
            """Test function _emit_init_body.

            Returns:
                bool: Computed output value from `_emit_init_body`.
            """
            return False

        def get_fallback_prefix(self) -> str:
            """Test function get_fallback_prefix.

            Returns:
                str: Computed output value from `get_fallback_prefix`.
            """
            return "numpy"

        def get_language(self) -> str:
            """Test function get_language.

            Returns:
                str: Computed output value from `get_language`.
            """
            return "python"

    gen = Gen6(IRGraph())
    code = gen.generate()
    assert "class " in code and "BaseClass" in code and "super().__init__()" in code


def test_graph_walker_lines() -> None:
    """Test function test_graph_walker_lines."""
    from ml_switcheroo_ir import IRGraph

    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator, IRGraphWalker

    class DummyVisitor:
        """Test class DummyVisitor."""

        def __init__(self, g: BackendArg) -> None:
            """Test function __init__.

            Args:
                g: Test parameter `g`.
            """
            self.g = g

        def generate_body(self, p: BackendArg) -> None:
            """Test function generate_body.

            Args:
                p: Test parameter `p`.
            """
            pass

    import ml_switcheroo_compiler.backends.base_generator as bg

    old = bg.CodeGeneratorVisitor
    pytest.MonkeyPatch().setattr(bg, "CodeGeneratorVisitor", DummyVisitor, raising=False)
    try:
        gen = BaseGenerator(IRGraph())
        w = IRGraphWalker(gen)
        w.walk()
    finally:
        pytest.MonkeyPatch().setattr(bg, "CodeGeneratorVisitor", old, raising=False)
    assert bg.CodeGeneratorVisitor is old


def test_base_generator_coverage_extras() -> None:
    """Test function test_base_generator_coverage_extras."""
    from ml_switcheroo_ir import IRGraph, IRNode

    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator

    class CovGen(BaseGenerator):
        """Test class CovGen."""

        def get_fallback_prefix(self) -> str:
            """Test function get_fallback_prefix.

            Returns:
                str: Computed output value from `get_fallback_prefix`.
            """
            return "numpy"

    gen = CovGen(IRGraph())

    # Hit line 243 by forcing get_ops_map to fetch from registry
    ops = gen.get_ops_map({})
    assert "transpose" in ops or "Transpose" in ops

    # Hit line 304-312 by forcing _format_operation with no format string
    node = IRNode(id="n1", op_type="UnknownCovOp", inputs=[])
    pytest.MonkeyPatch().setattr(gen, "ops_map", {}, raising=False)
    res = gen.generic_visit(node, ["in1"], **{})
    assert "unknowncovop" in res


def test_get_ops_map_fallbacks_coverage() -> None:
    """Test function test_get_ops_map_fallbacks_coverage."""
    import unittest.mock as mock

    from ml_switcheroo_ir import IRGraph

    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator

    class MyGen(BaseGenerator):
        """Test class MyGen."""

        def get_fallback_prefix(self) -> str:
            """Test function get_fallback_prefix.

            Returns:
                str: Computed output value from `get_fallback_prefix`.
            """
            return "test_prefix"

    gen = MyGen(IRGraph())

    with mock.patch("ml_switcheroo_compiler.ops.registry.backend_mapping_registry") as mock_reg:
        mock_reg.operations.keys.return_value = ["Dct", "Idct", "Mdct", "InverseMdct", "Frame", "OverlapAndAdd"]

        def mock_get(prefix: str, name: str) -> str | None:
            """Test function mock_get.

            Args:
                prefix (str): Test parameter `prefix`.
                name (str): Test parameter `name`.

            Returns:
                str | None: Computed output value from `mock_get`.
            """
            if prefix == "test_prefix":
                return "mapped_" + name
            return None

        mock_reg.get_generator_mapping.side_effect = mock_get

        ops = gen.get_ops_map({})

        assert ops["Dct"] == "mapped_Dct"
        assert ops["Idct"] == "mapped_Idct"
        assert ops["Mdct"] == "mapped_Mdct"
        assert ops["InverseMdct"] == "mapped_InverseMdct"
        assert ops["Frame"] == "mapped_Frame"
        assert ops["OverlapAndAdd"] == "mapped_OverlapAndAdd"


def test_base_generator_get_fallback_prefix() -> None:
    """Test function test_base_generator_get_fallback_prefix."""
    from ml_switcheroo_ir import IRGraph

    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator

    assert BaseGenerator(IRGraph()).get_fallback_prefix() == "np"


def test_base_generator_compile_aot() -> None:
    """Test function test_base_generator_compile_aot."""
    import pytest
    from ml_switcheroo_ir import IRGraph

    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator

    class CustomGen(BaseGenerator):
        """Test class CustomGen."""

        def _compile_aot_impl(self, graph: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function _compile_aot_impl.

            Args:
                graph (BackendArg): Test parameter `graph`.
                **kwargs (BackendArg): Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `_compile_aot_impl`.
            """
            return "compiled_artifact"

    g = IRGraph()
    # Call as classmethod
    res_cls = CustomGen.compile_aot(g)
    assert res_cls.callable_fn == "compiled_artifact"

    # Call on instance with graph=None falling back to self.graph
    inst = CustomGen(g)
    assert inst.compile_aot(None).callable_fn == "compiled_artifact"

    # Base implementation raises NotImplementedError
    base_inst = BaseGenerator(g)
    with pytest.raises(NotImplementedError, match="AOT compilation is not implemented"):
        base_inst.compile_aot(g)


class _TestBaseGeneratorVisitAndVisitOpnodeState(typing.TypedDict, total=False):
    """Typed shared state for `_test_base_generator_visit_and_visit_opnode` split helpers."""

    BaseGenerator: type[BaseGenerator]
    IRGraph: type[IRGraph]
    IRNode: type[IRNode]
    MagicMock: type[MagicMock]
    bg: BaseGenerator
    g: LogicalGraph
    patch: EagerBoundCallable


def _test_base_generator_visit_and_visit_opnode_part_1(_state: _TestBaseGeneratorVisitAndVisitOpnodeState) -> None:
    """Execute part 1 of test_base_generator_visit_and_visit_opnode.

    Args:
        _state (_TestBaseGeneratorVisitAndVisitOpnodeState): Shared test state dictionary.
    """
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_ir import IRGraph as _IRGraphCls
    from ml_switcheroo_ir import IRNode as _IRNodeCls

    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator as _BaseGenCls

    g = _IRGraphCls()
    bg = _BaseGenCls(g)

    # 1. target_api call string path with kwarg translations (e.g. Argmin in cupy: dim -> axis)
    cupy_bg = _BaseGenCls(g)
    cupy_bg.backend_name = "cupy"
    argmin_node = _IRNodeCls(
        id="n1",
        op_type="Argmin",
        attributes={"dim": 0, "keepdim": True, "_internal_attr": "hidden"},
    )
    argmin_res = cupy_bg.visit_OpNode(argmin_node, ["x"])
    assert "cupy.argmin" in argmin_res
    assert "x" in argmin_res
    assert "axis=0" in argmin_res
    assert "keepdim=True" in argmin_res
    assert "_internal_attr" not in argmin_res

    # 2. ast_template substitution path (e.g. Atan2 in pytorch)
    torch_bg = _BaseGenCls(g)
    torch_bg.backend_name = "pytorch"
    atan2_node = _IRNodeCls(id="n2", op_type="Atan2", attributes={"out_var": "out_val"})
    atan2_res = torch_bg.visit_OpNode(atan2_node, ["in_x", "in_y"])
    assert "out_val = torch.atan2(in_x, in_y)" in atan2_res

    # ast_template with default out variable name from node.id
    atan2_node_default_out = _IRNodeCls(id="atan2_id", op_type="Atan2")
    atan2_res_default = torch_bg.visit_OpNode(atan2_node_default_out, ["in_x", "in_y"])
    assert "tensor_0 = torch.atan2(in_x, in_y)" in atan2_res_default
    _state.update({"BaseGenerator": _BaseGenCls, "IRGraph": _IRGraphCls, "IRNode": _IRNodeCls, "MagicMock": MagicMock})
    _state.update({"bg": bg})
    _state.update({"g": g, "patch": require_callable(patch)})


def _test_base_generator_visit_and_visit_opnode_part_2(_state: _TestBaseGeneratorVisitAndVisitOpnodeState) -> None:
    """Execute part 2 of test_base_generator_visit_and_visit_opnode.

    Args:
        _state (_TestBaseGeneratorVisitAndVisitOpnodeState): Shared test state dictionary.
    """
    BaseGenCls, IRNodeCls, MagicMockCls, bg = (
        _state["BaseGenerator"],
        _state["IRNode"],
        _state["MagicMock"],
        _state["bg"],
    )
    g, patch = _state["g"], _state["patch"]
    mock_op = MagicMockCls()
    pytest.MonkeyPatch().setattr(mock_op, "ast_template", None, raising=False)
    pytest.MonkeyPatch().setattr(mock_op, "kwarg_map", {"drop_me": None}, raising=False)
    pytest.MonkeyPatch().setattr(mock_op, "target_api", "np.dropped", raising=False)
    mock_schema = MagicMockCls()
    pytest.MonkeyPatch().setattr(mock_schema, "operations", {"DroppedOp": mock_op}, raising=False)

    with patch(
        "ml_switcheroo_compiler.backends.mapping_loader.load_backend_mappings",
        return_value=mock_schema,
    ):
        drop_node = IRNodeCls(id="n_drop", op_type="DroppedOp", attributes={"drop_me": 123})
        drop_res = bg.visit_OpNode(drop_node, ["in_val"])
        assert drop_res == "np.dropped(in_val)"

    # 4. Exception in load_backend_mappings falling back to generic_visit
    bad_bg = BaseGenCls(g)
    with patch(
        "ml_switcheroo_compiler.backends.mapping_loader.load_backend_mappings",
        side_effect=RuntimeError("disk error"),
    ):
        unknown_node = IRNodeCls(id="n3", op_type="Add")
        fallback_res = bad_bg.visit_OpNode(unknown_node, ["arg1", "arg2"])
        assert "np.add(arg1, arg2)" in fallback_res

    # 5. Target op not in schema operations falling back to generic_visit
    unmapped_node = IRNodeCls(id="n4", op_type="CustomOpNotInSchema")
    unmapped_res = bg.visit_OpNode(unmapped_node, ["arg0"])
    assert "customopnotinschema" in unmapped_res.lower()
    _state.update({"BaseGenerator": BaseGenCls, "IRNode": IRNodeCls, "MagicMock": MagicMockCls})
    _state.update({"bg": bg, "g": g, "patch": require_callable(patch)})


def _test_base_generator_visit_and_visit_opnode_part_3(_state: _TestBaseGeneratorVisitAndVisitOpnodeState) -> None:
    """Execute part 3 of test_base_generator_visit_and_visit_opnode.

    Args:
        _state (_TestBaseGeneratorVisitAndVisitOpnodeState): Shared test state dictionary.
    """
    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator as _BaseGeneratorCls

    BaseGenCls, IRNodeCls, bg, g = _state["BaseGenerator"], _state["IRNode"], _state["bg"], _state["g"]

    class CustomVisitor:
        """Test class CustomVisitor."""

        def visit_VisitorOp(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> str:
            """Test function visit_VisitorOp.

            Args:
                node (IRNode): Test parameter `node`.
                input_vars (list[str]): Test parameter `input_vars`.
                **kwargs (BackendArg): Test parameter `kwargs`.

            Returns:
                str: Computed output value from `visit_VisitorOp`.
            """
            return f"visitor_handled({', '.join(input_vars)})"

    pytest.MonkeyPatch().setattr(bg, "visitors", [CustomVisitor()], raising=False)
    v_node = IRNodeCls(id="n5", op_type="VisitorOp")
    assert bg.visit(v_node, ["v1", "v2"]) == "visitor_handled(v1, v2)"

    class CustomGenerator(_BaseGeneratorCls):
        """Test class CustomGenerator."""

        def visit_SubclassOp(self, node: IRNode, input_vars: list[str], **kwargs: BackendArg) -> str:
            """Test function visit_SubclassOp.

            Args:
                node (IRNode): Test parameter `node`.
                input_vars (list[str]): Test parameter `input_vars`.
                **kwargs (BackendArg): Test parameter `kwargs`.

            Returns:
                str: Computed output value from `visit_SubclassOp`.
            """
            return f"subclass_handled({', '.join(input_vars)})"

    cg = CustomGenerator(g)
    pytest.MonkeyPatch().setattr(cg, "visitors", [], raising=False)
    s_node = IRNodeCls(id="n6", op_type="SubclassOp")
    assert cg.visit(s_node, ["s1"]) == "subclass_handled(s1)"

    # 8. visit dispatch: generic_visit fallback and visit_OpNode dispatch
    bg_clean = BaseGenCls(g)
    pytest.MonkeyPatch().setattr(bg_clean, "visitors", [], raising=False)
    fallback_node = IRNodeCls(id="n_fallback", op_type="NonexistentOpFallback")
    assert bg_clean.visit(fallback_node, ["x"]) == "np.nonexistentopfallback(x)"
    _state.update({"BaseGenerator": BaseGenCls, "IRNode": IRNodeCls, "bg": bg})
    _state.update({"g": g})


def _test_base_generator_visit_and_visit_opnode_part_4(_state: _TestBaseGeneratorVisitAndVisitOpnodeState) -> None:
    """Execute part 4 of test_base_generator_visit_and_visit_opnode.

    Args:
        _state (_TestBaseGeneratorVisitAndVisitOpnodeState): Shared test state dictionary.
    """
    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator as _BaseGeneratorCls

    BaseGenCls, IRNodeCls, g = _state["BaseGenerator"], _state["IRNode"], _state["g"]

    class NoOpNodeGenerator(_BaseGeneratorCls):
        """Test class NoOpNodeGenerator."""

        pass

    with pytest.MonkeyPatch.context() as mp:
        mp.delattr(BaseGenCls, "visit_OpNode")
        nog = NoOpNodeGenerator(g)
        g_node = IRNodeCls(id="n7", op_type="GenericOp")
        assert nog.visit(g_node, ["g1"]) == "np.genericop(g1)"


def test_base_generator_visit_and_visit_opnode() -> None:
    """Test visit_OpNode declarative mapping, ast_template, target_api, and visit dispatch."""
    _state: _TestBaseGeneratorVisitAndVisitOpnodeState = {}
    _test_base_generator_visit_and_visit_opnode_part_1(_state)
    _test_base_generator_visit_and_visit_opnode_part_2(_state)
    _test_base_generator_visit_and_visit_opnode_part_3(_state)
    _test_base_generator_visit_and_visit_opnode_part_4(_state)
    assert len(_state) > 0
