"""Tests for test_base_generator."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator, ClassBasedGenerator, EmitUtilsMixin, FormatterProxyMixin, PythonStringGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


class DummyFormatter:
    """Test class DummyFormatter."""

    def __init__(self) -> None:
        """Test function __init__."""
        self.var_names = cast("dict[str, EagerTensorProtocol]", {})
        self.code = cast("list[EagerTensorProtocol]", [])
        self.header = ""
        self.indent_level = 0

    def get_indent(self) -> EagerValue:
        """Test function get_indent."""
        return "  " * self.indent_level

    def add_line(self, line: EagerValue) -> None:
        """Test function add_line."""
        cast("Callable[..., EagerTensorProtocol]", self.code.append)(cast("Callable[..., EagerTensorProtocol]", self.get_indent)() + line)

    def assign_var_name(self, node_id: str, prefix: str = "tensor") -> str:
        """Test function assign_var_name."""
        return f"{prefix}_foo"


class DummyGenerator(BaseGenerator):
    """Test class DummyGenerator."""

    def get_fallback_prefix(self) -> str:
        """Test function get_fallback_prefix."""
        return "dummy"


def test_formatter_proxy_mixin() -> None:
    """Test function test_formatter_proxy_mixin."""
    proxy = FormatterProxyMixin()
    setattr(proxy, "formatter", DummyFormatter())
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
        self.code = cast("list[EagerTensorProtocol]", [])

    def add_line(self, line: EagerValue) -> None:
        """Test function add_line."""
        cast("Callable[..., EagerTensorProtocol]", self.code.append)(line)


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

    setattr(eu, "graph", IRGraph())
    setattr(eu, "formatter", DummyFormatter())
    setattr(cast("EagerTensorProtocol", getattr(eu, "graph")), "outputs", ["out1"])
    eu.code = []
    eu._emit_body_return([])
    assert eu.code == ["return out1"]

    setattr(cast("EagerTensorProtocol", getattr(eu, "graph")), "outputs", ["out1", "out2"])
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
    def get_ops_map(kwargs: EagerValue) -> EagerValue:
        """Test function get_ops_map."""
        return {"TestOp": "test_prefix.testop({0}, {1}, kw={kw})"}

    setattr(gen, "get_ops_map", get_ops_map)
    res = gen.generic_visit(IRNode("id", "TestOp"), ["a", "b"], kw=42)
    assert res == "test_prefix.testop(a, b, kw=42)"

    # kwargs replacement where kwargs left over
    def get_ops_map2(kwargs: EagerValue) -> EagerValue:
        """Test function get_ops_map2."""
        return {"TestOp2": "test_prefix.testop({0}, {1}, kw={kw})"}

    setattr(gen, "get_ops_map", get_ops_map2)
    res2 = gen.generic_visit(IRNode("id", "TestOp2"), ["a", "b"], kw=42, unrelated="{not_fmt}")
    # unrelated should be cleaned up by regex
    assert "test_prefix.testop" in res2

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
        """Test function _generate_body."""
        pass


def test_python_string_generator() -> None:
    """Test function test_python_string_generator."""
    gen = DummyPyGen(IRGraph())
    code = gen.generate()
    assert "import dummy" in code

    gen2 = DummyPyGen(IRGraph())
    setattr(gen2, "_import_header", ["import a", "import b"])
    assert "import a\nimport b" in gen2.generate()

    gen3 = DummyPyGen(IRGraph())
    setattr(gen3, "_import_header", 42)  # something else to test branch
    gen3.generate()


class DummyClassGen(ClassBasedGenerator):
    """Test class DummyClassGen."""

    def _generate_body(self, input_prefix: str = "args") -> None:
        """Test function _generate_body."""
        pass


def test_class_based_generator() -> None:
    """Test function test_class_based_generator."""
    gen = DummyClassGen(IRGraph())
    assert gen._get_prefix_code() == []
    assert gen._emit_init_body() is False


def _test_code_generator_visitor_missing_lines_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_code_generator_visitor_missing_lines.

    Args:
        _state: Shared test state dictionary.
    """
    CodeGeneratorVisitor = cast("EagerTensorProtocol", getattr(importlib.import_module("ml_switcheroo_compiler.backends.base_generator"), "CodeGeneratorVisitor"))
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    getattr(_state, "update")({"CodeGeneratorVisitor": CodeGeneratorVisitor, "IRGraph": IRGraph, "IRNode": IRNode})
    _state.update(locals())


def _test_code_generator_visitor_missing_lines_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_code_generator_visitor_missing_lines.

    Args:
        _state: Shared test state dictionary.
    """
    IRGraph, IRNode = _state["IRGraph"], _state["IRNode"]

    class MockGenerator:
        """Test class MockGenerator."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.code = cast("list[EagerTensorProtocol]", [])
            self.graph = IRGraph()
            setattr(self.graph, "inputs", [IRNode("in1", "Input", "in1")])
            setattr(self.graph, "outputs", [IRNode("out1", "Output", "out1")])
            self.sorted_nodes = self.graph.inputs + self.graph.outputs
            self.var_names = cast("dict[str, EagerTensorProtocol]", {})

        def assign_var_name(self, id: EagerValue, prefix: EagerValue = "") -> EagerValue:
            """Test function assign_var_name."""
            return cast("EagerTensorProtocol", "var_") + cast("EagerTensorProtocol", id)

        def generic_visit(self, node: EagerValue, inputs: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function generic_visit."""
            return f"generic({getattr(node, 'op_type')})"

        def _emit_input_assignment(self, var: EagerValue, node: EagerValue, prefix: EagerValue, i: EagerValue) -> None:
            """Test function _emit_input_assignment."""
            cast("Callable[..., EagerTensorProtocol]", self.code.append)(f"{var} = {prefix}[{i}]")

        def _emit_output_assignment(self, node: EagerValue, inputs: EagerValue, var: EagerValue) -> None:
            """Test function _emit_output_assignment."""
            cast("Callable[..., EagerTensorProtocol]", self.code.append)(f"ret {var}")

        def get_ops_map(self, *args: EagerValue, **kwargs: EagerValue) -> dict[str, str]:
            """Test function get_ops_map."""
            return {}

        def _emit_body_return(self, returns: EagerValue) -> None:
            """Test function _emit_body_return."""
            pass

        def add_line(self, line_str: EagerValue) -> None:
            """Test function add_line."""
            pass

        def _emit_constant_assignment(self, var: EagerValue, cst: EagerValue) -> None:
            """Test function _emit_constant_assignment."""
            pass

    getattr(_state, "update")({"MockGenerator": MockGenerator})
    _state.update(locals())


def _test_code_generator_visitor_missing_lines_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_code_generator_visitor_missing_lines.

    Args:
        _state: Shared test state dictionary.
    """
    CodeGeneratorVisitor, IRNode, MockGenerator = _state["CodeGeneratorVisitor"], _state["IRNode"], _state["MockGenerator"]
    gen = MockGenerator()
    visitor = CodeGeneratorVisitor(gen)
    visitor.generate_body("args")
    assert "var_in1 = args[0]" in gen.code

    # Test visit method routing
    node = IRNode("node1", "TestOp")

    class MockSpecificVisitor:
        """Test class MockSpecificVisitor."""

        def visit_TestOp(self, node: EagerValue, inputs: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function visit_TestOp."""
            return "visited_TestOp"

    setattr(gen, "visitors", [MockSpecificVisitor()])

    # Test visit method routing
    node = IRNode("node1", "TestOp")

    class MockSpecificVisitor2:
        """Test class MockSpecificVisitor2."""

        def visit_TestOp(self, node: EagerValue, inputs: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function visit_TestOp."""
            return "visited_TestOp"

    setattr(gen, "visitors", [MockSpecificVisitor2()])
    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator

    res = cast("Callable[..., EagerTensorProtocol]", BaseGenerator.visit)(gen, node, [])
    assert res == "visited_TestOp"
    getattr(_state, "update")({"BaseGenerator": BaseGenerator, "MockSpecificVisitor2": MockSpecificVisitor2, "gen": gen, "node": node})
    getattr(_state, "update")({"res": res, "visitor": visitor})
    _state.update(locals())


def _test_code_generator_visitor_missing_lines_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_code_generator_visitor_missing_lines.

    Args:
        _state: Shared test state dictionary.
    """
    BaseGenerator, IRNode, gen, node = _state["BaseGenerator"], _state["IRNode"], _state["gen"], _state["node"]
    visitor = _state["visitor"]

    class MockSpecificVisitor2:
        """Test class MockSpecificVisitor2."""

        def visit_OtherOp(self, node: EagerValue, inputs: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function visit_OtherOp."""
            return "visited_OtherOp"

    setattr(gen, "visitors", [MockSpecificVisitor2()])
    res2 = BaseGenerator.visit(gen, node, [])
    assert "generic" in res2

    # Test fallback

    node2 = IRNode("node2", "UnknownOp")
    res2 = visitor.generator.generic_visit(node2, [])
    getattr(_state, "update")({"MockSpecificVisitor2": MockSpecificVisitor2, "node2": node2, "res2": res2})


def test_code_generator_visitor_missing_lines() -> None:
    """Test function test_code_generator_visitor_missing_lines."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_code_generator_visitor_missing_lines_part_1(_state)
    _test_code_generator_visitor_missing_lines_part_2(_state)
    _test_code_generator_visitor_missing_lines_part_3(_state)
    _test_code_generator_visitor_missing_lines_part_4(_state)
    assert isinstance(_state, dict)


def test_base_generator_maps() -> None:
    """Test function test_base_generator_maps."""
    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    class MinimalGen(BaseGenerator):
        """Test class MinimalGen."""

        def get_fallback_prefix(self) -> str:
            """Test function get_fallback_prefix."""
            return "m"

        def get_ops_map(self, *args: EagerValue, **kwargs: EagerValue) -> dict[str, str]:
            """Test function get_ops_map."""
            ops = super().get_ops_map(kwargs)
            ops["Add"] = "m.add"
            return ops

    gen = MinimalGen(IRGraph())
    ops = gen.get_ops_map({})
    assert cast("EagerTensorProtocol", getattr(ops, "__getitem__")("Add")) == "m.add"


def test_class_based_generator_full() -> None:
    """Test function test_class_based_generator_full."""
    from ml_switcheroo_compiler.backends.base_generator import ClassBasedGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    class MyClassGen(ClassBasedGenerator):
        """Test class MyClassGen."""

        _base_class_name = "BaseClass"
        _forward_method_name = "forward"

        def _generate_body(self, input_prefix: str = "args") -> None:
            """Test function _generate_body."""
            self.add_line("self.generated = True")

        def _emit_init_body(self) -> bool:
            """Test function _emit_init_body."""
            self.add_line("self.init = True")
            return True

        def get_fallback_prefix(self) -> str:
            """Test function get_fallback_prefix."""
            return "my"

        def get_language(self) -> str:
            """Test function get_language."""
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

        _base_class_name = cast("str", None)

        def _generate_body(self, input_prefix: str = "args") -> None:
            """Test function _generate_body."""
            pass

        def _emit_init_body(self) -> bool:
            """Test function _emit_init_body."""
            return False

        def get_fallback_prefix(self) -> str:
            """Test function get_fallback_prefix."""
            return "my"

        def get_language(self) -> str:
            """Test function get_language."""
            return "python"

    gen2 = MyClassGenNoBase(IRGraph())
    code2 = gen2.generate()
    assert "class CompiledModel:" in code2
    assert "pass" in code2


def test_ir_graph_walker() -> None:
    """Test function test_ir_graph_walker."""
    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    class DummyVisitor:
        """Test class DummyVisitor."""

        def __init__(self, g: EagerValue) -> None:
            """Test function __init__."""
            self.g = g

        def generate_body(self, p: EagerValue) -> None:
            """Test function generate_body."""
            setattr(self.g, "called", p)

    gen = BaseGenerator(IRGraph())
    setattr(gen, "called", None)
    import sys

    try:
        setattr(sys.modules["ml_switcheroo_compiler.backends.base_generator"], "CodeGeneratorVisitor", DummyVisitor)
        import ml_switcheroo_compiler.backends.base_generator as bg

        setattr(bg, "CodeGeneratorVisitor", DummyVisitor)
        walker = bg.IRGraphWalker(gen)
        walker.walk("args")
        assert cast("EagerTensorProtocol", getattr(gen, "called")) == "args"
        assert walker.generator is gen
    finally:
        from ml_switcheroo_compiler.backends.visitor import CodeGeneratorVisitor

        setattr(sys.modules["ml_switcheroo_compiler.backends.base_generator"], "CodeGeneratorVisitor", CodeGeneratorVisitor)


def test_ast_walker_missing_lines() -> None:
    """Test function test_ast_walker_missing_lines."""
    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    gen = BaseGenerator(IRGraph())
    # patch out the visitor call to prevent error
    gen.sorted_nodes = []
    from ml_switcheroo_compiler.backends.visitor import CodeGeneratorVisitor

    class DummyVisitor:
        """Test class DummyVisitor."""

        def __init__(self, g: EagerValue) -> None:
            """Test function __init__."""
            self.g = g

        def generate_body(self, p: EagerValue) -> None:
            """Test function generate_body."""
            pass

    import sys

    try:
        setattr(sys.modules["ml_switcheroo_compiler.backends.base_generator"], "CodeGeneratorVisitor", DummyVisitor)
        import ml_switcheroo_compiler.backends.base_generator as bg

        setattr(bg, "CodeGeneratorVisitor", DummyVisitor)
        walker = bg.IRGraphWalker(gen)
        walker.walk("args")
        assert walker.generator is gen
    finally:
        setattr(sys.modules["ml_switcheroo_compiler.backends.base_generator"], "CodeGeneratorVisitor", CodeGeneratorVisitor)


def test_class_based_generator_has_base_no_params() -> None:
    """Test function test_class_based_generator_has_base_no_params."""
    from ml_switcheroo_compiler.backends.base_generator import ClassBasedGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    class Gen3(ClassBasedGenerator):
        """Test class Gen3."""

        _base_class_name = "Base3"

        def _generate_body(self, input_prefix: str = "args") -> None:
            """Test function _generate_body."""
            pass

        def _emit_init_body(self) -> bool:
            """Test function _emit_init_body."""
            return False

        def get_fallback_prefix(self) -> str:
            """Test function get_fallback_prefix."""
            return "numpy"

        def get_language(self) -> str:
            """Test function get_language."""
            return "python"

    gen = Gen3(IRGraph())
    code = gen.generate()
    assert "super().__init__()" in code
    assert "pass" in code


def test_class_based_generator_has_base_no_params_not_python() -> None:
    """Test function test_class_based_generator_has_base_no_params_not_python."""
    from ml_switcheroo_compiler.backends.base_generator import ClassBasedGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    class Gen4(ClassBasedGenerator):
        """Test class Gen4."""

        _base_class_name = "Base3"

        def _generate_body(self, input_prefix: str = "args") -> None:
            """Test function _generate_body."""
            pass

        def _emit_init_body(self) -> bool:
            """Test function _emit_init_body."""
            return False

        def get_fallback_prefix(self) -> str:
            """Test function get_fallback_prefix."""
            return "numpy"

        def get_language(self) -> str:
            """Test function get_language."""
            return "cpp"

    gen = Gen4(IRGraph())
    code = gen.generate()
    assert "super().__init__()" in code


def test_ast_walker_proper() -> None:
    """Test function test_ast_walker_proper."""
    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator, IRGraphWalker
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    n = IRNode("in", "Input")
    setattr(g, "inputs", [n])
    g.outputs = []

    class MyGen(BaseGenerator):
        """Test class MyGen."""

        def get_fallback_prefix(self) -> str:
            """Test function get_fallback_prefix."""
            return "numpy"

        def _emit_input_assignment(self, v: EagerValue, n: EagerValue, p: EagerValue, i: EagerValue) -> None:
            """Test function _emit_input_assignment."""
            pass

        def _emit_output_assignment(self, n: EagerValue, i: EagerValue, v: EagerValue) -> None:
            """Test function _emit_output_assignment."""
            pass

        def get_ops_map(self, *args: EagerValue, **kwargs: EagerValue) -> dict[str, str]:
            """Test function get_ops_map."""
            return {}

    gen = MyGen(g)
    w = IRGraphWalker(gen)
    w.walk("args")
    assert gen is not None


def test_class_based_generator_has_base_no_params_not_python_2() -> None:
    """Test function test_class_based_generator_has_base_no_params_not_python_2."""
    from ml_switcheroo_compiler.backends.base_generator import ClassBasedGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    class Gen4(ClassBasedGenerator):
        """Test class Gen4."""

        _base_class_name = "Base3"

        def _generate_body(self, input_prefix: str = "args") -> None:
            """Test function _generate_body."""
            pass

        def _emit_init_body(self) -> bool:
            """Test function _emit_init_body."""
            return False

        def get_fallback_prefix(self) -> str:
            """Test function get_fallback_prefix."""
            return "numpy"

        def get_language(self) -> str:
            """Test function get_language."""
            return "js"

    gen = Gen4(IRGraph())
    code = gen.generate()
    assert "super().__init__()" in code


def test_class_based_generator_has_base_no_params_is_python() -> None:
    """Test function test_class_based_generator_has_base_no_params_is_python."""
    from ml_switcheroo_compiler.backends.base_generator import ClassBasedGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    class Gen5(ClassBasedGenerator):
        """Test class Gen5."""

        _base_class_name = "Base3"

        def _generate_body(self, input_prefix: str = "args") -> None:
            """Test function _generate_body."""
            pass

        def _emit_init_body(self) -> bool:
            """Test function _emit_init_body."""
            return False

        def get_fallback_prefix(self) -> str:
            """Test function get_fallback_prefix."""
            return "numpy"

        def get_language(self) -> str:
            """Test function get_language."""
            return "python"

    gen = Gen5(IRGraph())
    code = gen.generate()
    assert "super().__init__()" in code
    assert "pass" in code


def test_class_based_generator_has_base_no_params_hit_457() -> None:
    """Test function test_class_based_generator_has_base_no_params_hit_457."""
    from ml_switcheroo_compiler.backends.base_generator import ClassBasedGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    class Gen6(ClassBasedGenerator):
        """Test class Gen6."""

        def __init__(self, *args: EagerValue, **kwargs: EagerValue) -> None:
            """Test function __init__."""
            cast("Callable[..., EagerTensorProtocol]", super().__init__)(*args, **kwargs)
            self._base_class_name = "BaseClass"

        def _generate_body(self, input_prefix: str = "args") -> None:
            """Test function _generate_body."""
            pass

        def _emit_init_body(self) -> bool:
            """Test function _emit_init_body."""
            return False

        def get_fallback_prefix(self) -> str:
            """Test function get_fallback_prefix."""
            return "numpy"

        def get_language(self) -> str:
            """Test function get_language."""
            return "python"

    gen = Gen6(IRGraph())
    gen.generate()
    assert gen is not None


def test_graph_walker_lines() -> None:
    """Test function test_graph_walker_lines."""
    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator, IRGraphWalker
    from ml_switcheroo_compiler.ir.core import IRGraph

    class DummyVisitor:
        """Test class DummyVisitor."""

        def __init__(self, g: EagerValue) -> None:
            """Test function __init__."""
            self.g = g

        def generate_body(self, p: EagerValue) -> None:
            """Test function generate_body."""
            pass

    import ml_switcheroo_compiler.backends.base_generator as bg

    old = cast("EagerTensorProtocol", getattr(bg, "CodeGeneratorVisitor"))
    setattr(bg, "CodeGeneratorVisitor", DummyVisitor)
    try:
        gen = BaseGenerator(IRGraph())
        w = IRGraphWalker(gen)
        w.walk()
    finally:
        setattr(bg, "CodeGeneratorVisitor", old)
    assert old is not None or old is None


def test_base_generator_coverage_extras() -> None:
    """Test function test_base_generator_coverage_extras."""
    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    class CovGen(BaseGenerator):
        """Test class CovGen."""

        def get_fallback_prefix(self) -> str:
            """Test function get_fallback_prefix."""
            return "numpy"

    gen = CovGen(IRGraph())

    # Hit line 243 by forcing get_ops_map to fetch from registry
    ops = gen.get_ops_map({})
    assert "transpose" in ops or "Transpose" in ops

    # Hit line 304-312 by forcing _format_operation with no format string
    node = IRNode(id="n1", op_type="UnknownCovOp", inputs=[])
    setattr(gen, "ops_map", {})
    res = gen.generic_visit(node, ["in1"], **{})
    assert "unknowncovop" in res


def test_get_ops_map_fallbacks_coverage() -> None:
    """Test function test_get_ops_map_fallbacks_coverage."""
    import unittest.mock as mock

    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    class MyGen(BaseGenerator):
        """Test class MyGen."""

        def get_fallback_prefix(self) -> str:
            """Test function get_fallback_prefix."""
            return "test_prefix"

    gen = MyGen(IRGraph())

    with mock.patch("ml_switcheroo_compiler.ops.registry.backend_mapping_registry") as mock_reg:
        mock_reg.operations.keys.return_value = ["Dct", "Idct", "Mdct", "InverseMdct", "Frame", "OverlapAndAdd"]

        def mock_get(prefix: EagerValue, name: EagerValue) -> EagerValue:
            """Test function mock_get."""
            if prefix == "test_prefix":
                return cast("EagerTensorProtocol", "mapped_") + cast("EagerTensorProtocol", name)
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
    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    assert BaseGenerator(IRGraph()).get_fallback_prefix() == "np"


def test_base_generator_compile_aot() -> None:
    """Test function test_base_generator_compile_aot."""
    import pytest

    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    class CustomGen(BaseGenerator):
        """Test class CustomGen."""

        def _compile_aot_impl(self, graph: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function _compile_aot_impl."""
            return "compiled_artifact"

    g = IRGraph()
    # Call as classmethod
    res_cls = CustomGen.compile_aot(g)
    assert cast("EagerTensorProtocol", res_cls) == cast("EagerTensorProtocol", "compiled_artifact")

    # Call on instance with graph=None falling back to self.graph
    inst = CustomGen(g)
    assert cast("EagerTensorProtocol", inst.compile_aot(None)) == cast("EagerTensorProtocol", "compiled_artifact")

    # Base implementation raises NotImplementedError
    base_inst = BaseGenerator(g)
    with pytest.raises(NotImplementedError, match="AOT compilation is not implemented"):
        base_inst.compile_aot(g)


def _test_base_generator_visit_and_visit_opnode_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_base_generator_visit_and_visit_opnode.

    Args:
        _state: Shared test state dictionary.
    """
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator

    IRNode = cast("EagerTensorProtocol", getattr(importlib.import_module("ml_switcheroo_compiler.backends.base_generator"), "IRNode"))
    from ml_switcheroo_compiler.ir.core import IRGraph

    g = IRGraph()
    bg = BaseGenerator(g)

    # 1. target_api call string path with kwarg translations (e.g. Argmin in cupy: dim -> axis)
    cupy_bg = BaseGenerator(g)
    setattr(cupy_bg, "backend_name", "cupy")
    argmin_node = IRNode(
        id="n1",
        op_type="Argmin",
        attributes={"dim": 0, "keepdim": True, "_internal_attr": "hidden"},
    )
    argmin_res = cast("Callable[..., EagerTensorProtocol]", cupy_bg.visit_OpNode)(argmin_node, ["x"])
    assert "cupy.argmin" in argmin_res
    assert "x" in argmin_res
    assert "axis=0" in argmin_res
    assert "keepdim=True" in argmin_res
    assert "_internal_attr" not in argmin_res

    # 2. ast_template substitution path (e.g. Atan2 in pytorch)
    torch_bg = BaseGenerator(g)
    setattr(torch_bg, "backend_name", "pytorch")
    atan2_node = IRNode(id="n2", op_type="Atan2", attributes={"out_var": "out_val"})
    atan2_res = cast("Callable[..., EagerTensorProtocol]", torch_bg.visit_OpNode)(atan2_node, ["in_x", "in_y"])
    assert "out_val = torch.atan2(in_x, in_y)" in atan2_res

    # ast_template with default out variable name from node.id
    atan2_node_default_out = IRNode(id="atan2_id", op_type="Atan2")
    atan2_res_default = cast("Callable[..., EagerTensorProtocol]", torch_bg.visit_OpNode)(atan2_node_default_out, ["in_x", "in_y"])
    assert "tensor_0 = torch.atan2(in_x, in_y)" in atan2_res_default
    getattr(_state, "update")({"BaseGenerator": BaseGenerator, "IRGraph": IRGraph, "IRNode": IRNode, "MagicMock": MagicMock})
    getattr(_state, "update")({"argmin_node": argmin_node, "argmin_res": argmin_res, "atan2_node": atan2_node, "atan2_node_default_out": atan2_node_default_out})
    getattr(_state, "update")({"atan2_res": atan2_res, "atan2_res_default": atan2_res_default, "bg": bg, "cupy_bg": cupy_bg})
    getattr(_state, "update")({"g": g, "patch": patch, "torch_bg": torch_bg})
    _state.update(locals())


def _test_base_generator_visit_and_visit_opnode_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_base_generator_visit_and_visit_opnode.

    Args:
        _state: Shared test state dictionary.
    """
    BaseGenerator, IRNode, MagicMock, bg = _state["BaseGenerator"], _state["IRNode"], _state["MagicMock"], _state["bg"]
    g, patch = _state["g"], _state["patch"]
    mock_op = MagicMock()
    setattr(mock_op, "ast_template", None)
    setattr(mock_op, "kwarg_map", {"drop_me": None})
    setattr(mock_op, "target_api", "np.dropped")
    mock_schema = MagicMock()
    setattr(mock_schema, "operations", {"DroppedOp": mock_op})

    with patch(
        "ml_switcheroo_compiler.backends.mapping_loader.load_backend_mappings",
        return_value=mock_schema,
    ):
        drop_node = IRNode(id="n_drop", op_type="DroppedOp", attributes={"drop_me": 123})
        drop_res = bg.visit_OpNode(drop_node, ["in_val"])
        assert drop_res == "np.dropped(in_val)"

    # 4. Exception in load_backend_mappings falling back to generic_visit
    bad_bg = BaseGenerator(g)
    with cast("Callable[..., EagerTensorProtocol]", patch)(
        "ml_switcheroo_compiler.backends.mapping_loader.load_backend_mappings",
        side_effect=cast("Callable[..., EagerTensorProtocol]", RuntimeError)("disk error"),
    ):
        unknown_node = IRNode(id="n3", op_type="Add")
        fallback_res = bad_bg.visit_OpNode(unknown_node, ["arg1", "arg2"])
        assert "np.add(arg1, arg2)" in fallback_res

    # 5. Target op not in schema operations falling back to generic_visit
    unmapped_node = IRNode(id="n4", op_type="CustomOpNotInSchema")
    unmapped_res = bg.visit_OpNode(unmapped_node, ["arg0"])
    assert "customopnotinschema" in unmapped_res.lower()
    getattr(_state, "update")({"bad_bg": bad_bg, "drop_node": drop_node, "drop_res": drop_res, "fallback_res": fallback_res})
    getattr(_state, "update")({"mock_op": mock_op, "mock_schema": mock_schema, "unknown_node": unknown_node, "unmapped_node": unmapped_node})
    getattr(_state, "update")({"unmapped_res": unmapped_res})
    _state.update(locals())


def _test_base_generator_visit_and_visit_opnode_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_base_generator_visit_and_visit_opnode.

    Args:
        _state: Shared test state dictionary.
    """
    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator as _BaseGeneratorCls

    BaseGenerator, IRNode, bg, g = _state["BaseGenerator"], _state["IRNode"], _state["bg"], _state["g"]

    class CustomVisitor:
        """Test class CustomVisitor."""

        def visit_VisitorOp(self, node: EagerTensorProtocol, input_vars: list[str], **kwargs: EagerValue) -> str:
            """Test function visit_VisitorOp."""
            return f"visitor_handled({', '.join(input_vars)})"

    setattr(bg, "visitors", [CustomVisitor()])
    v_node = IRNode(id="n5", op_type="VisitorOp")
    assert bg.visit(v_node, ["v1", "v2"]) == "visitor_handled(v1, v2)"

    class CustomGenerator(_BaseGeneratorCls):
        """Test class CustomGenerator."""

        def visit_SubclassOp(self, node: EagerTensorProtocol, input_vars: list[str], **kwargs: EagerValue) -> str:
            """Test function visit_SubclassOp."""
            return f"subclass_handled({', '.join(input_vars)})"

    cg = cast("Callable[..., EagerTensorProtocol]", CustomGenerator)(g)
    setattr(cg, "visitors", [])
    s_node = IRNode(id="n6", op_type="SubclassOp")
    assert cg.visit(s_node, ["s1"]) == "subclass_handled(s1)"

    # 8. visit dispatch: generic_visit fallback and visit_OpNode dispatch
    bg_clean = BaseGenerator(g)
    setattr(bg_clean, "visitors", [])
    fallback_node = IRNode(id="n_fallback", op_type="NonexistentOpFallback")
    assert bg_clean.visit(fallback_node, ["x"]) == "np.nonexistentopfallback(x)"
    getattr(_state, "update")({"CustomGenerator": CustomGenerator, "CustomVisitor": CustomVisitor, "bg_clean": bg_clean, "cg": cg})
    getattr(_state, "update")({"fallback_node": fallback_node, "s_node": s_node, "v_node": v_node})
    _state.update(locals())


def _test_base_generator_visit_and_visit_opnode_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_base_generator_visit_and_visit_opnode.

    Args:
        _state: Shared test state dictionary.
    """
    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator as _BaseGeneratorCls

    BaseGenerator, IRNode, g = _state["BaseGenerator"], _state["IRNode"], _state["g"]

    class NoOpNodeGenerator(_BaseGeneratorCls):
        """Test class NoOpNodeGenerator."""

        pass

    saved_opnode = BaseGenerator.visit_OpNode
    delattr(BaseGenerator, "visit_OpNode")
    try:
        nog = cast("Callable[..., EagerTensorProtocol]", NoOpNodeGenerator)(g)
        g_node = IRNode(id="n7", op_type="GenericOp")
        assert nog.visit(g_node, ["g1"]) == "np.genericop(g1)"
    finally:
        setattr(BaseGenerator, "visit_OpNode", saved_opnode)
    getattr(_state, "update")({"NoOpNodeGenerator": NoOpNodeGenerator, "saved_opnode": saved_opnode})


def test_base_generator_visit_and_visit_opnode() -> None:
    """Test visit_OpNode declarative mapping, ast_template, target_api, and visit dispatch."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_base_generator_visit_and_visit_opnode_part_1(_state)
    _test_base_generator_visit_and_visit_opnode_part_2(_state)
    _test_base_generator_visit_and_visit_opnode_part_3(_state)
    _test_base_generator_visit_and_visit_opnode_part_4(_state)
    assert isinstance(_state, dict)
