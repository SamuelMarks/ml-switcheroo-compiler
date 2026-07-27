from ml_switcheroo_compiler.backends.base_generator import BaseGenerator, ClassBasedGenerator, EmitUtilsMixin, FormatterProxyMixin, PythonStringGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


class DummyFormatter:
    def __init__(self):
        self.var_names = {}
        self.code = []
        self.header = ""
        self.indent_level = 0

    def get_indent(self):
        return "  " * self.indent_level

    def add_line(self, line):
        self.code.append(self.get_indent() + line)

    def assign_var_name(self, node_id: str, prefix: str = "tensor") -> str:
        return f"{prefix}_foo"


class DummyGenerator(BaseGenerator):
    def _get_backend_prefix(self):
        return "dummy"


def test_formatter_proxy_mixin():
    proxy = FormatterProxyMixin()
    proxy.formatter = DummyFormatter()
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
    def __init__(self):
        self.code = []

    def add_line(self, line):
        self.code.append(line)


def test_emit_utils_mixin():
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


def test_base_generator_methods():
    graph = IRGraph()
    gen = DummyGenerator(graph)
    node = IRNode("id", "op", attributes={"value": [1, 2, 3]})
    assert gen.emit_constant(node) == "[1, 2, 3]"

    assert gen.get_fallback_prefix() == "np"
    assert gen.get_fallback_axis_kwarg() == "axis"
    assert gen.get_fallback_keepdims_kwarg() == "keepdims"

    # kwargs mapping substitution
    # op_type in ops_map
    def get_ops_map(kwargs):
        return {"TestOp": "test_prefix.testop({0}, {1}, kw={kw})"}

    gen.get_ops_map = get_ops_map
    res = gen.generic_visit(IRNode("id", "TestOp"), ["a", "b"], kw=42)
    assert res == "test_prefix.testop(a, b, kw=42)"

    # kwargs replacement where kwargs left over
    def get_ops_map2(kwargs):
        return {"TestOp2": "test_prefix.testop({0}, {1}, kw={kw})"}

    gen.get_ops_map = get_ops_map2
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
    _import_header = "import dummy"

    def _generate_body(self, prefix):
        pass


def test_python_string_generator():
    gen = DummyPyGen(IRGraph())
    code = gen.generate()
    assert "import dummy" in code

    gen2 = DummyPyGen(IRGraph())
    gen2._import_header = ["import a", "import b"]
    assert "import a\nimport b" in gen2.generate()

    gen3 = DummyPyGen(IRGraph())
    gen3._import_header = 42  # something else to test branch
    gen3.generate()


class DummyClassGen(ClassBasedGenerator):
    def _generate_body(self, prefix):
        pass


def test_class_based_generator():
    gen = DummyClassGen(IRGraph())
    assert gen._get_prefix_code() == []
    assert gen._emit_init_body() is False
