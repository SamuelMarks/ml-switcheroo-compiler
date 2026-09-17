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
    def get_fallback_prefix(self):
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

    eu.graph = IRGraph()
    eu.formatter = DummyFormatter()
    eu.graph.outputs = ["out1"]
    eu.code = []
    eu._emit_body_return([])
    assert eu.code == ["return out1"]

    eu.graph.outputs = ["out1", "out2"]
    eu.code = []
    eu._emit_body_return([])
    assert eu.code == ["return (out1, out2)"]


def test_base_generator_methods():
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


def test_code_generator_visitor_missing_lines():
    from ml_switcheroo_compiler.backends.base_generator import CodeGeneratorVisitor
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    class MockGenerator:
        def __init__(self):
            self.code = []
            self.graph = IRGraph()
            self.graph.inputs = [IRNode("in1", "Input", "in1")]
            self.graph.outputs = [IRNode("out1", "Output", "out1")]
            self.sorted_nodes = self.graph.inputs + self.graph.outputs
            self.var_names = {}

        def assign_var_name(self, id, prefix=""):
            return "var_" + id

        def generic_visit(self, node, inputs, **kwargs):
            return f"generic({node.op_type})"

        def _emit_input_assignment(self, var, node, prefix, i):
            self.code.append(f"{var} = {prefix}[{i}]")

        def _emit_output_assignment(self, node, inputs, var):
            self.code.append(f"ret {var}")

        def get_ops_map(self, kwargs):
            return {}

        def _emit_body_return(self, returns):
            pass

        def add_line(self, l):
            pass

        def _emit_constant_assignment(self, var, cst):
            pass

    gen = MockGenerator()
    visitor = CodeGeneratorVisitor(gen)
    visitor.generate_body("args")
    assert "var_in1 = args[0]" in gen.code

    # Test visit method routing
    node = IRNode("node1", "TestOp")

    class MockSpecificVisitor:
        def visit_TestOp(self, node, inputs, **kwargs):
            return "visited_TestOp"

    gen.visitors = [MockSpecificVisitor()]

    # Test visit method routing
    node = IRNode("node1", "TestOp")

    class MockSpecificVisitor:
        def visit_TestOp(self, node, inputs, **kwargs):
            return "visited_TestOp"

    gen.visitors = [MockSpecificVisitor()]
    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator

    res = BaseGenerator.visit(gen, node, [])
    assert res == "visited_TestOp"

    # Test missing method mapping branch
    class MockSpecificVisitor2:
        def visit_OtherOp(self, node, inputs, **kwargs):
            return "visited_OtherOp"

    gen.visitors = [MockSpecificVisitor2()]
    res2 = BaseGenerator.visit(gen, node, [])
    assert "generic" in res2

    # Test fallback

    node2 = IRNode("node2", "UnknownOp")
    res2 = visitor.generator.generic_visit(node2, [])


def test_base_generator_maps():
    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    class MinimalGen(BaseGenerator):
        def get_fallback_prefix(self):
            return "m"

        def get_ops_map(self, kwargs):
            ops = super().get_ops_map(kwargs)
            ops["Add"] = "m.add"
            return ops

    gen = MinimalGen(IRGraph())
    ops = gen.get_ops_map({})
    assert ops["Add"] == "m.add"


def test_class_based_generator_full():
    from ml_switcheroo_compiler.backends.base_generator import ClassBasedGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    class MyClassGen(ClassBasedGenerator):
        _base_class_name = "BaseClass"
        _forward_method_name = "forward"

        def _generate_body(self):
            self.add_line("self.generated = True")

        def _emit_init_body(self):
            self.add_line("self.init = True")
            return True

        def get_fallback_prefix(self):
            return "my"

        def get_language(self):
            return "python"

    gen = MyClassGen(IRGraph())
    code = gen.generate()
    assert "class CompiledModel(BaseClass):" in code
    assert "super().__init__()" in code
    assert "self.init = True" in code
    assert "def forward(self, *args, **kwargs):" in code
    assert "self.generated = True" in code

    class MyClassGenNoBase(ClassBasedGenerator):
        _base_class_name = None

        def _generate_body(self):
            pass

        def _emit_init_body(self):
            return False

        def get_fallback_prefix(self):
            return "my"

        def get_language(self):
            return "python"

    gen2 = MyClassGenNoBase(IRGraph())
    code2 = gen2.generate()
    assert "class CompiledModel:" in code2
    assert "pass" in code2


def test_ir_graph_walker():
    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    class DummyVisitor:
        def __init__(self, g):
            self.g = g

        def generate_body(self, p):
            self.g.called = p

    gen = BaseGenerator(IRGraph())
    gen.called = None
    import sys

    try:
        sys.modules["ml_switcheroo_compiler.backends.base_generator"].CodeGeneratorVisitor = DummyVisitor
        import ml_switcheroo_compiler.backends.base_generator as bg

        bg.CodeGeneratorVisitor = DummyVisitor
        walker = bg.IRGraphWalker(gen)
        walker.walk("args")
        assert gen.called == "args"
        assert walker.generator is gen
    finally:
        from ml_switcheroo_compiler.backends.visitor import CodeGeneratorVisitor

        sys.modules["ml_switcheroo_compiler.backends.base_generator"].CodeGeneratorVisitor = CodeGeneratorVisitor


def test_ast_walker_missing_lines():

    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    gen = BaseGenerator(IRGraph())
    # patch out the visitor call to prevent error
    gen.sorted_nodes = []
    from ml_switcheroo_compiler.backends.visitor import CodeGeneratorVisitor

    class DummyVisitor:
        def __init__(self, g):
            self.g = g

        def generate_body(self, p):
            pass

    import sys

    try:
        sys.modules["ml_switcheroo_compiler.backends.base_generator"].CodeGeneratorVisitor = DummyVisitor
        import ml_switcheroo_compiler.backends.base_generator as bg

        bg.CodeGeneratorVisitor = DummyVisitor
        walker = bg.IRGraphWalker(gen)
        walker.walk("args")
        assert walker.generator is gen
    finally:
        sys.modules["ml_switcheroo_compiler.backends.base_generator"].CodeGeneratorVisitor = CodeGeneratorVisitor


def test_class_based_generator_has_base_no_params():
    from ml_switcheroo_compiler.backends.base_generator import ClassBasedGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    class Gen3(ClassBasedGenerator):
        _base_class_name = "Base3"

        def _generate_body(self):
            pass

        def _emit_init_body(self):
            return False

        def get_fallback_prefix(self):
            return "numpy"

        def get_language(self):
            return "python"

    gen = Gen3(IRGraph())
    code = gen.generate()
    assert "super().__init__()" in code
    assert "pass" in code


def test_class_based_generator_has_base_no_params_not_python():
    from ml_switcheroo_compiler.backends.base_generator import ClassBasedGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    class Gen4(ClassBasedGenerator):
        _base_class_name = "Base3"

        def _generate_body(self):
            pass

        def _emit_init_body(self):
            return False

        def get_fallback_prefix(self):
            return "numpy"

        def get_language(self):
            return "cpp"

    gen = Gen4(IRGraph())
    code = gen.generate()
    assert "super().__init__()" in code


def test_ast_walker_proper():
    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator, IRGraphWalker
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    n = IRNode("in", "Input")
    g.inputs = [n]
    g.outputs = []

    class MyGen(BaseGenerator):
        def get_fallback_prefix(self):
            return "numpy"

        def _emit_input_assignment(self, v, n, p, i):
            pass

        def _emit_output_assignment(self, n, i, v):
            pass

        def get_ops_map(self, k):
            return {}

    gen = MyGen(g)
    w = IRGraphWalker(gen)
    w.walk("args")


def test_class_based_generator_has_base_no_params_not_python_2():
    from ml_switcheroo_compiler.backends.base_generator import ClassBasedGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    class Gen4(ClassBasedGenerator):
        _base_class_name = "Base3"

        def _generate_body(self):
            pass

        def _emit_init_body(self):
            return False

        def get_fallback_prefix(self):
            return "numpy"

        def get_language(self):
            return "js"

    gen = Gen4(IRGraph())
    code = gen.generate()
    assert "super().__init__()" in code


def test_class_based_generator_has_base_no_params_is_python():
    from ml_switcheroo_compiler.backends.base_generator import ClassBasedGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    class Gen5(ClassBasedGenerator):
        _base_class_name = "Base3"

        def _generate_body(self):
            pass

        def _emit_init_body(self):
            return False

        def get_fallback_prefix(self):
            return "numpy"

        def get_language(self):
            return "python"

    gen = Gen5(IRGraph())
    code = gen.generate()
    assert "super().__init__()" in code
    assert "pass" in code


def test_class_based_generator_has_base_no_params_hit_457():
    from ml_switcheroo_compiler.backends.base_generator import ClassBasedGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    class Gen6(ClassBasedGenerator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._base_class_name = "BaseClass"

        def _generate_body(self):
            pass

        def _emit_init_body(self):
            return False

        def get_fallback_prefix(self):
            return "numpy"

        def get_language(self):
            return "python"

    gen = Gen6(IRGraph())
    gen.generate()


def test_graph_walker_lines():
    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator, IRGraphWalker
    from ml_switcheroo_compiler.ir.core import IRGraph

    class DummyVisitor:
        def __init__(self, g):
            self.g = g

        def generate_body(self, p):
            pass

    import ml_switcheroo_compiler.backends.base_generator as bg

    old = bg.CodeGeneratorVisitor
    bg.CodeGeneratorVisitor = DummyVisitor
    try:
        gen = BaseGenerator(IRGraph())
        w = IRGraphWalker(gen)
        w.walk()
    finally:
        bg.CodeGeneratorVisitor = old


def test_base_generator_coverage_extras():
    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    class CovGen(BaseGenerator):
        def get_fallback_prefix(self):
            return "numpy"

    gen = CovGen(IRGraph())

    # Hit line 243 by forcing get_ops_map to fetch from registry
    ops = gen.get_ops_map({})
    assert "transpose" in ops or "Transpose" in ops

    # Hit line 304-312 by forcing _format_operation with no format string
    node = IRNode(id="n1", op_type="UnknownCovOp", inputs=[])
    gen.ops_map = {}
    res = gen.generic_visit(node, ["in1"], **{})
    assert "unknowncovop" in res


def test_get_ops_map_fallbacks_coverage():
    import unittest.mock as mock

    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    class MyGen(BaseGenerator):
        def get_fallback_prefix(self):
            return "test_prefix"

    gen = MyGen(IRGraph())

    with mock.patch("ml_switcheroo_compiler.ops.registry.backend_mapping_registry") as mock_reg:
        mock_reg.operations.keys.return_value = ["Dct", "Idct", "Mdct", "InverseMdct", "Frame", "OverlapAndAdd"]

        def mock_get(prefix, name):
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


def test_base_generator_get_fallback_prefix():
    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    assert BaseGenerator(IRGraph()).get_fallback_prefix() == "np"


def test_base_generator_compile_aot():
    import pytest

    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    class CustomGen(BaseGenerator):
        def _compile_aot_impl(self, graph, **kwargs):
            return "compiled_artifact"

    g = IRGraph()
    # Call as classmethod
    res_cls = CustomGen.compile_aot(g)
    assert res_cls == "compiled_artifact"

    # Call on instance with graph=None falling back to self.graph
    inst = CustomGen(g)
    assert inst.compile_aot(None) == "compiled_artifact"

    # Base implementation raises NotImplementedError
    base_inst = BaseGenerator(g)
    with pytest.raises(NotImplementedError, match="AOT compilation is not implemented"):
        base_inst.compile_aot(g)


def test_base_generator_visit_and_visit_opnode() -> None:
    """Test visit_OpNode declarative mapping, ast_template, target_api, and visit dispatch."""
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator, IRNode
    from ml_switcheroo_compiler.ir.core import IRGraph

    g = IRGraph()
    bg = BaseGenerator(g)

    # 1. target_api call string path with kwarg translations (e.g. Argmin in cupy: dim -> axis)
    cupy_bg = BaseGenerator(g)
    cupy_bg.backend_name = "cupy"
    argmin_node = IRNode(
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
    torch_bg = BaseGenerator(g)
    torch_bg.backend_name = "pytorch"
    atan2_node = IRNode(id="n2", op_type="Atan2", attributes={"out_var": "out_val"})
    atan2_res = torch_bg.visit_OpNode(atan2_node, ["in_x", "in_y"])
    assert "out_val = torch.atan2(in_x, in_y)" in atan2_res

    # ast_template with default out variable name from node.id
    atan2_node_default_out = IRNode(id="atan2_id", op_type="Atan2")
    atan2_res_default = torch_bg.visit_OpNode(atan2_node_default_out, ["in_x", "in_y"])
    assert "tensor_0 = torch.atan2(in_x, in_y)" in atan2_res_default

    # 3. target_k is None branch (dropped kwarg)
    mock_op = MagicMock()
    mock_op.ast_template = None
    mock_op.kwarg_map = {"drop_me": None}
    mock_op.target_api = "np.dropped"
    mock_schema = MagicMock()
    mock_schema.operations = {"DroppedOp": mock_op}

    with patch(
        "ml_switcheroo_compiler.backends.mapping_loader.load_backend_mappings",
        return_value=mock_schema,
    ):
        drop_node = IRNode(id="n_drop", op_type="DroppedOp", attributes={"drop_me": 123})
        drop_res = bg.visit_OpNode(drop_node, ["in_val"])
        assert drop_res == "np.dropped(in_val)"

    # 4. Exception in load_backend_mappings falling back to generic_visit
    bad_bg = BaseGenerator(g)
    with patch(
        "ml_switcheroo_compiler.backends.mapping_loader.load_backend_mappings",
        side_effect=RuntimeError("disk error"),
    ):
        unknown_node = IRNode(id="n3", op_type="Add")
        fallback_res = bad_bg.visit_OpNode(unknown_node, ["arg1", "arg2"])
        assert "np.add(arg1, arg2)" in fallback_res

    # 5. Target op not in schema operations falling back to generic_visit
    unmapped_node = IRNode(id="n4", op_type="CustomOpNotInSchema")
    unmapped_res = bg.visit_OpNode(unmapped_node, ["arg0"])
    assert "customopnotinschema" in unmapped_res.lower()

    # 6. visit dispatch: visitor in self.visitors
    class CustomVisitor:
        def visit_VisitorOp(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
            return f"visitor_handled({', '.join(input_vars)})"

    bg.visitors = [CustomVisitor()]
    v_node = IRNode(id="n5", op_type="VisitorOp")
    assert bg.visit(v_node, ["v1", "v2"]) == "visitor_handled(v1, v2)"

    # 7. visit dispatch: method on self (subclass)
    class CustomGenerator(BaseGenerator):
        def visit_SubclassOp(self, node: IRNode, input_vars: list[str], **kwargs: object) -> str:
            return f"subclass_handled({', '.join(input_vars)})"

    cg = CustomGenerator(g)
    cg.visitors = []
    s_node = IRNode(id="n6", op_type="SubclassOp")
    assert cg.visit(s_node, ["s1"]) == "subclass_handled(s1)"

    # 8. visit dispatch: generic_visit fallback and visit_OpNode dispatch
    bg_clean = BaseGenerator(g)
    bg_clean.visitors = []
    fallback_node = IRNode(id="n_fallback", op_type="NonexistentOpFallback")
    assert bg_clean.visit(fallback_node, ["x"]) == "np.nonexistentopfallback(x)"

    # 9. visit dispatch: generic_visit when visit_OpNode is removed
    class NoOpNodeGenerator(BaseGenerator):
        pass

    saved_opnode = BaseGenerator.visit_OpNode
    delattr(BaseGenerator, "visit_OpNode")
    try:
        nog = NoOpNodeGenerator(g)
        g_node = IRNode(id="n7", op_type="GenericOp")
        assert nog.visit(g_node, ["g1"]) == "np.genericop(g1)"
    finally:
        BaseGenerator.visit_OpNode = saved_opnode
