"""Test module."""

from ml_switcheroo_compiler.backends.visitor import CodeGeneratorVisitor


class DummyGenerator:
    def __init__(self):
        self.sorted_nodes = []
        self.var_names = {"in1": "var_in1"}
        self.input_idx = 0
        self._output_returns = []
        self.lines = []

    def _emit_body_return(self, returns):
        self.lines.append(f"return {returns}")

    def emit_constant(self, node):
        return "42.0"

    def assign_var_name(self, id, prefix="tensor"):
        return f"{prefix}_1"

    def _emit_constant_assignment(self, var, val):
        self.lines.append(f"{var} = {val}")

    def _emit_input_assignment(self, var, node, pref, idx):
        self.lines.append(f"{var} = {pref}[{idx}]")

    def _emit_output_assignment(self, node, invars, returns):
        self._output_returns.append(returns)

    def visit(self, node, invars, **kwargs):
        return "visited"

    def add_line(self, line):
        self.lines.append(line)


class DummyNode:
    def __init__(self, op_type, id, inputs=None):
        self.op_type = op_type
        self.id = id
        self.inputs = inputs or []
        self.attributes = {}
        self.shape_metadata = ()


def test_visitor():
    gen = DummyGenerator()
    vis = CodeGeneratorVisitor(gen)

    n_const = DummyNode("Constant", "c1")
    n_in = DummyNode("Input", "i1")
    n_out = DummyNode("Output", "o1", inputs=["in1"])
    n_op = DummyNode("Op", "op1", inputs=["in1"])
    n_op.attributes = {"stream_id": 1, "async_check": True}
    n_op.shape_metadata = (1,)

    gen.sorted_nodes = [n_const, n_in, n_op, n_out]

    vis.generate_body("args")

    assert "const_1 = 42.0" in gen.lines
    assert "input_1 = args[0]" in gen.lines
    assert "tensor_1 = visited" in gen.lines
    assert "return ['var_in1']" in gen.lines


def test_visitor_compute_node_branches():
    gen = DummyGenerator()
    vis = CodeGeneratorVisitor(gen)

    n_op = DummyNode("Op", "op2")
    n_op.attributes = {}
    n_op.shape_metadata = ()
    vis.handle_compute_node(n_op)
    assert "tensor_1 = visited" in gen.lines
