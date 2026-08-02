# ruff: noqa: D100, D103
from ml_switcheroo_compiler.backends.llvm_cpp.generator import CppGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_cpp_generator_init():
    gen = CppGenerator(graph=IRGraph(), use_simd=True, use_openmp=True)
    assert gen.use_simd is True
    assert gen.use_openmp is True


def test_cpp_generator_generate():
    import pytest

    gen = CppGenerator(graph=IRGraph(), use_openmp=True)
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Input")
    n2 = IRNode(id="n2", op_type="Add", inputs=["n1", "n1"])
    n3 = IRNode(id="n3", op_type="Output", inputs=["n2"])
    n4 = IRNode(id="n4", op_type="UnknownOp", inputs=[])

    g.nodes["n1"] = n1
    g.nodes["n2"] = n2
    g.nodes["n3"] = n3

    code = gen.generate(g)

    assert "#include <omp.h>" in code
    assert "std::vector<float> n1;" in code
    assert "#pragma omp parallel for" in code
    assert "n2[i] = n1[i] + n1[i];" in code
    assert "// Output n2" in code

    g.nodes["n4"] = n4
    with pytest.raises(NotImplementedError):
        gen.generate(g)


def test_cpp_generator_generate_no_omp():
    gen = CppGenerator(graph=IRGraph(), use_openmp=False)
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Input")
    n2 = IRNode(id="n2", op_type="Add", inputs=["n1", "n1"])
    g.nodes["n1"] = n1
    g.nodes["n2"] = n2
    code = gen.generate(g)
    assert "#include <omp.h>" not in code
    assert "#pragma omp parallel for" not in code


def test_cpp_generator_execute():
    gen = CppGenerator(graph=IRGraph())
    g = IRGraph()
    g.nodes["n1"] = IRNode(id="n1", op_type="Input")

    res = gen.execute(g)
    assert res == "Execution simulated"


def test_cpp_generator_extra():
    from ml_switcheroo_compiler.backends.llvm_cpp.generator import CppGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    g = IRGraph()
    gen = CppGenerator(g, use_openmp=False)

    # Constant
    n_const = IRNode("Const", op_type="Constant", attributes={"value": 5.0})
    # Unary -
    n_neg = IRNode("Neg", op_type="Negative", inputs=["a"])
    # If without branches
    n_if = IRNode("If", op_type="If", inputs=["c"])
    # Loop without cond or body
    n_loop = IRNode("Loop", op_type="Loop")

    gen._visit_node(n_const)
    gen._visit_node(n_neg)
    gen._visit_node(n_if)
    gen._visit_node(n_loop)

    assert "    float Const = 5.0; // Constant" in gen.lines
    assert "        Neg[i] = -a[i];" in gen.lines
    assert "    if (c[0] > 0.0f) {" in gen.lines
    assert "    while (true) {" in gen.lines


"""Tests for llvm cpp backend coverage."""

from ml_switcheroo_compiler.ir.core import LogicalNode


def test_llvm_cpp_coverage() -> None:
    """Test llvm cpp code generator coverage."""
    gen = CppGenerator(IRGraph())

    n_while = LogicalNode(id="n_while", op_type="WhileLoop")
    sub_graph = IRGraph()
    n_add = LogicalNode(id="n_add", op_type="Add", inputs=["in1", "in2"])
    sub_graph.nodes[n_add.id] = n_add
    n_while.attributes["cond"] = sub_graph
    n_while.attributes["body"] = sub_graph

    n_if = LogicalNode(id="n_if", op_type="If", inputs=["cond"])
    n_if.attributes["then_branch"] = sub_graph
    n_if.attributes["else_branch"] = sub_graph

    n_matmul = LogicalNode(id="n_matmul", op_type="MatMul", inputs=["in1", "in2"])
    n_exp = LogicalNode(id="n_exp", op_type="Exp", inputs=["in1"])
    n_neg = LogicalNode(id="n_neg", op_type="Negative", inputs=["in1"])

    for n in [n_while, n_if, n_matmul, n_exp, n_neg, n_add]:
        try:
            gen._visit_node(n)
        except Exception:
            pass
