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
