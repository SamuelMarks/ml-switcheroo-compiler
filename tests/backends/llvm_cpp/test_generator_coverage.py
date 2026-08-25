from ml_switcheroo_compiler.backends.llvm_cpp.generator import CppGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode


def test_cpp_generator_tanh_explicit():
    graph = IRGraph()
    gen = CppGenerator(graph)
    node = LogicalNode(id="n1", op_type="Tanh", inputs=["in1"])
    gen._visit_node(node, graph)
    assert "std::tanh" in "".join(gen.lines)


def test_cpp_generator_matmul_openmp():
    graph = IRGraph()
    in1 = LogicalNode(id="in1", op_type="Input")
    in1.shape_metadata = [2, 2]
    in2 = LogicalNode(id="in2", op_type="Input")
    in2.shape_metadata = [2, 2]
    graph.nodes = {"in1": in1, "in2": in2}

    node = LogicalNode(id="n1", op_type="MatMul", inputs=["in1", "in2"])
    node.shape_metadata = [2, 2]

    gen = CppGenerator(graph, use_openmp=True)
    gen._visit_node(node, graph)

    # Test without openmp
    gen2 = CppGenerator(graph, use_openmp=False)
    gen2._visit_node(node, graph)


def test_cpp_generator_reduce_min():
    graph = IRGraph()
    gen = CppGenerator(graph)
    node = LogicalNode(id="n1", op_type="ReduceMin", inputs=["in1"])
    gen._visit_node(node, graph)
    assert "std::min" in "".join(gen.lines)

    # test unknown reduce
    node2 = LogicalNode(id="n2", op_type="UnknownReduce", inputs=["in1"])
    gen._visit_node(node2, graph)


def test_cpp_generator_activation_fallback():
    graph = IRGraph()
    gen = CppGenerator(graph)
    node = LogicalNode(id="n1", op_type="UnknownActivation", inputs=["in1"])
    gen._visit_node(node, graph)
    assert "Fallback Unimplemented UnknownActivation" in "".join(gen.lines)


def test_llvm_cpp_strides_num_elem():
    from ml_switcheroo_compiler.backends.llvm_cpp.generator import CppGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    gen = CppGenerator(graph=IRGraph())
    assert gen._num_elements([2, 3]) == 6
    assert gen._get_strides([2, 3, 4]) == [12, 4, 1]


def test_cpp_generator_if_with_else():
    graph = IRGraph()
    gen = CppGenerator(graph)

    then_graph = IRGraph()
    then_graph.nodes = {"n_then": LogicalNode(id="n_then", op_type="Add", inputs=["a", "b"])}

    else_graph = IRGraph()
    else_graph.nodes = {"n_else": LogicalNode(id="n_else", op_type="Sub", inputs=["a", "b"])}

    node = LogicalNode(id="n1", op_type="If", inputs=["cond"])
    node.attributes = {"then_branch": then_graph, "else_branch": else_graph}

    gen._visit_node(node, graph)
    lines = "".join(gen.lines)
    assert "Fallback Unimplemented Sub" in lines


def test_cpp_generator_while_with_body():
    graph = IRGraph()
    gen = CppGenerator(graph)

    cond_graph = IRGraph()
    cond_graph.nodes = {"n_cond": LogicalNode(id="n_cond", op_type="Equal", inputs=["a", "b"])}

    body_graph = IRGraph()
    body_graph.nodes = {"n_body": LogicalNode(id="n_body", op_type="Add", inputs=["a", "b"])}

    node = LogicalNode(id="n1", op_type="WhileLoop", inputs=["cond"])
    node.attributes = {"cond": cond_graph, "body": body_graph}

    gen._visit_node(node, graph)
    lines = "".join(gen.lines)
    assert "Fallback Unimplemented Equal" in lines


def test_cpp_generator_arena_buffer():
    from ml_switcheroo_compiler.backends.llvm_cpp.generator import CppGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode

    graph = IRGraph()
    node = LogicalNode(id="n1", op_type="Input")
    node.attributes = {"buffer_offset": 10, "buffer_size": 15}
    graph.nodes = {"n1": node}

    gen = CppGenerator(graph)
    code = gen.generate(graph)
    assert "Allocate global arena buffer of size 25 bytes" in code


def test_cpp_generator_conv2d_low_rank():
    from ml_switcheroo_compiler.backends.llvm_cpp.generator import CppGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode

    graph = IRGraph()
    in1 = LogicalNode(id="in1", op_type="Input")
    in1.shape_metadata = [2]
    in2 = LogicalNode(id="in2", op_type="Input")
    in2.shape_metadata = [2, 3]
    graph.nodes = {"in1": in1, "in2": in2}

    node = LogicalNode(id="n1", op_type="Conv2D", inputs=["in1", "in2"])
    node.shape_metadata = [1]

    gen = CppGenerator(graph)
    gen._visit_node(node, graph)


def test_cpp_generator_compile_subprocess_error():
    import subprocess
    from unittest.mock import patch

    import pytest

    from ml_switcheroo_compiler.backends.llvm_cpp.generator import CppGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    gen = CppGenerator(graph=IRGraph())

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, ["clang++"], stderr=b"clang++ error")
        with pytest.raises(RuntimeError, match="Compilation failed: clang"):
            gen.compile("int main() {}")


def test_cpp_generator_compile_cdll_error():
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.llvm_cpp.generator import CppGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    gen = CppGenerator(graph=IRGraph())

    with patch("subprocess.run"):
        with patch("ctypes.CDLL", side_effect=Exception("CDLL failed")):
            executable = gen.compile("int main() {}")
            assert executable() == "Execution simulated (compiled)"


def test_cpp_generator_compile_compute_func_error():
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.backends.llvm_cpp.generator import CppGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    gen = CppGenerator(graph=IRGraph())

    with patch("subprocess.run"):
        mock_lib = MagicMock()
        mock_lib.compute_graph.side_effect = Exception("compute_graph failed")
        with patch("ctypes.CDLL", return_value=mock_lib):
            executable = gen.compile("int main() {}")
            assert executable() == "Execution simulated (compiled)"


def test_cpp_generator_compile_execution_successful():
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.backends.llvm_cpp.generator import CppGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    gen = CppGenerator(graph=IRGraph())

    with patch("subprocess.run"):
        mock_lib = MagicMock()
        mock_lib.compute_graph.return_value = 0  # non-string
        with patch("ctypes.CDLL", return_value=mock_lib):
            executable = gen.compile("int main() {}")
            assert executable() == "Execution successful"
