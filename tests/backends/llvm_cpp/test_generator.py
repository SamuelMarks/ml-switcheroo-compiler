from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo_compiler.backends.llvm_cpp.generator import CppGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_llvm_cpp_generator_init():
    g = IRGraph()
    gen = CppGenerator(g)
    assert gen.graph == g
    assert gen.use_simd
    assert gen.use_openmp


def test_llvm_cpp_helpers():
    gen = CppGenerator(IRGraph())
    assert gen._num_elements([2, 3]) == 6
    assert gen._num_elements([]) == 1

    assert gen._get_strides([2, 3]) == [3, 1]
    assert gen._get_strides([5]) == [1]

    node = IRNode("Add", "add_1", [])
    node.shape_metadata = [2, 3]
    assert gen._get_shape(node) == [2, 3]
    node.shape_metadata = 5
    assert gen._get_shape(node) == [5]
    node.shape_metadata = None
    assert gen._get_shape(node) == [1]


def test_llvm_cpp_generate():
    g = IRGraph()
    n_in = IRNode("Input", "in_1", [])
    n_in.op_type = "Input"
    n_in.shape_metadata = [2, 2]
    n_in.attributes = {"buffer_offset": 0, "buffer_size": 16}

    n_const = IRNode("Constant", "c_1", [])
    n_const.op_type = "Constant"
    n_const.attributes = {"value": 5.0}

    n_add = IRNode("Add", "add_1", ["in_1", "c_1"])
    n_add.op_type = "Add"
    n_add.attributes = {"buffer_offset": 16, "buffer_size": 16}

    n_out = IRNode("Output", "out_1", ["add_1"])
    n_out.inputs = ["add_1"]
    n_out.op_type = "Output"

    g.nodes = {"in_1": n_in, "c_1": n_const, "add_1": n_add, "out_1": n_out}
    g.inputs = ["in_1"]
    g.outputs = ["out_1"]
    g.sorted_nodes = [n_in, n_const, n_add, n_out]

    gen = CppGenerator(g)

    with patch("ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider.get_cpp_template", return_value={"body": "body_code"}):
        with patch.dict("ml_switcheroo_compiler.ops.registry._YAML_REGISTRY", {"Add": {"variants": {"llvm_cpp": {"template": "elem"}}}}):
            out = gen.generate()
            assert "body_code" in out
            assert "Input" in out
            assert "Output add_1" in out


def test_llvm_cpp_generate_missing_mapping():
    g = IRGraph()
    n_add = IRNode("MissingOp", "miss_1", [])
    g.nodes = {"miss_1": n_add}
    g.sorted_nodes = [n_add]

    gen = CppGenerator(g)
    out = gen.generate()
    assert "Fallback Unimplemented miss_1" in out


def test_llvm_cpp_generate_visit_methods():
    g = IRGraph()
    n_cond = IRNode("Cond", "cond_1", ["cond"])
    n_cond.inputs = ["cond"]
    n_cond.op_type = "Cond"
    sg = IRGraph()
    sn = IRNode("Add", "a_1", [])
    sg.nodes = {"a_1": sn}
    n_cond.attributes = {"then_branch": sg, "else_branch": sg}

    n_loop = IRNode("WhileLoop", "loop_1", [])
    n_loop.op_type = "WhileLoop"
    n_loop.attributes = {"cond": sg, "body": sg}

    n_conv = IRNode("Conv2D", "conv_1", ["in_1", "in_2"])
    n_conv.inputs = ["in_1", "in_2"]
    n_conv.op_type = "Conv2D"
    n_conv.attributes = {"stride": [1, 1]}

    n_conv2 = IRNode("Conv2D", "conv_2", ["in_1", "in_2"])
    n_conv2.inputs = ["in_1", "in_2"]
    n_conv2.op_type = "Conv2D"
    n_conv2.attributes = {"stride": 1}

    g.nodes = {"cond_1": n_cond, "loop_1": n_loop, "conv_1": n_conv, "conv_2": n_conv2}
    g.sorted_nodes = [n_cond, n_loop, n_conv, n_conv2]

    gen = CppGenerator(g)

    def mock_get_template(name):
        if name == "if_op":
            return {"body": "if ({cond_var}.data[0] > 0.0f) {{\n{then_body}}} else {{\n{else_body}}}\n"}
        elif name == "loop_op":
            return {"body": "while (true) {{\n{cond_body}{loop_body}}}\n"}
        else:
            return {"body": "conv_body"}

    with patch("ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider.get_cpp_template", side_effect=mock_get_template):
        out = gen.generate()
        assert "if" in out
        assert "while" in out
        assert "conv_body" in out


def test_llvm_cpp_compile_and_exec():
    g = IRGraph()
    gen = CppGenerator(g)

    import subprocess

    with patch("subprocess.run") as mock_run:
        with patch("ctypes.CDLL") as mock_cdll:
            mock_lib = MagicMock()
            mock_lib.compute_graph = MagicMock()
            mock_cdll.return_value = mock_lib

            with patch("builtins.open", MagicMock()):
                fn = gen.compile("int compute_graph() {}")
                res = fn()
                assert res == "Execution successful"

                # Test execute
                with patch.object(gen, "generate", return_value="code"):
                    gen.execute(g)

    # Error compilation
    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "cmd", b"out", b"err")):
        with patch("builtins.open", MagicMock()):
            with pytest.raises(RuntimeError, match="Compilation failed: err"):
                gen.compile("code")

    # Error finding func
    with patch("subprocess.run"):
        with patch("builtins.open", MagicMock()):
            with patch("ctypes.CDLL") as mock_cdll:
                mock_cdll.return_value = object()  # No compute_graph
                with pytest.raises(RuntimeError):
                    gen.compile("code")


def test_llvm_cpp_generate_missing_branches():
    gen = CppGenerator(IRGraph())
    out = gen.generate(None)
    assert "compute_graph" in out

    g = IRGraph()
    n_conv = IRNode("Conv2D", "conv_3", ["in_1", "in_2"])
    n_conv.inputs = ["in_1", "in_2"]
    n_conv.op_type = "Conv2D"

    n_in1 = IRNode("Input", "in_1", [])
    n_in1.shape_metadata = [2, 3]  # len < 4

    n_in2 = IRNode("Input", "in_2", [])
    n_in2.shape_metadata = [3, 3]  # len < 4

    n_conv.shape_metadata = [2, 2]  # len < 4

    g.nodes = {"in_1": n_in1, "in_2": n_in2, "conv_3": n_conv}
    g.sorted_nodes = [n_in1, n_in2, n_conv]
    gen2 = CppGenerator(g)
    with patch("ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider.get_cpp_template", return_value={"body": "body"}):
        out2 = gen2.generate()
        assert "body" in out2

    n_add = IRNode("Add", "add_1", ["in_1"])
    n_add.inputs = ["in_1"]
    n_add.op_type = "Add"
    g.nodes["add_1"] = n_add
    g.sorted_nodes.append(n_add)
    with patch.dict("ml_switcheroo_compiler.ops.registry._YAML_REGISTRY", {"Add": {"variants": {"llvm_cpp": {"template": "elem"}}}}):
        with patch("ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider.get_cpp_template", return_value={"body": "elem_body"}):
            out3 = gen2.generate()
            assert "elem_body" in out3


def test_llvm_cpp_generate_missing_branches_2():
    # Trigger graph_to_use = IRGraph()
    gen = CppGenerator(IRGraph())
    gen.graph = None
    gen.generate(None)


from ml_switcheroo_compiler.ir.core import LogicalNode


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
            import pytest

            with pytest.raises(RuntimeError, match="Compilation or load failed"):
                executable = gen.compile("int main() {}")


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
            import pytest

            with pytest.raises(Exception, match="compute_graph failed"):
                executable()


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


def test_cpp_generator_if_missing_branches():
    from ml_switcheroo_compiler.backends.llvm_cpp.generator import CppGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode

    graph = IRGraph()
    gen = CppGenerator(graph)

    node = LogicalNode(id="n1", op_type="If", inputs=["cond"])
    # 134-138: no then_branch, no else_branch
    node.attributes = {}
    gen._visit_node(node, graph)

    # empty branches
    node.attributes = {"then_branch": IRGraph(), "else_branch": IRGraph()}
    gen._visit_node(node, graph)


def test_cpp_generator_while_missing_branches():
    from ml_switcheroo_compiler.backends.llvm_cpp.generator import CppGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode

    graph = IRGraph()
    gen = CppGenerator(graph)

    node = LogicalNode(id="n1", op_type="WhileLoop", inputs=["cond"])
    node.attributes = {}
    gen._visit_node(node, graph)

    node.attributes = {"cond": IRGraph(), "body": IRGraph()}
    gen._visit_node(node, graph)


def test_cpp_generator_conv2d_no_graph_to_use():
    from ml_switcheroo_compiler.backends.llvm_cpp.generator import CppGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode

    graph = IRGraph()
    node = LogicalNode(id="n1", op_type="Conv2D", inputs=["in1", "in2"])
    node.attributes = {"stride": 1}  # line 187 -> 190
    gen = CppGenerator(graph)
    # line 177: graph_to_use = None
    gen.visit_Conv2D(node, None)


def test_cpp_generator_conv2d_shape_4():
    from ml_switcheroo_compiler.backends.llvm_cpp.generator import CppGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode

    graph = IRGraph()
    in1 = LogicalNode(id="in1", op_type="Input")
    in1.shape_metadata = [1, 2, 3, 4]
    in2 = LogicalNode(id="in2", op_type="Input")
    in2.shape_metadata = [1, 2, 3, 4]
    graph.nodes = {"in1": in1, "in2": in2}

    node = LogicalNode(id="n1", op_type="Conv2D", inputs=["in1", "in2"])
    node.shape_metadata = [1, 2, 3, 4]
    node.attributes = {"stride": 1}

    gen = CppGenerator(graph)
    gen.visit_Conv2D(node, graph)


def test_cpp_generate_no_graph():
    gen = CppGenerator(IRGraph())
    gen.graph = None
    res = gen.generate(None)
    assert 'extern "C" void compute_graph()' in res


@patch("subprocess.run")
def test_cpp_compile_load_failure(mock_run):
    gen = CppGenerator(IRGraph())
    mock_run.return_value = MagicMock()

    with patch("ctypes.CDLL", side_effect=Exception("mocked cdld error")):
        with pytest.raises(RuntimeError, match="Compilation or load failed: mocked cdld error"):
            gen.compile("some code")
