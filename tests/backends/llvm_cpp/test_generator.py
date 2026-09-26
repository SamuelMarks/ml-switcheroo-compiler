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
    else_graph.nodes = {"n_else": LogicalNode(id="n_else", op_type="UnsupportedOp", inputs=["a", "b"])}

    node = LogicalNode(id="n1", op_type="If", inputs=["cond"])
    node.attributes = {"then_branch": then_graph, "else_branch": else_graph}

    gen._visit_node(node, graph)
    lines = "".join(gen.lines)
    assert "Fallback Unimplemented UnsupportedOp" in lines


def test_cpp_generator_while_with_body():
    graph = IRGraph()
    gen = CppGenerator(graph)

    cond_graph = IRGraph()
    cond_graph.nodes = {"n_cond": LogicalNode(id="n_cond", op_type="UnsupportedOp", inputs=["a", "b"])}

    body_graph = IRGraph()
    body_graph.nodes = {"n_body": LogicalNode(id="n_body", op_type="Add", inputs=["a", "b"])}

    node = LogicalNode(id="n1", op_type="WhileLoop", inputs=["cond"])
    node.attributes = {"cond": cond_graph, "body": body_graph}

    gen._visit_node(node, graph)
    lines = "".join(gen.lines)
    assert "Fallback Unimplemented UnsupportedOp" in lines


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
    assert 'extern "C" void compute_graph(' in res


@patch("subprocess.run")
def test_cpp_compile_load_failure(mock_run):
    gen = CppGenerator(IRGraph())
    mock_run.return_value = MagicMock()

    with patch("ctypes.CDLL", side_effect=Exception("mocked cdld error")):
        with pytest.raises(RuntimeError, match="Compilation or load failed: mocked cdld error"):
            gen.compile("some code")


def test_llvm_cpp_runner_and_compile_aot():
    """Test LLVMCPPRunner compiler selection and compile_aot method."""
    from ml_switcheroo_compiler.backends.llvm_cpp.generator import LLVMCPPRunner

    # Explicit compiler
    runner_custom = LLVMCPPRunner(compiler="custom-clang++")
    assert runner_custom.compiler == "custom-clang++"

    # Fallback to g++ when clang++ not found
    with patch("shutil.which", side_effect=lambda x: "/usr/bin/g++" if x == "g++" else None):
        runner_gpp = LLVMCPPRunner()
        assert runner_gpp.compiler == "g++"

    # Fallback to default clang++ when neither found
    with patch("shutil.which", return_value=None):
        runner_default = LLVMCPPRunner()
        assert runner_default.compiler == "clang++"

    # compile_aot with and without compiler kwargs
    gen = CppGenerator(IRGraph())
    with patch.object(LLVMCPPRunner, "compile_and_load", return_value=lambda: "success"):
        res_fn1 = gen.compile_aot(IRGraph(), compiler="g++")
        assert res_fn1() == "success"

        res_fn2 = gen.compile_aot(IRGraph())
        assert res_fn2() == "success"


def test_cpp_generator_strict_and_genuine_math():
    """Verify strict mode raises UnimplementedMathError and genuine math expressions are generated."""
    from ml_switcheroo_compiler.core.errors import UnimplementedMathError

    g = IRGraph()
    n_sin = IRNode("Sin", "s_1", ["in1"])
    n_sin.inputs = ["in1"]
    n_sin.op_type = "Sin"
    g.nodes = {"s_1": n_sin}
    g.sorted_nodes = [n_sin]

    gen = CppGenerator(g)
    code = gen.generate()
    assert "std::sin(in0_val)" in code

    g_unk = IRGraph()
    n_unk = IRNode("UnsupportedMathOp", "u_1", ["in1"])
    n_unk.inputs = ["in1"]
    n_unk.op_type = "UnsupportedMathOp"
    g_unk.nodes = {"u_1": n_unk}
    g_unk.sorted_nodes = [n_unk]

    gen_strict = CppGenerator(g_unk, strict=True)
    with pytest.raises(UnimplementedMathError, match="does not support operation: UnsupportedMathOp"):
        gen_strict.generate()


def test_cpp_generator_genuine_expr_override():
    """Verify that genuine expressions override placeholder scalar_expr in C++ generator."""
    g = IRGraph()
    gen = CppGenerator(g)
    node = IRNode("Relu", "relu_1", ["in_1"])
    node.op_type = "Relu"
    node.inputs = ["in_1"]
    node.shape_metadata = [2, 2]
    node.attributes = {"buffer_offset": 0, "buffer_size": 16}
    g.nodes = {"relu_1": node}

    mock_registry = {
        "Relu": {
            "variants": {
                "llvm_cpp": {
                    "template": "unary",
                    "scalar_expr": "in0_val",
                }
            }
        }
    }
    with patch("ml_switcheroo_compiler.ops.registry._YAML_REGISTRY", mock_registry):
        with patch("ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider.get_cpp_template", return_value={"body": "{scalar_expr}"}):
            gen._visit_node(node)
            assert any("std::max(0.0f, in0_val)" in l for l in gen.lines)

    # Cover line 342: mapping is empty in _YAML_REGISTRY but decl_op exists in get_cpp_operation
    gen2 = CppGenerator(g)
    with patch("ml_switcheroo_compiler.ops.registry._YAML_REGISTRY", {}):
        with patch("ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider.get_cpp_template", return_value={"body": "{scalar_expr}"}):
            gen2._visit_node(node)
            assert any("std::max(0.0f, in0_val)" in l for l in gen2.lines)


def test_cpp_generator_missing_output_and_executable_arg_counts():
    """Verify branch 138->137 (missing output node) and lines 500, 502, 504 (executable arg branches)."""
    # 1. Output in graph.nodes followed by output not in graph.nodes (branch 138->137)
    g = IRGraph()
    n = IRNode("in1", "Input", [])
    n.shape_metadata = [2, 2]
    g.nodes["in1"] = n
    g.outputs = ["in1", "missing_node"]
    gen = CppGenerator(g)
    code = gen.generate()
    assert "void compute_graph" in code

    # 2. compile_aot executable arg branches: len(args) == 2, 1, 0
    mock_compute = MagicMock()
    mock_dll = MagicMock()
    mock_dll.compute_graph = mock_compute

    with patch("subprocess.run"):
        with patch("ctypes.CDLL", return_value=mock_dll):
            exec_fn = gen.compile_aot(g)
            # len(args) == 3
            assert exec_fn("in", "out", "shapes") == "Execution successful"
            mock_compute.assert_called_with("in", "out", "shapes")

            # len(args) == 2 (line 500)
            assert exec_fn("in", "out") == "Execution successful"
            mock_compute.assert_called_with("in", "out", None)

            # len(args) == 1 (line 502)
            assert exec_fn("in") == "Execution successful"
            mock_compute.assert_called_with("in", None, None)

            # len(args) == 0 (line 504)
            assert exec_fn() == "Execution successful"
            mock_compute.assert_called_with(None, None, None)


def test_cpp_generator_multi_output_indexed_routing() -> None:
    """Verify indexed routing from multi-output nodes via LogicalEdge."""
    g = IRGraph(name="multi_out_graph")
    n_in = IRNode(id="x", op_type="Input", inputs=[])
    n_in.shape_metadata = [2, 2]
    n_split = IRNode(id="split_op", op_type="Relu", inputs=["x"], outputs=["split_out_0", "split_out_1"])
    n_split.shape_metadata = [2, 2]
    n_add = IRNode(id="add_op", op_type="Add", inputs=["split_out_0", "split_out_1"])
    n_add.shape_metadata = [2, 2]

    g.nodes = {"x": n_in, "split_op": n_split, "add_op": n_add}
    g.outputs = ["add_op"]

    gen = CppGenerator(g)
    code = gen.generate(g)
    assert "split_out_1" in code
    assert any(e.source_idx == 1 for e in g.edges)


def test_mlir_dialect_validation() -> None:
    """Verify MLIR dialect operation validation against canonical MLIR_REGISTRY."""
    from ml_switcheroo_compiler.backends.llvm_cpp.generator import (
        get_supported_mlir_dialects,
        validate_mlir_operation,
    )

    dialects = get_supported_mlir_dialects()
    assert "arith" in dialects
    assert "math" in dialects
    assert "tensor" in dialects
    assert "linalg" in dialects
    assert "scf" in dialects

    assert validate_mlir_operation("arith.addi") is True
    assert validate_mlir_operation("math.exp") is True
    assert validate_mlir_operation("unknown_dialect.fake_op") is False


def test_cpp_generator_unmapped_op_strict_and_polyfill() -> None:
    """Verify strict error reporting and safe zero-init polyfill for unmapped operations in LLVMCPPGenerator."""
    import pytest

    from ml_switcheroo_compiler.backends.llvm_cpp.generator import CppGenerator, LLVMCPPGenerator
    from ml_switcheroo_compiler.core.errors import UnimplementedMathError

    assert LLVMCPPGenerator is CppGenerator

    g = IRGraph()
    unmapped_node = IRNode(id="bad_op", op_type="NonExistentCustomOp", inputs=[])
    unmapped_node.shape_metadata = [2, 3]
    g.nodes = {"bad_op": unmapped_node}

    # Strict mode must raise UnimplementedMathError
    gen_strict = CppGenerator(g, strict=True)
    with pytest.raises(UnimplementedMathError, match="C\\+\\+ code generator does not support operation"):
        gen_strict.generate(g)

    # Non-strict mode must emit UserWarning and runtime polyfill loop
    gen_lenient = CppGenerator(g, strict=False)
    with pytest.warns(UserWarning, match="Emitting safe runtime polyfill"):
        code = gen_lenient.generate(g)

    assert "NDArrayView<float> bad_op({2,3}); // Fallback Unimplemented NonExistentCustomOp" in code
    assert "for(size_t i = 0; i < bad_op.size(); ++i)" in code
    assert "1.0f" in code


def test_llvm_cpp_validate_mlir_import_error_and_edge_target_idx1() -> None:
    """Verify validate_mlir_operation on ImportError and CppGenerator incoming edge with target_idx=1."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.llvm_cpp.generator import CppGenerator, validate_mlir_operation

    # 1. validate_mlir_operation ImportError
    with patch("ml_switcheroo_compiler.backends.llvm_cpp.generator.get_supported_mlir_dialects", side_effect=ImportError("mock")):
        assert validate_mlir_operation("custom.op") is False

    # 2. CppGenerator edge target_idx=0 and target_idx=1 with source_idx>0
    g = IRGraph()
    prod = IRNode(id="prod", op_type="Split", inputs=[], outputs=["out0", "out1"], shape_metadata=[2, 2])
    n_add = IRNode(id="add_node", op_type="Add", inputs=["out1", "out1"], shape_metadata=[2, 2])
    g.nodes = {"prod": prod, "add_node": n_add}
    g.inputs = []
    g.outputs = ["add_node"]

    gen = CppGenerator(g)
    code = gen.generate(g)
    assert "out1" in code


def test_llvm_cpp_execution_verification() -> None:
    """Verify execution of compiled C++ binaries matching NumPy eager numerical results."""
    import ctypes
    import shutil

    import numpy as np

    compiler = "clang++" if shutil.which("clang++") else "g++" if shutil.which("g++") else None
    if not compiler:
        pytest.skip("No C++ compiler (clang++ or g++) found.")

    from ml_switcheroo_compiler.backends.llvm_cpp.generator import LLVMCPPRunner

    runner = LLVMCPPRunner(compiler=compiler)

    # 1. LayerNorm
    g_ln = IRGraph(name="test_layernorm")
    in_ln = IRNode(id="in0", op_type="Input", shape_metadata=(2, 4))
    node_ln = IRNode(id="out0", op_type="LayerNorm", inputs=["in0"], shape_metadata=(2, 4))
    g_ln.nodes = {"in0": in_ln, "out0": node_ln}
    g_ln.inputs = ["in0"]
    g_ln.outputs = ["out0"]
    g_ln.sorted_nodes = [in_ln, node_ln]

    code_ln = CppGenerator(g_ln).generate(g_ln)
    fn_ln = runner.compile_and_load(code_ln)
    x_ln = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    y_ln = np.zeros_like(x_ln)
    fn_ln((ctypes.c_void_p * 1)(x_ln.ctypes.data), (ctypes.c_void_p * 1)(y_ln.ctypes.data))
    mean = x_ln.mean(axis=-1, keepdims=True)
    var = x_ln.var(axis=-1, keepdims=True)
    ref_ln = (x_ln - mean) / np.sqrt(var + 1e-5)
    assert np.allclose(y_ln, ref_ln, atol=1e-4)
    assert not np.all(y_ln == 0.0)

    # 2. RMSNorm
    g_rms = IRGraph(name="test_rmsnorm")
    in_rms = IRNode(id="in0", op_type="Input", shape_metadata=(2, 4))
    node_rms = IRNode(id="out0", op_type="RMSNorm", inputs=["in0"], shape_metadata=(2, 4))
    g_rms.nodes = {"in0": in_rms, "out0": node_rms}
    g_rms.inputs = ["in0"]
    g_rms.outputs = ["out0"]
    g_rms.sorted_nodes = [in_rms, node_rms]

    code_rms = CppGenerator(g_rms).generate(g_rms)
    fn_rms = runner.compile_and_load(code_rms)
    y_rms = np.zeros_like(x_ln)
    fn_rms((ctypes.c_void_p * 1)(x_ln.ctypes.data), (ctypes.c_void_p * 1)(y_rms.ctypes.data))
    ref_rms = x_ln / np.sqrt(np.mean(x_ln**2, axis=-1, keepdims=True) + 1e-5)
    assert np.allclose(y_rms, ref_rms, atol=1e-4)
    assert not np.all(y_rms == 0.0)

    # 3. Conv3D
    g_conv3d = IRGraph(name="test_conv3d")
    in_x3d = IRNode(id="in0", op_type="Input", shape_metadata=(1, 1, 3, 3, 3))
    in_w3d = IRNode(id="in1", op_type="Input", shape_metadata=(1, 1, 2, 2, 2))
    node_c3d = IRNode(id="out0", op_type="Conv3D", inputs=["in0", "in1"], shape_metadata=(1, 1, 2, 2, 2))
    g_conv3d.nodes = {"in0": in_x3d, "in1": in_w3d, "out0": node_c3d}
    g_conv3d.inputs = ["in0", "in1"]
    g_conv3d.outputs = ["out0"]
    g_conv3d.sorted_nodes = [in_x3d, in_w3d, node_c3d]

    code_c3d = CppGenerator(g_conv3d).generate(g_conv3d)
    fn_c3d = runner.compile_and_load(code_c3d)
    x_3d = np.ones((1, 1, 3, 3, 3), dtype=np.float32)
    w_3d = np.ones((1, 1, 2, 2, 2), dtype=np.float32)
    y_3d = np.zeros((1, 1, 2, 2, 2), dtype=np.float32)
    in_3d_ptrs = (ctypes.c_void_p * 2)(x_3d.ctypes.data, w_3d.ctypes.data)
    fn_c3d(in_3d_ptrs, (ctypes.c_void_p * 1)(y_3d.ctypes.data))
    assert np.allclose(y_3d, 8.0, atol=1e-4)
    assert not np.all(y_3d == 0.0)

    # 4. Slice
    g_sl = IRGraph(name="test_slice")
    in_sl = IRNode(id="in0", op_type="Input", shape_metadata=(4, 4))
    node_sl = IRNode(id="out0", op_type="Slice", inputs=["in0"], shape_metadata=(2, 2), attributes={"starts": [1, 1], "strides": [1, 1]})
    g_sl.nodes = {"in0": in_sl, "out0": node_sl}
    g_sl.inputs = ["in0"]
    g_sl.outputs = ["out0"]
    g_sl.sorted_nodes = [in_sl, node_sl]

    code_sl = CppGenerator(g_sl).generate(g_sl)
    fn_sl = runner.compile_and_load(code_sl)
    x_sl = np.arange(16, dtype=np.float32).reshape(4, 4)
    y_sl = np.zeros((2, 2), dtype=np.float32)
    fn_sl((ctypes.c_void_p * 1)(x_sl.ctypes.data), (ctypes.c_void_p * 1)(y_sl.ctypes.data))
    assert np.allclose(y_sl, x_sl[1:3, 1:3], atol=1e-4)
    assert not np.all(y_sl == 0.0)

    # 5. Pad
    g_pad = IRGraph(name="test_pad")
    in_pad = IRNode(id="in0", op_type="Input", shape_metadata=(2, 2))
    node_pad = IRNode(id="out0", op_type="Pad", inputs=["in0"], shape_metadata=(4, 4), attributes={"paddings": [1, 1, 1, 1], "constant_value": 7.0})
    g_pad.nodes = {"in0": in_pad, "out0": node_pad}
    g_pad.inputs = ["in0"]
    g_pad.outputs = ["out0"]
    g_pad.sorted_nodes = [in_pad, node_pad]

    code_pad = CppGenerator(g_pad).generate(g_pad)
    fn_pad = runner.compile_and_load(code_pad)
    x_pad = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y_pad = np.zeros((4, 4), dtype=np.float32)
    fn_pad((ctypes.c_void_p * 1)(x_pad.ctypes.data), (ctypes.c_void_p * 1)(y_pad.ctypes.data))
    ref_pad = np.full((4, 4), 7.0, dtype=np.float32)
    ref_pad[1:3, 1:3] = x_pad
    assert np.allclose(y_pad, ref_pad, atol=1e-4)
    assert not np.all(y_pad == 0.0)


def test_cpp_generator_additional_coverage() -> None:
    """Verify missing branches in visit_Conv3D, visit_MaxPool3D, visit_AvgPool3D, visit_BatchNorm, visit_Pad, visit_GatherND, and visit_ScatterND."""
    # 1. visit_Conv3D with graph_to_use having 3D shape nodes (< 5), stride as int, and stride as 1-element list
    g_3d = IRGraph(name="test_3d")
    in_3d_0 = IRNode(id="in0", op_type="Input", shape_metadata=(4, 4, 4))
    in_3d_1 = IRNode(id="in1", op_type="Input", shape_metadata=(4, 4, 4))
    node_conv_3d = IRNode(id="c_out", op_type="Conv3D", inputs=["in0", "in1"], shape_metadata=(4, 4, 4), attributes={"stride": 2})
    g_3d.nodes = {"in0": in_3d_0, "in1": in_3d_1, "c_out": node_conv_3d}

    gen = CppGenerator(g_3d)
    gen.visit_Conv3D(node_conv_3d, graph_to_use=g_3d)
    gen.visit_Conv3D(node_conv_3d, graph_to_use=None)
    assert len(gen.lines) > 0

    gen_stride1 = CppGenerator(g_3d)
    node_conv_stride1 = IRNode(id="c_out", op_type="Conv3D", inputs=["in0", "in1"], shape_metadata=(4, 4, 4), attributes={"stride": [1]})
    gen_stride1.visit_Conv3D(node_conv_stride1, graph_to_use=g_3d)

    # 2. visit_MaxPool3D with ksize as 1-element list [2], stride as 1-element list [1], and ksize/stride as int
    node_mp = IRNode(id="mp_out", op_type="MaxPool3D", inputs=["in0"], shape_metadata=(4, 4, 4), attributes={"kernel_size": [2], "stride": [1]})
    g_3d.nodes["mp_out"] = node_mp
    gen_maxpool = CppGenerator(g_3d)
    gen_maxpool.visit_MaxPool3D(node_mp, graph_to_use=g_3d)
    gen_maxpool._visit_node(node_mp, graph_to_use=g_3d)

    node_mp_scalar = IRNode(id="mp_scalar", op_type="MaxPool3D", inputs=["in0"], shape_metadata=(4, 4, 4), attributes={"kernel_size": 2, "stride": 1})
    gen_maxpool.visit_MaxPool3D(node_mp_scalar, graph_to_use=None)

    node_mp_2elem = IRNode(id="mp_2elem", op_type="MaxPool3D", inputs=["in0"], shape_metadata=(4, 4, 4), attributes={"kernel_size": [2, 2], "stride": [1, 1]})
    gen_maxpool.visit_MaxPool3D(node_mp_2elem, graph_to_use=None)

    node_mp_5d = IRNode(id="mp_5d", op_type="MaxPool3D", inputs=["in0"], shape_metadata=(1, 1, 4, 4, 4), attributes={"kernel_size": [2, 2, 2]})
    gen_maxpool.visit_MaxPool3D(node_mp_5d, graph_to_use=None)

    # 3. visit_AvgPool3D with ksize as 1-element list [2], stride as 1-element list [1], and ksize/stride as int
    node_ap = IRNode(id="ap_out", op_type="AvgPool3D", inputs=["in0"], shape_metadata=(4, 4, 4), attributes={"kernel_size": [2], "stride": [1]})
    g_3d.nodes["ap_out"] = node_ap
    gen_avgpool = CppGenerator(g_3d)
    gen_avgpool.visit_AvgPool3D(node_ap, graph_to_use=g_3d)
    gen_avgpool._visit_node(node_ap, graph_to_use=g_3d)

    node_ap_scalar = IRNode(id="ap_scalar", op_type="AvgPool3D", inputs=["in0"], shape_metadata=(4, 4, 4), attributes={"kernel_size": 2, "stride": 1})
    gen_avgpool.visit_AvgPool3D(node_ap_scalar, graph_to_use=None)

    node_ap_2elem = IRNode(id="ap_2elem", op_type="AvgPool3D", inputs=["in0"], shape_metadata=(4, 4, 4), attributes={"kernel_size": [2, 2], "stride": [1, 1]})
    gen_avgpool.visit_AvgPool3D(node_ap_2elem, graph_to_use=None)

    node_ap_5d = IRNode(id="ap_5d", op_type="AvgPool3D", inputs=["in0"], shape_metadata=(1, 1, 4, 4, 4), attributes={"kernel_size": [2, 2, 2]})
    gen_avgpool.visit_AvgPool3D(node_ap_5d, graph_to_use=None)

    # 4. visit_BatchNorm with 4D shape so spatial *= dim loop runs, and dispatch via _visit_node
    empty_g = IRGraph()
    gen_bn = CppGenerator(empty_g)
    node_bn = IRNode(id="bn_out", op_type="BatchNorm", inputs=["in0", "mean", "var", "gamma", "beta"], shape_metadata=(1, 3, 8, 8))
    gen_bn.visit_BatchNorm(node_bn, graph_to_use=None)
    gen_bn._visit_node(node_bn, graph_to_use=None)

    # 5. visit_Pad with paddings as int, and paddings as 1-element list
    gen_pad_int = CppGenerator(empty_g)
    node_pad_int = IRNode(id="pad_out", op_type="Pad", inputs=["in0"], shape_metadata=(4, 4), attributes={"paddings": 1})
    gen_pad_int.visit_Pad(node_pad_int, graph_to_use=None)

    gen_pad_list = CppGenerator(empty_g)
    node_pad_list = IRNode(id="pad_out", op_type="Pad", inputs=["in0"], shape_metadata=(4, 4), attributes={"paddings": [1]})
    gen_pad_list.visit_Pad(node_pad_list, graph_to_use=None)

    # 6. GatherND and ScatterND via _visit_node and directly
    g = IRGraph(name="test_gather_scatter")
    in_data = IRNode(id="in0", op_type="Input", shape_metadata=(4, 4))
    in_idx = IRNode(id="in1", op_type="Input", shape_metadata=(2, 1))
    in_updates = IRNode(id="in2", op_type="Input", shape_metadata=(2, 4))
    node_gather = IRNode(id="gather_out", op_type="gather_nd", inputs=["in0", "in1"], shape_metadata=(2, 4))
    node_scatter = IRNode(id="scatter_out", op_type="scatter_nd", inputs=["in0", "in1", "in2"], shape_metadata=(4, 4))
    g.nodes = {"in0": in_data, "in1": in_idx, "in2": in_updates, "gather_out": node_gather, "scatter_out": node_scatter}

    gen_gs = CppGenerator(g)
    gen_gs._visit_node(node_gather, g)
    gen_gs._visit_node(node_scatter, g)
    assert len(gen_gs.lines) > 0

    # 7. init_val branch (line 868) where init_val is numeric without "f" or "INFINITY"
    node_init = IRNode(id="init_op", op_type="CustomInitOp", inputs=["in0"], shape_metadata=(4, 4))
    gen_init = CppGenerator(empty_g)
    with (
        patch.dict("ml_switcheroo_compiler.ops.registry._YAML_REGISTRY", {"CustomInitOp": {"variants": {"llvm_cpp": {"template": "custom_init_tpl", "init_val": "0.0"}}}}),
        patch("ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider.get_cpp_template", return_value={"body": "float val = {init_val};"}),
    ):
        gen_init._visit_node(node_init, empty_g)


def test_cpp_generator_new_visitors_and_providers() -> None:
    """Verify code generation and coverage for new C++ visitors, operators, and dynamic providers."""
    from ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider import (
        get_cpp_operation,
        get_cpp_template,
        synthesize_cpp_operation,
    )

    # 1. Provider functions edge cases
    assert get_cpp_template("non_existent_template_xyz") == {}
    s_unary = synthesize_cpp_operation("custom_unary", num_inputs=1)
    assert s_unary.template == "unary"
    assert s_unary.scalar_expr == "in0_val"

    s_binary = synthesize_cpp_operation("custom_binary", num_inputs=2)
    assert s_binary.template == "binary"
    assert s_binary.scalar_expr == "in0_val + in1_val"

    s_ternary = synthesize_cpp_operation("custom_ternary", num_inputs=3)
    assert s_ternary.template == "ternary"

    s_custom_expr = synthesize_cpp_operation("custom_op", scalar_expr="std::sin(in0_val)")
    assert s_custom_expr.scalar_expr == "std::sin(in0_val)"

    s_where = synthesize_cpp_operation("where", num_inputs=3)
    assert s_where.template == "ternary"

    s_mse = synthesize_cpp_operation("mse_loss", num_inputs=2)
    assert s_mse.template == "binary"

    op_known = get_cpp_operation("add")
    assert op_known is not None

    op_synth_allowed = get_cpp_operation("unmapped_op_synth", num_inputs=1, allow_synth=True)
    assert op_synth_allowed is not None

    op_synth_disallowed = get_cpp_operation("unmapped_op_no_synth", num_inputs=1, allow_synth=False)
    assert op_synth_disallowed is None

    # 2. Visitors: MaxPool2D and AvgPool2D (2D and 4D shapes, kernel/stride as int and list, is_global)
    g = IRGraph(name="test_pool")
    in_node = IRNode(id="in0", op_type="Input", shape_metadata=(1, 3, 8, 8))
    g.nodes["in0"] = in_node

    gen = CppGenerator(g)

    # 4D with list attrs
    node_mp_list = IRNode(id="mp_list", op_type="MaxPool2D", inputs=["in0"], shape_metadata=(1, 3, 4, 4), attributes={"kernel_size": [2, 2], "stride": [2, 2]})
    gen.visit_MaxPool2D(node_mp_list, g)
    gen._visit_node(node_mp_list, g)

    # 2D with int attrs and global
    node_mp_int = IRNode(id="mp_int", op_type="global_maxpool2d", inputs=["in0"], shape_metadata=(4, 4), attributes={"kernel_size": 2, "stride": 2})
    gen.visit_MaxPool2D(node_mp_int, None, is_global=True)
    gen._visit_node(node_mp_int, g)

    # AvgPool2D
    node_ap_list = IRNode(id="ap_list", op_type="AvgPool2D", inputs=["in0"], shape_metadata=(1, 3, 4, 4), attributes={"kernel_size": [2, 2], "stride": [2, 2]})
    gen.visit_AvgPool2D(node_ap_list, g)
    gen._visit_node(node_ap_list, g)

    node_ap_int = IRNode(id="ap_int", op_type="global_avgpool2d", inputs=["in0"], shape_metadata=(4, 4), attributes={"kernel_size": 2, "stride": 2})
    gen.visit_AvgPool2D(node_ap_int, None, is_global=True)
    gen._visit_node(node_ap_int, g)

    # 3. GroupNorm with gamma, beta, num_groups <= 0
    node_gn_full = IRNode(id="gn_full", op_type="GroupNorm", inputs=["in0", "gamma", "beta"], shape_metadata=(2, 4, 8, 8), attributes={"num_groups": 2, "eps": 1e-4})
    gen.visit_GroupNorm(node_gn_full, g)
    gen._visit_node(node_gn_full, g)

    node_gn_nogamma = IRNode(id="gn_no_gamma", op_type="groupnorm", inputs=["in0"], shape_metadata=(2, 4), attributes={"num_groups": 0})
    gen.visit_GroupNorm(node_gn_nogamma, None)
    gen._visit_node(node_gn_nogamma, None)

    # 4. Softmax and LogSoftmax
    node_sm = IRNode(id="sm", op_type="softmax", inputs=["in0"], shape_metadata=(4, 4))
    gen.visit_Softmax(node_sm, g)
    gen._visit_node(node_sm, g)

    node_lsm = IRNode(id="lsm", op_type="logsoftmax", inputs=["in0"], shape_metadata=(4, 4))
    gen.visit_LogSoftmax(node_lsm, g)
    gen._visit_node(node_lsm, g)

    # 5. Advanced hardware patterns
    node_wsr = IRNode(id="wsr", op_type="warp_shuffle_reduce", inputs=["in0"], shape_metadata=(4, 4))
    gen.visit_WarpShuffleReduce(node_wsr, g)
    gen._visit_node(node_wsr, g)

    node_smtr = IRNode(id="smtr", op_type="shared_memory_tree_reduce", inputs=["in0"], shape_metadata=(4, 4))
    gen.visit_SharedMemoryTreeReduce(node_smtr, g)
    gen._visit_node(node_smtr, g)

    node_tmm = IRNode(id="tmm", op_type="tiled_matmul_2d", inputs=["in0"], shape_metadata=(4, 4))
    gen.visit_TiledMatMul2D(node_tmm, g)
    gen._visit_node(node_tmm, g)

    node_c2d_halo = IRNode(id="c2d_halo", op_type="conv2d_shared_halo", inputs=["in0"], shape_metadata=(1, 3, 8, 8), attributes={"stride": 1})
    gen.visit_Conv2DSharedHalo(node_c2d_halo, g)
    gen._visit_node(node_c2d_halo, g)

    node_mp_dyn = IRNode(id="mp_dyn", op_type="maxpool2d_dynamic", inputs=["in0"], shape_metadata=(1, 3, 4, 4), attributes={"kernel_size": 2, "stride": 2, "dilation_h": 1, "dilation_w": 1, "pad_h": 0, "pad_w": 0})
    gen.visit_MaxPool2DDynamic(node_mp_dyn, g)
    gen._visit_node(node_mp_dyn, g)

    node_ap_dyn = IRNode(id="ap_dyn", op_type="avgpool2d_dynamic", inputs=["in0"], shape_metadata=(1, 3, 4, 4), attributes={"kernel_size": 2, "stride": 2, "dilation_h": 1, "dilation_w": 1, "pad_h": 0, "pad_w": 0})
    gen.visit_AvgPool2DDynamic(node_ap_dyn, g)
    gen._visit_node(node_ap_dyn, g)


def test_llvm_cpp_missing_branches_and_edge_cases() -> None:
    """Test while loop annotations, 2D shape padding in conv/pooling, decl_op overrides, and edge indices."""
    g = IRGraph()
    gen = CppGenerator(g)

    # 1. While loop annotations and parallel iterations (#pragma omp)
    n_while = IRNode(
        id="while_opts",
        op_type="WhileLoop",
        inputs=["in0"],
        shape_metadata=(4,),
        attributes={
            "maximum_iterations": 10,
            "parallel_iterations": 4,
            "swap_memory": True,
            "shape_invariants": [(4,)],
        },
    )
    gen._visit_loop_op(n_while, g)
    joined_lines = "\n".join(gen.lines)
    assert "parallel_iterations=4" in joined_lines
    assert "swap_memory=True" in joined_lines
    assert "maximum_iterations=10" in joined_lines
    assert "shape_invariants=[(4,)]" in joined_lines
    assert "#pragma omp parallel for num_threads(4)" in joined_lines

    # 2. 2D shapes (< 4D) in MaxPool2D, AvgPool2D, Conv2D, MaxPool2DDynamic, AvgPool2DDynamic
    node_in_2d = IRNode(id="in_2d", op_type="Input", inputs=[], shape_metadata=(4, 4))
    node_w_2d = IRNode(id="w_2d", op_type="Weight", inputs=[], shape_metadata=(3, 3))
    g.nodes["in_2d"] = node_in_2d
    g.nodes["w_2d"] = node_w_2d

    node_mp_2d = IRNode(id="mp_2d", op_type="MaxPool2D", inputs=["in_2d"], shape_metadata=(4, 4), attributes={"kernel_size": [2, 2], "stride": [1, 1]})
    gen.visit_MaxPool2D(node_mp_2d, g)

    node_ap_2d = IRNode(id="ap_2d", op_type="AvgPool2D", inputs=["in_2d"], shape_metadata=(4, 4), attributes={"kernel_size": [2, 2], "stride": [1, 1]})
    gen.visit_AvgPool2D(node_ap_2d, g)

    node_conv_2d = IRNode(id="conv_2d", op_type="Conv2DSharedHalo", inputs=["in_2d", "w_2d"], shape_metadata=(4, 4), attributes={"stride": 1, "padding": "SAME"})
    gen.visit_Conv2DSharedHalo(node_conv_2d, g)

    node_mp_dyn_2d = IRNode(id="mp_dyn_2d", op_type="MaxPool2DDynamic", inputs=["in_2d"], shape_metadata=(4, 4), attributes={"kernel_size": 2, "stride": 2})
    gen.visit_MaxPool2DDynamic(node_mp_dyn_2d, g)

    node_ap_dyn_2d = IRNode(id="ap_dyn_2d", op_type="AvgPool2DDynamic", inputs=["in_2d"], shape_metadata=(4, 4), attributes={"kernel_size": 2, "stride": 2})
    gen.visit_AvgPool2DDynamic(node_ap_dyn_2d, g)

    # 3. decl_op init_val and final_combine with and without existing mapping
    mock_decl_op = MagicMock()
    mock_decl_op.template = "elementwise_1d"
    mock_decl_op.scalar_expr = "in0_val * 2"
    mock_decl_op.init_val = "0.0"
    mock_decl_op.final_combine = "out_val"

    with patch("ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider.get_cpp_operation", return_value=mock_decl_op):
        with patch("ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider.get_cpp_template", return_value={"body": "// body {clean_id}"}):
            # 3a. not mapping and decl_op is not None (lines 1290, 1292)
            with patch.dict("ml_switcheroo_compiler.ops.registry._YAML_REGISTRY", {"NonExistentOp123": {}}):
                n_unmapped = IRNode(id="unmapped_decl", op_type="NonExistentOp123", inputs=["in_2d"], shape_metadata=(4, 4))
                gen._visit_node(n_unmapped, g)

            # 3b. mapping with scalar_expr == 'in0_val' and decl_op is not None (lines 1297, 1299)
            with patch.dict("ml_switcheroo_compiler.ops.registry._YAML_REGISTRY", {"IdentityCustom": {"variants": {"llvm_cpp": {"template": "elementwise_1d", "scalar_expr": "in0_val"}}}}):
                n_scalar_in0 = IRNode(id="ident_custom", op_type="IdentityCustom", inputs=["in_2d"], shape_metadata=(4, 4))
                gen._visit_node(n_scalar_in0, g)

    # 4. Target index 2 edge input mapping (line 1330: in2_var = str(src))
    class DummyGraphWithEdges:
        """Dummy graph container supporting edges attribute."""

        def __init__(self) -> None:
            """Initialize dummy graph with nodes and edges."""
            self.nodes: dict[str, IRNode] = {"in_2d": node_in_2d}
            self.edges: list[MagicMock] = [MagicMock(target="ternary_op", source="src_node_2", target_idx=2, source_idx=1)]

    g_edges = DummyGraphWithEdges()
    node_ternary = IRNode(id="ternary_op", op_type="Select", inputs=["in_2d", "in_2d", "in_2d"], shape_metadata=(4, 4))
    with patch("ml_switcheroo_compiler.backends.llvm_cpp.cpp_provider.get_cpp_template", return_value={"body": "// body {clean_id} {in2}"}):
        with patch.dict("ml_switcheroo_compiler.ops.registry._YAML_REGISTRY", {"Select": {"variants": {"llvm_cpp": {"template": "elementwise_1d"}}}}):
            gen._visit_node(node_ternary, g_edges)
            assert any("src_node_2" in line for line in gen.lines)
