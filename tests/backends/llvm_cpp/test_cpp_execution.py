import os
import subprocess
import tempfile

from ml_switcheroo_compiler.backends.llvm_cpp.generator import CppGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode


def test_cpp_compile_and_execute():
    graph = IRGraph()
    n1 = LogicalNode(id="in1", op_type="Input")
    n1.shape_metadata = (1, 1)
    n2 = LogicalNode(id="out", op_type="Add", inputs=["in1", "in1"])
    n2.shape_metadata = (1, 1)
    graph.nodes = {"in1": n1, "out": n2}

    gen = CppGenerator(graph)
    code = gen.generate(graph)

    # Let's see if g++ or clang++ is available
    compiler = None
    import shutil

    if shutil.which("clang++"):
        compiler = "clang++"
    elif shutil.which("g++"):
        compiler = "g++"

    if not compiler:
        return  # Skip if no compiler

    main_func = """
int main() {
    compute_graph();
    std::cout << "SUCCESS" << std::endl;
    return 0;
}
"""
    full_code = code + main_func

    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "test.cpp")
        exe = os.path.join(td, "test.out")
        with open(src, "w") as f:
            f.write(full_code)

        try:
            subprocess.run([compiler, "-std=c++11", src, "-o", exe], check=True, capture_output=True)
            res = subprocess.run([exe], check=True, capture_output=True, text=True)
            assert "SUCCESS" in res.stdout
        except subprocess.CalledProcessError as e:
            print("Compile/Run Failed:", e.stderr)
            raise


def test_cpp_compile_and_execute_matmul():
    graph = IRGraph()
    n1 = LogicalNode(id="in1", op_type="Constant", attributes={"value": 2.0})
    n1.shape_metadata = (2, 2)
    n2 = LogicalNode(id="in2", op_type="Constant", attributes={"value": 3.0})
    n2.shape_metadata = (2, 2)
    n3 = LogicalNode(id="out", op_type="MatMul", inputs=["in1", "in2"])
    n3.shape_metadata = (2, 2)
    graph.nodes = {"in1": n1, "in2": n2, "out": n3}

    gen = CppGenerator(graph)
    code = gen.generate(graph)

    compiler = None
    import shutil

    if shutil.which("clang++"):
        compiler = "clang++"
    elif shutil.which("g++"):
        compiler = "g++"

    if not compiler:
        return

    main_func = """
int main() {
    compute_graph();
    std::cout << "SUCCESS" << std::endl;
    return 0;
}
"""
    full_code = code + main_func

    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "test.cpp")
        exe = os.path.join(td, "test.out")
        with open(src, "w") as f:
            f.write(full_code)

        subprocess.run([compiler, "-std=c++11", src, "-o", exe], check=True, capture_output=True)
        res = subprocess.run([exe], check=True, capture_output=True, text=True)
        assert "SUCCESS" in res.stdout
