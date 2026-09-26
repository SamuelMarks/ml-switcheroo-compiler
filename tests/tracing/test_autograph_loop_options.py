"""Tests for Autograph dynamic loop options and backend loop annotation propagation."""

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.cuda.cuda import CudaCodeGenerator
from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
from ml_switcheroo_compiler.backends.llvm_cpp.generator import LLVMCPPGenerator
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow.tracing import while_loop_tracing
from ml_switcheroo_compiler.tracing.autograph import LoopOptions, set_loop_options
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor


def test_loop_options_init_and_context_manager() -> None:
    """Test LoopOptions initialization and context manager state restoration."""
    opts1 = LoopOptions(
        parallel_iterations=8,
        swap_memory=True,
        maximum_iterations=64,
        shape_invariants=[(10, 20)],
    )
    assert opts1.parallel_iterations == 8
    assert opts1.swap_memory is True
    assert opts1.maximum_iterations == 64
    assert opts1.shape_invariants == [(10, 20)]

    opts2 = LoopOptions(
        parallel_iterations=4,
        swap_memory=False,
        maximum_iterations=32,
        shape_invariants=None,
    )

    assert global_tracing_state.current_loop_options is None

    with opts1:
        assert global_tracing_state.current_loop_options is opts1
        with opts2:
            assert global_tracing_state.current_loop_options is opts2
        assert global_tracing_state.current_loop_options is opts1

    assert global_tracing_state.current_loop_options is None


def test_set_loop_options_injection_on_active_graph() -> None:
    """Test that set_loop_options injects options into active LogicalGraph."""
    graph = global_tracing_state.start_tracing("test_loop_options_graph")
    try:
        graph.metadata = {}
        opts = set_loop_options(
            parallel_iterations=16,
            swap_memory=True,
            maximum_iterations=128,
            shape_invariants=[(32, 64)],
        )
        assert global_tracing_state.current_loop_options is opts
        assert getattr(graph, "loop_options", None) is opts
        assert graph.metadata["loop_options"] is opts

        loop_node = LogicalNode(
            id="while_loop_1",
            op_type="WhileLoop",
            inputs=[],
            attributes={},
        )
        global_tracing_state.add_node(loop_node)

        assert loop_node.attributes["parallel_iterations"] == 16
        assert loop_node.attributes["swap_memory"] is True
        assert loop_node.attributes["maximum_iterations"] == 128
        assert loop_node.attributes["shape_invariants"] == [(32, 64)]
        assert loop_node.attributes["loop_options"] is opts

        with LoopOptions(parallel_iterations=32, maximum_iterations=256) as inner_opts:
            assert getattr(graph, "loop_options", None) is inner_opts
            assert graph.metadata["loop_options"] is inner_opts

        assert getattr(graph, "loop_options", None) is opts
        assert graph.metadata["loop_options"] is opts
    finally:
        global_tracing_state.stop_tracing()

    assert global_tracing_state.current_loop_options is None

    # Test context manager when active graph has no metadata attribute
    graph_no_meta = global_tracing_state.start_tracing("test_no_meta")
    try:
        with LoopOptions(parallel_iterations=2):
            assert getattr(graph_no_meta, "loop_options", None).parallel_iterations == 2
    finally:
        global_tracing_state.stop_tracing()


def test_while_loop_tracing_with_loop_options() -> None:
    """Test that while_loop_tracing attaches active loop options to the resulting Loop node."""
    global_tracing_state.start_tracing("test_while_loop_trace")
    try:
        set_loop_options(
            parallel_iterations=4,
            swap_memory=True,
            maximum_iterations=50,
            shape_invariants=[(2, 2)],
        )

        proxy = ProxyTensor(id="init_val", shape=(2, 2), dtype="float32")
        init_tensor = Tensor(proxy, TensorConfig((2, 2), DType.Float32, "cpu"))

        def cond_fn(x: Tensor) -> Tensor:
            """Condition predicate function."""
            return x

        def body_fn(x: Tensor) -> Tensor:
            """Loop body transformation function."""
            return x

        res = while_loop_tracing(cond_fn, body_fn, init_tensor)
        assert isinstance(res, Tensor)

        active_g = global_tracing_state.active_graph
        assert active_g is not None
        loop_node = next(n for n in active_g.nodes.values() if n.op_type == "Loop")
        assert loop_node.attributes["parallel_iterations"] == 4
        assert loop_node.attributes["swap_memory"] is True
        assert loop_node.attributes["maximum_iterations"] == 50
        assert loop_node.attributes["shape_invariants"] == [(2, 2)]
    finally:
        global_tracing_state.stop_tracing()


def test_llvm_cpp_propagation() -> None:
    """Test that LLVM C++ generator propagates loop annotations and OpenMP pragmas."""
    graph = LogicalGraph(name="test_llvm_loop", outputs=["loop_1"])
    loop_node = LogicalNode(
        id="loop_1",
        op_type="Loop",
        inputs=[],
        attributes={
            "parallel_iterations": 8,
            "swap_memory": True,
            "maximum_iterations": 100,
            "shape_invariants": [(1, 4)],
        },
    )
    graph.nodes["loop_1"] = loop_node

    generator = LLVMCPPGenerator(graph)
    code = generator.generate(graph)

    assert "Loop annotations:" in code
    assert "parallel_iterations=8" in code
    assert "swap_memory=True" in code
    assert "maximum_iterations=100" in code
    assert "#pragma omp parallel for num_threads(8)" in code


def test_cuda_propagation() -> None:
    """Test that CUDA generator propagates loop annotations, unroll pragmas, and swap memory policy."""
    graph = LogicalGraph(name="test_cuda_loop", outputs=["loop_1"])
    in_node = LogicalNode(id="in_0", op_type="Input", shape_metadata=[4, 4])
    loop_node = LogicalNode(
        id="loop_1",
        op_type="WhileLoop",
        inputs=["in_0"],
        shape_metadata=[4, 4],
        attributes={
            "parallel_iterations": 4,
            "swap_memory": True,
            "maximum_iterations": 50,
            "shape_invariants": [(4, 4)],
        },
    )
    graph.nodes["in_0"] = in_node
    graph.nodes["loop_1"] = loop_node

    generator = CudaCodeGenerator(graph)
    code = generator.generate()

    assert "Loop annotations:" in code
    assert "parallel_iterations=4" in code
    assert "swap_memory=True" in code
    assert "maximum_iterations=50" in code
    assert "Swap memory policy enabled" in code
    assert "#pragma unroll 4" in code
    assert "for (int _iter_" in code and "< 50; ++_iter_" in code


def test_wasm_propagation() -> None:
    """Test that WASM generator propagates loop annotations for both WhileLoop and Loop."""
    graph = LogicalGraph(name="test_wasm_loop", outputs=["loop_1"])
    in_node = LogicalNode(id="in_0", op_type="Input", shape_metadata=[1])
    loop_node = LogicalNode(
        id="loop_1",
        op_type="WhileLoop",
        inputs=["in_0"],
        shape_metadata=[1],
        attributes={
            "parallel_iterations": 2,
            "swap_memory": False,
            "maximum_iterations": 25,
            "shape_invariants": [(1,)],
        },
    )
    graph.nodes["in_0"] = in_node
    graph.nodes["loop_1"] = loop_node

    generator = WasmCodeGenerator(graph)
    code = generator.generate()

    assert "Loop annotations:" in code
    assert "parallel_iterations=2" in code
    assert "swap_memory=False" in code
    assert "maximum_iterations=25" in code
    assert "while" in code or "for" in code

    graph_alias = LogicalGraph(name="test_wasm_loop_alias", outputs=["loop_alias"])
    in_node2 = LogicalNode(id="in_0", op_type="Input", shape_metadata=[1])
    loop_node_alias = LogicalNode(
        id="loop_alias",
        op_type="Loop",
        inputs=["in_0"],
        shape_metadata=[1],
        attributes={
            "maximum_iterations": 15,
        },
    )
    graph_alias.nodes["in_0"] = in_node2
    graph_alias.nodes["loop_alias"] = loop_node_alias

    generator_alias = WasmCodeGenerator(graph_alias)
    code_alias = generator_alias.generate()
    assert "Loop annotations:" in code_alias
    assert "maximum_iterations=15" in code_alias
