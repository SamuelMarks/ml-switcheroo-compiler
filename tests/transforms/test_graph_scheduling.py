"""Unit tests for concrete hardware graph scheduling cost models and passes."""

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.graph_scheduling import (
    CpuCostModel,
    EdgeCostModel,
    GpuCostModel,
    GraphSchedulingPass,
)


def test_cpu_cost_model_properties_and_memory() -> None:
    """Test CpuCostModel cache-aware memory cost calculations and properties."""
    model = CpuCostModel(
        l1_cache_bytes=32 * 1024,
        l2_cache_bytes=256 * 1024,
        l3_cache_bytes=1024 * 1024,
        simd_width_bytes=32,
        compute_heavy_threshold=200,
        heavy_interleave_penalty=250,
        light_interleave_penalty=40,
    )
    assert model.compute_heavy_threshold == 200
    assert model.heavy_interleave_penalty == 250
    assert model.light_interleave_penalty == 40

    # None shape
    node_none = IRNode("n_none", "Add", shape_metadata=None, attributes={"dtype": DType.Float32.value})
    assert model.get_memory_cost(node_none) == 4

    # Fitting in L1/L2: 1000 float32 = 4000 bytes (< 256KB)
    node_small = IRNode("n_small", "Add", shape_metadata=(1000,), attributes={"dtype": DType.Float32.value})
    assert model.get_memory_cost(node_small) == 4000

    # Exceeding L2 but fitting in L3: 100,000 float32 = 400,000 bytes (> 256KB, < 1MB)
    node_mid = IRNode("n_mid", "Add", shape_metadata=(100000,), attributes={"dtype": DType.Float32.value})
    assert model.get_memory_cost(node_mid) == int(400000 * 1.1)

    # Exceeding L3: 400,000 float32 = 1,600,000 bytes (> 1MB)
    node_large = IRNode("n_large", "Add", shape_metadata=(400000,), attributes={"dtype": DType.Float32.value})
    assert model.get_memory_cost(node_large) == int(1600000 * 1.25)

    # Symbolic dimension
    node_sym = IRNode("n_sym", "Add", shape_metadata=("B", 128), attributes={"dtype": DType.Float32.value})
    assert model.get_memory_cost(node_sym) == "B * 128 * 4"

    # Empty dynamic shape
    node_dyn_empty = IRNode("n_dyn", "Add", shape_metadata=(), attributes={"dtype": DType.Float32.value, "is_dynamic_shape": True})
    assert model.get_memory_cost(node_dyn_empty) == 4

    # Float64 dtype size
    node_f64 = IRNode("n_f64", "Add", shape_metadata=(10,), attributes={"dtype": DType.Float64.value})
    assert model.get_memory_cost(node_f64) == 80


def test_cpu_cost_model_compute() -> None:
    """Test CpuCostModel SIMD-calibrated compute costs."""
    model = CpuCostModel(simd_width_bytes=64)

    node_matmul = IRNode("mm", "MatMul", shape_metadata=(32, 32))
    assert model.get_compute_cost(node_matmul) >= 500

    node_conv = IRNode("conv", "Conv2D", shape_metadata=(1, 16, 14, 14))
    assert model.get_compute_cost(node_conv) >= 800

    node_reduce = IRNode("red", "ReduceSum", shape_metadata=(1000,))
    assert model.get_compute_cost(node_reduce) >= 20

    node_elem = IRNode("add", "Add", shape_metadata=(1000,))
    assert model.get_compute_cost(node_elem) >= 5

    node_default = IRNode("custom", "CustomOp", shape_metadata=(100,))
    assert model.get_compute_cost(node_default) == 50


def test_gpu_cost_model_properties_and_memory() -> None:
    """Test GpuCostModel roofline and HBM memory cost calculations."""
    model = GpuCostModel(
        hbm_bandwidth_gbps=800.0,
        peak_tflops=250.0,
        launch_latency_ns=4000,
        sm_count=80,
    )
    assert model.compute_heavy_threshold == 400
    assert model.heavy_interleave_penalty == 800
    assert model.light_interleave_penalty == 150

    node_none = IRNode("n_none", "Add", shape_metadata=None, attributes={"dtype": DType.Float32.value})
    assert model.get_memory_cost(node_none) == 4

    node_static = IRNode("n_static", "MatMul", shape_metadata=(100, 100), attributes={"dtype": DType.Float32.value})
    assert model.get_memory_cost(node_static) == 100 * 100 * 4

    node_f16 = IRNode("n_f16", "MatMul", shape_metadata=(100, 100), attributes={"dtype": DType.Float16.value})
    assert model.get_memory_cost(node_f16) == 100 * 100 * 2

    node_sym = IRNode("n_sym", "MatMul", shape_metadata=("Seq", 512), attributes={"dtype": DType.Float32.value})
    assert model.get_memory_cost(node_sym) == "Seq * 512 * 4"

    node_dyn_empty = IRNode("n_dyn", "Add", shape_metadata=(), attributes={"dtype": DType.Float32.value, "is_dynamic_shape": True})
    assert model.get_memory_cost(node_dyn_empty) == 4


def test_gpu_cost_model_compute() -> None:
    """Test GpuCostModel launch latency and kernel execution costs."""
    model = GpuCostModel(launch_latency_ns=5000, sm_count=100)
    launch_overhead = 500

    node_matmul = IRNode("mm", "MatMul", shape_metadata=(64, 64))
    cost_mm = model.get_compute_cost(node_matmul)
    assert cost_mm >= launch_overhead + 400

    node_conv = IRNode("conv", "Conv2D", shape_metadata=(1, 32, 16, 16))
    cost_conv = model.get_compute_cost(node_conv)
    assert cost_conv >= launch_overhead + 600

    node_comm = IRNode("comm", "AllReduce", shape_metadata=(1024,))
    assert model.get_compute_cost(node_comm) == launch_overhead + 1000

    node_add = IRNode("add", "Add", shape_metadata=(1024,))
    assert model.get_compute_cost(node_add) >= launch_overhead + 1

    node_other = IRNode("other", "CustomNode", shape_metadata=(100,))
    assert model.get_compute_cost(node_other) == launch_overhead + 50


def test_edge_cost_model_properties_and_memory() -> None:
    """Test EdgeCostModel memory constraint penalties and compute costs."""
    model = EdgeCostModel(
        max_linear_memory_bytes=1024 * 1024,  # 1MB limit
        allocation_penalty=100,
        compute_heavy_threshold=120,
    )
    assert model.compute_heavy_threshold == 120
    assert model.heavy_interleave_penalty == 250
    assert model.light_interleave_penalty == 40

    node_none = IRNode("none", "Add", shape_metadata=None, attributes={"dtype": "float32"})
    assert model.get_memory_cost(node_none) == 4

    # Small tensor: 10,000 float32 = 40,000 bytes (< 512KB)
    node_small = IRNode("small", "Add", shape_metadata=(10000,), attributes={"dtype": "float32"})
    assert model.get_memory_cost(node_small) == 40000 + 100

    # Big tensor exceeding 512KB limit: 200,000 float32 = 800,000 bytes (> 512KB)
    node_big = IRNode("big", "Add", shape_metadata=(200000,), attributes={"dtype": "float32"})
    assert model.get_memory_cost(node_big) == 800000 * 2 + 100

    node_sym = IRNode("sym", "Add", shape_metadata=("T", 10), attributes={"dtype": "float32"})
    assert model.get_memory_cost(node_sym) == "T * 10 * 4"

    node_dyn_empty = IRNode("dyn", "Add", shape_metadata=(), attributes={"dtype": "float32", "is_dynamic_shape": True})
    assert model.get_memory_cost(node_dyn_empty) == 4 + 100


def test_edge_cost_model_compute() -> None:
    """Test EdgeCostModel scalar compute costs."""
    model = EdgeCostModel()

    node_mm = IRNode("mm", "MatMul", shape_metadata=(16, 16))
    assert model.get_compute_cost(node_mm) >= 300

    node_red = IRNode("red", "ReduceSum", shape_metadata=(100,))
    assert model.get_compute_cost(node_red) >= 15

    node_add = IRNode("add", "Add", shape_metadata=(100,))
    assert model.get_compute_cost(node_add) >= 2

    node_other = IRNode("other", "CustomEdgeOp", shape_metadata=(10,))
    assert model.get_compute_cost(node_other) == 40


def test_scheduling_with_concrete_cost_models() -> None:
    """Test graph scheduling execution with CPU, GPU, and Edge cost models."""
    for cost_model in (CpuCostModel(), GpuCostModel(), EdgeCostModel()):
        graph = IRGraph()
        graph.nodes["in0"] = IRNode("in0", "Input", shape_metadata=(10, 10))
        graph.nodes["mm"] = IRNode("mm", "MatMul", inputs=["in0"], shape_metadata=(10, 10))
        graph.nodes["add"] = IRNode("add", "Add", inputs=["in0"], shape_metadata=(10, 10))
        graph.nodes["out"] = IRNode("out", "Add", inputs=["mm", "add"], shape_metadata=(10, 10))

        scheduler = GraphSchedulingPass(cost_model=cost_model)
        sched = scheduler.schedule(graph)
        assert len(sched) == 4
        assert sched[0] == "in0"
        assert sched[-1] == "out"

        peak_mem = scheduler.calculate_peak_memory(graph, sched)
        assert peak_mem > 0

        modified = scheduler.run(graph)
        assert isinstance(modified, bool)


def test_symbolic_compute_costs_and_stream_merging() -> None:
    """Test compute cost evaluation with symbolic shapes and multi-parent stream merging."""
    sym_node = IRNode("sym", "MatMul", shape_metadata=("B", "K"))

    assert CpuCostModel().get_compute_cost(sym_node) >= 500
    assert GpuCostModel().get_compute_cost(sym_node) >= 400
    assert EdgeCostModel().get_compute_cost(sym_node) >= 300

    # Test multi-parent stream merging
    graph = IRGraph()
    n1 = IRNode("n1", "Input", stream="stream_1")
    n2 = IRNode("n2", "Input", stream="stream_2")
    n3 = IRNode("n3", "Add", inputs=["n1", "n2"])
    graph.nodes = {"n1": n1, "n2": n2, "n3": n3}

    scheduler = GraphSchedulingPass()
    scheduler.run(graph)
    assert graph.nodes["n3"].stream in ("stream_1", "stream_2")
