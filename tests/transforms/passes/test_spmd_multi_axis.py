"""Integration tests for SPMD multi-axis mesh partitioning and automated collective insertion."""

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.config_models import (
    DataParallelStrategyConfig,
    DeviceMeshPartitioningConfig,
    MeshPartitioningDefaultMesh,
    MeshPartitioningStrategiesConfig,
    PipelineParallelStrategyConfig,
    TensorModelParallelStrategyConfig,
    load_mesh_partitioning,
)
from ml_switcheroo_compiler.transforms.passes.spmd import (
    SPMDShardingAnnotation,
    propagate_sharding,
    spmd_partitioning_pass,
)


def test_spmd_config_models_and_loader() -> None:
    """Test loading and validating declarative device mesh partitioning configuration."""
    cfg = load_mesh_partitioning()
    assert isinstance(cfg, DeviceMeshPartitioningConfig)
    assert cfg.default_mesh.shape == [2, 2, 1]
    assert cfg.default_mesh.axis_names == ["dp", "tp", "pp"]
    assert cfg.strategies.data_parallel.mesh_axis == "dp"
    assert cfg.strategies.tensor_model_parallel.mesh_axis == "tp"
    assert cfg.strategies.pipeline_parallel.mesh_axis == "pp"

    manual_cfg = DeviceMeshPartitioningConfig(
        default_mesh=MeshPartitioningDefaultMesh(
            shape=[4, 2],
            axis_names=["dp", "tp"],
            devices=[0, 1, 2, 3, 4, 5, 6, 7],
        ),
        strategies=MeshPartitioningStrategiesConfig(
            data_parallel=DataParallelStrategyConfig(mesh_axis="dp", tensor_dim=0),
            tensor_model_parallel=TensorModelParallelStrategyConfig(mesh_axis="tp", row_parallel_dim=0, col_parallel_dim=1),
            pipeline_parallel=PipelineParallelStrategyConfig(mesh_axis="pp"),
        ),
    )
    assert manual_cfg.default_mesh.shape == [4, 2]


def test_spmd_sharding_annotation_multi_axis() -> None:
    """Test SPMDShardingAnnotation with multi-axis mesh metadata and properties."""
    ann = SPMDShardingAnnotation(
        mesh="mesh_2x2",
        mesh_mapping=["dp", "tp"],
        mesh_shape=[2, 2],
        mesh_axes=["dp", "tp"],
    )
    assert ann.is_sharded is True
    assert ann.mesh_shape == (2, 2)
    assert ann.mesh_axes == ("dp", "tp")
    assert "SPMDShardingAnnotation" in repr(ann)

    replicated_ann = SPMDShardingAnnotation(
        mesh="mesh_2x2",
        mesh_mapping=[None, None],
    )
    assert replicated_ann.is_sharded is False


def test_spmd_matmul_contracting_parallel_all_reduce() -> None:
    """Test automated AllReduce injection on contracting-parallel MatMul."""
    graph = IRGraph()
    x = IRNode(id="x", op_type="Input")
    w = IRNode(id="w", op_type="Input")
    mm = IRNode(id="mm", op_type="MatMul", inputs=["x", "w"])

    x.sharding = SPMDShardingAnnotation(mesh="grid", mesh_mapping=[None, "tp"])
    w.sharding = SPMDShardingAnnotation(mesh="grid", mesh_mapping=["tp", None])

    graph.nodes = {"x": x, "w": w, "mm": mm}
    graph.outputs = ["mm"]

    assert spmd_partitioning_pass(graph) is True
    assert "mm_all_reduce" in graph.nodes
    assert graph.nodes["mm_all_reduce"].op_type == "AllReduce"
    assert graph.outputs == ["mm_all_reduce"]


def test_spmd_matmul_contracting_parallel_reduce_scatter() -> None:
    """Test automated ReduceScatter injection when consumer requires scattered layout."""
    graph = IRGraph()
    x = IRNode(id="x", op_type="Input")
    w = IRNode(id="w", op_type="Input")
    mm = IRNode(id="mm", op_type="MatMul", inputs=["x", "w"], attributes={"requires_scatter": True})

    x.sharding = SPMDShardingAnnotation(mesh="grid", mesh_mapping=[None, "tp"])
    w.sharding = SPMDShardingAnnotation(mesh="grid", mesh_mapping=["tp", None])

    graph.nodes = {"x": x, "w": w, "mm": mm}
    graph.outputs = ["mm"]

    assert spmd_partitioning_pass(graph) is True
    assert "mm_reducescatter" in graph.nodes
    assert graph.nodes["mm_reducescatter"].op_type == "ReduceScatter"
    assert graph.outputs == ["mm_reducescatter"]


def test_spmd_reduction_reduce_scatter() -> None:
    """Test automated ReduceScatter injection on reduction when requires_reduce_scatter is set."""
    graph = IRGraph()
    x = IRNode(id="x", op_type="Input")
    red = IRNode(id="red", op_type="ReduceSum", inputs=["x"], attributes={"axis": 1, "requires_reduce_scatter": True})

    x.sharding = SPMDShardingAnnotation(mesh="grid", mesh_mapping=["dp", "tp"])

    graph.nodes = {"x": x, "red": red}
    graph.outputs = ["red"]

    assert spmd_partitioning_pass(graph) is True
    assert "red_reducescatter" in graph.nodes
    assert graph.nodes["red_reducescatter"].op_type == "ReduceScatter"
    assert graph.outputs == ["red_reducescatter"]


def test_spmd_multi_axis_dp_tp_propagation() -> None:
    """Test multi-axis sharding propagation across 3D/4D tensors with data and model parallelism."""
    graph = IRGraph()
    act = IRNode(id="act", op_type="Input")
    weight = IRNode(id="weight", op_type="Input")
    mm = IRNode(id="mm", op_type="MatMul", inputs=["act", "weight"])
    add = IRNode(id="add", op_type="Add", inputs=["mm", "act"])

    # act is sharded along dp on batch (dim 0) and tp on row (dim 1)
    act.sharding = SPMDShardingAnnotation(mesh="mesh_4x2", mesh_mapping=["dp", "tp"])
    # weight is replicated along dp, row-parallel
    weight.sharding = SPMDShardingAnnotation(mesh="mesh_4x2", mesh_mapping=[None, "tp"])

    graph.nodes = {"act": act, "weight": weight, "mm": mm, "add": add}
    graph.outputs = ["add"]

    assert propagate_sharding(graph) is True
    assert mm.sharding is not None
    assert add.sharding is not None
