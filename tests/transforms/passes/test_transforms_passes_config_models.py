"""Tests for test_transforms_passes_config_models."""

from __future__ import annotations

import os

from ml_switcheroo_ir import LogicalGraph, LogicalNode

import ml_switcheroo_compiler.backends.generator_mixins as gen_mixins
import ml_switcheroo_compiler.transforms.passes.config_models as passes_cfg


class _MockDummyGenerator(gen_mixins.GeneratorLifecycleMixin):
    """Mock generator class using GeneratorLifecycleMixin."""

    def __init__(self, graph: LogicalGraph) -> None:
        """Initialize mock generator.

        Args:
            graph (LogicalGraph): Logical graph to simulate.
        """
        self.graph = graph
        self.header = "# Header comment"
        self.code: list[str] = []
        self.indent_level: int = 0
        self.sorted_nodes: list[LogicalNode] = list(graph.nodes.values())
        self.input_idx: int = 0
        self._output_returns: list[str] = []
        self.var_names: dict[str, str] = {"in1": "args[0]"}

    def assign_var_name(self, node_id: str) -> str:
        """Assign variable name.

        Args:
            node_id (str): Node id.

        Returns:
            str: Assigned variable name.
        """
        name = f"var_{node_id}"
        self.var_names[node_id] = name
        return name

    def visit(self, node: LogicalNode, input_vars: list[str]) -> str:
        """Generate code for node.

        Args:
            node (LogicalNode): The node.
            input_vars (list[str]): Input variables.

        Returns:
            str: Code string.
        """
        return f"Identity({', '.join(input_vars)})"

    def _emit_body_return(self, returns: list[str]) -> None:
        """Emit return statement for body.

        Args:
            returns (list[str]): Return variables.
        """
        self.add_line(f"return {', '.join(returns) if returns else 'None'}")

    def add_line(self, line: str) -> None:
        """Add line to code buffer.

        Args:
            line (str): Line string.
        """
        self.code.append("    " * self.indent_level + line)


class _MockCustomModule:
    """Mock backend module exposing zeros, array, asarray, and item."""

    def zeros(self, shape: tuple[int, ...]) -> tuple[str, tuple[int, ...]]:
        """Mock zeros.

        Args:
            shape (tuple[int, ...]): Target shape.

        Returns:
            tuple[str, tuple[int, ...]]: Descriptor.
        """
        return ("custom_zeros", shape)

    def array(self, data: object, dtype: object | None = None) -> tuple[str, object, object | None]:
        """Mock array.

        Args:
            data (object): Input data.
            dtype (object | None): Data type.

        Returns:
            tuple[str, object, object | None]: Descriptor.
        """
        return ("custom_array", data, dtype)

    def asarray(self, data: object) -> tuple[str, object]:
        """Mock asarray.

        Args:
            data (object): Input data.

        Returns:
            tuple[str, object]: Descriptor.
        """
        return ("custom_asarray", data)

    def item(self, data: object) -> float:
        """Mock item.

        Args:
            data (object): Input object.

        Returns:
            float: 99.9.
        """
        return 99.9


class _MockEagerBackendWithModule(gen_mixins.EagerExecutionMixin):
    """Eager backend class providing custom module."""

    @classmethod
    def get_module(cls: type) -> _MockCustomModule:
        """Return custom backend module.

        Returns:
            _MockCustomModule: Mock module.
        """
        return _MockCustomModule()


class _MockEagerBackendDefault(gen_mixins.EagerExecutionMixin):
    """Eager backend class using default numpy module."""

    pass


class _MockScalarHolder:
    """Mock scalar container providing item()."""

    def item(self) -> float:
        """Return scalar value.

        Returns:
            float: Scalar value.
        """
        return 33.3


def test_transforms_passes_config_models() -> None:
    """Verify all Pydantic models and helper loaders in transforms/passes/config_models.py."""
    # Pattern & fusion
    node_pat = passes_cfg.NodePatternConfig(
        op_type="Mul",
        capture="m1",
        inputs=[passes_cfg.NodePatternConfig(op_type="Add", capture="a1")],
        wildcard=False,
        commutative=True,
        broadcast_dims=True,
    )
    assert node_pat.op_type == "Mul"
    assert node_pat.commutative is True

    repl = passes_cfg.ReplacementConfig(
        op_type="FusedMulAdd",
        inputs=["x", "y", "z"],
        capture_to_replace="m1",
    )
    assert repl.op_type == "FusedMulAdd"

    fusion = passes_cfg.FusionPatternConfig(pattern=node_pat, replacement=repl)
    assert fusion.pattern == node_pat

    # Cost model & PassConfig
    costs = passes_cfg.ComputeCosts(
        heavy_ops=["MatMul", "Conv2D"],
        light_ops=["Add", "Sub"],
        heavy_cost=100,
        light_cost=1,
        default_cost=10,
    )
    cost_model = passes_cfg.CostModelConfig(
        memory_sizes={"host": 1024},
        compute_costs=costs,
        compute_heavy_threshold=50,
        heavy_interleave_penalty=5,
        light_interleave_penalty=1,
    )
    pass_cfg = passes_cfg.PassConfig(
        execution_order=["dce", "cse"],
        cost_model=cost_model,
        fusion_patterns={"fuse_fma": fusion},
    )
    assert pass_cfg.execution_order == ["dce", "cse"]

    # Convergence, PipelineStageModel, PassConfigModel
    conv = passes_cfg.ConvergenceCriteria(max_iterations=15, detect_cyclic_oscillation=True, require_fixpoint=True)
    assert conv.max_iterations == 15

    stage = passes_cfg.PipelineStageModel(
        stage_name="canonicalize",
        passes=["dce", "cse"],
        fixpoint_iteration=True,
        max_iterations=5,
    )
    assert stage.stage_name == "canonicalize"

    pass_cfg_model = passes_cfg.PassConfigModel(
        pass_name="cse",
        enabled=True,
        prerequisites=["dce"],
        preserves=["type_info"],
        options={"aggressive": True},
    )
    assert pass_cfg_model.pass_name == "cse"

    opt_level = passes_cfg.OptLevelConfig(execution_order=["dce"], fixpoint_iteration=False, max_iterations=1)
    contract = passes_cfg.PassAnalysisContract(
        prerequisites=["dce"],
        consumed_properties=["shapes"],
        invalidated_analyses=["liveness"],
    )

    pipeline = passes_cfg.PassPipelineConfig(
        execution_order=["stage1"],
        convergence_criteria=conv,
        prerequisites={"cse": ["dce"]},
        stages=[stage],
        pass_configs={"cse": pass_cfg_model},
        optimization_levels={"O1": opt_level},
        pass_analyses_and_invalidation={"cse": contract},
    )
    assert len(pipeline.stages) == 1

    # Remat, heuristics, behavior
    remat_thresh = passes_cfg.RematerializationThresholds(min_memory_bytes=4096, max_compute_to_memory_ratio=2.5)
    remat_rules = passes_cfg.RematerializationRulesConfig(
        target_ops=["Conv2D"],
        high_cost_ops=["MatMul"],
        thresholds=remat_thresh,
    )
    assert remat_rules.thresholds.min_memory_bytes == 4096

    heuristics = passes_cfg.OptimizationHeuristicsConfig(in_place_safe_ops=["AddInPlace"])
    assert heuristics.in_place_safe_ops == ["AddInPlace"]

    behaviors = passes_cfg.BehaviorDescriptorsConfig(side_effect_ops=["Print", "Assert"])
    assert behaviors.side_effect_ops == ["Print", "Assert"]

    # Runtime shapes & payloads
    obs_shape = passes_cfg.ObservedNodeShape(
        node_id="n1",
        shape=[1, 3, 224, 224],
        dtype="float32",
        strides=[150528, 50176, 224, 1],
        offset=0,
        byte_length=602112,
    )
    assert obs_shape.node_id == "n1"

    rt_meta = passes_cfg.RuntimeTensorMetadata(
        buffer_id="buf0",
        element_count=150528,
        is_gradient=False,
        aliased_to=None,
        is_contiguous=True,
    )
    assert rt_meta.buffer_id == "buf0"

    payload = passes_cfg.ShapeInspectionPayload(
        runtime="wasm_simd",
        execution_id="exec_1",
        observed_shapes={"n1": [1, 3, 224, 224]},
        detailed_nodes={"n1": obs_shape},
        tensor_metadata={"buf0": rt_meta},
        execution_time_ms=1.25,
        memory_usage_bytes=1048576,
    )
    assert payload.runtime == "wasm_simd"

    packet = passes_cfg.RuntimeShapePacket(
        packet_id="pkt_1",
        runtime="webgpu",
        graph_name="resnet50",
        observations=[obs_shape],
        execution_time_ms=0.8,
        memory_usage_bytes=524288,
    )
    assert packet.packet_id == "pkt_1"

    # Shape learning protocol loader (default and custom path)
    loaded_protocol = passes_cfg.load_shape_learning_protocol()
    assert loaded_protocol.protocol_version != ""
    assert len(loaded_protocol.supported_runtimes) > 0

    default_proto_path = os.path.join(os.path.dirname(passes_cfg.__file__), "shape_learning_protocol.yaml")
    loaded_custom_proto = passes_cfg.load_shape_learning_protocol(default_proto_path)
    assert loaded_custom_proto.protocol_version == loaded_protocol.protocol_version

    # Device mesh & SPMD strategy models
    mesh_grid = passes_cfg.MeshPartitioningDefaultMesh(
        shape=[2, 4],
        axis_names=["dp", "tp"],
        devices=[0, 1, 2, 3, 4, 5, 6, 7],
    )
    assert mesh_grid.shape == [2, 4]

    dp_strat = passes_cfg.DataParallelStrategyConfig(mesh_axis="dp", tensor_dim=0)
    tp_strat = passes_cfg.TensorModelParallelStrategyConfig(mesh_axis="tp", row_parallel_dim=0, col_parallel_dim=1, contracting_dim=-1)
    pp_strat = passes_cfg.PipelineParallelStrategyConfig(mesh_axis="pp", stage_dim=None)
    strategies = passes_cfg.MeshPartitioningStrategiesConfig(
        data_parallel=dp_strat,
        tensor_model_parallel=tp_strat,
        pipeline_parallel=pp_strat,
    )
    mesh_cfg = passes_cfg.DeviceMeshPartitioningConfig(default_mesh=mesh_grid, strategies=strategies)
    root_mesh_cfg = passes_cfg.DeviceMeshPartitioningRootConfig(device_mesh_partitioning=mesh_cfg)
    assert root_mesh_cfg.device_mesh_partitioning.default_mesh.shape == [2, 4]

    # SPMD Communication matrix
    comm_cond = passes_cfg.SpmdCommunicationCondition(
        inject="AllReduce",
        is_reduction=True,
        is_grad=False,
        axes_match=True,
        axes_length_match=True,
        default=False,
    )
    comm_rule = passes_cfg.SpmdCommunicationRule(state=[True, False], conditions=[comm_cond])
    matrix_cfg = passes_cfg.SpmdCommunicationMatrixConfig(communication_matrix=[comm_rule])
    assert len(matrix_cfg.communication_matrix) == 1

    # Load mesh partitioning helper (default and custom path)
    loaded_mesh = passes_cfg.load_mesh_partitioning()
    assert len(loaded_mesh.default_mesh.shape) > 0

    default_mesh_path = os.path.join(os.path.dirname(passes_cfg.__file__), "spmd_mappings", "mesh_partitioning.yaml")
    loaded_custom_mesh = passes_cfg.load_mesh_partitioning(default_mesh_path)
    assert loaded_custom_mesh.default_mesh.shape == loaded_mesh.default_mesh.shape
