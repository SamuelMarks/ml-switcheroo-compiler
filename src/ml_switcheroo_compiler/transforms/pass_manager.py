"""Pass Manager Infrastructure for Middle-End Transformations.

This module provides components for managing, validating, and executing optimization and
transformation passes on Intermediate Representation (IR) graphs. It includes
topological sorting, structural and shape validation, and a pass manager that can run
passes sequentially or until fixpoint convergence with cyclic oscillation detection.
"""

from __future__ import annotations

import hashlib
import json
import typing
from collections.abc import Callable, Iterable, Mapping, Sequence  # noqa: F401

from ml_switcheroo_ir import AttributeValue, NodeDict  # noqa: F401

from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.ir.core import as_node_list

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.transforms.passes.config_models import ConvergenceCriteria

from ml_switcheroo_compiler.core.errors import CompilationError
from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode, clone_logical_node


@typing.runtime_checkable
class _NamedPassProtocol(typing.Protocol):
    """Protocol for pass callables exposing a `__name__` attribute."""

    @property
    def __name__(self) -> str:
        """Return pass function name.

        Returns:
            str: Pass name.
        """
        ...


@typing.runtime_checkable
class _NodesMappingProtocol(typing.Protocol):
    """Protocol for node containers supporting `.values()` and `.items()`."""

    def values(self) -> Iterable[IRNode]:
        """Return node values.

        Returns:
            Iterable[IRNode]: Node values.
        """
        ...

    def items(self) -> Iterable[tuple[str, IRNode]]:
        """Return node key-value pairs.

        Returns:
            Iterable[tuple[str, IRNode]]: Node key-value pairs.
        """
        ...


@typing.runtime_checkable
class _MutableNodesMappingProtocol(typing.Protocol):
    """Protocol for mutable node containers supporting `.clear()` and `.update()`."""

    def clear(self) -> None:
        """Clear all nodes."""
        ...

    def update(self, other: dict[str, IRNode]) -> None:
        """Update with nodes dictionary.

        Args:
            other (dict[str, IRNode]): Nodes dictionary.
        """
        ...


@typing.runtime_checkable
class _GraphInputsAttrProtocol(typing.Protocol):
    """Protocol for graphs exposing `inputs` and `outputs`."""

    inputs: list[str]
    outputs: list[str]


class IRValidator:
    """Validate the structural integrity and metadata consistency of an IR graph."""

    @staticmethod
    def check_cycles(graph: IRGraph) -> None:
        """Validate that the graph has no cycles.

        Args:
            graph (IRGraph): The intermediate representation graph.
        """
        topological_sort(graph)

    @staticmethod
    def check_shapes(graph: IRGraph) -> None:
        """Validate shape consistency.

        Args:
            graph (IRGraph): The intermediate representation graph.

        Raises:
            CompilationError: If a node is missing shape metadata.
        """
        raw_nodes: TensorOrScalar = graph.nodes
        nodes_iterable: Iterable[IRNode] = raw_nodes.values() if isinstance(raw_nodes, _NodesMappingProtocol) else as_node_list(raw_nodes)
        for node in nodes_iterable:
            if node.shape_metadata is None:
                msg = f"Node {node.id} is missing shape_metadata."
                raise CompilationError(msg)


def _graph_hash(graph: IRGraph) -> str:
    """Compute a deterministic hash representing graph nodes and edges.

    Args:
        graph (IRGraph): The graph parameter.

    Returns:
        str: MD5 hash digest of the serialized graph structure.
    """
    state: dict[str, dict[str, str | list[str]]] = {}
    raw_nodes: TensorOrScalar = graph.nodes
    nodes_iterable: Iterable[IRNode] = raw_nodes.values() if isinstance(raw_nodes, _NodesMappingProtocol) else as_node_list(raw_nodes)
    for node in nodes_iterable:
        state[node.id] = {
            "op": str(node.op_type),
            "inputs": list(node.inputs),
        }
    return hashlib.md5(json.dumps(state, sort_keys=True).encode("utf-8")).hexdigest()


def _snapshot_graph(graph: IRGraph) -> dict[str, dict[str, IRNode] | list[str]]:
    """Create a deep snapshot of graph nodes and topology for rollback.

    Args:
        graph (IRGraph): The intermediate representation graph to snapshot.

    Returns:
        dict[str, dict[str, IRNode] | list[str]]: Snapshot containing cloned nodes and io lists.
    """
    nodes_copy: dict[str, IRNode] = {}
    nodes_iterable: TensorOrScalar = graph.nodes
    if isinstance(nodes_iterable, _NodesMappingProtocol):
        for nid, n in nodes_iterable.items():
            nodes_copy[nid] = clone_logical_node(n)
    else:
        for n in as_node_list(nodes_iterable):
            nodes_copy[n.id] = clone_logical_node(n)

    inputs_copy: list[str] = list(graph.inputs or [])
    outputs_copy: list[str] = list(graph.outputs or [])
    return {"nodes": nodes_copy, "inputs": inputs_copy, "outputs": outputs_copy}


def _restore_graph(graph: IRGraph, snapshot: dict[str, dict[str, IRNode] | list[str]]) -> None:
    """Restore graph nodes and topology from a snapshot.

    Args:
        graph (IRGraph): The target graph to restore in place.
        snapshot (dict[str, dict[str, IRNode] | list[str]]): The snapshot to restore from.
    """
    nodes_data = snapshot["nodes"]
    raw_nodes: TensorOrScalar = graph.nodes
    if isinstance(nodes_data, dict):
        if isinstance(raw_nodes, _MutableNodesMappingProtocol):
            raw_nodes.clear()
            raw_nodes.update(nodes_data)
        else:
            assert isinstance(raw_nodes, list)
            raw_nodes.clear()
            raw_nodes.extend(nodes_data.values())
    in_snap = snapshot.get("inputs", [])
    out_snap = snapshot.get("outputs", [])
    assert isinstance(in_snap, list)
    assert isinstance(out_snap, list)
    graph.inputs = list(in_snap)
    graph.outputs = list(out_snap)


class PassManager:
    """Manages and executes optimization and transformation passes on an IR graph.

    The PassManager maintains a list of transformation passes and validation checks,
    ensuring that the graph remains structurally valid before and after
    transformations.
    """

    def __init__(self) -> None:
        """Initialize the PassManager."""
        self.passes: list[Callable[[IRGraph], IRGraph | bool]] = []
        self.pass_names: list[str] = []
        self.convergence_criteria: ConvergenceCriteria | None = None
        self.validators: list[Callable[[IRGraph], None]] = [
            IRValidator.check_cycles,
            IRValidator.check_shapes,
        ]

    def add_pass(self, ir_pass: Callable[[IRGraph], IRGraph | bool], name: str | None = None) -> None:
        """Add a pass to the manager.

        A pass should return True if it modified the graph.

        Args:
            ir_pass (Callable[[IRGraph], IRGraph | bool]): The transformation pass.
            name (str, optional): The name of the pass.
        """
        self.passes.append(ir_pass)

        raw_pass: TensorOrScalar = ir_pass
        pass_name = name if name is not None else (str(raw_pass.__name__) if isinstance(raw_pass, _NamedPassProtocol) and raw_pass.__name__ else "unknown_pass")
        self.pass_names.append(pass_name)

    @staticmethod
    def _resolve_config_path(config_path: str | None) -> str:
        """Resolve the configuration YAML path for pass loading.

        Args:
            config_path (str | None): Explicit configuration path or None for default lookup.

        Returns:
            str: Resolved filesystem path to the YAML configuration file.
        """
        import os

        if config_path is not None:
            return config_path
        default_pipeline = os.path.join(os.path.dirname(__file__), "passes", "pass_pipeline.yaml")
        if os.path.exists(default_pipeline):
            return default_pipeline
        return os.path.join(os.path.dirname(__file__), "pass_config.yaml")

    @staticmethod
    def _validate_prerequisites(execution_order: list[str], prerequisites: dict[str, list[str]]) -> None:
        """Validate that all pass prerequisites appear earlier in the execution order.

        Args:
            execution_order (list[str]): Ordered list of pass names to execute.
            prerequisites (dict[str, list[str]]): Mapping of pass name to required prior passes.

        Raises:
            CompilationError: If a required prerequisite pass has not executed prior to its dependent.
        """
        seen_passes: set[str] = set()
        for p_name in execution_order:
            for prereq in prerequisites.get(p_name, []):
                if prereq not in seen_passes:
                    msg = f"Pass '{p_name}' requires prerequisite '{prereq}' which has not executed."
                    raise CompilationError(msg)
            seen_passes.add(p_name)

    def _extract_execution_order(
        self,
        res: AttributeValue,
        opt_level: str | None,
    ) -> list[str]:
        """Extract the pass execution order from parsed YAML configuration data.

        Args:
            res (AttributeValue): Parsed YAML config object.
            opt_level (str | None): Optional optimization level ('O0', 'O1', 'O2', 'O3').

        Returns:
            list[str]: Ordered list of pass names to register.
        """
        from ml_switcheroo_compiler.transforms.passes.config_models import (
            PassConfig,
            PassPipelineConfig,
        )

        if not isinstance(res, dict):
            return []

        if "prerequisites" in res:
            pipeline_config = PassPipelineConfig(**res)
            self.convergence_criteria = pipeline_config.convergence_criteria
            if opt_level and opt_level in pipeline_config.optimization_levels:
                execution_order = pipeline_config.optimization_levels[opt_level].execution_order
            else:
                execution_order = pipeline_config.execution_order
            self._validate_prerequisites(execution_order, pipeline_config.prerequisites)
            return execution_order

        if "cost_model" in res:
            return PassConfig(**res).execution_order

        if "execution_order" in res and isinstance(res["execution_order"], list):
            return [str(p) for p in res["execution_order"]]

        return []

    def load_from_config(self, config_path: str | None = None, opt_level: str | None = None) -> None:
        """Load passes based on pass pipeline configuration or pass_config.yaml.

        Args:
            config_path (str, optional): Path to YAML configuration file.
            opt_level (str, optional): Optimization level identifier ('O0', 'O1', 'O2', 'O3').

        Raises:
            CompilationError: If prerequisite passes are missing.
        """
        import os

        import yaml

        import ml_switcheroo_compiler.transforms.passes as passes_module

        yaml_path = self._resolve_config_path(config_path)
        if not os.path.exists(yaml_path):
            return

        with open(yaml_path) as f:
            res = yaml.safe_load(f)

        self.passes = []
        self.pass_names = []
        execution_order = self._extract_execution_order(res, opt_level)

        pass_map: dict[str, Callable[[IRGraph], IRGraph | bool]] = {
            "axis_translation": passes_module.axis_translation_pass,
            "axis_translation_pass": passes_module.axis_translation_pass,
            "batch_norm_folding": passes_module.batch_norm_folding_pass,
            "batch_norm_folding_pass": passes_module.batch_norm_folding_pass,
            "broadcast_explicitizer": passes_module.broadcast_explicitizer_pass,
            "broadcast_explicitizer_pass": passes_module.broadcast_explicitizer_pass,
            "buffer_allocation": passes_module.buffer_allocation_pass,
            "buffer_allocation_pass": passes_module.buffer_allocation_pass,
            "common_subexpression_elimination": passes_module.common_subexpression_elimination_pass,
            "common_subexpression_elimination_pass": passes_module.common_subexpression_elimination_pass,
            "constant_folding": passes_module.constant_folding_pass,
            "constant_folding_pass": passes_module.constant_folding_pass,
            "control_flow_optimization": passes_module.control_flow_optimization_pass,
            "control_flow_optimization_pass": passes_module.control_flow_optimization_pass,
            "dead_code_elimination": passes_module.dead_code_elimination_pass,
            "dead_code_elimination_pass": passes_module.dead_code_elimination_pass,
            "dtype_inference": passes_module.dtype_inference_pass,
            "dtype_inference_pass": passes_module.dtype_inference_pass,
            "graph_scheduling": passes_module.graph_scheduling_pass,
            "graph_scheduling_pass": passes_module.graph_scheduling_pass,
            "inject_spmd_communication": passes_module.inject_spmd_communication_pass,
            "inject_spmd_communication_pass": passes_module.inject_spmd_communication_pass,
            "lift_state": passes_module.lift_state_pass,
            "lift_state_pass": passes_module.lift_state_pass,
            "loop_tiling": passes_module.loop_tiling_pass,
            "loop_tiling_pass": passes_module.loop_tiling_pass,
            "loop_unrolling": passes_module.loop_unrolling_pass,
            "loop_unrolling_pass": passes_module.loop_unrolling_pass,
            "mixed_precision": passes_module.mixed_precision_pass,
            "mixed_precision_pass": passes_module.mixed_precision_pass,
            "operator_fusion": passes_module.operator_fusion_pass,
            "operator_fusion_pass": passes_module.operator_fusion_pass,
            "parallel_scan": passes_module.parallel_scan_pass,
            "parallel_scan_pass": passes_module.parallel_scan_pass,
            "polyfill_lowering": passes_module.polyfill_lowering_pass,
            "polyfill_lowering_pass": passes_module.polyfill_lowering_pass,
            "rematerialization": passes_module.rematerialization_pass,
            "rematerialization_pass": passes_module.rematerialization_pass,
            "shape_inference": passes_module.shape_inference_pass,
            "shape_inference_pass": passes_module.shape_inference_pass,
            "spmd_partitioning": passes_module.spmd_partitioning_pass,
            "spmd_partitioning_pass": passes_module.spmd_partitioning_pass,
            "state_lowering": passes_module.state_lowering_pass,
            "state_lowering_pass": passes_module.state_lowering_pass,
            "strip_offline_diagnostic_nodes": passes_module.strip_offline_diagnostic_nodes_pass,
            "strip_offline_diagnostic_nodes_pass": passes_module.strip_offline_diagnostic_nodes_pass,
            "type_promotion_explicitizer": passes_module.type_promotion_explicitizer_pass,
            "type_promotion_explicitizer_pass": passes_module.type_promotion_explicitizer_pass,
            "vectorization": passes_module.vectorization_pass,
            "vectorization_pass": passes_module.vectorization_pass,
        }

        for pass_name in execution_order:
            pass_func = pass_map.get(pass_name)
            if pass_func is not None:
                self.add_pass(pass_func, name=pass_name)

    def validate(self, graph: IRGraph) -> None:
        """Run all validators on the graph.

        Args:
            graph (IRGraph): The intermediate representation graph.
        """
        for validator in self.validators:
            validator(graph)

    def run(self, graph: IRGraph) -> IRGraph:
        """Run all passes sequentially on the graph.

        Args:
            graph (IRGraph): The intermediate representation graph.

        Returns:
            IRGraph: The transformed graph.
        """
        self.validate(graph)
        for ir_pass in self.passes:
            ir_pass(graph)
            self.validate(graph)
        return graph

    def run_until_converged(self, graph: IRGraph, max_iterations: int = 10) -> IRGraph:
        """Run passes until the graph stops changing or max_iterations reached.

        Includes cyclic oscillation detection and rollback to the initial state
        of the cycle.

        Args:
            graph (IRGraph): The intermediate representation graph.
            max_iterations (int): The maximum number of iterations.

        Returns:
            IRGraph: The transformed graph.
        """
        self.validate(graph)

        if self.convergence_criteria is not None:
            max_iterations = self.convergence_criteria.max_iterations

        history_hashes: list[str] = [_graph_hash(graph)]
        history_snapshots: list[dict[str, dict[str, IRNode] | list[str]]] = [_snapshot_graph(graph)]

        for _ in range(max_iterations):
            for ir_pass in self.passes:
                if ir_pass(graph):
                    from ml_switcheroo_compiler.transforms.passes.dtype_inference import (
                        dtype_inference_pass,
                    )
                    from ml_switcheroo_compiler.transforms.passes.shape_inference import (
                        shape_inference_pass,
                    )

                    shape_inference_pass(graph)
                    dtype_inference_pass(graph)
                self.validate(graph)

            new_hash = _graph_hash(graph)

            if new_hash == history_hashes[-1]:
                break

            if new_hash in history_hashes:
                earlier_index = history_hashes.index(new_hash)
                _restore_graph(graph, history_snapshots[earlier_index])
                self.validate(graph)
                break

            history_hashes.append(new_hash)
            history_snapshots.append(_snapshot_graph(graph))

        return graph


class DAGTopologicalSorter:
    """Alias for topological sorter."""

    @staticmethod
    def sort(graph: IRGraph) -> list[IRNode]:
        """Sort the graph topologically.

        Args:
            graph (IRGraph): The intermediate representation graph.

        Returns:
            list[IRNode]: The topologically sorted list of nodes.
        """
        return list(topological_sort(graph))
