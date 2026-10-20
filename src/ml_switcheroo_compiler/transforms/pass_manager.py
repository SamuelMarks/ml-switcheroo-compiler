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
from collections.abc import Callable, Iterable, Sequence  # noqa: F401

from ml_switcheroo_ir import AttributeValue

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.transforms.passes.config_models import ConvergenceCriteria

from ml_switcheroo_compiler.core.errors import CompilationError
from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode, clone_logical_node


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
        nodes_iterable: Iterable[IRNode] = graph.nodes.values() if hasattr(graph.nodes, "values") else typing.cast(Iterable[IRNode], graph.nodes)
        for node in nodes_iterable:
            if getattr(node, "shape_metadata", None) is None:
                msg = f"Node {getattr(node, 'id', '')} is missing shape_metadata."
                raise CompilationError(msg)


def _graph_hash(graph: IRGraph) -> str:
    """Compute a deterministic hash representing graph nodes and edges.

    Args:
        graph (IRGraph): The graph parameter.

    Returns:
        str: MD5 hash digest of the serialized graph structure.
    """
    state: dict[str, dict[str, str | list[str]]] = {}
    nodes_iterable: Iterable[IRNode] = graph.nodes.values() if hasattr(graph.nodes, "values") else typing.cast(Iterable[IRNode], graph.nodes)
    for node in nodes_iterable:
        state[getattr(node, "id", "")] = {
            "op": str(getattr(node, "op_type", "")),
            "inputs": list(getattr(node, "inputs", [])),
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
    nodes_iterable = graph.nodes
    if hasattr(nodes_iterable, "items"):
        for nid, n in nodes_iterable.items():
            nodes_copy[nid] = clone_logical_node(n)
    else:
        for n in typing.cast(Iterable[IRNode], nodes_iterable):
            nodes_copy[getattr(n, "id", "")] = clone_logical_node(n)

    inputs_copy: list[str] = list(graph.inputs) if hasattr(graph, "inputs") and graph.inputs is not None else []
    outputs_copy: list[str] = list(graph.outputs) if hasattr(graph, "outputs") and graph.outputs is not None else []
    return {"nodes": nodes_copy, "inputs": inputs_copy, "outputs": outputs_copy}


def _restore_graph(graph: IRGraph, snapshot: dict[str, dict[str, IRNode] | list[str]]) -> None:
    """Restore graph nodes and topology from a snapshot.

    Args:
        graph (IRGraph): The target graph to restore in place.
        snapshot (dict[str, dict[str, IRNode] | list[str]]): The snapshot to restore from.
    """
    nodes_data = snapshot["nodes"]
    if isinstance(nodes_data, dict):
        if hasattr(graph.nodes, "clear") and hasattr(graph.nodes, "update"):
            graph.nodes.clear()
            graph.nodes.update(nodes_data)
        else:
            object.__setattr__(graph, "nodes", list(nodes_data.values()))
    if hasattr(graph, "inputs") and "inputs" in snapshot:
        graph.inputs = list(snapshot["inputs"])
    if hasattr(graph, "outputs") and "outputs" in snapshot:
        graph.outputs = list(snapshot["outputs"])


class PassManager:
    """Manages and executes optimization and transformation passes on an IR graph.

    The PassManager maintains a list of transformation passes and validation checks,
    ensuring that the graph remains structurally valid before and after
    transformations.
    """

    def __init__(self) -> None:
        """Initialize the PassManager."""
        self.passes: list[Callable[[IRGraph], bool]] = []
        self.pass_names: list[str] = []
        self.convergence_criteria: ConvergenceCriteria | None = None
        self.validators: list[Callable[[IRGraph], None]] = [
            IRValidator.check_cycles,
            IRValidator.check_shapes,
        ]

    def add_pass(self, ir_pass: Callable[[IRGraph], bool], name: str | None = None) -> None:
        """Add a pass to the manager.

        A pass should return True if it modified the graph.

        Args:
            ir_pass (Callable[[IRGraph], bool]): The transformation pass.
            name (str, optional): The name of the pass.
        """
        self.passes.append(ir_pass)
        self.pass_names.append(name or str(getattr(ir_pass, "__name__", "unknown_pass")))

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

        for pass_name in execution_order:
            pass_func = getattr(passes_module, f"{pass_name}_pass", None) or getattr(passes_module, pass_name, None)
            if pass_func and callable(pass_func):
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
            max_iterations = getattr(self.convergence_criteria, "max_iterations", max_iterations)

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
