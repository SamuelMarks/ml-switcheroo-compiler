# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module pass_manager.py."""

from typing import Any

"""Pass Manager Infrastructure for Middle-End Transformations.

This module provides components for managing, validating, and executing optimization and
transformation passes on Intermediate Representation (IR) graphs. It includes
topological sorting, structural and shape validation, and a pass manager that can run
passes sequentially or until convergence
"""

import hashlib
import json
from typing import Callable

from ml_switcheroo_compiler.core.errors import CompilationError
from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


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
        # Simple validator: ensure every node has shape metadata
        for node_id, node in graph.nodes.items():
            if getattr(node, "shape_metadata", None) is None:
                msg = f"Node {node_id} is missing shape_metadata."
                raise CompilationError(msg)


def _graph_hash(graph: IRGraph) -> str:
    """Evaluate _graph_hash operation.

    Args:
        graph (IRGraph): The graph parameter.

    Returns:
        str: Result.
    """
    state = {}
    for node_id, node in graph.nodes.items():
        state[node_id] = {
            "op": node.op_type,
            "inputs": node.inputs,
            # Ignore attributes that might not be easily serializable for basic hash
        }
    return hashlib.md5(json.dumps(state, sort_keys=True).encode("utf-8")).hexdigest()


class PassManager:
    """Manages and executes optimization and transformation passes on an IR graph.

    The PassManager maintains a list of transformation passes and validation checks,
    ensuring that the graph remains structurally valid before and after
    transformations
    """

    def __init__(self) -> None:
        """Initialize the PassManager."""
        self.passes: list[Callable[[IRGraph], bool]] = []
        self.pass_names: list[str] = []
        self.validators: list[Callable[[IRGraph], None]] = [
            IRValidator.check_cycles,
            IRValidator.check_shapes,
        ]

    def add_pass(self, ir_pass: Callable[[IRGraph], bool], name: Any = None) -> None:
        """Add a pass to the manager.

        A pass should return True if it modified the graph

        Args:
            ir_pass (Callable[[IRGraph], bool]): The transformation pass.
            name (str, optional): The name of the pass.
        """
        self.passes.append(ir_pass)
        self.pass_names.append(name or ir_pass.__name__)

    def load_from_config(self) -> None:
        """Load passes based on pass_config.yaml execution_order."""
        import os

        import yaml

        from ml_switcheroo_compiler.transforms.passes.config_models import PassConfig

        yaml_path = os.path.join(os.path.dirname(__file__), "pass_config.yaml")
        if not os.path.exists(yaml_path):
            return

        with open(yaml_path) as f:
            res = yaml.safe_load(f)
            config = PassConfig(**res)

        import importlib

        import ml_switcheroo_compiler.transforms.passes as passes_module

        self.passes = []
        self.pass_names = []

        # Keep a mapping of known passes, normally this would use a registry
        # We will dynamically look them up from the passes package
        for pass_name in config.execution_order:
            pass_func = getattr(passes_module, f"{pass_name}_pass", None)
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

        Args:
            graph (IRGraph): The intermediate representation graph.
            max_iterations (int): The maximum number of iterations.

        Returns:
            IRGraph: The transformed graph.
        """
        self.validate(graph)

        for _ in range(max_iterations):
            prev_hash = _graph_hash(graph)

            for ir_pass in self.passes:
                if ir_pass(graph):
                    from ml_switcheroo_compiler.transforms.passes.dtype_inference import dtype_inference_pass
                    from ml_switcheroo_compiler.transforms.passes.shape_inference import shape_inference_pass

                    shape_inference_pass(graph)
                    dtype_inference_pass(graph)
                self.validate(graph)

            new_hash = _graph_hash(graph)
            if new_hash == prev_hash:
                break

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
        import typing

        return list(topological_sort(graph))
