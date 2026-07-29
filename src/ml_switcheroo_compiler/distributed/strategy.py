# ruff: noqa: D107, ANN401
"""Distributed training strategies for ML Switcheroo Compiler."""

from typing import Any

from ml_switcheroo_compiler.distributed import Distribution


class ParameterServerStrategy(Distribution):
    """Parameter server strategy."""

    def __init__(self, cluster_resolver: Any = None) -> None:
        super().__init__()
        self.cluster_resolver = cluster_resolver


class MultiWorkerMirroredStrategy(Distribution):
    """Multi-worker mirrored strategy."""

    def __init__(self, cluster_resolver: Any = None) -> None:
        super().__init__()
        self.cluster_resolver = cluster_resolver


class CentralStorageStrategy(Distribution):
    """Central storage strategy."""

    def __init__(self) -> None:
        super().__init__()


class TPUStrategy(Distribution):
    """TPU strategy."""

    def __init__(self, tpu_cluster_resolver: Any = None) -> None:
        super().__init__()
        self.tpu_cluster_resolver = tpu_cluster_resolver


class PreemptionCheckpointHandler:
    """Handles asynchronous checkpointing for preemptible instances."""

    def __init__(self, cluster_resolver: Any, checkpoint_dir: str) -> None:
        self.cluster_resolver = cluster_resolver
        self.checkpoint_dir = checkpoint_dir


class Server:
    """Distributed execution server."""

    def __init__(self, server_def: Any, job_name: str = None, task_index: int = None) -> None:
        self.server_def = server_def
        self.job_name = job_name
        self.task_index = task_index

    def start(self) -> None:
        """Start the server."""
        import ml_switcheroo_compiler.backends.registry as registry
        from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

        backend = registry.get_active_backend()
        if hasattr(backend, "start_server"):
            backend.start_server(self)
        else:
            raise BackendNotSupportedError(f"Active backend '{getattr(backend, '__name__', type(backend).__name__)}' does not support start_server()")

    def join(self) -> None:
        """Block until the server terminates."""
        import ml_switcheroo_compiler.backends.registry as registry
        from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

        backend = registry.get_active_backend()
        if hasattr(backend, "join_server"):
            backend.join_server(self)
        else:
            raise BackendNotSupportedError(f"Active backend '{getattr(backend, '__name__', type(backend).__name__)}' does not support join_server()")


class Coordinator:
    """Coordinator for distributed synchronization."""

    def __init__(self) -> None:
        """Initialize coordinator."""
        self.joined = False

    def join(self) -> None:
        """Wait for all threads to terminate."""
        self.joined = True


class TFConfigClusterResolver:
    """Resolves cluster topology from TF_CONFIG env var."""

    def __init__(self) -> None:
        """Initialize."""
        import json
        import os

        self.cluster = {}
        tf_config = os.environ.get("TF_CONFIG")
        if tf_config:
            try:
                self.cluster = json.loads(tf_config).get("cluster", {})
            except json.JSONDecodeError as e:
                import warnings

                warnings.warn(f"Failed to parse TF_CONFIG env var as JSON: {e}", stacklevel=2)


class KubernetesClusterResolver:
    """Resolves cluster topology from Kubernetes env vars."""

    def __init__(self) -> None:
        """Initialize."""
        import os

        self.cluster = {}
        if "KUBERNETES_SERVICE_HOST" in os.environ:
            # Basic dummy implementation
            self.cluster = {"worker": [f"{os.environ.get('HOSTNAME', 'localhost')}:8080"]}


class SlurmClusterResolver:
    """Resolves cluster topology from Slurm env vars."""

    def __init__(self) -> None:
        """Initialize."""
        import os

        self.cluster = {}
        if "SLURM_JOB_NODELIST" in os.environ:
            self.cluster = {"worker": os.environ["SLURM_JOB_NODELIST"].split(",")}


class PerWorkerValue:
    """Represents a value that varies across workers."""

    def __init__(self, values: list[Any]) -> None:
        self.values = values


class RemoteValue:
    """Represents a value that resides on a remote worker."""

    def __init__(self) -> None:
        """Initialize."""
        self.value = None


class MeshShardingStrategy(Distribution):
    """1D/2D Mesh Sharding Strategy for SPMD Graph Partitioning and Lowering."""

    def __init__(self, mesh: Any = None, layout_map: Any = None) -> None:
        """Initialize MeshShardingStrategy.

        Args:
            mesh (Any, optional): The device mesh (e.g., DeviceMesh).
            layout_map (Any, optional): LayoutMap specifying sharding specs.
        """
        super().__init__()
        self.mesh = mesh
        self.layout_map = layout_map

    def propagate_layouts(self, graph: Any) -> None:
        """Propagate sharding specifications along the graph dimensions.

        This pass traverses the graph, and if a node's inputs have sharding constraints
        or are sharded, it propagates that layout to the node itself if not already specified.

        Args:
            graph (Any): The IR graph to process.
        """
        for node in graph.nodes.values():
            if getattr(node, "sharding", None) is not None:
                continue

            # Check if any input has sharding to propagate
            for inp_id in node.inputs:
                inp_node = graph.nodes.get(inp_id)
                if inp_node and getattr(inp_node, "sharding", None) is not None:
                    node.sharding = inp_node.sharding
                    break

            # Check layout_map if available
            if self.layout_map:
                spec = self.layout_map.get(node.id)
                if spec:
                    node.sharding = spec

    def lower_sharding(self, graph: Any) -> bool:
        """Execute SPMD graph partitioning, lowering 1D/2D mesh sharding to explicit collectives.

        Args:
            graph (Any): The IR graph to partition.

        Returns:
            bool: True if the graph was modified, False otherwise.
        """
        # 1. Propagate layouts
        self.propagate_layouts(graph)

        # 2. Lower/inject collectives across boundary transitions
        from ml_switcheroo_compiler.transforms.passes.spmd import inject_spmd_communication_pass

        return inject_spmd_communication_pass(graph)
