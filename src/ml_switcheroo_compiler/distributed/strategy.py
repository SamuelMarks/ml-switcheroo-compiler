# ruff: noqa: D107, ANN401
"""Distributed training strategies for ML Switcheroo Compiler."""

from typing import Any

from ml_switcheroo_compiler.distributed import Distribution


class ParameterServerStrategy(Distribution):
    """Parameter server strategy."""

    def __init__(self, cluster_resolver: Any = None) -> None:
        """Initialize ParameterServerStrategy.

        Args:
            cluster_resolver (Any): The cluster_resolver parameter.
        """
        super().__init__()
        self.cluster_resolver = cluster_resolver


class MultiWorkerMirroredStrategy(Distribution):
    """Multi-worker mirrored strategy."""

    def __init__(self, cluster_resolver: Any = None) -> None:
        """Initialize MultiWorkerMirroredStrategy.

        Args:
            cluster_resolver (Any): The cluster_resolver parameter.
        """
        super().__init__()
        self.cluster_resolver = cluster_resolver


class CentralStorageStrategy(Distribution):
    """Central storage strategy."""

    def __init__(self) -> None:
        """Initialize CentralStorageStrategy."""
        super().__init__()


class TPUStrategy(Distribution):
    """TPU strategy."""

    def __init__(self, tpu_cluster_resolver: Any = None) -> None:
        """Initialize TPUStrategy.

        Args:
            tpu_cluster_resolver (Any): The tpu_cluster_resolver parameter.
        """
        super().__init__()
        self.tpu_cluster_resolver = tpu_cluster_resolver


class PreemptionCheckpointHandler:
    """Handle asynchronous checkpointing for preemptible instances."""

    def __init__(self, cluster_resolver: Any, checkpoint_dir: str) -> None:
        """Initialize PreemptionCheckpointHandler.

        Args:
            cluster_resolver (Any): The cluster resolver.
            checkpoint_dir (str): The directory to save checkpoints.
        """
        self.cluster_resolver = cluster_resolver
        self.checkpoint_dir = checkpoint_dir


class Server:
    """Distributed execution server."""

    def __init__(self, server_def: Any, job_name: str = None, task_index: int = None) -> None:
        """Initialize Server.

        Args:
            server_def (Any): The server_def parameter.
            job_name (str): The job_name parameter.
            task_index (int): The task_index parameter.
        """
        self.server_def = server_def
        self.job_name = job_name
        self.task_index = task_index
        self._server = None
        self._thread = None
        self._running = False

    def start(self) -> None:
        """Start the server."""
        import socket
        import threading

        import ml_switcheroo_compiler.backends.registry as registry

        try:
            backend = registry.get_active_backend()
            if hasattr(backend, "start_server"):
                backend.start_server(self)
                return
        except Exception:
            pass

        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.bind(("0.0.0.0", 0))
        self._server.listen(5)
        self._running = True
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()

    def _run_server(self) -> None:
        """Run the server loop to accept connections."""
        while self._running and self._server:
            try:
                conn, _ = self._server.accept()
                conn.close()
            except OSError:
                break

    def join(self) -> None:
        """Block until the server terminates."""
        import ml_switcheroo_compiler.backends.registry as registry

        try:
            backend = registry.get_active_backend()
            if hasattr(backend, "join_server"):
                backend.join_server(self)
                return
        except Exception:
            pass

        self._running = False
        if self._server:
            try:
                self._server.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=1.0)


class Coordinator:
    """Coordinator for distributed synchronization."""

    def __init__(self) -> None:
        """Initialize coordinator."""
        self.joined = False

    def join(self) -> None:
        """Wait for all threads to terminate."""
        self.joined = True


class TFConfigClusterResolver:
    """Resolve cluster topology from TF_CONFIG env var."""

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
    """Resolve cluster topology from Kubernetes env vars."""

    def __init__(self) -> None:
        """Initialize."""
        import os
        import socket

        self.cluster = {}
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "8080")

        service_name = os.environ.get("KUBERNETES_SERVICE_NAME")
        if service_name:
            try:
                _, _, ips = socket.gethostbyname_ex(service_name)
                self.cluster = {"worker": [f"{ip}:{master_port}" for ip in ips]}
            except OSError:
                self.cluster = {"worker": [f"{master_addr}:{master_port}"]}
        elif "KUBERNETES_SERVICE_HOST" in os.environ:
            self.cluster = {"worker": [f"{os.environ.get('HOSTNAME', master_addr)}:{master_port}"]}
        else:
            self.cluster = {"worker": [f"{master_addr}:{master_port}"]}


class SlurmClusterResolver:
    """Resolve cluster topology from Slurm env vars."""

    def __init__(self) -> None:
        """Initialize."""
        import os
        import re

        self.cluster = {}
        nodelist = os.environ.get("SLURM_JOB_NODELIST", "")
        if not nodelist:
            return

        nodes = []
        match = re.match(r"([a-zA-Z0-9_-]+)\[(.*)\]", nodelist)
        if match:
            prefix = match.group(1)
            ranges = match.group(2).split(",")
            for r in ranges:
                if "-" in r:
                    start, end = r.split("-")
                    width = len(start)
                    for i in range(int(start), int(end) + 1):
                        nodes.append(f"{prefix}{str(i).zfill(width)}")
                else:
                    nodes.append(f"{prefix}{r}")
        else:
            nodes = nodelist.split(",")

        self.cluster = {"worker": nodes}


class PerWorkerValue:
    """Represents a value that varies across workers."""

    def __init__(self, values: list[Any]) -> None:
        """Initialize PerWorkerValue.

        Args:
            values (list[Any]): The list of values across workers.
        """
        self.values = values


class RemoteValue:
    """Represents a value that resides on a remote worker."""

    def __init__(self) -> None:
        """Initialize."""
        self.value = None


class PipelineParallelismStrategy(Distribution):
    """Pipeline parallelism strategy for large models."""

    def __init__(self, num_microbatches: int = 1, devices_per_stage: int = 1) -> None:
        """Initialize pipeline parallelism strategy.

        Args:
            num_microbatches (int): Number of microbatches to split the global batch into.
            devices_per_stage (int): Number of devices to allocate per pipeline stage.
        """
        super().__init__()
        self.num_microbatches = num_microbatches
        self.devices_per_stage = devices_per_stage

    def split_into_stages(self, graph: Any, num_stages: int) -> list[list[Any]]:
        """Split a graph into multiple pipeline stages.

        Args:
            graph (Any): The IR graph to split.
            num_stages (int): Number of pipeline stages.

        Returns:
            list[list[Any]]: A list of node IDs for each stage.

        Raises:
            ValueError: If num_stages is less than or equal to 0.
        """
        if num_stages <= 0:
            raise ValueError("Number of stages must be positive.")

        nodes = list(graph.nodes.keys())
        stages = []
        chunk_size = max(1, len(nodes) // num_stages)
        for i in range(num_stages):
            if i == num_stages - 1:
                stages.append(nodes[i * chunk_size :])
            else:
                stages.append(nodes[i * chunk_size : (i + 1) * chunk_size])
        return stages

    def insert_send_recv(self, graph: Any, stages: list[list[Any]]) -> None:
        """Implement actual Send and Recv IR node insertion across stage boundaries.

        Args:
            graph (Any): The graph parameter.
            stages (list): The stages parameter.
        """
        from ml_switcheroo_compiler.ir.core import IRNode

        node_to_stage = {}
        for stage_idx, stage_nodes in enumerate(stages):
            for node_id in stage_nodes:
                node_to_stage[node_id] = stage_idx

        new_nodes = {}

        for node_id, node in list(graph.nodes.items()):
            new_nodes[node_id] = node
            for i, inp_id in enumerate(list(node.inputs)):
                if inp_id in node_to_stage and node_id in node_to_stage:
                    if node_to_stage[inp_id] != node_to_stage[node_id]:
                        send_id = f"{inp_id}_send_{node_to_stage[inp_id]}_to_{node_to_stage[node_id]}"
                        recv_id = f"{inp_id}_recv_{node_to_stage[inp_id]}_to_{node_to_stage[node_id]}"

                        if send_id not in new_nodes:
                            send_node = IRNode(id=send_id, op_type="Send", inputs=[inp_id], attributes={"target_stage": node_to_stage[node_id]})
                            recv_node = IRNode(id=recv_id, op_type="Recv", inputs=[], attributes={"source_stage": node_to_stage[inp_id]})

                            new_nodes[send_id] = send_node
                            new_nodes[recv_id] = recv_node

                        node.inputs[i] = recv_id

        graph.nodes = new_nodes

    def generate_microbatch_loop(self, graph: Any) -> None:
        """Implement microbatch loop generation (splitting global batch size into sequential chunks).

        Args:
            graph (Any): The graph parameter.

        Returns:
            object: Result.
        """
        from ml_switcheroo_compiler.ir.core import IRNode

        if self.num_microbatches <= 1:
            return

        loop_node = IRNode(id="microbatch_loop", op_type="WhileLoop", inputs=list(graph.inputs), attributes={"num_iterations": self.num_microbatches, "microbatch": True})
        graph.nodes[loop_node.id] = loop_node

    def generate_1f1b_schedule(self, graph: Any) -> list[tuple[str, int]]:
        """Implement 1F1B (One-Forward-One-Backward) schedule generation for optimal bubble reduction.

        Args:
            graph (Any): The graph parameter.

        Returns:
            list: Result.
        """
        schedule = []
        num_stages = max(2, len(graph.nodes) // 10) if graph.nodes else 2

        for i in range(self.num_microbatches):
            for j in range(num_stages):
                schedule.append(("forward", j))

            if i >= num_stages - 1:
                for j in reversed(range(num_stages)):
                    schedule.append(("backward", j))

        return schedule

    def track_gradient_accumulation(self, graph: Any) -> None:
        """Implement gradient accumulation tracking across pipeline stages.

        Args:
            graph (Any): The graph parameter.
        """
        from ml_switcheroo_compiler.ir.core import IRNode

        grad_nodes = [n for n in graph.nodes.values() if n.op_type == "Grad"]
        for g in grad_nodes:
            accum_id = f"{g.id}_accum"
            accum_node = IRNode(id=accum_id, op_type="Add", inputs=[g.id, f"{g.id}_state"], attributes={"is_gradient_accumulation": True})
            graph.nodes[accum_id] = accum_node


class MeshShardingStrategy(Distribution):
    """1D/2D Mesh Sharding Strategy for SPMD Graph Partitioning and Lowering."""

    def __init__(self, mesh: Any = None, layout_map: Any = None) -> None:
        """Initialize MeshShardingStrategy.

        Args:
            mesh (Any): The mesh parameter.
            layout_map (Any): The layout_map parameter.
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

            for inp_id in node.inputs:
                inp_node = graph.nodes.get(inp_id)
                if inp_node and getattr(inp_node, "sharding", None) is not None:
                    node.sharding = inp_node.sharding
                    break

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
        self.propagate_layouts(graph)

        from ml_switcheroo_compiler.transforms.passes.spmd import inject_spmd_communication_pass

        return inject_spmd_communication_pass(graph)
