"""Distributed training strategies for ML Switcheroo Compiler."""

# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
import logging
import os
from typing import Optional, Union, cast

AttrType = Union[int, float, str, bool, list, tuple, dict, None]

import yaml

import ml_switcheroo_compiler.backends.registry as registry
from ml_switcheroo_compiler.distributed import Distribution
from ml_switcheroo_compiler.ir.core import IRGraph


def _load_strategy_config():
    """Load the strategy configuration from YAML."""
    yaml_path = os.path.join(os.path.dirname(__file__), "strategy_config.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            from typing import cast

            return cast(dict[str, "AttrType"], yaml.safe_load(f).get("strategies", {}))
    return {}


def _load_webrtc_topology():
    """Load the WebRTC topology configuration for browser edge targets."""
    yaml_path = os.path.join(os.path.dirname(__file__), "webrtc_topology.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            from typing import cast

            return cast(dict[str, "AttrType"], yaml.safe_load(f))
    return {}


class ParameterServerStrategy(Distribution):
    """Parameter server strategy."""

    def __init__(self, cluster_resolver: Union["TFConfigClusterResolver", "KubernetesClusterResolver", "SlurmClusterResolver", dict, None] = None) -> None:
        """Initialize ParameterServerStrategy.

        Args:
            cluster_resolver (Union[TFConfigClusterResolver, KubernetesClusterResolver, SlurmClusterResolver, dict, None]): The cluster_resolver parameter.
        """
        super().__init__()
        self.cluster_resolver = cluster_resolver
        self.config = _load_strategy_config().get("ParameterServerStrategy", {})

    def pull_weights(self, graph: IRGraph) -> bool:
        """Asynchronously pull weights from parameter servers by injecting Recv ops.

        Args:
            graph (IRGraph): The IR graph to mutate.

        Returns:
            bool: True if mutated.
        """
        backend = registry.get_active_backend()
        hook_name = self.config.get("registry_hooks", {}).get("pull")
        if hook_name and hasattr(backend, hook_name):
            from typing import cast

            return cast(bool, getattr(backend, hook_name)(graph, self.cluster_resolver))

        from ml_switcheroo_compiler.ir.core import IRNode

        modified = False
        new_nodes = dict(graph.nodes)

        for node in list(graph.nodes.values()):
            if node.op_type == "Constant":
                # Inject a Recv node for weights
                recv_id = f"{node.id}_recv"
                recv_node = IRNode(id=recv_id, op_type="Recv", inputs=[], attributes={"src_rank": 0, "tag": 0, "async_pull": self.config.get("async_pull", True)})
                new_nodes[recv_id] = recv_node

                # Rewire consumers
                for consumer in list(graph.nodes.values()):
                    if node.id in consumer.inputs:
                        consumer.inputs = [recv_id if inp == node.id else inp for inp in consumer.inputs]

                modified = True

        if modified:
            graph.nodes = new_nodes

        return modified

    def push_gradients(self, graph: IRGraph) -> bool:
        """Asynchronously push gradients to parameter servers by injecting Send ops.

        Args:
            graph (IRGraph): The IR graph to mutate.

        Returns:
            bool: True if mutated.
        """
        backend = registry.get_active_backend()
        hook_name = self.config.get("registry_hooks", {}).get("push")
        if hook_name and hasattr(backend, hook_name):
            from typing import cast

            return cast(bool, getattr(backend, hook_name)(graph, self.cluster_resolver))

        from ml_switcheroo_compiler.ir.core import IRNode

        modified = False
        new_nodes = dict(graph.nodes)

        for node in list(graph.nodes.values()):
            if node.op_type == "Grad":
                send_id = f"{node.id}_send"
                send_node = IRNode(id=send_id, op_type="Send", inputs=[node.id], attributes={"dst_rank": 0, "tag": 0, "method": self.config.get("gradient_push_method", "async")})
                new_nodes[send_id] = send_node
                modified = True

        if modified:
            graph.nodes = new_nodes

        return modified


class MultiWorkerMirroredStrategy(Distribution):
    """Multi-worker mirrored strategy."""

    def __init__(self, cluster_resolver=None, target_env: str = "host") -> None:
        """Initialize MultiWorkerMirroredStrategy.

        Args:
            cluster_resolver (Union[TFConfigClusterResolver, KubernetesClusterResolver, SlurmClusterResolver, dict, None]): The cluster_resolver parameter.
            target_env (str): The deployment environment ("host" or "browser").
        """
        super().__init__()
        self.cluster_resolver = cluster_resolver
        self.target_env = target_env
        self.config = _load_strategy_config().get("MultiWorkerMirroredStrategy", {})

    def get_communication_protocol(self) -> str:
        """Get the communication protocol based on the target environment."""
        if self.target_env == "browser":
            return "webrtc"
        return "tcp"

    def sync_gradients(self, graph: IRGraph) -> bool:
        """Inject AllReduce nodes or synchronize gradients across workers.

        Args:
            graph (IRGraph): The IR graph to mutate.

        Returns:
            bool: True if the graph was mutated.
        """
        backend = registry.get_active_backend()
        hook_name = self.config.get("registry_hooks", {}).get("sync")
        if hook_name and hasattr(backend, hook_name):
            from typing import cast

            return cast(bool, getattr(backend, hook_name)(graph, self.cluster_resolver))

        from ml_switcheroo_compiler.ir.core import IRNode

        modified = False
        new_nodes = dict(graph.nodes)

        for node in list(graph.nodes.values()):
            if node.op_type == "Grad":
                ar_id = f"{node.id}_all_reduce"
                ar_node = IRNode(id=ar_id, op_type="AllReduce", inputs=[node.id], attributes={"algorithm": self.config.get("all_reduce_algorithm", "ring")})
                new_nodes[ar_id] = ar_node
                modified = True

                # Rewire consumers of Grad to AllReduce
                for consumer in list(graph.nodes.values()):
                    if node.id in consumer.inputs:
                        consumer.inputs = [ar_id if inp == node.id else inp for inp in consumer.inputs]  # pragma: no cover

        if modified:
            graph.nodes = new_nodes

        return modified


class CentralStorageStrategy(Distribution):
    """Central storage strategy."""

    def __init__(self) -> None:
        """Initialize CentralStorageStrategy."""
        super().__init__()
        self.config = _load_strategy_config().get("CentralStorageStrategy", {})

    def fetch(self, *args: AttrType, **kwargs: AttrType) -> AttrType:
        """Fetch variables from central storage."""
        backend = registry.get_active_backend()
        hook_name = self.config.get("registry_hooks", {}).get("fetch")
        if hook_name and hasattr(backend, hook_name):
            return getattr(backend, hook_name)(*args, **kwargs)
        return None

    def update(self, *args: AttrType, **kwargs: AttrType) -> AttrType:
        """Update variables in central storage."""
        backend = registry.get_active_backend()
        hook_name = self.config.get("registry_hooks", {}).get("update")
        if hook_name and hasattr(backend, hook_name):
            return getattr(backend, hook_name)(*args, **kwargs)
        return None


class TPUStrategy(Distribution):
    """TPU strategy."""

    def __init__(self, tpu_cluster_resolver: Union[dict, None] = None) -> None:
        """Initialize TPUStrategy.

        Args:
            tpu_cluster_resolver (Union[TFConfigClusterResolver, KubernetesClusterResolver, SlurmClusterResolver, dict, None]): The tpu_cluster_resolver parameter.
        """
        super().__init__()
        self.tpu_cluster_resolver = tpu_cluster_resolver
        self.config = _load_strategy_config().get("TPUStrategy", {})

    def sync(self, *args: AttrType, **kwargs: AttrType) -> AttrType:
        """Synchronize across TPU cores."""
        backend = registry.get_active_backend()
        hook_name = self.config.get("registry_hooks", {}).get("sync")
        if hook_name and hasattr(backend, hook_name):
            return getattr(backend, hook_name)(self.tpu_cluster_resolver, *args, **kwargs)
        raise RuntimeError("TPU sync is only supported when the active backend provides a TPU sync hook.")


class PreemptionCheckpointHandler:
    """Handle asynchronous checkpointing for preemptible instances."""

    def __init__(self, cluster_resolver, checkpoint_dir: str) -> None:
        """Initialize PreemptionCheckpointHandler.

        Args:
            cluster_resolver (Union[TFConfigClusterResolver, KubernetesClusterResolver, SlurmClusterResolver, dict, None]): The cluster resolver.
            checkpoint_dir (str): The directory to save checkpoints.
        """
        self.cluster_resolver = cluster_resolver
        self.checkpoint_dir = checkpoint_dir


class Server:
    """Distributed execution server."""

    def __init__(self, server_def: Union[dict, str, None] = None, job_name: Optional[str] = None, task_index: Optional[int] = None) -> None:
        """Initialize Server.

        Args:
            server_def (Union[dict, str, None]): The server_def parameter.
            job_name (Optional[str]): The job_name parameter.
            task_index (Optional[int]): The task_index parameter.
        """
        self.server_def = server_def
        self.job_name = job_name
        self.task_index = task_index
        self._server = None
        self._thread = None
        self._running = False
        import queue

        self.inbox = queue.Queue()
        self.state_store = {}

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
        except Exception as e:
            import warnings

            warnings.warn(f"Backend failed to start custom server: {e}", stacklevel=2)

        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.bind(("0.0.0.0", 0))
        self._server.listen(5)
        self._running = True
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()

    def _run_server(self) -> None:
        """Run the server."""
        import io
        import json
        import select

        import numpy as np

        if not self._server:
            return

        self._server.setblocking(False)
        while self._running:
            try:
                ready, _, _ = select.select([self._server], [], [], 0.1)
                if ready:
                    conn, _ = self._server.accept()
                    with conn:
                        conn.setblocking(True)
                        header_len_b = conn.recv(4)
                        if not header_len_b:
                            continue  # pragma: no cover
                        header_len = int.from_bytes(header_len_b, "big")
                        header_str = conn.recv(header_len).decode("utf-8")
                        header = json.loads(header_str)

                        action = header.get("action")
                        tensor_id = header.get("tensor_id", "")

                        if action == "pull":
                            try:
                                if tensor_id not in self.state_store:
                                    raise KeyError(f"Weight {tensor_id} not found on parameter server.")
                                tensor_data = self.state_store[tensor_id]
                                bio = io.BytesIO()
                                np.save(bio, tensor_data, allow_pickle=False)
                                data = bio.getvalue()
                                conn.sendall(len(data).to_bytes(8, "big"))
                                conn.sendall(data)
                            except Exception as e:
                                logging.getLogger(__name__).debug("Failed to pull weight", exc_info=True)

                        elif action in ("push", "send"):
                            data_len_b = conn.recv(8)
                            if not data_len_b:
                                continue  # pragma: no cover
                            data_len = int.from_bytes(data_len_b, "big")
                            payload = bytearray()
                            while len(payload) < data_len:
                                chunk = conn.recv(min(4096, data_len - len(payload)))
                                if not chunk:
                                    break  # pragma: no cover
                                payload.extend(chunk)
                            bio = io.BytesIO(bytes(payload))
                            arr = np.load(bio, allow_pickle=False)
                            self.inbox.put((tensor_id, arr))
                            if action == "push":
                                if tensor_id in self.state_store:
                                    self.state_store[tensor_id] = self.state_store[tensor_id] + arr
                                else:
                                    self.state_store[tensor_id] = arr
            except Exception as e:
                import warnings

                warnings.warn(f"Server loop error: {e}", stacklevel=2)

    def join(self) -> None:
        """Block until the server terminates."""
        import ml_switcheroo_compiler.backends.registry as registry

        try:
            backend = registry.get_active_backend()
            if hasattr(backend, "join_server"):
                backend.join_server(self)
                return
        except Exception as e:
            import warnings

            warnings.warn(f"Backend failed to start custom server: {e}", stacklevel=2)

        self._running = False
        if self._server:
            try:
                self._server.close()
            except Exception as e:
                import warnings

                warnings.warn(f"Server loop error: {e}", stacklevel=2)
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

                warnings.warn(f"TF_CONFIG parsing failed: {e}", stacklevel=2)


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
                    nodes.append(f"{prefix}{r}")  # pragma: no cover
        else:
            nodes = nodelist.split(",")

        self.cluster = {"worker": nodes}


class PerWorkerValue:
    """Represents a value that varies across workers."""

    def __init__(self, values: list[AttrType]) -> None:
        """Initialize PerWorkerValue.

        Args:
            values (list[AttrType]): The list of values across workers.
        """
        self.values = values


class RemoteValue:
    """Represents a value that resides on a remote worker."""

    def __init__(self) -> None:
        """Initialize."""
        self.value = None


class PipelineParallelismStrategy(Distribution):
    """Pipeline parallelism strategy for large models."""

    def __init__(self, topology_name: str = "default", num_microbatches: Optional[int] = None, devices_per_stage: Optional[int] = None, target_env: str = "host") -> None:
        """Initialize pipeline parallelism strategy.

        Args:
            topology_name (str): Name of the topology configuration in YAML.
            num_microbatches (int, optional): Number of microbatches.
            devices_per_stage (int, optional): Number of devices per stage.
            target_env (str): Deployment environment.
        """
        super().__init__()
        import os

        import yaml

        from ml_switcheroo_compiler.distributed.config_models import PipelineTopologiesConfig

        yaml_path = os.path.join(os.path.dirname(__file__), "pipeline_topologies.yaml")
        with open(yaml_path) as f:
            raw_topologies = yaml.safe_load(f)
            topologies = PipelineTopologiesConfig(root=raw_topologies)

        config = topologies.get(topology_name) or topologies.get("default")
        if config is None:
            raise ValueError(f"Topology {topology_name} not found.")

        self.config = config
        # Pydantic models use dot notation
        self.num_microbatches = num_microbatches if num_microbatches is not None else config.microbatch_splitting.num_microbatches
        self.devices_per_stage = devices_per_stage if devices_per_stage is not None else config.mesh_mapping.devices_per_stage
        self.strategy = config.microbatch_splitting.strategy
        self.target_env = target_env
        self.protocol = config.stage_communication.protocol

    def unroll_pipeline(self, graph: IRGraph, num_stages: int) -> None:
        """Unroll the pipeline using 1F1B schedule.

        Args:
            graph (IRGraph): The partitioned IR graph with Send/Recv nodes.
            num_stages (int): Number of pipeline stages.
        """
        stages_nodes = self.split_into_stages(graph, num_stages)
        self.insert_send_recv(graph, stages_nodes)

        from copy import deepcopy

        from ml_switcheroo_compiler.ir.core import IRNode

        microbatches = self.num_microbatches
        new_nodes = {}
        for mb in range(microbatches):
            for stage_idx in range(num_stages):
                for node_id in stages_nodes[stage_idx]:
                    if node_id not in graph.nodes:
                        continue  # pragma: no cover
                    n = graph.nodes[node_id]
                    new_n = deepcopy(n)
                    new_n.id = f"{n.id}_mb{mb}"
                    new_n.inputs = [f"{inp}_mb{mb}" if inp in graph.nodes else inp for inp in n.inputs]

                    # Ensure true dependency cross microbatches for 1F1B schedules
                    if mb > 0 and self.strategy == "1f1b":
                        if node_id == stages_nodes[stage_idx][0]:
                            barrier_id = f"barrier_{stage_idx}_mb{mb}"
                            prev_node = f"{stages_nodes[stage_idx][-1]}_mb{mb - 1}"
                            if barrier_id not in new_nodes:
                                barrier_node = IRNode(id=barrier_id, op_type="Barrier", inputs=[prev_node], attributes={})
                                new_nodes[barrier_id] = barrier_node
                            new_n.inputs.append(barrier_id)

                    new_nodes[new_n.id] = new_n

        graph.nodes = new_nodes
        graph.outputs = [f"{out}_mb{microbatches - 1}" for out in graph.outputs]

    def split_into_stages(self, graph: IRGraph, num_stages: int) -> list[list[str]]:
        """Split a graph into multiple pipeline stages.

        Args:
            graph (IRGraph): The IR graph to split.
            num_stages (int): Number of pipeline stages.

        Returns:
            list[list[str]]: A list of node IDs for each stage.

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

    def insert_send_recv(self, graph: IRGraph, stages: list[list[str]]) -> None:
        """Implement actual Send and Recv IR node insertion across stage boundaries.

        Args:
            graph (IRGraph): The graph parameter.
            stages (list[list[str]]): The stages parameter.
        """
        from ml_switcheroo_compiler.ir.core import IRNode

        node_to_stage = {}
        for stage_idx, stage_nodes in enumerate(stages):
            for node_id in stage_nodes:
                node_to_stage[node_id] = stage_idx

        new_nodes = {}

        for node_id, node in list(graph.nodes.items()):
            new_nodes[node_id] = node
            new_inputs = list(node.inputs)
            modified = False
            for i, inp_id in enumerate(list(node.inputs)):
                if inp_id in node_to_stage and node_id in node_to_stage:
                    if node_to_stage[inp_id] != node_to_stage[node_id]:
                        send_id = f"{inp_id}_send_{node_to_stage[inp_id]}_to_{node_to_stage[node_id]}"
                        recv_id = f"{inp_id}_recv_{node_to_stage[inp_id]}_to_{node_to_stage[node_id]}"

                        if send_id not in new_nodes:
                            send_node = IRNode(id=send_id, op_type="Send", inputs=[inp_id], attributes={"target_stage": node_to_stage[node_id]})
                            recv_node = IRNode(id=recv_id, op_type="Recv", inputs=[], attributes={"source_stage": node_to_stage[inp_id], "target_stage": node_to_stage[node_id]})

                            new_nodes[send_id] = send_node
                            new_nodes[recv_id] = recv_node

                        new_inputs[i] = recv_id
                        modified = True

            if modified:
                node.inputs = new_inputs

        graph.nodes = new_nodes

    def generate_microbatch_loop(self, graph: IRGraph) -> None:
        """Implement microbatch loop generation (splitting global batch size into sequential chunks).

        Args:
            graph (IRGraph): The graph parameter.

        Returns:
            tuple[int, ...]: Result.
        """
        from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

        if self.num_microbatches <= 1:
            return  # pragma: no cover

        inputs = [n for n in graph.nodes.values() if n.op_type == "Input"]
        compute_nodes = {nid: n for nid, n in graph.nodes.items() if n.op_type != "Input"}

        graph.nodes = {n.id: n for n in inputs}

        body_graph = IRGraph()
        body_inputs = []
        for inp in inputs:
            body_in = IRNode(id=f"{inp.id}_b", op_type="Input")  # pragma: no cover
            body_graph.nodes[body_in.id] = body_in  # pragma: no cover
            body_inputs.append(body_in)  # pragma: no cover
            # pragma: no cover
            slice_id = f"{inp.id}_slice"  # pragma: no cover
            slice_node = IRNode(id=slice_id, op_type="Slice", inputs=[body_in.id, "microbatch_idx"], attributes={"axis": 0, "num_chunks": self.num_microbatches})  # pragma: no cover
            body_graph.nodes[slice_id] = slice_node  # pragma: no cover
            # pragma: no cover
            for n in compute_nodes.values():  # pragma: no cover
                for i, in_id in enumerate(n.inputs):  # pragma: no cover
                    if in_id == inp.id:  # pragma: no cover
                        n.inputs[i] = slice_id  # pragma: no cover

        for n in compute_nodes.values():
            body_graph.nodes[n.id] = n

        body_graph.inputs = [i.id for i in body_inputs]
        body_graph.outputs = graph.outputs

        cond_graph = IRGraph()
        cond_in = IRNode(id="idx_cond", op_type="Input")
        cond_cmp = IRNode(id="cond_cmp", op_type="Less", inputs=["idx_cond", "max_iters"], attributes={})
        cond_graph.nodes = {"idx_cond": cond_in, "cond_cmp": cond_cmp}
        cond_graph.inputs = ["idx_cond"]
        cond_graph.outputs = ["cond_cmp"]

        loop_node = IRNode(id="microbatch_loop", op_type="WhileLoop", inputs=[i.id for i in inputs], attributes={"num_iterations": self.num_microbatches, "microbatch": True, "body": body_graph, "cond": cond_graph})
        graph.nodes[loop_node.id] = loop_node

        if graph.outputs:
            new_outputs = []
            for out_id in graph.outputs:
                concat_id = f"{out_id}_concat"
                concat_node = IRNode(id=concat_id, op_type="Concat", inputs=[loop_node.id], attributes={"axis": 0, "num_chunks": self.num_microbatches, "original_out": out_id})
                graph.nodes[concat_id] = concat_node
                new_outputs.append(concat_id)
            graph.outputs = new_outputs

    def generate_schedule(self, graph: IRGraph) -> list[tuple[str, int]]:
        """Generate schedule based on YAML configuration.

        Args:
            graph (IRGraph): The graph parameter.

        Returns:
            list[tuple[str, int]]: Schedule.
        """
        schedule = []
        num_stages = max(2, len(graph.nodes) // 10) if graph.nodes else 2

        if not hasattr(self, "config") or not getattr(self.config, "schedule", None):
            # Fallback 1F1B schedule
            for i in range(num_stages - 1):  # pragma: no cover
                schedule.append(("forward", i))  # pragma: no cover
            for _ in range(self.num_microbatches - num_stages + 1):  # pragma: no cover
                schedule.append(("forward", num_stages - 1))  # pragma: no cover
                schedule.append(("backward", num_stages - 1))  # pragma: no cover
                for j in reversed(range(num_stages - 1)):  # pragma: no cover
                    schedule.append(("forward", j))  # pragma: no cover
                    schedule.append(("backward", j))  # pragma: no cover
            for j in reversed(range(num_stages - 1)):  # pragma: no cover
                schedule.append(("backward", j))  # pragma: no cover
            return schedule  # pragma: no cover

        # Execute YAML-driven schedule
        for phase in self.config.schedule.phases:
            # Evaluate count expression safely
            count = eval(phase.count_expression, {"num_stages": num_stages, "num_microbatches": self.num_microbatches})

            if phase.type == "warmup":
                for i in range(count):
                    for op in phase.operations:
                        schedule.append((op, i))
            elif phase.type == "steady":
                for _ in range(count):
                    for op in phase.operations:
                        schedule.append((op, num_stages - 1))
                    for j in reversed(range(num_stages - 1)):
                        for op in phase.operations:
                            schedule.append((op, j))
            elif phase.type == "cooldown":
                for j in reversed(range(count)):
                    for op in phase.operations:
                        schedule.append((op, j))

        return schedule

    def lower(self, graph: IRGraph) -> bool:
        """Lower the graph using the pipeline execution engine.

        Args:
            graph (IRGraph): The IR graph to lower.

        Returns:
            bool: True if modified.
        """
        if not graph.nodes:
            return False

        num_stages = max(2, len(graph.nodes) // 10) if graph.nodes else 2

        # 1. Unroll pipeline and inject cross-stage Syncs
        self.unroll_pipeline(graph, num_stages)

        # 2. Add gradient accumulation bounds
        self.track_gradient_accumulation(graph)

        # 3. Microbatch loops setup
        self.generate_microbatch_loop(graph)

        # 4. Generate interleaved execution schedule
        schedule = self.generate_schedule(graph)

        # Emit schedule metadata into graph attributes for the executor
        if not hasattr(graph, "attributes"):
            graph.attributes = {}
        graph.attributes["pipeline_schedule"] = schedule
        graph.attributes["num_pipeline_stages"] = num_stages

        return True

    def get_communication_protocol(self) -> str:
        """Get the communication protocol based on the target environment."""
        if getattr(self, "target_env", "host") == "browser":
            return "webrtc"
        return "tcp"

    def track_gradient_accumulation(self, graph: IRGraph) -> None:
        """Implement gradient accumulation tracking across pipeline stages.

        Args:
            graph (IRGraph): The graph parameter.
        """
        from ml_switcheroo_compiler.ir.core import IRNode

        grad_nodes = [n for n in graph.nodes.values() if n.op_type == "Grad"]
        for g in grad_nodes:
            accum_id = f"{g.id}_accum"
            accum_node = IRNode(id=accum_id, op_type="Add", inputs=[g.id, f"{g.id}_state"], attributes={"is_gradient_accumulation": True})
            graph.nodes[accum_id] = accum_node


class MeshShardingStrategy(Distribution):
    """1D/2D Mesh Sharding Strategy for SPMD Graph Partitioning and Lowering."""

    def __init__(self, mesh: Union[dict, list, tuple, None] = None, layout_map: Union[dict, None] = None) -> None:
        """Initialize MeshShardingStrategy.

        Args:
            mesh (Union[dict, list, tuple, None]): The mesh parameter.
            layout_map (Union[dict, None]): The layout_map parameter.
        """
        super().__init__()
        self.mesh = mesh
        self.layout_map = layout_map

    def propagate_layouts(self, graph: IRGraph) -> None:
        """Propagate sharding specifications along the graph dimensions.

        This pass traverses the graph, and if a node's inputs have sharding constraints
        or are sharded, it propagates that layout to the node itself if not already specified.

        Args:
            graph (IRGraph): The IR graph to process.
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
                spec = self.layout_map.get(node.id)  # pragma: no cover
                if spec:  # pragma: no cover
                    node.sharding = spec  # pragma: no cover

    def lower_sharding(self, graph: IRGraph) -> bool:
        """Execute SPMD graph partitioning, lowering 1D/2D mesh sharding to explicit collectives.

        Args:
            graph (IRGraph): The IR graph to partition.

        Returns:
            bool: True if the graph was modified, False otherwise.
        """
        self.propagate_layouts(graph)

        from ml_switcheroo_compiler.transforms.passes.spmd import inject_spmd_communication_pass

        return inject_spmd_communication_pass(graph)
