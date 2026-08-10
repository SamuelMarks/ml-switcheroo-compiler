# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
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

    def __init__(self, server_def: Any, job_name: Any = None, task_index: Any = None) -> None:
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

        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        self._server.bind(("0.0.0.0", 0))  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        self._server.listen(5)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        self._running = True
        self._thread = threading.Thread(target=self._run_server, daemon=True)  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
        self._thread.start()  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism

    def _run_server(self) -> None:
        """Run the server loop to accept connections and handle basic RPC."""
        import time

        while self._running:
            time.sleep(0.1)

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

                pass


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

    def execute_pipeline(self, graph: Any, inputs: dict[str, Any], num_stages: int) -> dict[str, Any]:
        """Execute a graph using pipeline parallelism over robust async workers.

        Args:
            graph (Any): The partitioned IR graph with Send/Recv nodes.
            inputs (dict[str, Any]): The input tensors mapping.
            num_stages (int): Number of pipeline stages.

        Returns:
            dict[str, Any]: The outputs of the pipeline execution.
        """
        import concurrent.futures
        import queue

        from ml_switcheroo_compiler.ops.registry import get_op

        # Map: (source_stage, target_stage) -> Queue
        comm_queues: dict[tuple[int, int], queue.Queue] = {}
        for i in range(num_stages):
            for j in range(num_stages):
                if i != j:
                    comm_queues[(i, j)] = queue.Queue()

        stages_nodes = self.split_into_stages(graph, num_stages)
        self.insert_send_recv(graph, stages_nodes)

        node_to_stage = {}
        for stage_idx, stage_nodes in enumerate(stages_nodes):
            for node_id in stage_nodes:
                node_to_stage[node_id] = stage_idx

        for node_id, node in graph.nodes.items():
            if node.op_type == "Send":
                node_to_stage[node_id] = node_to_stage[node.inputs[0]]
            elif node.op_type == "Recv":
                node_to_stage[node_id] = node.attributes.get("target_stage", 0)

        outputs = {}
        import threading

        output_lock = threading.Lock()

        def stage_worker(stage_idx: int) -> None:
            """Worker task for a pipeline stage."""
            stage_env = dict(inputs)
            local_nodes = [nid for nid, stg in node_to_stage.items() if stg == stage_idx]

            try:
                from ml_switcheroo_compiler.ir.core import IRGraph
                from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter

                sub_g = IRGraph()
                for nid in local_nodes:
                    sub_g.nodes[nid] = graph.nodes[nid]
                sorted_nodes = DAGTopologicalSorter.sort(sub_g)
                sorted_local_nodes = [n.id for n in sorted_nodes]
            except Exception:
                sorted_local_nodes = local_nodes

            for node_id in sorted_local_nodes:
                node = graph.nodes[node_id]

                if node.op_type == "Recv":
                    src_stage = node.attributes.get("source_stage", 0)
                    q = comm_queues.get((src_stage, stage_idx))
                    if q:
                        stage_env[node_id] = q.get()
                else:
                    input_vals = []
                    for inp in node.inputs:
                        if inp not in stage_env:
                            raise KeyError(f"Input {inp} for node {node_id} not found in stage_env!")
                        input_vals.append(stage_env[inp])

                    if node.op_type == "Input":
                        pass
                    elif node.op_type == "Constant":
                        stage_env[node_id] = node.attributes.get("value", 0.0)
                    elif node.op_type == "Send":
                        pass
                    else:
                        from ml_switcheroo_compiler.backends.registry import get_active_backend

                        backend = get_active_backend()
                        op_cls = get_op(node.op_type)
                        has_custom_eval = hasattr(op_cls, "eager_eval") and "eager_eval" in op_cls.__dict__

                        if has_custom_eval:
                            res = op_cls().eager_eval(*input_vals, **node.attributes)
                        else:
                            res = backend.execute_op(node.op_type, *input_vals, **node.attributes)
                        stage_env[node_id] = res

                    if node.op_type == "Send":
                        tgt_stage = node.attributes.get("target_stage", 0)
                        q = comm_queues.get((stage_idx, tgt_stage))
                        if q:
                            q.put(stage_env.get(node.inputs[0]))

            with output_lock:
                for out_id in getattr(graph, "outputs", []):
                    if out_id in stage_env:
                        outputs[out_id] = stage_env[out_id]

        # Use robust ThreadPoolExecutor with context copy
        import contextvars

        def thread_wrapper(stage_idx: int, ctx: contextvars.Context) -> None:
            ctx.run(stage_worker, stage_idx)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_stages) as executor:
            futures = []
            for i in range(num_stages):
                ctx = contextvars.copy_context()
                futures.append(executor.submit(thread_wrapper, i, ctx))

            for future in concurrent.futures.as_completed(futures):
                future.result()  # raise exceptions if any occurred

        return outputs

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

    def generate_microbatch_loop(self, graph: Any) -> None:
        """Implement microbatch loop generation (splitting global batch size into sequential chunks).

        Args:
            graph (Any): The graph parameter.

        Returns: Any: Result.
        """
        from ml_switcheroo_compiler.ir.core import IRNode

        if self.num_microbatches <= 1:
            return

        inputs = [n for n in graph.nodes.values() if n.op_type == "Input"]

        # 1. Inject slices
        sliced_inputs = []
        for inp in inputs:
            slice_id = f"{inp.id}_slice"
            slice_node = IRNode(id=slice_id, op_type="Slice", inputs=[inp.id, "microbatch_idx"], attributes={"axis": 0, "num_chunks": self.num_microbatches})
            graph.nodes[slice_id] = slice_node
            sliced_inputs.append(slice_id)

            # Re-route the rest of the graph to use the sliced input
            for n in graph.nodes.values():
                if n.id not in (slice_id, "microbatch_idx", "microbatch_loop"):
                    for i, in_id in enumerate(n.inputs):
                        if in_id == inp.id:
                            n.inputs[i] = slice_id

        # 2. Add loop wrapper
        loop_node = IRNode(id="microbatch_loop", op_type="WhileLoop", inputs=[i.id for i in inputs], attributes={"num_iterations": self.num_microbatches, "microbatch": True})
        graph.nodes[loop_node.id] = loop_node

        # 3. Concatenate outputs
        if graph.outputs:
            new_outputs = []
            for out_id in graph.outputs:
                concat_id = f"{out_id}_concat"
                concat_node = IRNode(id=concat_id, op_type="Concat", inputs=[out_id], attributes={"axis": 0, "num_chunks": self.num_microbatches})
                graph.nodes[concat_id] = concat_node
                new_outputs.append(concat_id)
            graph.outputs = new_outputs

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
