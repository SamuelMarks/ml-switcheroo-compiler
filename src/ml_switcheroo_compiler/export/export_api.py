# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Export API."""

import os
from typing import Callable, Optional

import yaml

from ml_switcheroo_compiler.export.pb_utils import ProtobufWriter


class ExportArchive:
    """ExportArchive is used to write SavedModel artifacts (e.g. for TF Serving)."""

    def __init__(self) -> None:
        """Initialize."""
        self.trackables: dict[int, object] = {}
        self.endpoints: dict[str, Callable[..., object]] = {}
        self.collections: dict[str, object] = {}

        yaml_path: object = os.path.join(os.path.dirname(__file__), "tf_schema.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                self.schema = yaml.safe_load(f)
        else:
            self.schema = {"types": {}, "operations": {}}

    def track(self, resource: object) -> None:
        """Track a resource.

        Args:
            resource: Resource to track.
        """
        self.trackables[id(resource)] = resource

    def add_endpoint(self, name: str, fn: Callable[..., object], **kwargs: object) -> None:
        """Add an endpoint.

        Args:
            name: Name of endpoint.
            fn: Function endpoint.
            kwargs: Kwargs.
        """
        self.endpoints[name] = fn

    def _get_tf_dtype(self, dtype_str: str) -> int:
        """Map IR dtype string to TF DT_* enum."""
        return int(self.schema.get("types", {}).get(str(dtype_str).lower(), 1))  # Default DT_FLOAT

    def _get_tf_op(self, op_type: str) -> str:
        """Map IR op_type to TF NodeDef op."""
        return str(self.schema.get("operations", {}).get(op_type, self.schema.get("operations", {}).get("fallback", "Placeholder")))

    def _build_signature_def(self, name: str, graph: object = None) -> ProtobufWriter:
        """Build a SignatureDef protobuf message.

        Args:
            name (str): The name parameter.
            graph (object, optional): The IR graph to inspect.

        Returns:
            ProtobufWriter: Result.
        """
        sig: object = ProtobufWriter()
        sig.add_string(3, name)  # method_name

        if graph is not None:
            # Dynamically build inputs
            input_nodes: object = [n for n in graph.nodes.values() if n.op_type == "Input"]
            for i, node in enumerate(input_nodes):
                inp_tensor: object = ProtobufWriter()
                inp_tensor.add_string(1, node.id)  # name
                inp_tensor.add_varint(2, self._get_tf_dtype(getattr(node, "dtype", "float32")))  # dtype
                # Note: Adding shape TensorShapeProto would go here (field 3)

                inp_map: object = ProtobufWriter()
                inp_map.add_string(1, f"input_{i}")  # Logical name
                inp_map.add_message(2, inp_tensor)
                sig.add_message(1, inp_map)  # inputs

            # Dynamically build outputs
            if hasattr(graph, "outputs") and graph.outputs:
                for i, out_id in enumerate(graph.outputs):
                    out_node: object = graph.nodes.get(out_id)
                    dtype: object = getattr(out_node, "dtype", "float32") if out_node else "float32"

                    out_tensor: object = ProtobufWriter()
                    out_tensor.add_string(1, out_id)  # name
                    out_tensor.add_varint(2, self._get_tf_dtype(dtype))  # dtype

                    out_map: object = ProtobufWriter()
                    out_map.add_string(1, f"output_{i}")  # Logical name
                    out_map.add_message(2, out_tensor)
                    sig.add_message(2, out_map)  # outputs
        else:
            # Dummy fallback if no graph provided
            inp_tensor: object = ProtobufWriter()
            inp_tensor.add_string(1, "input")
            inp_tensor.add_varint(2, 1)  # DT_FLOAT

            inp_map: object = ProtobufWriter()
            inp_map.add_string(1, "x")
            inp_map.add_message(2, inp_tensor)

            sig.add_message(1, inp_map)

        return sig

    def _build_graph_def(self, graph: object = None) -> ProtobufWriter:
        """Build a GraphDef protobuf message.

        Args:
            graph (object, optional): The IR graph to serialize.

        Returns:
            ProtobufWriter: Result.
        """
        graph_def: object = ProtobufWriter()

        if graph is not None:
            from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter

            sorted_nodes: object = DAGTopologicalSorter.sort(graph)

            for node in sorted_nodes:
                node_def: object = ProtobufWriter()
                node_def.add_string(1, node.id)  # name
                node_def.add_string(2, self._get_tf_op(node.op_type))  # op

                for inp in node.inputs:
                    node_def.add_string(3, inp)  # input

                # Attributes (field 5, map<string, AttrValue>) would go here for completeness
                graph_def.add_message(1, node_def)
        else:
            # Dummy node just to be compliant
            dummy_node: object = ProtobufWriter()
            dummy_node.add_string(1, "dummy_node")
            dummy_node.add_string(2, "Placeholder")
            graph_def.add_message(1, dummy_node)

        # versions
        versions: object = ProtobufWriter()
        versions.add_varint(1, 1)  # producer
        graph_def.add_message(4, versions)
        return graph_def

    def _build_saved_model(self, graph: object = None) -> bytes:
        """Build the SavedModel protobuf bytes.

        Args:
            graph (object, optional): The IR graph.

        Returns:
            bytes: Result.
        """
        saved_model: object = ProtobufWriter()
        saved_model.add_varint(1, 1)  # saved_model_schema_version

        meta_graph: object = ProtobufWriter()
        meta_graph.add_message(2, self._build_graph_def(graph))  # graph_def

        for name in self.endpoints:
            sig_map: object = ProtobufWriter()
            sig_map.add_string(1, name)
            sig_map.add_message(2, self._build_signature_def(name, graph))
            meta_graph.add_message(5, sig_map)  # signature_def

        saved_model.add_message(2, meta_graph)  # meta_graphs
        return saved_model.get_bytes()

    def write_out(self, filepath: str, options: Optional[object] = None, graph: object = None) -> None:
        """Write the archive to a directory.

        Args:
            filepath: Target path.
            options: Save options.
            graph: The IR graph to serialize.
        """
        os.makedirs(filepath, exist_ok=True)

        # Serialize weights/variables
        var_dir: object = os.path.join(filepath, "variables")
        os.makedirs(var_dir, exist_ok=True)

        with open(os.path.join(var_dir, "variables.data-00000-of-00001"), "wb") as f:
            if self.collections:
                import pickle

                pickle.dump(self.collections, f)
            else:
                f.write(b"")

        with open(os.path.join(var_dir, "variables.index"), "wb") as f:
            f.write(b"")

        # Save the SavedModel protobuf
        with open(os.path.join(filepath, "saved_model.pb"), "wb") as f:
            f.write(self._build_saved_model(graph))

    def add_variable_collection(self, name: str, variables: object) -> None:
        """Add a variable collection.

        Args:
            name: Collection name.
            variables: Variables.
        """
        self.collections[name] = variables
