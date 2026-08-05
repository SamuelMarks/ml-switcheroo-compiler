"""Export API."""

import os
from typing import Callable, Optional

from ml_switcheroo_compiler.export.pb_utils import ProtobufWriter


class ExportArchive:
    """ExportArchive is used to write SavedModel artifacts (e.g. for TF Serving)."""

    def __init__(self) -> None:
        """Initialize."""
        self.trackables: dict[int, object] = {}
        self.endpoints: dict[str, Callable[..., object]] = {}
        self.collections: dict[str, object] = {}

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

    def _build_signature_def(self, name: str) -> ProtobufWriter:
        """Build a SignatureDef protobuf message.

        Args:
        name (str): The name parameter.

        Returns:
        ProtobufWriter: Result.
        """
        sig = ProtobufWriter()
        sig.add_string(3, name)  # method_name

        # Add dummy input
        inp_tensor = ProtobufWriter()
        inp_tensor.add_string(1, "input")  # name
        inp_tensor.add_varint(2, 1)  # dtype (DT_FLOAT)

        inp_map = ProtobufWriter()
        inp_map.add_string(1, "x")
        inp_map.add_message(2, inp_tensor)

        sig.add_message(1, inp_map)  # inputs

        return sig

    def _build_graph_def(self) -> ProtobufWriter:
        """Build a GraphDef protobuf message.

        Returns:
        ProtobufWriter: Result.
        """
        graph = ProtobufWriter()
        # Add a dummy node just to be compliant
        node = ProtobufWriter()
        node.add_string(1, "dummy_node")
        node.add_string(2, "Placeholder")
        graph.add_message(1, node)

        # versions
        versions = ProtobufWriter()
        versions.add_varint(1, 1)  # producer
        graph.add_message(4, versions)
        return graph

    def _build_saved_model(self) -> bytes:
        """Build the SavedModel protobuf bytes.

        Returns:
        bytes: Result.
        """
        saved_model = ProtobufWriter()
        saved_model.add_varint(1, 1)  # saved_model_schema_version

        meta_graph = ProtobufWriter()
        meta_graph.add_message(2, self._build_graph_def())  # graph_def

        for name in self.endpoints:
            sig_map = ProtobufWriter()
            sig_map.add_string(1, name)
            sig_map.add_message(2, self._build_signature_def(name))
            meta_graph.add_message(5, sig_map)  # signature_def

        saved_model.add_message(2, meta_graph)  # meta_graphs
        return saved_model.get_bytes()

    def write_out(self, filepath: str, options: Optional[object] = None) -> None:
        """Write the archive to a directory.

        Args:
            filepath: Target path.
            options: Save options.
        """
        os.makedirs(filepath, exist_ok=True)

        # Serialize weights/variables
        var_dir = os.path.join(filepath, "variables")
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
            f.write(self._build_saved_model())

    def add_variable_collection(self, name: str, variables: object) -> None:
        """Add a variable collection.

        Args:
            name: Collection name.
            variables: Variables.
        """
        self.collections[name] = variables
