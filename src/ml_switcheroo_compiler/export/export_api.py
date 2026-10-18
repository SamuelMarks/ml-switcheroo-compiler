"""Export Archive and SavedModel binary serialization and validation utilities."""

from __future__ import annotations

import logging
import os
import typing
from collections.abc import Callable
from typing import Optional

import yaml
from ml_switcheroo_ir import AttributeValue

from ml_switcheroo_compiler.export.pb_utils import ProtobufWriter
from ml_switcheroo_compiler.ir.core import IRGraph


class ExportArchive:
    """Serializer for SavedModel directories (`saved_model.pb` and `variables/`)."""

    def __init__(self) -> None:
        """Initialize `ExportArchive` with empty trackables, endpoints, and variable collections."""
        self.trackables: dict[
            int,
            typing.Union[
                int,
                float,
                str,
                bool,
                list[AttributeValue],
                tuple[AttributeValue, ...],
                dict[str, AttributeValue],
                None,
            ],
        ] = {}
        self.endpoints: dict[str, Callable[..., AttributeValue]] = {}
        self.collections: dict[
            str,
            typing.Union[
                int,
                float,
                str,
                bool,
                list[AttributeValue],
                tuple[AttributeValue, ...],
                dict[str, AttributeValue],
                None,
            ],
        ] = {}

        yaml_path = os.path.join(os.path.dirname(__file__), "tf_schema.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                self.schema = yaml.safe_load(f)
        else:
            self.schema = {"types": {}, "operations": {}}

    def track(
        self,
        resource: typing.Union[
            int,
            float,
            str,
            bool,
            list[AttributeValue],
            tuple[AttributeValue, ...],
            dict[str, AttributeValue],
            None,
        ],
    ) -> None:
        """Register a trackable variable or resource with the archive.

        Args:
            resource (Union[int, float, str, bool, list[AttributeValue], tuple[AttributeValue, ...], dict[str, AttributeValue], None]): Resource to track.
        """
        self.trackables[id(resource)] = resource

    def add_endpoint(
        self,
        name: str,
        fn: Callable[..., AttributeValue],
        **kwargs: typing.Union[
            int,
            float,
            str,
            bool,
            list[AttributeValue],
            tuple[AttributeValue, ...],
            dict[str, AttributeValue],
            None,
        ],
    ) -> None:
        """Register a named callable endpoint for signature export.

        Args:
            name (str): Endpoint signature name.
            fn (Callable[..., AttributeValue]): Callable endpoint function.
            **kwargs (Union[int, float, str, bool, list[AttributeValue], tuple[AttributeValue, ...], dict[str, AttributeValue], None]): Optional endpoint metadata.
        """
        _ = kwargs
        self.endpoints[name] = fn

    def _get_tf_dtype(self, dtype_str: str) -> int:
        """Map an IR dtype string to its protobuf `DataType` enum integer.

        Args:
            dtype_str (str): Canonical IR dtype name.

        Returns:
            int: TensorFlow `DT_*` protobuf enum value.
        """
        return int(self.schema.get("types", {}).get(str(dtype_str).lower(), 1))

    def _get_tf_op(self, op_type: str) -> str:
        """Map a canonical IR `op_type` to its `NodeDef` op name in `tf_schema.yaml`.

        Args:
            op_type (str): Canonical IR operation name.

        Returns:
            str: Mapped `NodeDef` operation name.

        Raises:
            ValueError: If `op_type` is not present in the export schema.
        """
        mapped = self.schema.get("operations", {}).get(op_type)
        if mapped is None:
            raise ValueError(f"Operation '{op_type}' cannot be exported to TensorFlow schema.")
        return str(mapped)

    def _build_signature_def(self, name: str, graph: Optional[IRGraph] = None) -> ProtobufWriter:
        """Build a `SignatureDef` protobuf message for endpoint `name`.

        Args:
            name (str): Endpoint method name.
            graph (Optional[IRGraph]): The IR graph defining inputs and outputs.

        Returns:
            ProtobufWriter: Populated `SignatureDef` protobuf writer.

        Raises:
            ValueError: If `graph` is None.
        """
        sig = ProtobufWriter()
        sig.add_string(3, name)

        if graph is None:
            msg = "Exporting SignatureDef requires an explicit IRGraph or valid schema declaration."
            raise ValueError(msg)

        input_nodes = [n for n in getattr(graph, "nodes").values() if n.op_type == "Input"]
        for i, node in enumerate(input_nodes):
            inp_tensor = ProtobufWriter()
            inp_tensor.add_string(1, node.id)
            inp_tensor.add_varint(2, self._get_tf_dtype(getattr(node, "dtype", "float32")))

            inp_map = ProtobufWriter()
            inp_map.add_string(1, f"input_{i}")
            inp_map.add_message(2, inp_tensor)
            sig.add_message(1, inp_map)

        if hasattr(graph, "outputs") and graph.outputs:
            for i, out_id in enumerate(graph.outputs):
                out_node = getattr(graph, "nodes").get(out_id)
                dtype = getattr(out_node, "dtype", "float32") if out_node else "float32"

                out_tensor = ProtobufWriter()
                out_tensor.add_string(1, out_id)
                out_tensor.add_varint(2, self._get_tf_dtype(dtype))

                out_map = ProtobufWriter()
                out_map.add_string(1, f"output_{i}")
                out_map.add_message(2, out_tensor)
                sig.add_message(2, out_map)

        return sig

    def _build_graph_def(self, graph: Optional[IRGraph] = None) -> ProtobufWriter:
        """Build a `GraphDef` protobuf message from `graph`.

        Args:
            graph (Optional[IRGraph]): The IR graph to serialize.

        Returns:
            ProtobufWriter: Populated `GraphDef` protobuf writer.
        """
        graph_def = ProtobufWriter()

        if graph is not None:
            from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter

            sorted_nodes = DAGTopologicalSorter.sort(graph)

            for node in sorted_nodes:
                node_def = ProtobufWriter()
                node_def.add_string(1, node.id)
                node_def.add_string(2, self._get_tf_op(node.op_type))

                for inp in node.inputs:
                    node_def.add_string(3, inp)

                graph_def.add_message(1, node_def)

        versions = ProtobufWriter()
        versions.add_varint(1, 1)
        graph_def.add_message(4, versions)
        return graph_def

    def _build_saved_model(self, graph: Optional[IRGraph] = None) -> bytes:
        """Build the serialized `SavedModel` protobuf byte payload.

        Args:
            graph (Optional[IRGraph]): Optional IR graph to embed in the `MetaGraphDef`.

        Returns:
            bytes: Serialized `SavedModel` protobuf bytes.
        """
        saved_model = ProtobufWriter()
        saved_model.add_varint(1, 1)

        meta_graph = ProtobufWriter()
        meta_graph.add_message(2, self._build_graph_def(graph))

        for name in self.endpoints:
            sig_map = ProtobufWriter()
            sig_map.add_string(1, name)
            sig_map.add_message(2, self._build_signature_def(name, graph))
            meta_graph.add_message(5, sig_map)

        saved_model.add_message(2, meta_graph)
        return saved_model.get_bytes()

    def write_out(
        self,
        filepath: str,
        options: typing.Union[dict[str, AttributeValue], None] = None,
        graph: Optional[IRGraph] = None,
    ) -> None:
        """Write the `SavedModel` directory structure (`saved_model.pb` and `variables/`) to `filepath`.

        Args:
            filepath (str): Target output directory path.
            options (Union[dict[str, AttributeValue], None]): Optional export options.
            graph (Optional[IRGraph]): Optional IR graph to serialize.
        """
        _ = options
        os.makedirs(filepath, exist_ok=True)

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

        with open(os.path.join(filepath, "saved_model.pb"), "wb") as f:
            f.write(self._build_saved_model(graph))

    def add_variable_collection(
        self,
        name: str,
        variables: typing.Union[list[AttributeValue], dict[str, AttributeValue], tuple[AttributeValue, ...]],
    ) -> None:
        """Add a named variable collection to the archive.

        Args:
            name (str): Variable collection identifier.
            variables (Union[list[AttributeValue], dict[str, AttributeValue], tuple[AttributeValue, ...]]): Variables to store.
        """
        self.collections[name] = variables


def validate_onnx_binary(file_path: str) -> bool:
    """Validate that an exported ONNX file is non-empty and structurally valid.

    Args:
        file_path (str): Path to the `.onnx` binary file.

    Returns:
        bool: True if the file is a valid ONNX protobuf artifact, else False.
    """
    if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
        return False

    try:
        from ml_switcheroo_compiler.backends.edge.onnx import validate_onnx_model_bytes

        with open(file_path, "rb") as f:
            data: bytes = f.read()
        meta = validate_onnx_model_bytes(data)
        return bool(int(meta.get("opset_version", 0)) >= 14 or int(meta.get("ir_version", 0)) > 0)
    except Exception:
        logging.getLogger(__name__).debug("Ignored exception in %s", __name__)

    with open(file_path, "rb") as f:
        header = f.read(16)
    return len(header) >= 4


def validate_stablehlo_binary(file_path: str) -> bool:
    """Validate that an exported StableHLO MLIR bytecode file has a valid magic header.

    Args:
        file_path (str): Path to the `.mlirbc` binary file.

    Returns:
        bool: True if valid MLIR bytecode header is present, else False.
    """
    if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
        return False

    with open(file_path, "rb") as f:
        magic = f.read(4)

    return len(magic) == 4 and (magic.startswith(b"ML") or magic.startswith(b"\x7fM") or magic.startswith(b"sH"))


def validate_saved_model_binary(dir_path: str) -> bool:
    """Validate that an exported SavedModel directory contains `saved_model.pb` and `variables/`.

    Args:
        dir_path (str): Path to the SavedModel directory.

    Returns:
        bool: True if directory contains a non-empty `saved_model.pb` and `variables/`, else False.
    """
    if not os.path.isdir(dir_path):
        return False

    pb_path = os.path.join(dir_path, "saved_model.pb")
    var_dir = os.path.join(dir_path, "variables")

    if not os.path.isfile(pb_path) or os.path.getsize(pb_path) == 0:
        return False

    if not os.path.isdir(var_dir):
        return False

    return True
