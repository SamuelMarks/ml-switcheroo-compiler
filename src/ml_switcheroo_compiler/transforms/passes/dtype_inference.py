"""DType Inference Pass."""

from typing import Optional, cast

from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.core.type_promotion import promote_types
from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


def _get_value_dtype(val: TensorOrScalar) -> Optional[DType]:
    """Get the dtype from a constant value.

    Args:
        val (TensorOrScalar): The val parameter.

    Returns:
        Optional[DType]: Inferred `DType` or None.
    """
    if hasattr(val, "dtype"):
        return DType(str(getattr(val, "dtype")))
    if isinstance(val, bool):
        return DType.Bool
    if isinstance(val, int):
        return DType.Int32
    if isinstance(val, float):
        return DType.Float32
    return None


def _infer_constant_dtype(node: LogicalNode, dtypes: dict[str, str]) -> bool:
    """Infer dtype for a Constant node.

    Args:
        node (LogicalNode): The node parameter for the operation.
        dtypes (dict[str, str]): The dtypes dict.

    Returns:
        bool: True if node was modified.
    """
    val = node.attributes.get("value")
    dt = _get_value_dtype(val)

    if dt is not None:
        dtypes[node.id] = dt.value
        if node.attributes.get("dtype") != dt.value:
            node.attributes["dtype"] = dt.value
            return True
        return False

    dtypes[node.id] = cast(str, node.attributes.get("dtype", DType.Float32.value))
    return False


def _infer_input_dtype(node: LogicalNode, dtypes: dict[str, str]) -> bool:
    """Infer dtype for an Input node.

    Args:
        node (LogicalNode): The node parameter for the operation.
        dtypes (dict[str, str]): The dtypes dict.

    Returns:
        bool: True if node was modified.
    """
    dtypes[node.id] = cast(str, node.attributes.get("dtype", DType.Float32.value))
    return False


def _infer_output_dtype(node: LogicalNode, dtypes: dict[str, str]) -> bool:
    """Infer dtype for an Output node.

    Args:
        node (LogicalNode): The node parameter for the operation.
        dtypes (dict[str, str]): The dtypes dict.

    Returns:
        bool: True if node was modified.
    """
    modified = False
    inp_dtype = None
    if node.inputs:
        inp_dtype = dtypes.get(node.inputs[0])
    if inp_dtype is not None:
        dtypes[node.id] = inp_dtype
    if inp_dtype and node.attributes.get("dtype") != inp_dtype:
        modified = True
        node.attributes["dtype"] = inp_dtype
    return modified


def _get_promoted_dtype(valid_dtypes: list[str]) -> str:
    """Compute the promoted common data type across `valid_dtypes`.

    Args:
        valid_dtypes (list[str]): The valid_dtypes parameter.

    Returns:
        str: Promoted dtype string.
    """
    if len(valid_dtypes) == 1:
        return valid_dtypes[0]
    try:
        promoted = promote_types(DType(valid_dtypes[0]), DType(valid_dtypes[1]))
        for i in range(2, len(valid_dtypes)):
            promoted = promote_types(promoted, DType(valid_dtypes[i]))
        return promoted.value
    except (TypeError, ValueError):
        return valid_dtypes[0]


def _handle_cast_dtype(node: LogicalNode, valid_dtypes: list[str]) -> Optional[str]:
    """Resolve output dtype for Cast and Bitcast nodes.

    Args:
        node (LogicalNode): The node parameter.
        valid_dtypes (list[str]): The valid_dtypes parameter.

    Returns:
        Optional[str]: Target cast dtype string or None.
    """
    _ = valid_dtypes
    if "dtype" in node.attributes:
        val = node.attributes["dtype"]
        return val.value if isinstance(val, DType) else str(val)
    return None


def _handle_boolean_dtype(node: LogicalNode, valid_dtypes: list[str]) -> str:
    """Resolve output dtype for boolean comparison and logical nodes.

    Args:
        node (LogicalNode): The node parameter.
        valid_dtypes (list[str]): The valid_dtypes parameter.

    Returns:
        str: Boolean dtype string `'bool'`.
    """
    _ = (node, valid_dtypes)
    return DType.Bool.value


DTYPE_INFERENCE_REGISTRY = {
    "Cast": _handle_cast_dtype,
    "Bitcast": _handle_cast_dtype,
    "Equal": _handle_boolean_dtype,
    "NotEqual": _handle_boolean_dtype,
    "Greater": _handle_boolean_dtype,
    "GreaterEqual": _handle_boolean_dtype,
    "Less": _handle_boolean_dtype,
    "LessEqual": _handle_boolean_dtype,
    "Isnan": _handle_boolean_dtype,
    "Isinf": _handle_boolean_dtype,
    "Isfinite": _handle_boolean_dtype,
    "Allclose": _handle_boolean_dtype,
    "Isclose": _handle_boolean_dtype,
    "LogicalAnd": _handle_boolean_dtype,
    "LogicalOr": _handle_boolean_dtype,
    "LogicalNot": _handle_boolean_dtype,
    "LogicalXor": _handle_boolean_dtype,
}


def _get_node_valid_dtypes(node: LogicalNode, dtypes: dict[str, str]) -> list[str]:
    """Collect known input dtypes for `node` from `dtypes`.

    Args:
        node (LogicalNode): The node parameter.
        dtypes (dict[str, str]): The dtypes parameter.

    Returns:
        list[str]: Known input dtype strings.
    """
    valid = []
    for inp in getattr(node, "inputs"):
        dt = dtypes.get(inp)
        if dt is not None:
            valid.append(dt)
    return valid


def _infer_op_dtype(node: LogicalNode, dtypes: dict[str, str]) -> bool:
    """Infer dtype for a generic Op node.

    Args:
        node (LogicalNode): The node parameter for the operation.
        dtypes (dict[str, str]): The dtypes dict.

    Returns:
        bool: True if node was modified.
    """
    valid_dtypes = _get_node_valid_dtypes(node, dtypes)

    out_dtype_val = None
    if node.op_type in DTYPE_INFERENCE_REGISTRY:
        out_dtype_val = DTYPE_INFERENCE_REGISTRY[node.op_type](node, valid_dtypes)

    if out_dtype_val is None:
        if valid_dtypes:
            out_dtype_val = _get_promoted_dtype(valid_dtypes)
        else:
            out_dtype_val = DType.Float32.value

    dtypes[node.id] = out_dtype_val
    curr_dtype = node.attributes.get("dtype")
    if curr_dtype != out_dtype_val or type(curr_dtype) is not str:
        node.attributes["dtype"] = out_dtype_val
        return True

    return False


_INFERENCE_HANDLERS = {
    "Constant": _infer_constant_dtype,
    "Input": _infer_input_dtype,
    "Output": _infer_output_dtype,
}


def dtype_inference_pass(graph: IRGraph) -> bool:
    """In-place dtype inference.

    Args:
        graph (IRGraph): The graph parameter for the operation.

    Returns:
        bool: A boolean indicating the result of the check.
    """
    modified = False
    sorted_nodes = DAGTopologicalSorter.sort(graph)
    dtypes: dict[str, str] = {}

    for node in sorted_nodes:
        handler = _INFERENCE_HANDLERS.get(node.op_type, _infer_op_dtype)
        if handler(node, dtypes):
            modified = True

    return modified
