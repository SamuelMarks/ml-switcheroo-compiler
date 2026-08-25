# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""DType Inference Pass."""

from typing import Optional

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.type_promotion import promote_types
from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


def _get_value_dtype(val: object) -> Optional[DType]:
    """Get the dtype from a constant value.

    Args:
        val (object): The val parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    if hasattr(val, "dtype"):
        return DType(str(val.dtype))
    if isinstance(val, bool):
        return DType.Bool
    if isinstance(val, int):
        return DType.Int32
    if isinstance(val, float):
        return DType.Float32
    return None


def _infer_constant_dtype(node: object, dtypes: dict[str, str]) -> bool:
    """Infer dtype for a Constant node.

    Args:
        node (object): The node parameter for the operation.
        dtypes (dict[str, str]): The dtypes dict.

    Returns:
        bool: True if node was modified.
    """
    val: object = node.attributes.get("value")
    dt: object = _get_value_dtype(val)

    if dt is not None:
        dtypes[node.id] = dt.value
        if node.attributes.get("dtype") != dt.value:
            node.attributes["dtype"] = dt.value
            return True
        return False

    dtypes[node.id] = node.attributes.get("dtype", DType.Float32.value)
    return False


def _infer_input_dtype(node: object, dtypes: dict[str, str]) -> bool:
    """Infer dtype for an Input node.

    Args:
        node (object): The node parameter for the operation.
        dtypes (dict[str, str]): The dtypes dict.

    Returns:
        bool: True if node was modified.
    """
    dtypes[node.id] = node.attributes.get("dtype", DType.Float32.value)
    return False


def _infer_output_dtype(node: object, dtypes: dict[str, str]) -> bool:
    """Infer dtype for an Output node.

    Args:
        node (object): The node parameter for the operation.
        dtypes (dict[str, str]): The dtypes dict.

    Returns:
        bool: True if node was modified.
    """
    modified: object = False
    inp_dtype: object = None
    if node.inputs:
        inp_dtype: object = dtypes.get(node.inputs[0])
    if inp_dtype is not None:
        dtypes[node.id] = inp_dtype
    if inp_dtype and node.attributes.get("dtype") != inp_dtype:
        modified: object = True
        node.attributes["dtype"] = inp_dtype
    return modified


def _get_promoted_dtype(valid_dtypes: list[str]) -> str:
    """Evaluate _get_promoted_dtype operation.

    Args:
        valid_dtypes (object): The valid_dtypes parameter.

    Returns:
        str: Result.
    """
    if len(valid_dtypes) == 1:
        return valid_dtypes[0]
    try:
        promoted: object = promote_types(DType(valid_dtypes[0]), DType(valid_dtypes[1]))
        for i in range(2, len(valid_dtypes)):
            promoted: object = promote_types(promoted, DType(valid_dtypes[i]))
        return promoted.value
    except (TypeError, ValueError):
        return valid_dtypes[0]


def _handle_cast_dtype(node: object, valid_dtypes: list[str]) -> Optional[str]:
    """Evaluate _handle_cast_dtype operation.

    Args:
        node (object): The node parameter.
        valid_dtypes (object): The valid_dtypes parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    if "dtype" in node.attributes:
        val: object = node.attributes["dtype"]
        return val.value if isinstance(val, DType) else str(val)
    return None


def _handle_boolean_dtype(node: object, valid_dtypes: list[str]) -> str:
    """Evaluate _handle_boolean_dtype operation.

    Args:
        node (object): The node parameter.
        valid_dtypes (object): The valid_dtypes parameter.

    Returns:
        str: Result.
    """
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


def _get_node_valid_dtypes(node: object, dtypes: dict[str, str]) -> list[str]:
    """Evaluate _get_node_valid_dtypes operation.

    Args:
        node (object): The node parameter.
        dtypes (object): The dtypes parameter.

    Returns:
            tuple[int, ...]: Result.
    """
    valid: object = []
    for inp in node.inputs:
        dt: object = dtypes.get(inp)
        if dt is not None:
            valid.append(dt)
    return valid


def _infer_op_dtype(node: object, dtypes: dict[str, str]) -> bool:
    """Infer dtype for a generic Op node.

    Args:
        node (object): The node parameter for the operation.
        dtypes (dict[str, str]): The dtypes dict.

    Returns:
        bool: True if node was modified.
    """
    valid_dtypes: object = _get_node_valid_dtypes(node, dtypes)

    out_dtype_val: object = None
    if node.op_type in DTYPE_INFERENCE_REGISTRY:
        out_dtype_val: object = DTYPE_INFERENCE_REGISTRY[node.op_type](node, valid_dtypes)

    if out_dtype_val is None:
        if valid_dtypes:
            out_dtype_val: object = _get_promoted_dtype(valid_dtypes)
        else:
            out_dtype_val: object = DType.Float32.value

    dtypes[node.id] = out_dtype_val
    if node.attributes.get("dtype") != out_dtype_val:
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
    modified: object = False
    sorted_nodes: object = DAGTopologicalSorter.sort(graph)
    dtypes: dict[str, str] = {}

    for node in sorted_nodes:
        handler: object = _INFERENCE_HANDLERS.get(node.op_type, _infer_op_dtype)
        if handler(node, dtypes):
            modified: object = True

    return modified
