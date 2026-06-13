"""DType Inference Pass."""

import numpy as np

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.transforms.pass_manager import DAGTopologicalSorter


def dtype_inference_pass(graph: IRGraph) -> bool:
    """In-place dtype inference.

    Args:
        graph (IRGraph): The graph.

    Returns:
        bool: The computed result.
    """
    modified = False
    sorted_nodes = DAGTopologicalSorter.sort(graph)
    dtypes = {}

    for node in sorted_nodes:
        if node.op_type == "Constant":
            val = node.attributes.get("value")
            dt = None
            if hasattr(val, "dtype"):
                dt = DType(str(val.dtype))
            elif isinstance(val, bool):
                dt = DType.Bool
            elif isinstance(val, int):
                dt = DType.Int32
            elif isinstance(val, float):
                dt = DType.Float32

            if dt is not None:
                dtypes[node.id] = dt.value
                if node.attributes.get("dtype") != dt.value:
                    node.attributes["dtype"] = dt.value
                    modified = True
            else:
                dtypes[node.id] = node.attributes.get("dtype", DType.Float32.value)

        elif node.op_type == "Input":
            dtypes[node.id] = node.attributes.get("dtype", DType.Float32.value)

        elif node.op_type == "Output":
            inp_dtype = None
            if node.inputs:
                inp_dtype = dtypes.get(node.inputs[0])
            dtypes[node.id] = inp_dtype
            if inp_dtype and node.attributes.get("dtype") != inp_dtype:
                modified = True
                node.attributes["dtype"] = inp_dtype

        else:
            in_dtypes = [dtypes.get(inp) for inp in node.inputs]
            out_dtype_val = None

            if "dtype" in node.attributes and node.op_type in ["Cast", "Bitcast"]:
                val = node.attributes["dtype"]
                out_dtype_val = val.value if isinstance(val, DType) else str(val)
            elif "Logical" in node.op_type or node.op_type in [
                "Equal",
                "NotEqual",
                "Greater",
                "GreaterEqual",
                "Less",
                "LessEqual",
                "Isnan",
                "Isinf",
                "Isfinite",
                "Allclose",
                "Isclose",
            ]:
                out_dtype_val = DType.Bool.value
            elif in_dtypes and any(dt is not None for dt in in_dtypes):
                valid_dtypes = [dt for dt in in_dtypes if dt is not None]
                if len(valid_dtypes) == 1:
                    out_dtype_val = valid_dtypes[0]
                else:
                    try:
                        np_types = [np.dtype(dt) for dt in valid_dtypes]
                        promoted = np.promote_types(*np_types)
                        out_dtype_val = str(promoted)
                    except TypeError:
                        out_dtype_val = valid_dtypes[0]

            if out_dtype_val is None:
                out_dtype_val = DType.Float32.value

            dtypes[node.id] = out_dtype_val

            if node.attributes.get("dtype") != out_dtype_val:
                node.attributes["dtype"] = out_dtype_val
                modified = True

    return modified
