from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops.creation.frontend_basic import _infer_dtype


def test_infer_dtype_from_array_object():
    class ObjArr:
        dtype = "O"

    assert _infer_dtype(ObjArr()) == DType.Object
