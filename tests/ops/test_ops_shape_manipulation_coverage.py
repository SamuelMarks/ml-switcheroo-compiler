from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.ops.shape.manipulation import transpose
from ml_switcheroo_compiler.tracing.tracer import _tracer


def test_transpose_coverage():
    _tracer.start_tracing()
    t = Tensor(data=None, device=Device("cpu"), dtype=DType.Float32, shape=(2,))
    # dim0 and dim1 >= len
    transpose(t, 2, 3)
    _tracer.stop_tracing()


def test_extract_helpers_coverage():
    from ml_switcheroo_compiler.ops.shape.manipulation import _try_extract_tolist, _try_extract_item
    from unittest.mock import MagicMock

    # tolist list
    m1 = MagicMock()
    m1.data.tolist.return_value = [1]
    assert _try_extract_tolist([m1]) == [1]

    # tolist int
    m2 = MagicMock()
    m2.data.tolist.return_value = 2
    assert _try_extract_tolist([m2]) == [2]

    # TypeError
    m3 = MagicMock()
    m3.data.tolist.side_effect = TypeError
    assert _try_extract_tolist([m3]) is None

    # item
    m4 = MagicMock()
    m4.data.item.return_value = 3
    assert _try_extract_item([m4]) == [3]

    m5 = MagicMock()
    m5.data.item.side_effect = TypeError
    assert _try_extract_item([m5]) is None
