"""Module docstring."""

from unittest.mock import MagicMock

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.shape.manipulation import _try_extract_item, _try_extract_tolist, transpose
from ml_switcheroo_compiler.tracing.state import global_tracing_state


def test_transpose_coverage() -> object:
    """Function docstring."""
    global_tracing_state.start_tracing()
    t = Tensor(None, TensorConfig((2,), DType.Float32, Device("cpu")))
    # dim0 and dim1 >= len
    transpose(t, 2, 3)
    global_tracing_state.stop_tracing()


def test_extract_helpers_coverage() -> object:
    """Function docstring."""
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
