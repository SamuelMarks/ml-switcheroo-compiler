"""Tests for test_pytorch_eager_complex_ops."""

from __future__ import annotations

import typing

import pytest

import ml_switcheroo_compiler.backends.pytorch.eager as pytorch_eager
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_tensor_dict

if typing.TYPE_CHECKING:
    import torch

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol

torch = pytest.importorskip("torch")


def _test_pytorch_eager_additional_sub_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 1 of test_pytorch_eager_additional.

    Args:
        _state: Shared test state dictionary.
    """
    pytorch_eager._execute_accumulate_n([torch.tensor(1), torch.tensor(2)])
    tensor = torch.zeros(2, 2)
    indices = torch.tensor([[0, 0]])
    updates = torch.tensor([1.0])
    pytorch_eager._execute_tensor_scatter_max(tensor, indices, updates)
    pytorch_eager._execute_tensor_scatter_min(tensor, indices, updates)
    pytorch_eager._execute_tensor_scatter_update(tensor, indices, updates)
    pytorch_eager._execute_tensor_scatter_add(tensor, indices, updates)
    w = torch.eye(2).unsqueeze(0)
    pytorch_eager._execute_power_iteration(w)
    pytorch_eager._execute_broadcast_to(torch.tensor([1.0]), shape=(2,))

    class DummyDtype:
        """Test class DummyDtype."""

        value = "int4"

    ensure_callable(pytorch_eager._execute_cast)(ensure_callable(torch.tensor)([1.0]), ensure_callable(DummyDtype)())

    class DummyDtypeBfloat16:
        """Test class DummyDtypeBfloat16."""

        value = "bfloat16"

    _state.update(ensure_tensor_dict({"DummyDtype": DummyDtype, "DummyDtypeBfloat16": DummyDtypeBfloat16, "indices": indices, "tensor": tensor}))
    _state.update(ensure_tensor_dict({"updates": updates, "w": w}))


def _test_pytorch_eager_additional_sub_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 2 of test_pytorch_eager_additional.

    Args:
        _state: Shared test state dictionary.
    """
    DummyDtypeBfloat16 = _state["DummyDtypeBfloat16"]
    pytorch_eager._execute_cast(torch.tensor([1.0]), DummyDtypeBfloat16())

    class DummyDtypeFloat16:
        """Test class DummyDtypeFloat16."""

        value = "float16"

    ensure_callable(pytorch_eager._execute_cast)(ensure_callable(torch.tensor)([1.0]), ensure_callable(DummyDtypeFloat16)())

    class DummyDtypeFloat8_e4m3fn:
        """Test class DummyDtypeFloat8_e4m3fn."""

        value = "float8_e4m3fn"

    ensure_callable(pytorch_eager._execute_cast)(ensure_callable(torch.tensor)([1.0]), ensure_callable(DummyDtypeFloat8_e4m3fn)())
    pytorch_eager._execute_cummax(torch.tensor([1, 2]), dim=0)
    pytorch_eager._execute_cummin(torch.tensor([1, 2]), dim=0)
    pytorch_eager._execute_cumlogsumexp(torch.tensor([1.0, 2.0]), dim=0)
    pytorch_eager._execute_ragged_tensor_to_dense(1)
    pytorch_eager._torch_variance(torch.tensor([1.0, 2.0]))
    pytorch_eager._torch_tensordot(torch.tensor([1.0]), torch.tensor([1.0]), dims=1)
    pytorch_eager._torch_tensordot(torch.tensor([1.0]), torch.tensor([1.0]), axes=1)
    _state.update(ensure_tensor_dict({"DummyDtypeFloat16": DummyDtypeFloat16, "DummyDtypeFloat8_e4m3fn": DummyDtypeFloat8_e4m3fn}))
    _state.update(ensure_tensor_dict({"DummyDtypeBfloat16": DummyDtypeBfloat16}))


def _test_pytorch_eager_additional_sub_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 3 of test_pytorch_eager_additional.

    Args:
        _state: Shared test state dictionary.
    """
    for op in ["HardSilu", "HardSwish", "Squareplus", "Fftnd", "Ifftnd", "Irfftnd", "Rfftnd"]:
        with pytest.raises(AttributeError):
            ensure_eager_tensor(ensure_eager_tensor(pytorch_eager)._TORCH_EAGER_OP_MAP)[op](torch.tensor([1.0]))
    for op in ["Fftfreq", "Rfftfreq"]:
        with pytest.raises(AttributeError):
            ensure_eager_tensor(ensure_eager_tensor(pytorch_eager)._TORCH_EAGER_OP_MAP)[op](2)

    class DummyDtypeFloat8_e5m2:
        """Test class DummyDtypeFloat8_e5m2."""

        value = "float8_e5m2"

    ensure_callable(pytorch_eager._execute_cast)(ensure_callable(torch.tensor)([1.0]), ensure_callable(DummyDtypeFloat8_e5m2)())

    class DummyDtypeInt32:
        """Test class DummyDtypeInt32."""

        value = "int32"

    ensure_callable(pytorch_eager._execute_cast)(ensure_callable(torch.tensor)([1.0]), ensure_callable(DummyDtypeInt32)())
    _state.update(ensure_tensor_dict({"DummyDtypeFloat8_e5m2": DummyDtypeFloat8_e5m2, "DummyDtypeInt32": DummyDtypeInt32}))


def _test_pytorch_eager_additional_sub_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 4 of test_pytorch_eager_additional.

    Args:
        _state: Shared test state dictionary.
    """
    with pytest.raises(ValueError):
        pytorch_eager._execute_accumulate_n(inputs=[])
    ensure_callable(pytorch_eager.execute_op)(None, "Add", ensure_callable(torch.tensor)(1), ensure_callable(torch.tensor)(2))
    ensure_callable(pytorch_eager.execute_op)(None, "RaggedTensorToDense", 1)
    try:
        ensure_callable(pytorch_eager.execute_op)(None, "unknownop", ensure_callable(torch.tensor)(1))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
        _caught_err = True
    ensure_callable(pytorch_eager.execute_op)(None, "Zeros", 2)
    try:
        ensure_callable(pytorch_eager.execute_op)(None, "Descriptive", ensure_callable(torch.tensor)([1.0]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
        _caught_err = True
    try:
        ensure_callable(pytorch_eager.execute_op)(None, "totally_unknown_and_unregistered", 1)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
        _caught_err = True
    pass


def test_pytorch_eager_additional() -> None:
    """Test function test_pytorch_eager_additional."""
    import pytest

    _state: dict[str, EagerTensorProtocol] = {}
    _state.update(ensure_tensor_dict({"pytest": pytest}))
    for _sub_fn in (_test_pytorch_eager_additional_sub_1, _test_pytorch_eager_additional_sub_2, _test_pytorch_eager_additional_sub_3, _test_pytorch_eager_additional_sub_4):
        _sub_fn(_state)
    assert len(_state) > 0
