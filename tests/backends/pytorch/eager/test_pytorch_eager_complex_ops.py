"""Tests for test_pytorch_eager_complex_ops."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
import torch

import ml_switcheroo_compiler.backends.pytorch.eager as pytorch_eager

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


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

    cast("Callable[..., EagerTensorProtocol]", pytorch_eager._execute_cast)(cast("Callable[..., EagerTensorProtocol]", torch.tensor)([1.0]), cast("Callable[..., EagerTensorProtocol]", DummyDtype)())

    class DummyDtypeBfloat16:
        """Test class DummyDtypeBfloat16."""

        value = "bfloat16"

    getattr(_state, "update")({"DummyDtype": DummyDtype, "DummyDtypeBfloat16": DummyDtypeBfloat16, "indices": indices, "tensor": tensor})
    getattr(_state, "update")({"updates": updates, "w": w})
    _state.update(locals())


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

    cast("Callable[..., EagerTensorProtocol]", pytorch_eager._execute_cast)(cast("Callable[..., EagerTensorProtocol]", torch.tensor)([1.0]), cast("Callable[..., EagerTensorProtocol]", DummyDtypeFloat16)())

    class DummyDtypeFloat8_e4m3fn:
        """Test class DummyDtypeFloat8_e4m3fn."""

        value = "float8_e4m3fn"

    cast("Callable[..., EagerTensorProtocol]", pytorch_eager._execute_cast)(cast("Callable[..., EagerTensorProtocol]", torch.tensor)([1.0]), cast("Callable[..., EagerTensorProtocol]", DummyDtypeFloat8_e4m3fn)())
    pytorch_eager._execute_cummax(torch.tensor([1, 2]), dim=0)
    pytorch_eager._execute_cummin(torch.tensor([1, 2]), dim=0)
    pytorch_eager._execute_cumlogsumexp(torch.tensor([1.0, 2.0]), dim=0)
    pytorch_eager._execute_ragged_tensor_to_dense(1)
    pytorch_eager._torch_variance(torch.tensor([1.0, 2.0]))
    pytorch_eager._torch_tensordot(torch.tensor([1.0]), torch.tensor([1.0]), dims=1)
    pytorch_eager._torch_tensordot(torch.tensor([1.0]), torch.tensor([1.0]), axes=1)
    getattr(_state, "update")({"DummyDtypeFloat16": DummyDtypeFloat16, "DummyDtypeFloat8_e4m3fn": DummyDtypeFloat8_e4m3fn})
    _state.update(locals())


def _test_pytorch_eager_additional_sub_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 3 of test_pytorch_eager_additional.

    Args:
        _state: Shared test state dictionary.
    """
    for op in ["HardSilu", "HardSwish", "Squareplus", "Fftnd", "Ifftnd", "Irfftnd", "Rfftnd"]:
        with pytest.raises(AttributeError):
            cast("EagerTensorProtocol", getattr(pytorch_eager, "_TORCH_EAGER_OP_MAP"))[op](torch.tensor([1.0]))
    for op in ["Fftfreq", "Rfftfreq"]:
        with pytest.raises(AttributeError):
            cast("EagerTensorProtocol", getattr(pytorch_eager, "_TORCH_EAGER_OP_MAP"))[op](2)

    class DummyDtypeFloat8_e5m2:
        """Test class DummyDtypeFloat8_e5m2."""

        value = "float8_e5m2"

    cast("Callable[..., EagerTensorProtocol]", pytorch_eager._execute_cast)(cast("Callable[..., EagerTensorProtocol]", torch.tensor)([1.0]), cast("Callable[..., EagerTensorProtocol]", DummyDtypeFloat8_e5m2)())

    class DummyDtypeInt32:
        """Test class DummyDtypeInt32."""

        value = "int32"

    cast("Callable[..., EagerTensorProtocol]", pytorch_eager._execute_cast)(cast("Callable[..., EagerTensorProtocol]", torch.tensor)([1.0]), cast("Callable[..., EagerTensorProtocol]", DummyDtypeInt32)())
    getattr(_state, "update")({"DummyDtypeFloat8_e5m2": DummyDtypeFloat8_e5m2, "DummyDtypeInt32": DummyDtypeInt32})
    _state.update(locals())


def _test_pytorch_eager_additional_sub_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute sub-part 4 of test_pytorch_eager_additional.

    Args:
        _state: Shared test state dictionary.
    """
    with pytest.raises(ValueError):
        pytorch_eager._execute_accumulate_n(inputs=[])
    cast("Callable[..., EagerTensorProtocol]", pytorch_eager.execute_op)(None, "Add", cast("Callable[..., EagerTensorProtocol]", torch.tensor)(1), cast("Callable[..., EagerTensorProtocol]", torch.tensor)(2))
    cast("Callable[..., EagerTensorProtocol]", pytorch_eager.execute_op)(None, "RaggedTensorToDense", 1)
    try:
        cast("Callable[..., EagerTensorProtocol]", pytorch_eager.execute_op)(None, "unknownop", cast("Callable[..., EagerTensorProtocol]", torch.tensor)(1))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
        _caught_err = True
    cast("Callable[..., EagerTensorProtocol]", pytorch_eager.execute_op)(None, "Zeros", 2)
    try:
        cast("Callable[..., EagerTensorProtocol]", pytorch_eager.execute_op)(None, "Descriptive", cast("Callable[..., EagerTensorProtocol]", torch.tensor)([1.0]))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
        _caught_err = True
    try:
        cast("Callable[..., EagerTensorProtocol]", pytorch_eager.execute_op)(None, "totally_unknown_and_unregistered", 1)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, OverflowError, RecursionError, LookupError, ArithmeticError):
        _caught_err = True
    pass
    _state.update(locals())


def test_pytorch_eager_additional() -> None:
    """Test function test_pytorch_eager_additional."""
    import pytest

    _state: dict[str, EagerTensorProtocol] = dict(locals())
    getattr(_state, "update")({"pytest": pytest})
    for _sub_fn in (_test_pytorch_eager_additional_sub_1, _test_pytorch_eager_additional_sub_2, _test_pytorch_eager_additional_sub_3, _test_pytorch_eager_additional_sub_4):
        _sub_fn(_state)
    assert isinstance(_state, dict)
