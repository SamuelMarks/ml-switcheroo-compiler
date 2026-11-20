"""Tests for test_pytorch_eager_complex_ops."""

from __future__ import annotations

import typing

import pytest

import ml_switcheroo_compiler.backends.pytorch.eager as pytorch_eager
from tests.testing_utils import MockTestStub, adapt_backend_module

if typing.TYPE_CHECKING:
    import torch


torch = pytest.importorskip("torch")


class _TestPytorchEagerAdditionalState(typing.TypedDict, total=False):
    """Typed shared state for `_test_pytorch_eager_additional` split helpers."""

    DummyDtypeBfloat16: type


def _test_pytorch_eager_additional_sub_1(_state: _TestPytorchEagerAdditionalState) -> None:
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

    class DummyDtype(MockTestStub):
        """Test class DummyDtype."""

        value = "int4"

    pytorch_eager._execute_cast(torch.tensor([1.0]), DummyDtype())

    class DummyDtypeBfloat16:
        """Test class DummyDtypeBfloat16."""

        value = "bfloat16"

    _state.update({"DummyDtypeBfloat16": DummyDtypeBfloat16})


def _test_pytorch_eager_additional_sub_2(_state: _TestPytorchEagerAdditionalState) -> None:
    """Execute sub-part 2 of test_pytorch_eager_additional.

    Args:
        _state: Shared test state dictionary.
    """
    DummyDtypeBfloat16 = _state["DummyDtypeBfloat16"]
    pytorch_eager._execute_cast(torch.tensor([1.0]), DummyDtypeBfloat16())

    class DummyDtypeFloat16(MockTestStub):
        """Test class DummyDtypeFloat16."""

        value = "float16"

    pytorch_eager._execute_cast(torch.tensor([1.0]), DummyDtypeFloat16())

    class DummyDtypeFloat8_e4m3fn(MockTestStub):
        """Test class DummyDtypeFloat8_e4m3fn."""

        value = "float8_e4m3fn"

    pytorch_eager._execute_cast(torch.tensor([1.0]), DummyDtypeFloat8_e4m3fn())
    pytorch_eager._execute_cummax(torch.tensor([1, 2]), dim=0)
    pytorch_eager._execute_cummin(torch.tensor([1, 2]), dim=0)
    pytorch_eager._execute_cumlogsumexp(torch.tensor([1.0, 2.0]), dim=0)
    pytorch_eager._execute_ragged_tensor_to_dense(1)
    pytorch_eager._torch_variance(torch.tensor([1.0, 2.0]))
    pytorch_eager._torch_tensordot(torch.tensor([1.0]), torch.tensor([1.0]), dims=1)
    pytorch_eager._torch_tensordot(torch.tensor([1.0]), torch.tensor([1.0]), axes=1)
    _state.update({"DummyDtypeBfloat16": DummyDtypeBfloat16})


def _test_pytorch_eager_additional_sub_3(_state: _TestPytorchEagerAdditionalState) -> None:
    """Execute sub-part 3 of test_pytorch_eager_additional.

    Args:
        _state: Shared test state dictionary.
    """
    for op in ["HardSilu", "HardSwish", "Squareplus", "Fftnd", "Ifftnd", "Irfftnd", "Rfftnd"]:
        with pytest.raises(AttributeError):
            adapt_backend_module(pytorch_eager).require_symbol("_TORCH_EAGER_OP_MAP")[op](torch.tensor([1.0]))
    for op in ["Fftfreq", "Rfftfreq"]:
        with pytest.raises(AttributeError):
            adapt_backend_module(pytorch_eager).require_symbol("_TORCH_EAGER_OP_MAP")[op](2)

    class DummyDtypeFloat8_e5m2(MockTestStub):
        """Test class DummyDtypeFloat8_e5m2."""

        value = "float8_e5m2"

    pytorch_eager._execute_cast(torch.tensor([1.0]), DummyDtypeFloat8_e5m2())

    class DummyDtypeInt32(MockTestStub):
        """Test class DummyDtypeInt32."""

        value = "int32"

    pytorch_eager._execute_cast(torch.tensor([1.0]), DummyDtypeInt32())


def _test_pytorch_eager_additional_sub_4(_state: _TestPytorchEagerAdditionalState) -> None:
    """Execute sub-part 4 of test_pytorch_eager_additional.

    Args:
        _state: Shared test state dictionary.
    """
    with pytest.raises(ValueError):
        pytorch_eager._execute_accumulate_n(inputs=[])
    pytorch_eager.execute_op(None, "Add", torch.tensor(1), torch.tensor(2))
    pytorch_eager.execute_op(None, "RaggedTensorToDense", 1)
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    with pytest.raises((BackendNotSupportedError, ValueError, TypeError, RuntimeError, NotImplementedError)):
        pytorch_eager.execute_op(None, "unknownop", torch.tensor(1))
    pytorch_eager.execute_op(None, "Zeros", 2)
    desc_out = pytorch_eager.execute_op(None, "Descriptive", torch.tensor([1.0, 3.0]))
    assert desc_out is not None
    with pytest.raises((BackendNotSupportedError, ValueError, TypeError, RuntimeError, NotImplementedError)):
        pytorch_eager.execute_op(None, "totally_unknown_and_unregistered", 1)


def test_pytorch_eager_additional() -> None:
    """Test function test_pytorch_eager_additional."""
    _state: _TestPytorchEagerAdditionalState = {}
    for _sub_fn in (
        _test_pytorch_eager_additional_sub_1,
        _test_pytorch_eager_additional_sub_2,
        _test_pytorch_eager_additional_sub_3,
        _test_pytorch_eager_additional_sub_4,
    ):
        _sub_fn(_state)
    assert len(_state) > 0
