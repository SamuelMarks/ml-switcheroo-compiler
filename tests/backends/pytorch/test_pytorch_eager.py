"""Tests for test_pytorch_eager."""

from __future__ import annotations

import typing

import torch

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_pytorch_ragged_tensor_to_tensor_eager() -> None:
    """Test function test_pytorch_ragged_tensor_to_tensor_eager."""
    from ml_switcheroo_compiler.backends.pytorch.eager import _execute_ragged_tensor_to_dense

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    DummyBackend()

    # 1. Nested tensor (mocked)
    class MockNested:
        """Test class MockNested."""

        is_nested = True

        def to_padded_tensor(self, padding: EagerValue) -> EagerValue:
            """Test function to_padded_tensor.

            Args:
                padding: Test parameter `padding`.
            """
            return "nested_padded"

    assert typing.cast("EagerCallableProtocol", _execute_ragged_tensor_to_dense)(typing.cast("EagerCallableProtocol", MockNested)()) == "nested_padded"

    # 2. List of tensors
    list_tensors = [torch.tensor([1, 2]), torch.tensor([3])]
    res = _execute_ragged_tensor_to_dense(list_tensors, default_value=0.0)
    assert isinstance(res, torch.Tensor)
    assert res.shape == (2, 2)

    # 3. Dictionary ragged encoding
    dict_named = {"values": torch.tensor([1, 2, 3]), "row_splits": torch.tensor([0, 2, 3])}

    res2 = _execute_ragged_tensor_to_dense(dict_named, default_value=0.0)
    assert typing.cast("EagerTensorProtocol", getattr(res2, "shape")) == (2, 2)

    # 4. Named tuple ragged encoding
    class MockNamed:
        """Test class MockNamed."""

        values = torch.tensor([1, 2, 3])
        row_splits = torch.tensor([0, 2, 3])

    res3 = typing.cast("EagerCallableProtocol", _execute_ragged_tensor_to_dense)(typing.cast("EagerCallableProtocol", MockNamed)(), default_value=0.0)
    assert typing.cast("EagerTensorProtocol", getattr(res3, "shape")) == (2, 2)

    # 5. Length <= 0
    dict_named_empty = {"values": torch.tensor([]), "row_splits": torch.tensor([0, 0, 0])}
    res4 = _execute_ragged_tensor_to_dense(dict_named_empty, default_value=0.0)
    assert typing.cast("EagerTensorProtocol", getattr(res4, "shape")) == (2, 0)
