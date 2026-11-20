"""Tests for test_pytorch_eager."""

from __future__ import annotations

import typing

import pytest

from tests.testing_utils import MockTestStub, extract_test_shape

if typing.TYPE_CHECKING:
    import torch

    from ml_switcheroo_compiler.backends.eager_registry import BackendArg

torch = pytest.importorskip("torch")


def test_pytorch_ragged_tensor_to_tensor_eager() -> None:
    """Test function test_pytorch_ragged_tensor_to_tensor_eager."""
    from ml_switcheroo_compiler.backends.pytorch.eager import _execute_ragged_tensor_to_dense

    class DummyBackend:
        """Test class DummyBackend."""

        pass

    DummyBackend()

    # 1. Nested tensor (mocked)
    class MockNested(MockTestStub):
        """Test class MockNested."""

        is_nested = True

        def to_padded_tensor(self, padding: BackendArg) -> BackendArg:
            """Test function to_padded_tensor.

            Args:
                padding: Test parameter `padding`.

            Returns:
                BackendArg: Computed output value from `to_padded_tensor`.
            """
            return "nested_padded"

    assert _execute_ragged_tensor_to_dense(MockNested()) == "nested_padded"

    # 2. List of tensors
    list_tensors = [torch.tensor([1, 2]), torch.tensor([3])]
    res = _execute_ragged_tensor_to_dense(list_tensors, default_value=0.0)
    assert isinstance(res, torch.Tensor)
    assert res.shape == (2, 2)

    # 3. Dictionary ragged encoding
    dict_named = {"values": torch.tensor([1, 2, 3]), "row_splits": torch.tensor([0, 2, 3])}

    res2 = _execute_ragged_tensor_to_dense(dict_named, default_value=0.0)
    assert extract_test_shape(res2) == (2, 2)

    # 4. Named tuple ragged encoding
    class MockNamed(MockTestStub):
        """Test class MockNamed."""

        values = torch.tensor([1, 2, 3])
        row_splits = torch.tensor([0, 2, 3])

    res3 = _execute_ragged_tensor_to_dense(MockNamed(), default_value=0.0)
    assert extract_test_shape(res3) == (2, 2)

    # 5. Length <= 0
    dict_named_empty = {"values": torch.tensor([]), "row_splits": torch.tensor([0, 0, 0])}
    res4 = _execute_ragged_tensor_to_dense(dict_named_empty, default_value=0.0)
    assert extract_test_shape(res4) == (2, 0)
