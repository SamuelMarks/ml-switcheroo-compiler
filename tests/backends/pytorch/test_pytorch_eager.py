import torch


def test_pytorch_ragged_tensor_to_tensor_eager():
    from ml_switcheroo_compiler.backends.pytorch.eager import _execute_ragged_tensor_to_dense

    class DummyBackend:
        pass

    db = DummyBackend()

    # 1. Nested tensor (mocked)
    class MockNested:
        is_nested = True

        def to_padded_tensor(self, padding):
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
    assert res2.shape == (2, 2)

    # 4. Named tuple ragged encoding
    class MockNamed:
        values = torch.tensor([1, 2, 3])
        row_splits = torch.tensor([0, 2, 3])

    res3 = _execute_ragged_tensor_to_dense(MockNamed(), default_value=0.0)
    assert res3.shape == (2, 2)

    # 5. Length <= 0
    dict_named_empty = {"values": torch.tensor([]), "row_splits": torch.tensor([0, 0, 0])}
    res4 = _execute_ragged_tensor_to_dense(dict_named_empty, default_value=0.0)
    assert res4.shape == (2, 0)
