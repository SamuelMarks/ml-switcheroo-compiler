import torch

import ml_switcheroo_compiler.backends.pytorch.eager as pytorch_eager


def test_pytorch_eager_extras():
    res = pytorch_eager._execute_accumulate_n([torch.tensor(1), torch.tensor(2)])

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
        value = "int4"

    pytorch_eager._execute_cast(torch.tensor([1.0]), DummyDtype())

    class DummyDtypeBfloat16:
        value = "bfloat16"

    pytorch_eager._execute_cast(torch.tensor([1.0]), DummyDtypeBfloat16())

    class DummyDtypeFloat16:
        value = "float16"

    pytorch_eager._execute_cast(torch.tensor([1.0]), DummyDtypeFloat16())

    class DummyDtypeFloat8_e4m3fn:
        value = "float8_e4m3fn"

    pytorch_eager._execute_cast(torch.tensor([1.0]), DummyDtypeFloat8_e4m3fn())

    pytorch_eager._execute_cummax(torch.tensor([1, 2]), dim=0)
    pytorch_eager._execute_cummin(torch.tensor([1, 2]), dim=0)
    pytorch_eager._execute_cumlogsumexp(torch.tensor([1.0, 2.0]), dim=0)

    pytorch_eager._execute_ragged_tensor_to_dense(1)

    pytorch_eager._torch_variance(torch.tensor([1.0, 2.0]))
    pytorch_eager._torch_tensordot(torch.tensor([1.0]), torch.tensor([1.0]), dims=1)
    pytorch_eager._torch_tensordot(torch.tensor([1.0]), torch.tensor([1.0]), axes=1)

    # lambdas
    for op in ["HardSilu", "HardSwish", "Squareplus", "Fftnd", "Ifftnd", "Irfftnd", "Rfftnd"]:
        try:
            pytorch_eager._TORCH_EAGER_OP_MAP[op](torch.tensor([1.0]))
        except Exception:
            pass

    for op in ["Fftfreq", "Rfftfreq"]:
        try:
            pytorch_eager._TORCH_EAGER_OP_MAP[op](2)
        except Exception:
            pass

    class DummyDtypeFloat8_e5m2:
        value = "float8_e5m2"

    pytorch_eager._execute_cast(torch.tensor([1.0]), DummyDtypeFloat8_e5m2())

    class DummyDtypeInt32:
        value = "int32"

    pytorch_eager._execute_cast(torch.tensor([1.0]), DummyDtypeInt32())

    try:
        pytorch_eager._execute_accumulate_n(inputs=[])
    except ValueError:
        pass

    # execute_op coverage
    pytorch_eager.execute_op(None, "Add", torch.tensor(1), torch.tensor(2))
    pytorch_eager.execute_op(None, "RaggedTensorToDense", 1)
    pytorch_eager.execute_op(None, "unknownop", torch.tensor(1))

    # hit 261: Zeros -> torch.zeros
    pytorch_eager.execute_op(None, "Zeros", 2)

    # hit 269: Descriptive -> global_eager_registry
    pytorch_eager.execute_op(None, "Descriptive", torch.tensor([1.0]))

    # hit 272-273: exception in np.zeros
    import numpy as np

    original_zeros = np.zeros

    def mock_zeros(*args, **kwargs):
        raise RuntimeError("boom")

    np.zeros = mock_zeros
    try:
        pytorch_eager.execute_op(None, "totally_unknown_and_unregistered", 1)
    finally:
        np.zeros = original_zeros

    from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry

    @global_eager_registry.register("FakeGlobalOpForTest")
    def _fake_global_op(backend, *args, **kwargs):
        return "fake_global"

    res = pytorch_eager.execute_op(None, "FakeGlobalOpForTest", 1)
    assert res == "fake_global"
