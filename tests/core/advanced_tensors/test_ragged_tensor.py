"""Test module."""

from ml_switcheroo_compiler.core.ragged_tensor import RaggedTensor


class DummyTensor:
    def __init__(self, shape, dtype="float32", device="cpu", requires_grad=True):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.requires_grad = requires_grad

    def __len__(self):
        return self.shape[0]


def test_ragged_tensor():
    v = DummyTensor((4, 2))
    r = DummyTensor((3,))

    rt = RaggedTensor(v, r)
    assert rt.values == v
    assert rt.row_splits == r

    assert rt.shape == (2, -1, 2)
    assert rt.dtype == "float32"
    assert rt.device == "cpu"
    assert rt.requires_grad is True

    # Empty
    v2 = DummyTensor(())
    r2 = DummyTensor((0,))
    rt2 = RaggedTensor(v2, r2)
    assert rt2.shape == (0,)
