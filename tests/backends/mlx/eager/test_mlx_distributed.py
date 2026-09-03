import pytest


def test_mlx_all_to_all_fallback(mocker):
    """Test all to all fallback."""
    from ml_switcheroo_compiler.backends.mlx.eager import _mlx_all_to_all

    class DummyCore:
        pass

    class DummyMLX:
        core = DummyCore()

    import sys

    mocker.patch.dict(sys.modules, {"mlx": DummyMLX, "mlx.core": DummyMLX.core})

    assert _mlx_all_to_all(None, "tensor") == "tensor"


def test_mlx_reduce_scatter_exception(mocker):
    """Test reduce scatter exception."""
    from ml_switcheroo_compiler.backends.mlx.eager import _mlx_reduce_scatter

    class DummyGroup:
        def rank(self):
            raise Exception("No")

    class DummyDistributed:
        def all_sum(self, tensor):
            class DummyTensor:
                shape = [10]
                ndim = 1

                def __getitem__(self, idx):
                    return "sliced"

            return DummyTensor()

    class DummyCore:
        distributed = DummyDistributed()

    class DummyMLX:
        core = DummyCore()

    import sys

    mocker.patch.dict(sys.modules, {"mlx": DummyMLX, "mlx.core": DummyMLX.core})

    res = _mlx_reduce_scatter(DummyMLX.core, "tensor", group=DummyGroup())
    assert res == "sliced"


def test_mlx_reduce_scatter_not_supported(mocker):
    """Test reduce scatter not supported."""
    from ml_switcheroo_compiler.backends.mlx.eager import _mlx_reduce_scatter
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    class DummyCore:
        pass

    class DummyMLX:
        core = DummyCore()

    import sys

    mocker.patch.dict(sys.modules, {"mlx": DummyMLX, "mlx.core": DummyMLX.core})

    with pytest.raises(BackendNotSupportedError, match="not supported"):
        _mlx_reduce_scatter(None, "tensor")
