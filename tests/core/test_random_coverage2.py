import numpy as np
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.random import categorical, choice, PRNGKey


def test_random_extra_coverage():
    device = Device(DeviceType.CPU, 0)

    with ConfigContext(eager_mode=True):
        key1 = PRNGKey(0)
        logits1d = Tensor(np.array([1.0, 2.0]), (2,), DType.Float32, device)
        res1 = categorical(key1, logits1d)
        assert res1 is not None

    with ConfigContext(eager_mode=False):
        from ml_switcheroo_compiler.tracing import _tracer
        from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

        _tracer.start_tracing()
        try:
            key2 = PRNGKey(0)
            a = Tensor(ProxyTensor("a", (3,), "int32"), (3,), DType.Int32, device)
            p = Tensor(ProxyTensor("p", (3,), "float32"), (3,), DType.Float32, device)
            res2 = choice(key2, a, shape=(10,), p=p)
            assert res2 is not None
        finally:
            _tracer.stop_tracing()


def test_permutation_eager_none():
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.random import permutation
    from unittest.mock import MagicMock

    config.eager_mode = True
    key = MagicMock()
    key.data = [0, 1]
    import numpy as np

    x = MagicMock()
    x.data = np.array([1, 2, 3])
    x.dtype = "float32"
    x.shape = (3,)

    # Test path where x has no shape/dtype getattr fallback
    class MockNoShape:
        data = np.array([1, 2, 3])

    out = permutation(key, MockNoShape())
    assert out is not None


def test_choice_eager_p():
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.random import choice
    from unittest.mock import MagicMock

    config.eager_mode = True
    key = MagicMock()
    key.data = [0, 1]
    import numpy as np

    a = MagicMock()
    a.data = np.array([1, 2, 3])
    a.dtype = "float32"

    p = MagicMock()
    p.data = np.array([0.1, 0.2, 0.7])

    out = choice(key, a, p=p)
    assert out is not None


def test_categorical_eager_2d():
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.random import categorical, PRNGKey
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.device import Device, DeviceType
    import numpy as np

    device = Device(DeviceType.CPU, 0)
    config.eager_mode = True
    key = PRNGKey(0)
    # 2D logits
    logits2d = Tensor(np.array([[1.0, 2.0], [0.5, 0.5]]), (2, 2), DType.Float32, device)
    res = categorical(key, logits2d)
    assert res.shape == ()
    assert res.data.shape == (2,)


def test_truncated_normal_eager_rejection():
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.random import truncated_normal, PRNGKey
    from ml_switcheroo_compiler.core.dtype import DType

    config.eager_mode = True
    key = PRNGKey(0)
    # Create very tight bounds so the rejection sampling loop has to run multiple times
    res = truncated_normal(key, lower=-0.0001, upper=0.0001, shape=(1000,), dtype=DType.Float32)
    assert res.shape == (1000,)
