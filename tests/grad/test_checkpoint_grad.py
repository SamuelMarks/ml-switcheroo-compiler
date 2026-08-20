def test_checkpoint_grad():
    import numpy as np

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.grad.api import value_and_grad
    from ml_switcheroo_compiler.grad.checkpointing import checkpoint

    config.backend = "numpy"
    config.eager_mode = False

    def my_fun(x):
        return x * x * x

    cp_fun = checkpoint(my_fun)

    vg_std = value_and_grad(my_fun)
    vg_cp = value_and_grad(cp_fun)

    x = Tensor(np.array([2.0], dtype=np.float32), TensorConfig(shape=(1,), dtype="float32", device="cpu"))

    from ml_switcheroo_compiler.grad.jit import jit

    jitted_std = jit(vg_std)
    jitted_cp = jit(vg_cp)

    v1, g1 = jitted_std(x)
    v2, g2 = jitted_cp(x)

    np.testing.assert_allclose(v1.numpy(), v2.numpy())
    np.testing.assert_allclose(g1.numpy(), g2.numpy())
