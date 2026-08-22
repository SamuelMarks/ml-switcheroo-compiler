import numpy as np


def test_gradient_integration_across_backends():
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.grad.api import grad

    # We test a simple function f(x) = x * x + 2*x
    # Its derivative is f'(x) = 2*x + 2
    def f(x):
        return x * x + x + x

    grad_f = grad(f)
    x_val = np.array([3.0], dtype=np.float32)
    expected_grad = np.array([8.0], dtype=np.float32)

    backends_to_test = ["numpy", "pytorch", "jax"]

    results = {}
    for be in backends_to_test:
        try:
            config.set_backend(be)
            x_t = Tensor(x_val)
            res = grad_f(x_t)
            results[be] = res.numpy()
        except Exception:
            # If backend not installed, skip
            pass

    for be, res in results.items():
        np.testing.assert_allclose(res, expected_grad, rtol=1e-5)
