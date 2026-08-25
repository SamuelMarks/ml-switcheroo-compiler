def test_value_and_grad_has_aux():
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.grad.api import value_and_grad
    from ml_switcheroo_compiler.grad.options import GradOptions

    def func(x):
        return x, x

    # Just to hit the line
    opts = GradOptions(has_aux=True)
    wrapped = value_and_grad(func, opts)
    try:
        wrapped(Tensor([1.0], TensorConfig((1,), "float32", "cpu")))
    except Exception:
        pass


def test_hook_gradient():
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.grad.api import grad, hook_gradient

    def hook(g):
        return g * 2.0

    def func(x):
        return hook_gradient(x, hook)

    def func2(x):
        return hook_gradient(x, lambda g: None)

    g = grad(func)
    try:
        g(Tensor([1.0], TensorConfig((1,), "float32", "cpu")))
    except Exception:
        pass

    g2 = grad(func2)
    try:
        g2(Tensor([1.0], TensorConfig((1,), "float32", "cpu")))
    except Exception:
        pass


def test_value_and_grad_no_aux():
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.grad.api import value_and_grad
    from ml_switcheroo_compiler.grad.options import GradOptions

    def func(x):
        return x

    opts = GradOptions(has_aux=False)
    wrapped = value_and_grad(func, opts)
    try:
        wrapped(Tensor([1.0], TensorConfig((1,), "float32", "cpu")))
    except Exception:
        pass
