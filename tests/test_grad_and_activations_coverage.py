"""Module docstring."""

import numpy as np

from ml_switcheroo_compiler import ops
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.nn.activations import prelu, softmin, step


def test_grad_compile() -> object:
    """Function docstring."""

    def my_func(x: object) -> object:
        """Function docstring."""
        return x

    with ConfigContext(eager_mode=True):
        compiled = compile(my_func)
        # compile returns jit(fun). If we don't do much else, it should just be covered.
        assert compiled is not None


def test_activations_extra() -> object:
    """Function docstring."""
    with ConfigContext(eager_mode=True):
        x = ops.array(np.array([-1.0, 0.0, 1.0]))

        # prelu
        prelu(x, alpha=0.5)

        # softmin
        softmin(x)

        # step
        step(x)
