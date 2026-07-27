from ml_switcheroo_compiler.ops._math_registry import __all__ as math_all
from ml_switcheroo_compiler.ops._nn_registry import __all__ as nn_all
from ml_switcheroo_compiler.ops._vision_registry import __all__ as vision_all


def test_registries():
    assert isinstance(math_all, list)
    assert isinstance(nn_all, list)
    assert isinstance(vision_all, list)
