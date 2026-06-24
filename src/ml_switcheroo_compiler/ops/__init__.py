"""Auto-generated ml_switcheroo_compiler.ops module exports. Refactored into registries."""

from ml_switcheroo_compiler.ops import _math_registry
from ml_switcheroo_compiler.ops import _nn_registry
from ml_switcheroo_compiler.ops import _core_registry
from ml_switcheroo_compiler.ops import _vision_registry

__all__ = []
__all__.extend(_math_registry.__all__)
__all__.extend(_nn_registry.__all__)
__all__.extend(_core_registry.__all__)
__all__.extend(_vision_registry.__all__)
__all__ = list(set(__all__))


def __getattr__(name: str) -> object:
    if hasattr(_math_registry, name):
        return getattr(_math_registry, name)
    if hasattr(_nn_registry, name):
        return getattr(_nn_registry, name)
    if hasattr(_core_registry, name):
        return getattr(_core_registry, name)
    if hasattr(_vision_registry, name):
        return getattr(_vision_registry, name)  # pragma: no cover
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
