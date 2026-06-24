# ruff: noqa: F403

"""NN registry."""

import ml_switcheroo_compiler.ops.nn as _nn
import ml_switcheroo_compiler.ops.normalization as _normalization

__all__ = []
__all__.extend(getattr(_nn, "__all__", [n for n in dir(_nn) if not n.startswith("_")]))
__all__.extend(
    getattr(_normalization, "__all__", [n for n in dir(_normalization) if not n.startswith("_")])
)
__all__ = list(set(__all__))

# Inject attributes into globals
for _mod in [_nn, _normalization]:
    for _name in getattr(_mod, "__all__", [n for n in dir(_mod) if not n.startswith("_")]):
        globals()[_name] = getattr(_mod, _name)
