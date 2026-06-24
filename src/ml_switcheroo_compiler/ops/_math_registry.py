# ruff: noqa: F403

"""Math registry."""

import ml_switcheroo_compiler.ops.binary as _binary
import ml_switcheroo_compiler.ops.unary as _unary
import ml_switcheroo_compiler.ops.linalg as _linalg
import ml_switcheroo_compiler.ops.reductions as _reductions
import ml_switcheroo_compiler.ops.random_ops as _random_ops

__all__ = []
__all__.extend(getattr(_binary, "__all__", [n for n in dir(_binary) if not n.startswith("_")]))
__all__.extend(getattr(_unary, "__all__", [n for n in dir(_unary) if not n.startswith("_")]))
__all__.extend(getattr(_linalg, "__all__", [n for n in dir(_linalg) if not n.startswith("_")]))
__all__.extend(
    getattr(_reductions, "__all__", [n for n in dir(_reductions) if not n.startswith("_")])
)
__all__.extend(
    getattr(_random_ops, "__all__", [n for n in dir(_random_ops) if not n.startswith("_")])
)
__all__ = list(set(__all__))

# Inject attributes into globals
for _mod in [_binary, _unary, _linalg, _reductions, _random_ops]:
    for _name in getattr(_mod, "__all__", [n for n in dir(_mod) if not n.startswith("_")]):
        globals()[_name] = getattr(_mod, _name)
