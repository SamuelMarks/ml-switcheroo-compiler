# ruff: noqa: F403

"""Vision registry."""

import ml_switcheroo_compiler.ops.vision as _vision
import ml_switcheroo_compiler.ops.image as _image
import ml_switcheroo_compiler.ops.vision.ops as _vision_ops

__all__ = []
__all__.extend(getattr(_vision, "__all__", [n for n in dir(_vision) if not n.startswith("_")]))
__all__.extend(getattr(_image, "__all__", [n for n in dir(_image) if not n.startswith("_")]))
__all__.extend(
    getattr(_vision_ops, "__all__", [n for n in dir(_vision_ops) if not n.startswith("_")])
)
__all__ = list(set(__all__))

# Inject attributes into globals
for _mod in [_vision, _image, _vision_ops]:
    for _name in getattr(_mod, "__all__", [n for n in dir(_mod) if not n.startswith("_")]):
        globals()[_name] = getattr(_mod, _name)
