# ruff: noqa: F403

"""Core registry."""

import ml_switcheroo_compiler.ops.aliases as _aliases
import ml_switcheroo_compiler.ops.audio as _audio
import ml_switcheroo_compiler.ops.base as _base
import ml_switcheroo_compiler.ops.control_flow as _control_flow
import ml_switcheroo_compiler.ops.creation as _creation
import ml_switcheroo_compiler.ops.distributed as _distributed
import ml_switcheroo_compiler.ops.shape as _shape
import ml_switcheroo_compiler.ops.text as _text
import ml_switcheroo_compiler.ops.sparse as _sparse
import ml_switcheroo_compiler.ops.ragged as _ragged
import ml_switcheroo_compiler.ops.tensor_array as _tensor_array
import ml_switcheroo_compiler.ops.state as _state
import ml_switcheroo_compiler.ops.creation.frontend as _creation_frontend

__all__ = []
__all__.extend(getattr(_aliases, "__all__", [n for n in dir(_aliases) if not n.startswith("_")]))
__all__.extend(getattr(_audio, "__all__", [n for n in dir(_audio) if not n.startswith("_")]))
__all__.extend(getattr(_base, "__all__", [n for n in dir(_base) if not n.startswith("_")]))
__all__.extend(
    getattr(_control_flow, "__all__", [n for n in dir(_control_flow) if not n.startswith("_")])
)
__all__.extend(getattr(_creation, "__all__", [n for n in dir(_creation) if not n.startswith("_")]))
__all__.extend(
    getattr(_distributed, "__all__", [n for n in dir(_distributed) if not n.startswith("_")])
)
__all__.extend(getattr(_shape, "__all__", [n for n in dir(_shape) if not n.startswith("_")]))
__all__.extend(getattr(_text, "__all__", [n for n in dir(_text) if not n.startswith("_")]))
__all__.extend(getattr(_sparse, "__all__", [n for n in dir(_sparse) if not n.startswith("_")]))
__all__.extend(getattr(_ragged, "__all__", [n for n in dir(_ragged) if not n.startswith("_")]))
__all__.extend(
    getattr(_tensor_array, "__all__", [n for n in dir(_tensor_array) if not n.startswith("_")])
)
__all__.extend(getattr(_state, "__all__", [n for n in dir(_state) if not n.startswith("_")]))
__all__.extend(
    getattr(
        _creation_frontend, "__all__", [n for n in dir(_creation_frontend) if not n.startswith("_")]
    )
)
__all__ = list(set(__all__))

# Inject attributes into globals
for _mod in [
    _aliases,
    _audio,
    _base,
    _control_flow,
    _creation,
    _distributed,
    _shape,
    _text,
    _sparse,
    _ragged,
    _tensor_array,
    _state,
    _creation_frontend,
]:
    for _name in getattr(_mod, "__all__", [n for n in dir(_mod) if not n.startswith("_")]):
        globals()[_name] = getattr(_mod, _name)
