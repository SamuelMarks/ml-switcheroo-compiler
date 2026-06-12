"""Creation operations package."""

from ml_switcheroo_compiler.ops.creation.basic import (
    Arange,
    CreationOp,
    Full,
    Ones,
    Zeros,
)

__all__ = [
    "Arange",
    "CreationOp",
    "Full",
    "Ones",
    "Zeros",
]
from .frontend import (
    arange as arange,
)
from .frontend import (
    diag as diag,
)
from .frontend import (
    empty as empty,
)
from .frontend import (
    eye as eye,
)
from .frontend import (
    full as full,
)
from .frontend import (
    full_like as full_like,
)
from .frontend import (
    identity as identity,
)
from .frontend import (
    linspace as linspace,
)
from .frontend import (
    ones as ones,
)
from .frontend import (
    ones_like as ones_like,
)
from .frontend import (
    zeros as zeros,
)
from .frontend import (
    zeros_like as zeros_like,
)
