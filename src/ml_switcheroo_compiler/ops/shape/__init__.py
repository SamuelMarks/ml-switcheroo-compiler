"""Shape manipulation operations package."""

from ml_switcheroo_compiler.ops.shape.basic import (
    BroadcastTo,
    Reshape,
    Transpose,
)

__all__ = [
    "BroadcastTo",
    "Reshape",
    "Transpose",
]
from .frontend import (
    broadcast_to as broadcast_to,
)
from .frontend import (
    concatenate as concatenate,
)
from .frontend import (
    dynamic_slice as dynamic_slice,
)
from .frontend import (
    expand as expand,
)
from .frontend import (
    flatten as flatten,
)
from .frontend import (
    gather as gather,
)
from .frontend import (
    gather_nd as gather_nd,
)
from .frontend import (
    meshgrid as meshgrid,
)
from .frontend import (
    moveaxis as moveaxis,
)
from .frontend import (
    pad as pad,
)
from .frontend import (
    permute as permute,
)
from .frontend import (
    repeat as repeat,
)
from .frontend import (
    reshape as reshape,
)
from .frontend import (
    roll as roll,
)
from .frontend import (
    scatter as scatter,
)
from .frontend import (
    scatter_add as scatter_add,
)
from .frontend import (
    scatter_nd as scatter_nd,
)
from .frontend import (
    slice as slice,
)
from .frontend import (
    split as split,
)
from .frontend import (
    squeeze as squeeze,
)
from .frontend import (
    stack as stack,
)
from .frontend import (
    strided_slice as strided_slice,
)
from .frontend import (
    swapaxes as swapaxes,
)
from .frontend import (
    take as take,
)
from .frontend import (
    take_along_axis as take_along_axis,
)
from .frontend import (
    tile as tile,
)
from .frontend import (
    transpose as transpose,
)
from .frontend import (
    tril as tril,
)
from .frontend import (
    triu as triu,
)
from .frontend import (
    unsqueeze as unsqueeze,
)
from .frontend import (
    unstack as unstack,
)
from .frontend import (
    update_slice as update_slice,
)
from .frontend import (
    where as where,
)
