"""Static structural protocol bindings shard 38 (outer..product)."""

from __future__ import annotations

import types
import typing
from collections.abc import Callable

from ml_switcheroo_compiler.backends.eager_registry.protocols import EagerTensorProtocol
from ml_switcheroo_compiler.backends.eager_registry.types_base import BackendOperand

MemberReader = Callable[
    [typing.Union[types.ModuleType, BackendOperand]],  # noqa: UP007
    tuple[bool, typing.Optional[EagerTensorProtocol]],  # noqa: UP045
]


@typing.runtime_checkable
class _HasMember38_00(typing.Protocol):
    """Structural protocol for objects exposing `outer`."""

    outer: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_01(typing.Protocol):
    """Structural protocol for objects exposing `outfeed`."""

    outfeed: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_02(typing.Protocol):
    """Structural protocol for objects exposing `output_shape_holder_protocol`."""

    output_shape_holder_protocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_03(typing.Protocol):
    """Structural protocol for objects exposing `outputs`."""

    outputs: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_04(typing.Protocol):
    """Structural protocol for objects exposing `outputshapeholderprotocol`."""

    outputshapeholderprotocol: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_05(typing.Protocol):
    """Structural protocol for objects exposing `packbits`."""

    packbits: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_06(typing.Protocol):
    """Structural protocol for objects exposing `pad`."""

    pad: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_07(typing.Protocol):
    """Structural protocol for objects exposing `pad_images`."""

    pad_images: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_08(typing.Protocol):
    """Structural protocol for objects exposing `pad_to_bounding_box`."""

    pad_to_bounding_box: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_09(typing.Protocol):
    """Structural protocol for objects exposing `parameters`."""

    parameters: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_10(typing.Protocol):
    """Structural protocol for objects exposing `params`."""

    params: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_11(typing.Protocol):
    """Structural protocol for objects exposing `parse_tensor`."""

    parse_tensor: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_12(typing.Protocol):
    """Structural protocol for objects exposing `parsetensor`."""

    parsetensor: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_13(typing.Protocol):
    """Structural protocol for objects exposing `partition`."""

    partition: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_14(typing.Protocol):
    """Structural protocol for objects exposing `partition_nth_indices`."""

    partition_nth_indices: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_15(typing.Protocol):
    """Structural protocol for objects exposing `pdf`."""

    pdf: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_16(typing.Protocol):
    """Structural protocol for objects exposing `pdot`."""

    pdot: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_17(typing.Protocol):
    """Structural protocol for objects exposing `percentile`."""

    percentile: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_18(typing.Protocol):
    """Structural protocol for objects exposing `permutation`."""

    permutation: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_19(typing.Protocol):
    """Structural protocol for objects exposing `permute`."""

    permute: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_20(typing.Protocol):
    """Structural protocol for objects exposing `perspective_transform`."""

    perspective_transform: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_21(typing.Protocol):
    """Structural protocol for objects exposing `perspectivetransform`."""

    perspectivetransform: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_22(typing.Protocol):
    """Structural protocol for objects exposing `piecewise`."""

    piecewise: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_23(typing.Protocol):
    """Structural protocol for objects exposing `pinv`."""

    pinv: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_24(typing.Protocol):
    """Structural protocol for objects exposing `pkg`."""

    pkg: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_25(typing.Protocol):
    """Structural protocol for objects exposing `pmap`."""

    pmap: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_26(typing.Protocol):
    """Structural protocol for objects exposing `pmean`."""

    pmean: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_27(typing.Protocol):
    """Structural protocol for objects exposing `pmf`."""

    pmf: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_28(typing.Protocol):
    """Structural protocol for objects exposing `poisson`."""

    poisson: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_29(typing.Protocol):
    """Structural protocol for objects exposing `polar`."""

    polar: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_30(typing.Protocol):
    """Structural protocol for objects exposing `poly_fn`."""

    poly_fn: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_31(typing.Protocol):
    """Structural protocol for objects exposing `polygamma`."""

    polygamma: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_32(typing.Protocol):
    """Structural protocol for objects exposing `polyint`."""

    polyint: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_33(typing.Protocol):
    """Structural protocol for objects exposing `polyval`."""

    polyval: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_34(typing.Protocol):
    """Structural protocol for objects exposing `pop`."""

    pop: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_35(typing.Protocol):
    """Structural protocol for objects exposing `positive`."""

    positive: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_36(typing.Protocol):
    """Structural protocol for objects exposing `posterize`."""

    posterize: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_37(typing.Protocol):
    """Structural protocol for objects exposing `pow`."""

    pow: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_38(typing.Protocol):
    """Structural protocol for objects exposing `power`."""

    power: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_39(typing.Protocol):
    """Structural protocol for objects exposing `power_iteration`."""

    power_iteration: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_40(typing.Protocol):
    """Structural protocol for objects exposing `ppermute`."""

    ppermute: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_41(typing.Protocol):
    """Structural protocol for objects exposing `prelu`."""

    prelu: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_42(typing.Protocol):
    """Structural protocol for objects exposing `printable_graph`."""

    printable_graph: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_43(typing.Protocol):
    """Structural protocol for objects exposing `prod`."""

    prod: EagerTensorProtocol


@typing.runtime_checkable
class _HasMember38_44(typing.Protocol):
    """Structural protocol for objects exposing `product`."""

    product: EagerTensorProtocol


SHARD_38_READERS: dict[str, MemberReader] = {
    "outer": lambda t: (True, t.outer) if isinstance(t, _HasMember38_00) else (False, None),
    "outfeed": lambda t: (True, t.outfeed) if isinstance(t, _HasMember38_01) else (False, None),
    "output_shape_holder_protocol": lambda t: (
        (True, t.output_shape_holder_protocol) if isinstance(t, _HasMember38_02) else (False, None)
    ),
    "outputs": lambda t: (True, t.outputs) if isinstance(t, _HasMember38_03) else (False, None),
    "outputshapeholderprotocol": lambda t: (
        (True, t.outputshapeholderprotocol) if isinstance(t, _HasMember38_04) else (False, None)
    ),
    "packbits": lambda t: (True, t.packbits) if isinstance(t, _HasMember38_05) else (False, None),
    "pad": lambda t: (True, t.pad) if isinstance(t, _HasMember38_06) else (False, None),
    "pad_images": lambda t: (True, t.pad_images) if isinstance(t, _HasMember38_07) else (False, None),
    "pad_to_bounding_box": lambda t: (True, t.pad_to_bounding_box) if isinstance(t, _HasMember38_08) else (False, None),
    "parameters": lambda t: (True, t.parameters) if isinstance(t, _HasMember38_09) else (False, None),
    "params": lambda t: (True, t.params) if isinstance(t, _HasMember38_10) else (False, None),
    "parse_tensor": lambda t: (True, t.parse_tensor) if isinstance(t, _HasMember38_11) else (False, None),
    "parsetensor": lambda t: (True, t.parsetensor) if isinstance(t, _HasMember38_12) else (False, None),
    "partition": lambda t: (True, t.partition) if isinstance(t, _HasMember38_13) else (False, None),
    "partition_nth_indices": lambda t: (
        (True, t.partition_nth_indices) if isinstance(t, _HasMember38_14) else (False, None)
    ),
    "pdf": lambda t: (True, t.pdf) if isinstance(t, _HasMember38_15) else (False, None),
    "pdot": lambda t: (True, t.pdot) if isinstance(t, _HasMember38_16) else (False, None),
    "percentile": lambda t: (True, t.percentile) if isinstance(t, _HasMember38_17) else (False, None),
    "permutation": lambda t: (True, t.permutation) if isinstance(t, _HasMember38_18) else (False, None),
    "permute": lambda t: (True, t.permute) if isinstance(t, _HasMember38_19) else (False, None),
    "perspective_transform": lambda t: (
        (True, t.perspective_transform) if isinstance(t, _HasMember38_20) else (False, None)
    ),
    "perspectivetransform": lambda t: (
        (True, t.perspectivetransform) if isinstance(t, _HasMember38_21) else (False, None)
    ),
    "piecewise": lambda t: (True, t.piecewise) if isinstance(t, _HasMember38_22) else (False, None),
    "pinv": lambda t: (True, t.pinv) if isinstance(t, _HasMember38_23) else (False, None),
    "pkg": lambda t: (True, t.pkg) if isinstance(t, _HasMember38_24) else (False, None),
    "pmap": lambda t: (True, t.pmap) if isinstance(t, _HasMember38_25) else (False, None),
    "pmean": lambda t: (True, t.pmean) if isinstance(t, _HasMember38_26) else (False, None),
    "pmf": lambda t: (True, t.pmf) if isinstance(t, _HasMember38_27) else (False, None),
    "poisson": lambda t: (True, t.poisson) if isinstance(t, _HasMember38_28) else (False, None),
    "polar": lambda t: (True, t.polar) if isinstance(t, _HasMember38_29) else (False, None),
    "poly_fn": lambda t: (True, t.poly_fn) if isinstance(t, _HasMember38_30) else (False, None),
    "polygamma": lambda t: (True, t.polygamma) if isinstance(t, _HasMember38_31) else (False, None),
    "polyint": lambda t: (True, t.polyint) if isinstance(t, _HasMember38_32) else (False, None),
    "polyval": lambda t: (True, t.polyval) if isinstance(t, _HasMember38_33) else (False, None),
    "pop": lambda t: (True, t.pop) if isinstance(t, _HasMember38_34) else (False, None),
    "positive": lambda t: (True, t.positive) if isinstance(t, _HasMember38_35) else (False, None),
    "posterize": lambda t: (True, t.posterize) if isinstance(t, _HasMember38_36) else (False, None),
    "pow": lambda t: (True, t.pow) if isinstance(t, _HasMember38_37) else (False, None),
    "power": lambda t: (True, t.power) if isinstance(t, _HasMember38_38) else (False, None),
    "power_iteration": lambda t: (True, t.power_iteration) if isinstance(t, _HasMember38_39) else (False, None),
    "ppermute": lambda t: (True, t.ppermute) if isinstance(t, _HasMember38_40) else (False, None),
    "prelu": lambda t: (True, t.prelu) if isinstance(t, _HasMember38_41) else (False, None),
    "printable_graph": lambda t: (True, t.printable_graph) if isinstance(t, _HasMember38_42) else (False, None),
    "prod": lambda t: (True, t.prod) if isinstance(t, _HasMember38_43) else (False, None),
    "product": lambda t: (True, t.product) if isinstance(t, _HasMember38_44) else (False, None),
}
