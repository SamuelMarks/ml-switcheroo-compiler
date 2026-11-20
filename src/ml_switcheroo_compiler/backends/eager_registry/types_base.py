"""Foundational operand and value type aliases for `ml_switcheroo_compiler.backends.eager_registry`."""

from __future__ import annotations

import typing
from collections.abc import Sequence

import pydantic

from ml_switcheroo_compiler.core.shape import OpValue, SliceProtocol
from ml_switcheroo_compiler.core.shape import ShapeProtocol as ShapeProtocol

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator, VisitorProtocol
    from ml_switcheroo_compiler.backends.common.mixins.common import FallbackPrefixProtocol
    from ml_switcheroo_compiler.backends.eager.types_utils import ZerosModule
    from ml_switcheroo_compiler.backends.eager_registry.protocols import EagerBoundCallable
    from ml_switcheroo_compiler.backends.edge.onnx import GraphProtoType, NodeProtoType


def _passthrough_tensor_val(val: BackendArg) -> BackendArg:
    """Return `val` unchanged for Pydantic plain-validator core schema serialization.

    Args:
        val (BackendArg): Candidate runtime tensor or scalar value.

    Returns:
        BackendArg: Unmodified runtime value.
    """
    return val


_AttributeValuePydanticAnnotation = pydantic.PlainValidator(_passthrough_tensor_val)

SliceType = SliceProtocol
BackendArg = OpValue
_GeneratorOrVisitor = typing.Union["BaseGenerator", "FallbackPrefixProtocol", "VisitorProtocol", "ZerosModule"]

BackendOperand = typing.Annotated[
    typing.Union[  # noqa: UP007
        BackendArg,
        "EagerBoundCallable",
        _GeneratorOrVisitor,
        "GraphProtoType",
        "NodeProtoType",
        "Sequence[NodeProtoType]",
    ],
    _AttributeValuePydanticAnnotation,
]
