"""Core type unions and Pydantic annotations for Shape Layer 2."""

from __future__ import annotations

import sys
import typing
from collections.abc import Callable, Mapping, Sequence

import pydantic
from ml_switcheroo_ir import (
    ArrayBufferProtocol as IRArrayBufferProtocol,
)
from ml_switcheroo_ir import (
    CustomRuleHandleProtocol as IRCustomRuleHandleProtocol,
)
from ml_switcheroo_ir import (
    DeviceLikeProtocol,
    DTypeLikeProtocol,
    IRGraph,
    LogicalGraph,
    LogicalNode,
    LoopOptionsProtocol,
    PartitionSpec,
    SubgraphLikeProtocol,
    SymbolicIntWrapperProtocol,
    SymbolicNodeProtocol,
    TensorShape,
    TensorSpec,
)
from ml_switcheroo_ir import (
    DType as IRDType,
)
from ml_switcheroo_ir import (
    SymInt as IRSymInt,
)
from ml_switcheroo_ir import (
    SymNode as IRSymNode,
)

from ml_switcheroo_compiler.core._shape_protocols import (
    CallableObjectProtocol,
    CtypesArrayProtocol,
    DataHolderProtocol,
    DeviceHolderProtocol,
    DTypeHolderProtocol,
    ShapeMetadataProtocol,
    ShapeProtocol,
    SliceProtocol,
    TypedVariadicCallable,
    VariadicCallableProtocol,
    _ItemMemberProtocol,
    _R_co,
)
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ir.shape_system import SymInt, SymNode, SymVar

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.core.tensor import Tensor

if sys.version_info >= (3, 10):  # noqa: UP036
    from types import EllipsisType as _EllipsisType

    _OpValueSelf: typing.TypeAlias = "OpValue"
    _IRAttrSelf: typing.TypeAlias = "IRAttrValue"
else:
    _EllipsisType = type(Ellipsis)
    _OpValueSelf = DType
    _IRAttrSelf = DType

_PassValT = typing.TypeVar("_PassValT")


def _passthrough_identity(val: _PassValT) -> _PassValT:
    """Return ``val`` unchanged for Pydantic core schema validation.

    Args:
        val (_PassValT): Candidate input value.

    Returns:
        _PassValT: Unmodified input value.
    """
    return val


class _PassthroughPydanticAnnotation:
    """Non-recursive Pydantic v2 annotation accepting validated compiler domain values."""

    def __get_pydantic_core_schema__(
        self,
        _source_type: type,
        _handler: pydantic.GetCoreSchemaHandler,
    ) -> Mapping[str, str | Mapping[str, str | Callable[[_PassValT], _PassValT]]]:
        """Build a plain pass-through Pydantic core schema without expanding recursive unions.

        Args:
            _source_type (type): Annotated union type.
            _handler (pydantic.GetCoreSchemaHandler): Pydantic schema handler.

        Returns:
            Mapping[str, str | Mapping[str, str | Callable[[_PassValT], _PassValT]]]: Plain validator schema dict.
        """
        return {"type": "function-plain", "function": {"type": "no-info", "function": _passthrough_identity}}


_AttributeValuePydanticAnnotation = _PassthroughPydanticAnnotation()

SliceType = SliceProtocol
CallableLike = typing.Union[  # noqa: UP007
    Callable[[], _R_co],
    Callable[[typing.NoReturn], _R_co],
    Callable[[typing.NoReturn, typing.NoReturn], _R_co],
    Callable[[typing.NoReturn, typing.NoReturn, typing.NoReturn], _R_co],
    Callable[[typing.NoReturn, typing.NoReturn, typing.NoReturn, typing.NoReturn], _R_co],
    Callable[[typing.NoReturn, typing.NoReturn, typing.NoReturn, typing.NoReturn, typing.NoReturn], _R_co],
    TypedVariadicCallable[_R_co],
]
IRSymbolicLeaf = typing.Union[  # noqa: UP007
    IRSymNode,
    IRSymInt,
    SymbolicNodeProtocol,
    SymbolicIntWrapperProtocol,
]
SymbolicDim = typing.Union[str, SymNode, SymInt, SymVar, IRSymbolicLeaf]  # noqa: UP007
ShapeLike = typing.Annotated[
    typing.Union[  # noqa: UP007
        tuple[typing.Union[int, SymbolicDim, None], ...],  # noqa: UP007
        Sequence[typing.Union[int, SymbolicDim, None]],  # noqa: UP007
        int,
        SymbolicDim,
        TensorShape,
        ShapeProtocol,
        None,
    ],
    _AttributeValuePydanticAnnotation,
]
AxesLike = typing.Annotated[
    typing.Union[int, Sequence[int], ShapeProtocol, None],  # noqa: UP007
    _AttributeValuePydanticAnnotation,
]
Scalar = typing.Union[int, float, bool, complex]  # noqa: UP007
ScalarLike = Scalar
TensorOperand = typing.Union["Tensor", Scalar]  # noqa: UP007
T_Tensor = typing.TypeVar("T_Tensor", bound=ShapeProtocol)


@typing.runtime_checkable
class CustomRuleHandleProtocol(typing.Protocol):
    """Protocol for serializable custom AD rule handles."""

    @property
    def rule_id(self) -> str:
        """Return unique registered rule identifier.

        Returns:
            str: Registered callback identifier.
        """
        ...


IRProtocolLeaf = typing.Union[  # noqa: UP007
    IRDType,
    DTypeLikeProtocol,
    DeviceLikeProtocol,
    IRCustomRuleHandleProtocol,
    IRArrayBufferProtocol,
    pydantic.BaseModel,
]
IRTypeMeta = typing.Union[  # noqa: UP007
    DType,
    Device,
    TensorShape,
    TensorSpec,
    PartitionSpec,
    IRSymbolicLeaf,
    IRProtocolLeaf,
]
IRGraphOrHook = typing.Union[  # noqa: UP007
    LogicalNode,
    LogicalGraph,
    IRGraph,
    SubgraphLikeProtocol,
    LoopOptionsProtocol,
    CustomRuleHandleProtocol,
]
IRDescriptor = typing.Union[IRTypeMeta, IRGraphOrHook]  # noqa: UP007
IRLeafOrDescriptor = typing.Union[Scalar, SymbolicDim, bytes, None, IRDescriptor, ShapeProtocol]  # noqa: UP007
IRAttrValue = typing.Annotated[
    typing.Union[  # noqa: UP007
        IRLeafOrDescriptor,
        Sequence[_IRAttrSelf],
        Mapping[str, _IRAttrSelf],
    ],
    _AttributeValuePydanticAnnotation,
]


@typing.runtime_checkable
class CompilerConfigProtocol(typing.Protocol):
    """Protocol for structured operator/compiler configuration models exposing `model_dump`."""

    def model_dump(self) -> dict[str, IRAttrValue]:
        """Serialize configuration fields into a dictionary.

        Returns:
            dict[str, IRAttrValue]: Serialized configuration mapping.
        """
        ...


@typing.runtime_checkable
class NdimHolderProtocol(typing.Protocol):
    """Protocol for multidimensional array buffers exposing `.ndim`."""

    @property
    def ndim(self) -> int:
        """Return tensor rank.

        Returns:
            int: Number of dimensions.
        """
        ...


ArrayBufferProtocol = typing.Union[  # noqa: UP007
    ShapeProtocol,
    NdimHolderProtocol,
    ShapeMetadataProtocol,
    DTypeHolderProtocol,
    DeviceHolderProtocol,
    DataHolderProtocol,
    CtypesArrayProtocol,
    _ItemMemberProtocol,
]
IndexSpec = typing.Union[SliceType, _EllipsisType]  # noqa: UP007
OpCallableOrConfig = typing.Union[  # noqa: UP007
    CallableObjectProtocol,
    VariadicCallableProtocol,
    CompilerConfigProtocol,
    type,
]
OpValue = typing.Annotated[
    typing.Union[  # noqa: UP007
        IRLeafOrDescriptor,
        ArrayBufferProtocol,
        IndexSpec,
        OpCallableOrConfig,
        TypedVariadicCallable[_OpValueSelf],
        Sequence[_OpValueSelf],
        Mapping[str, _OpValueSelf],
        Mapping[typing.Union[str, int], _OpValueSelf],  # noqa: UP007
    ],
    _AttributeValuePydanticAnnotation,
]
