"""Core runtime shape protocols, metadata extractors, and dynamic validation (Shape Layer 2).

Layer Responsibilities across the Shape Subsystem:
    1. ``ir.shape_system``: Symbolic algebra (``Polynomial``, ``SymNode``, ``SymInt``), constraint unification,
       canonical contiguous strides, and symbolic matrix/broadcast rules.
    2. ``core.shape``: Runtime structural protocols, shape/dtype/device extraction, and dynamic validation.
    3. ``ops.shape_inference``: Declarative YAML op signatures and per-operator transfer functions.
    4. ``transforms.passes.shape_inference``: Topological whole-graph shape propagation and runtime shape learning.
"""

from __future__ import annotations

import typing

from ml_switcheroo_ir import AttributeValue, IRGraph, LogicalGraph, LogicalNode, TensorShape, TensorSpec

from ml_switcheroo_compiler.core._shape_composites import (
    LinearOperatorOperandProtocol,
    RaggedOperandProtocol,
    SparseOperandProtocol,
    TensorArrayOperandProtocol,
    TensorOperandProtocol,
)
from ml_switcheroo_compiler.core._shape_extractors import (
    _has_data_holder_attr,
    _has_device_holder_attr,
    _has_dtype_holder_attr,
    _has_shape_attr,
    _has_shape_metadata_attr,
    extract_data,
    extract_device,
    extract_dtype,
    extract_int_shape,
    extract_shape,
    extract_symbolic_shape,
    is_backend_buffer,
    is_traced_data,
    is_variadic_callable,
)
from ml_switcheroo_compiler.core._shape_protocols import (
    BackendBufferProtocol,
    CallableObjectProtocol,
    CtypesArrayProtocol,
    DataHolderProtocol,
    DeviceHolderProtocol,
    DimT,
    DTypeHolderProtocol,
    ShapeDim,
    ShapeMetadataProtocol,
    ShapeProtocol,
    ShapeTuple,
    SliceProtocol,
    TracedDataProtocol,
    TypedVariadicCallable,
    UfuncProtocol,
    VariadicCallableProtocol,
    _IdMemberProtocol,
    _ItemMemberProtocol,
    _R_co,
    _SizeMemberProtocol,
    _TolistMemberProtocol,
)
from ml_switcheroo_compiler.core._shape_rules import (
    _broadcast_dim,
    _validate_strict_2d_matmul,
    _validate_vector_or_batched_matmul,
    broadcast_shapes,
    validate_matrix_multiply_shapes,
)
from ml_switcheroo_compiler.core._shape_types import (
    ArrayBufferProtocol,
    AxesLike,
    CallableLike,
    IRAttrValue,
    OpValue,
    Scalar,
    ScalarLike,
    ShapeLike,
    SliceType,
    SymbolicDim,
    T_Tensor,
    TensorOperand,
    _AttributeValuePydanticAnnotation,
    _IRAttrSelf,
    _OpValueSelf,
    _passthrough_identity,
    _PassthroughPydanticAnnotation,
    _PassValT,
)
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ir.shape_system import SymInt, SymNode, SymVar

if typing.TYPE_CHECKING:
    pass

__all__ = [
    "ArrayBufferProtocol",
    "AttributeValue",
    "AxesLike",
    "BackendBufferProtocol",
    "CallableLike",
    "CallableObjectProtocol",
    "CtypesArrayProtocol",
    "DType",
    "DTypeHolderProtocol",
    "DataHolderProtocol",
    "Device",
    "DeviceHolderProtocol",
    "DimT",
    "IRAttrValue",
    "IRGraph",
    "LinearOperatorOperandProtocol",
    "LogicalGraph",
    "LogicalNode",
    "OpValue",
    "RaggedOperandProtocol",
    "Scalar",
    "ScalarLike",
    "ShapeDim",
    "ShapeLike",
    "ShapeMetadataProtocol",
    "ShapeProtocol",
    "ShapeTuple",
    "SliceProtocol",
    "SliceType",
    "SparseOperandProtocol",
    "SymInt",
    "SymNode",
    "SymVar",
    "SymbolicDim",
    "T_Tensor",
    "TensorArrayOperandProtocol",
    "TensorOperand",
    "TensorOperandProtocol",
    "TensorShape",
    "TensorSpec",
    "TracedDataProtocol",
    "TypedVariadicCallable",
    "UfuncProtocol",
    "VariadicCallableProtocol",
    "_AttributeValuePydanticAnnotation",
    "_IRAttrSelf",
    "_IdMemberProtocol",
    "_ItemMemberProtocol",
    "_OpValueSelf",
    "_PassValT",
    "_PassthroughPydanticAnnotation",
    "_R_co",
    "_SizeMemberProtocol",
    "_TolistMemberProtocol",
    "_broadcast_dim",
    "_has_data_holder_attr",
    "_has_device_holder_attr",
    "_has_dtype_holder_attr",
    "_has_shape_attr",
    "_has_shape_metadata_attr",
    "_passthrough_identity",
    "_validate_strict_2d_matmul",
    "_validate_vector_or_batched_matmul",
    "broadcast_shapes",
    "extract_data",
    "extract_device",
    "extract_dtype",
    "extract_int_shape",
    "extract_shape",
    "extract_symbolic_shape",
    "is_backend_buffer",
    "is_traced_data",
    "is_variadic_callable",
    "validate_matrix_multiply_shapes",
]
