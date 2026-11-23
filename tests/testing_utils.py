"""Shared utilities and re-exported sound runtime helpers and fixtures for unit tests."""

from __future__ import annotations

import typing

import numpy as np
from ml_switcheroo_ir import AttributeValue as AttributeValue
from ml_switcheroo_ir import LogicalGraph as LogicalGraph
from ml_switcheroo_ir import LogicalNode as LogicalNode

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg as BackendArg,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArrayProtocol as BackendArrayProtocol,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    BackendOperand as BackendOperand,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    EagerBoundCallable as EagerBoundCallable,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    EagerCallableProtocol as EagerCallableProtocol,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    EagerTensorProtocol as EagerTensorProtocol,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    coerce_array as coerce_array,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    coerce_float as coerce_float,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    coerce_int as coerce_int,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    coerce_int_list as coerce_int_list,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    coerce_numeric as coerce_numeric,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    coerce_operand_list as coerce_operand_list,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    coerce_shape as coerce_shape,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    coerce_str as coerce_str,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    require_callable as require_callable,
)
from ml_switcheroo_compiler.backends.eager_registry import (
    unpack_pair as unpack_pair,
)
from ml_switcheroo_compiler.backends.protocols import (
    BackendModuleProtocol as BackendModuleProtocol,
)
from ml_switcheroo_compiler.backends.protocols import (
    StaticEagerBackendAdapter as StaticEagerBackendAdapter,
)
from ml_switcheroo_compiler.backends.protocols import (
    adapt_backend_module as adapt_backend_module,
)
from ml_switcheroo_compiler.core.tensor import (
    Tensor as Tensor,
)
from ml_switcheroo_compiler.core.tensor import (
    ensure_tensor as ensure_tensor,
)
from tests.fixtures.backend_stubs import (
    DummyDimSpecs as DummyDimSpecs,
)
from tests.fixtures.backend_stubs import (
    MockGraphStub as MockGraphStub,
)
from tests.fixtures.backend_stubs import (
    MockNodeStub as MockNodeStub,
)
from tests.fixtures.backend_stubs import (
    MockTensorStub as MockTensorStub,
)
from tests.fixtures.backend_stubs import (
    MockTestStub as MockTestStub,
)
from tests.fixtures.backend_stubs import (
    _MockJaxArray as _MockJaxArray,
)
from tests.fixtures.backend_stubs import (
    _MockNNLinear as _MockNNLinear,
)
from tests.fixtures.backend_stubs import (
    _MockNNModule as _MockNNModule,
)
from tests.fixtures.backend_stubs import (
    _MockTorchDevice as _MockTorchDevice,
)
from tests.fixtures.backend_stubs import (
    _MockTorchTensor as _MockTorchTensor,
)
from tests.fixtures.backend_stubs import (
    extract_test_data as extract_test_data,
)
from tests.fixtures.backend_stubs import (
    extract_test_dtype as extract_test_dtype,
)
from tests.fixtures.backend_stubs import (
    extract_test_shape as extract_test_shape,
)
from tests.fixtures.backend_stubs import (
    make_mock_module as make_mock_module,
)
from tests.fixtures.ir_builders import (
    build_test_graph as build_test_graph,
)
from tests.fixtures.ir_builders import (
    build_test_node as build_test_node,
)
from tests.fixtures.profiler_stubs import (
    _has_working_mlx as _has_working_mlx,
)
from tests.fixtures.profiler_stubs import (
    _install_jax_stub as _install_jax_stub,
)
from tests.fixtures.profiler_stubs import (
    _install_mlx_and_sparse_stubs as _install_mlx_and_sparse_stubs,
)
from tests.fixtures.profiler_stubs import (
    _install_torch_stub as _install_torch_stub,
)
from tests.fixtures.profiler_stubs import (
    install_profiler_backend_stubs as install_profiler_backend_stubs,
)

NumpyTestArray: typing.TypeAlias = np.ndarray[
    tuple[int, ...],
    typing.Union[  # noqa: UP007
        np.dtype[np.float64],
        np.dtype[np.float32],
        np.dtype[np.int64],
        np.dtype[np.int32],
        np.dtype[np.uint8],
    ],
]

GenericNDArray: typing.TypeAlias = np.ndarray[
    tuple[int, ...],
    typing.Union[  # noqa: UP007
        np.dtype[np.float16],
        np.dtype[np.float32],
        np.dtype[np.float64],
        np.dtype[np.int8],
        np.dtype[np.int16],
        np.dtype[np.int32],
        np.dtype[np.int64],
        np.dtype[np.uint8],
        np.dtype[np.uint16],
        np.dtype[np.uint32],
        np.dtype[np.uint64],
        np.dtype[np.complex64],
        np.dtype[np.complex128],
        np.dtype[np.bool_],
        np.dtype[np.str_],
        np.dtype[np.bytes_],
    ],
]

__all__ = [
    "AttributeValue",
    "BackendArg",
    "BackendArrayProtocol",
    "BackendModuleProtocol",
    "BackendOperand",
    "DummyDimSpecs",
    "EagerBoundCallable",
    "EagerCallableProtocol",
    "EagerTensorProtocol",
    "GenericNDArray",
    "LogicalGraph",
    "LogicalNode",
    "MockGraphStub",
    "MockNodeStub",
    "MockTensorStub",
    "MockTestStub",
    "NumpyTestArray",
    "StaticEagerBackendAdapter",
    "Tensor",
    "adapt_backend_module",
    "build_test_graph",
    "build_test_node",
    "coerce_array",
    "coerce_float",
    "coerce_int",
    "coerce_int_list",
    "coerce_numeric",
    "coerce_operand_list",
    "coerce_shape",
    "coerce_str",
    "ensure_tensor",
    "extract_test_data",
    "extract_test_dtype",
    "extract_test_shape",
    "install_profiler_backend_stubs",
    "make_mock_module",
    "np",
    "require_callable",
    "unpack_pair",
]
