"""Shared typed fixture builders and backend test doubles."""

from __future__ import annotations

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
from tests.fixtures.ir_builders import (
    build_test_graph as build_test_graph,
)
from tests.fixtures.ir_builders import (
    build_test_node as build_test_node,
)
from tests.fixtures.profiler_stubs import install_profiler_backend_stubs as install_profiler_backend_stubs

__all__ = [
    "_MockJaxArray",
    "_MockNNLinear",
    "_MockNNModule",
    "_MockTorchDevice",
    "_MockTorchTensor",
    "build_test_graph",
    "build_test_node",
    "install_profiler_backend_stubs",
]
