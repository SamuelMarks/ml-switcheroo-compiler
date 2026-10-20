"""Tests for test_aot."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.export.aot import compile_function

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_aot() -> None:
    """Test function test_aot."""

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.
        """
        return typing.cast("EagerTensorProtocol", x) + typing.cast("EagerTensorProtocol", 1)

    c = compile_function(f)
    assert typing.cast("EagerCallableProtocol", c)(1) == 2


def test_aot_tensor_caching() -> None:
    """Test function test_aot_tensor_caching."""

    def f(x: EagerValue, y: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.
            y: Test parameter `y`.
        """
        return typing.cast("EagerTensorProtocol", x) + typing.cast("EagerTensorProtocol", y)

    c = compile_function(f, backend="numpy")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    t2 = Tensor(np.array([2.0]), TensorConfig((1,), DType.Float32, "cpu"))

    # First call will trace and compile
    res = typing.cast("EagerCallableProtocol", c)(t1, t2)
    assert res is not None  # Actually, returning evaluated tensor

    # Second call uses cache
    res2 = typing.cast("EagerCallableProtocol", c)(t1, t2)
    assert res2 is not None


def test_aot_list_output() -> None:
    """Test function test_aot_list_output."""

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.
        """
        return [x, typing.cast("EagerTensorProtocol", x) + typing.cast("EagerTensorProtocol", x)]

    c = compile_function(f, backend="numpy")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))

    res = typing.cast("EagerCallableProtocol", c)(t1)
    assert res is not None


def test_aot_backend_missing_generator() -> None:
    """Test function test_aot_backend_missing_generator."""

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.
        """
        return x

    # backend doesn't exist
    c = compile_function(f, backend="invalid_backend")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    # it should just fallback
    res = typing.cast("EagerCallableProtocol", c)(t1)
    assert res is not None


def test_aot_compile_aot_hasattr() -> None:
    """Test function test_aot_compile_aot_hasattr."""
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    class MockBackend:
        """Test class MockBackend."""

        @staticmethod
        def compile_aot(graph: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function compile_aot.

            Args:
                graph: Test parameter `graph`.
                kwargs: Test parameter `kwargs`.
            """
            return lambda *args, **kw: "mock_aot"

    typing.cast("EagerCallableProtocol", BackendRegistry.register)("mock_aot_backend", MockBackend)

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.
        """
        return typing.cast("EagerTensorProtocol", x) + typing.cast("EagerTensorProtocol", x)

    c = compile_function(f, backend="mock_aot_backend")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    res = typing.cast("EagerCallableProtocol", c)(t1)
    assert res == "mock_aot"


def test_aot_apply_model() -> None:
    """Test function test_aot_apply_model."""
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    class MockBackendApply:
        """Test class MockBackendApply."""

        def __init__(self, graph: EagerValue) -> None:
            """Test function __init__.

            Args:
                graph: Test parameter `graph`.
            """
            self.graph = graph

        def generate(self) -> EagerValue:
            """Test function generate."""
            return "def apply_model(params, *args, **kwargs):\n    return 'apply_model'"

        @classmethod
        def get_generator(cls) -> EagerValue:
            """Test function get_generator."""
            return cls

    typing.cast("EagerCallableProtocol", BackendRegistry.register)("mock_apply", MockBackendApply)

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.
        """
        return x

    c = compile_function(f, backend="mock_apply")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    assert typing.cast("EagerCallableProtocol", c)(t1) == "apply_model"


def test_aot_execution_failure() -> None:
    """Test function test_aot_execution_failure."""
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    class MockBackendFail:
        """Test class MockBackendFail."""

        def __init__(self, graph: EagerValue) -> None:
            """Test function __init__.

            Args:
                graph: Test parameter `graph`.
            """
            self.graph = graph

        def generate(self) -> EagerValue:
            """Test function generate."""
            return "def apply_model(params, *args, **kwargs):\n    raise ValueError('fail')"

        @classmethod
        def get_generator(cls) -> EagerValue:
            """Test function get_generator."""
            return cls

    typing.cast("EagerCallableProtocol", BackendRegistry.register)("mock_fail", MockBackendFail)

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.
        """
        return "fallback"

    c = compile_function(f, backend="mock_fail")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    assert typing.cast("EagerCallableProtocol", c)(t1) == "fallback"


def test_aot_caching_success() -> None:
    """Test function test_aot_caching_success."""
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    class MockBackendSuccess:
        """Test class MockBackendSuccess."""

        @staticmethod
        def compile_aot(graph: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function compile_aot.

            Args:
                graph: Test parameter `graph`.
                kwargs: Test parameter `kwargs`.
            """
            return lambda *args, **kw: "success_cache"

    typing.cast("EagerCallableProtocol", BackendRegistry.register)("mock_cache", MockBackendSuccess)

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.
        """
        return x

    c = compile_function(f, backend="mock_cache")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    assert typing.cast("EagerCallableProtocol", c)(t1) == "success_cache"
    assert typing.cast("EagerCallableProtocol", c)(t1) == "success_cache"  # Hits cache


def test_aot_list_output_no_tensors() -> None:
    """Test function test_aot_list_output_no_tensors."""

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.
        """
        return [1, 2, 3]

    c = compile_function(f, backend="numpy")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    res = typing.cast("EagerCallableProtocol", c)(t1)
    assert res == [1, 2, 3]


def test_aot_tracing_exception() -> None:
    """Test function test_aot_tracing_exception."""

    def f(x: EagerValue) -> None:
        """Test function f.

        Args:
            x: Test parameter `x`.
        """
        raise ValueError("Trace error")

    c = compile_function(f, backend="numpy")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    # Should fallback and raise ValueError because fn raises it
    import pytest

    with pytest.raises(ValueError):
        typing.cast("EagerCallableProtocol", c)(t1)


def test_aot_fallback() -> None:
    """Test function test_aot_fallback."""
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    class MockBackendNoModule:
        """Test class MockBackendNoModule."""

        def __init__(self, graph: EagerValue) -> None:
            """Test function __init__.

            Args:
                graph: Test parameter `graph`.
            """
            self.graph = graph

        def generate(self) -> EagerValue:
            """Test function generate."""
            return "def apply_model(params, *args, **kwargs):\n    return 'no_mod'"

        @classmethod
        def get_generator(cls) -> EagerValue:
            """Test function get_generator."""
            return cls

    typing.cast("EagerCallableProtocol", BackendRegistry.register)("mock_no_mod", MockBackendNoModule)

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.
        """
        return x

    c = compile_function(f, backend="mock_no_mod")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    assert typing.cast("EagerCallableProtocol", c)(t1) == "no_mod"


def test_aot_evaluate_wrapper() -> None:
    """Test function test_aot_evaluate_wrapper."""
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    class MockBackendEvaluate:
        """Test class MockBackendEvaluate."""

        def __init__(self, graph: EagerValue) -> None:
            """Test function __init__.

            Args:
                graph: Test parameter `graph`.
            """
            self.graph = graph

        def generate(self) -> EagerValue:
            """Test function generate."""
            return "def evaluate(args):\n    return 'evaluate_success'"

        @classmethod
        def get_generator(cls) -> EagerValue:
            """Test function get_generator."""
            return cls

        @classmethod
        def get_module(cls) -> EagerValue:
            """Test function get_module."""
            return "mock_module"

    typing.cast("EagerCallableProtocol", BackendRegistry.register)("mock_eval", MockBackendEvaluate)

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.
        """
        return x

    c = compile_function(f, backend="mock_eval")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    assert typing.cast("EagerCallableProtocol", c)(t1) == "evaluate_success"


def test_aot_no_func_wrapper() -> None:
    """Test function test_aot_no_func_wrapper."""
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    class MockBackendEmpty:
        """Test class MockBackendEmpty."""

        def __init__(self, graph: EagerValue) -> None:
            """Test function __init__.

            Args:
                graph: Test parameter `graph`.
            """
            self.graph = graph

        def generate(self) -> EagerValue:
            """Test function generate."""
            return "x = 1"

        @classmethod
        def get_generator(cls) -> EagerValue:
            """Test function get_generator."""
            return cls

    typing.cast("EagerCallableProtocol", BackendRegistry.register)("mock_empty", MockBackendEmpty)

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.
        """
        return "fallback_empty"

    c = compile_function(f, backend="mock_empty")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    assert typing.cast("EagerCallableProtocol", c)(t1) == "fallback_empty"


def test_aot_custom_dtype_string_prefix() -> None:
    """Test _prepare_proxy_args with an input object whose dtype is a string starting with DType."""

    class CustomArray:
        """Test class CustomArray."""

        def __init__(self, dt: str) -> None:
            """Test function __init__.

            Args:
                dt: Test parameter `dt`.
            """
            self.shape = (2, 3)
            self.dtype = dt

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.
        """
        return x

    c = compile_function(f, backend="numpy")
    res1 = typing.cast("EagerCallableProtocol", c)(typing.cast("EagerCallableProtocol", CustomArray)("DType.float32"))
    assert res1 is not None

    res2 = typing.cast("EagerCallableProtocol", c)(typing.cast("EagerCallableProtocol", CustomArray)("float32"))
    assert res2 is not None

    res3 = typing.cast("EagerCallableProtocol", c)(typing.cast("EagerCallableProtocol", CustomArray)("unknown_custom_dtype"))
    assert res3 is not None


def test_aot_no_func_wrapper_tensor() -> None:
    """Test function test_aot_no_func_wrapper_tensor."""
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    class MockBackendEmptyTensor:
        """Test class MockBackendEmptyTensor."""

        def __init__(self, graph: EagerValue) -> None:
            """Test function __init__.

            Args:
                graph: Test parameter `graph`.
            """
            self.graph = graph

        def generate(self) -> EagerValue:
            """Test function generate."""
            return "x = 1"

        @classmethod
        def get_generator(cls) -> EagerValue:
            """Test function get_generator."""
            return cls

    typing.cast("EagerCallableProtocol", BackendRegistry.register)("mock_empty_tensor", MockBackendEmptyTensor)

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.
        """
        return x

    c = compile_function(f, backend="mock_empty_tensor")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    # Since f(x) returns x (a Tensor), graph.outputs will be populated!
    # But generator returns "x = 1", so apply_model and evaluate are missing!
    res = typing.cast("EagerCallableProtocol", c)(t1)
    assert id(res) == id(t1)


def _test_aot_all_remaining_branches_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_aot_all_remaining_branches.

    Args:
        _state: Shared test state dictionary.
    """
    import pytest

    from ml_switcheroo_compiler.backends.registry import BackendRegistry
    from ml_switcheroo_compiler.core.errors import CompilationError
    from ml_switcheroo_compiler.export.aot import _get_namespace

    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))

    def simple_identity(x: EagerValue) -> EagerValue:
        """Test function simple_identity.

        Args:
            x: Test parameter `x`.
        """
        return x

    # 1. _get_namespace fallback for numpy (lines 130-132)
    class GeneratorNoGetModule:
        """Test class GeneratorNoGetModule."""

        pass

    ns = _get_namespace("numpy", GeneratorNoGetModule)
    assert typing.cast("EagerTensorProtocol", "numpy") in typing.cast("EagerTensorProtocol", ns)
    getattr(_state, "update")({"BackendRegistry": BackendRegistry, "CompilationError": CompilationError, "GeneratorNoGetModule": GeneratorNoGetModule, "_get_namespace": _get_namespace})
    getattr(_state, "update")({"ns": ns, "pytest": pytest, "simple_identity": simple_identity, "t1": t1})
    _state.update(locals())


def _test_aot_all_remaining_branches_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_aot_all_remaining_branches.

    Args:
        _state: Shared test state dictionary.
    """
    BackendRegistry, simple_identity, t1 = _state["BackendRegistry"], _state["simple_identity"], _state["t1"]

    class BackendStaticNonCallable:
        """Test class BackendStaticNonCallable."""

        @staticmethod
        def compile_aot(graph: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function compile_aot.

            Args:
                graph: Test parameter `graph`.
                kwargs: Test parameter `kwargs`.
            """
            return "not_a_callable"

        def __init__(self, graph: EagerValue) -> None:
            """Test function __init__.

            Args:
                graph: Test parameter `graph`.
            """
            pass

        def generate(self) -> EagerValue:
            """Test function generate."""
            return "def apply_model(x): return x"

    BackendRegistry.register("backend_static_non_callable", BackendStaticNonCallable)
    c_non_call = compile_function(simple_identity, backend="backend_static_non_callable")
    assert typing.cast("EagerCallableProtocol", c_non_call)(t1) is not None
    getattr(_state, "update")({"BackendStaticNonCallable": BackendStaticNonCallable, "c_non_call": c_non_call})
    _state.update(locals())


def _test_aot_all_remaining_branches_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_aot_all_remaining_branches.

    Args:
        _state: Shared test state dictionary.
    """
    BackendRegistry, CompilationError, pytest, simple_identity = _state["BackendRegistry"], _state["CompilationError"], _state["pytest"], _state["simple_identity"]
    t1 = _state["t1"]

    class BackendStaticRaises:
        """Test class BackendStaticRaises."""

        @staticmethod
        def compile_aot(graph: EagerValue, **kwargs: EagerValue) -> None:
            """Test function compile_aot.

            Args:
                graph: Test parameter `graph`.
                kwargs: Test parameter `kwargs`.
            """
            raise RuntimeError("static aot error")

        def __init__(self, graph: EagerValue) -> None:
            """Test function __init__.

            Args:
                graph: Test parameter `graph`.
            """
            pass

        def generate(self) -> EagerValue:
            """Test function generate."""
            return "def apply_model(x): return x"

    BackendRegistry.register("backend_static_raises", BackendStaticRaises)
    c_raises = compile_function(simple_identity, backend="backend_static_raises")
    assert typing.cast("EagerCallableProtocol", c_raises)(t1) is not None

    # 3. generator_cls(graph) fails with strict=True and strict=False (lines 221-226)
    class BackendInitFails:
        """Test class BackendInitFails."""

        def __init__(self, graph: EagerValue) -> None:
            """Test function __init__.

            Args:
                graph: Test parameter `graph`.
            """
            raise RuntimeError("init failure")

    BackendRegistry.register("backend_init_fails", BackendInitFails)
    with pytest.raises(CompilationError, match="Failed to instantiate generator"):
        typing.cast("EagerCallableProtocol", compile_function)(simple_identity, backend="backend_init_fails", strict=True)(t1)
    getattr(_state, "update")({"BackendInitFails": BackendInitFails, "BackendStaticRaises": BackendStaticRaises, "c_raises": c_raises})
    _state.update(locals())


def _test_aot_all_remaining_branches_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_aot_all_remaining_branches.

    Args:
        _state: Shared test state dictionary.
    """
    BackendRegistry, simple_identity, t1 = _state["BackendRegistry"], _state["simple_identity"], _state["t1"]
    c_fallback_init = compile_function(simple_identity, backend="backend_init_fails", strict=False)
    assert typing.cast("EagerCallableProtocol", c_fallback_init)(t1) is t1

    # 4. Instance compile_aot hook (lines 232-242)
    class BackendInstanceAot:
        """Test class BackendInstanceAot."""

        def __init__(self, graph: EagerValue) -> None:
            """Test function __init__.

            Args:
                graph: Test parameter `graph`.
            """
            self.mode = "non_callable"

        def compile_aot(self, graph: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function compile_aot.

            Args:
                graph: Test parameter `graph`.
                kwargs: Test parameter `kwargs`.
            """
            if self.mode == "non_callable":
                return 123
            elif self.mode == "not_implemented":
                raise NotImplementedError()
            else:
                raise RuntimeError("instance aot fail")

        def generate(self) -> EagerValue:
            """Test function generate."""
            return "def apply_model(x): return x"

    BackendRegistry.register("backend_instance_aot", BackendInstanceAot)
    # 232->244: returns non-callable
    c_inst_noncall = compile_function(simple_identity, backend="backend_instance_aot")
    assert typing.cast("EagerCallableProtocol", c_inst_noncall)(t1) is not None
    getattr(_state, "update")({"BackendInstanceAot": BackendInstanceAot, "c_fallback_init": c_fallback_init, "c_inst_noncall": c_inst_noncall})
    _state.update(locals())


def _test_aot_all_remaining_branches_part_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 5 of test_aot_all_remaining_branches.

    Args:
        _state: Shared test state dictionary.
    """
    BackendInstanceAot_cls, BackendRegistry, CompilationError, pytest = _state["BackendInstanceAot"], _state["BackendRegistry"], _state["CompilationError"], _state["pytest"]
    from ml_switcheroo_compiler.backends.base_generator import BaseGenerator, CompiledArtifact

    simple_identity, t1 = _state["simple_identity"], _state["t1"]

    class BackendInstNotImpl(BaseGenerator):
        """Test class BackendInstNotImpl."""

        def __init__(self, graph: EagerValue) -> None:
            """Test function __init__.

            Args:
                graph: Test parameter `graph`.
            """
            typing.cast("EagerCallableProtocol", super().__init__)(graph)
            self.mode = "not_implemented"

        def compile_aot(self, *args: EagerValue, **kwargs: EagerValue) -> CompiledArtifact:
            """Test function compile_aot.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return typing.cast("CompiledArtifact", typing.cast("EagerCallableProtocol", getattr(BackendInstanceAot_cls, "compile_aot"))(self, *args, **kwargs))

    BackendRegistry.register("backend_inst_not_impl", BackendInstNotImpl)
    c_inst_notimpl = compile_function(simple_identity, backend="backend_inst_not_impl")
    assert typing.cast("EagerCallableProtocol", c_inst_notimpl)(t1) is not None

    # 238-242: Exception with strict=True and False
    class BackendInstError(BaseGenerator):
        """Test class BackendInstError."""

        def __init__(self, graph: EagerValue) -> None:
            """Test function __init__.

            Args:
                graph: Test parameter `graph`.
            """
            typing.cast("EagerCallableProtocol", super().__init__)(graph)
            self.mode = "error"

        def compile_aot(self, *args: EagerValue, **kwargs: EagerValue) -> CompiledArtifact:
            """Test function compile_aot.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return typing.cast("CompiledArtifact", typing.cast("EagerCallableProtocol", getattr(BackendInstanceAot_cls, "compile_aot"))(self, *args, **kwargs))

    BackendRegistry.register("backend_inst_error", BackendInstError)
    with pytest.raises(CompilationError, match="AOT compilation failed"):
        typing.cast("EagerCallableProtocol", compile_function)(simple_identity, backend="backend_inst_error", strict=True)(t1)

    c_inst_err_fallback = compile_function(simple_identity, backend="backend_inst_error", strict=False)
    assert typing.cast("EagerCallableProtocol", c_inst_err_fallback)(t1) is t1
    getattr(_state, "update")({"BackendInstError": BackendInstError, "BackendInstNotImpl": BackendInstNotImpl, "c_inst_err_fallback": c_inst_err_fallback, "c_inst_notimpl": c_inst_notimpl})
    _state.update(locals())


def _test_aot_all_remaining_branches_part_6(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 6 of test_aot_all_remaining_branches.

    Args:
        _state: Shared test state dictionary.
    """
    BackendRegistry, CompilationError, pytest, simple_identity = _state["BackendRegistry"], _state["CompilationError"], _state["pytest"], _state["simple_identity"]
    t1 = _state["t1"]

    class BackendNoEntrypoint:
        """Test class BackendNoEntrypoint."""

        def __init__(self, graph: EagerValue) -> None:
            """Test function __init__.

            Args:
                graph: Test parameter `graph`.
            """
            pass

        def generate(self) -> EagerValue:
            """Test function generate."""
            return "some_var = 1"

    BackendRegistry.register("backend_no_entrypoint", BackendNoEntrypoint)
    with pytest.raises(CompilationError, match="No executable entrypoint found"):
        typing.cast("EagerCallableProtocol", compile_function)(simple_identity, backend="backend_no_entrypoint", strict=True)(t1)

    class BackendExecFails:
        """Test class BackendExecFails."""

        def __init__(self, graph: EagerValue) -> None:
            """Test function __init__.

            Args:
                graph: Test parameter `graph`.
            """
            pass

        def generate(self) -> EagerValue:
            """Test function generate."""
            return "def apply_model(x):\n    raise RuntimeError('runtime execution fail')"

    BackendRegistry.register("backend_exec_fails", BackendExecFails)
    with pytest.raises(CompilationError, match="Execution of compiled code failed"):
        typing.cast("EagerCallableProtocol", compile_function)(simple_identity, backend="backend_exec_fails", strict=True)(t1)
    getattr(_state, "update")({"BackendExecFails": BackendExecFails, "BackendNoEntrypoint": BackendNoEntrypoint})
    _state.update(locals())


def _test_aot_all_remaining_branches_part_7(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 7 of test_aot_all_remaining_branches.

    Args:
        _state: Shared test state dictionary.
    """
    simple_identity, t1 = _state["simple_identity"], _state["t1"]
    c_exec_fallback = compile_function(simple_identity, backend="backend_exec_fails", strict=False)
    assert typing.cast("EagerCallableProtocol", c_exec_fallback)(t1) is t1
    getattr(_state, "update")({"c_exec_fallback": c_exec_fallback})


def test_aot_all_remaining_branches() -> None:
    """Test lines 130-132, 213-217, 221-226, 232-242, 280-282, and 288-293 in export/aot.py."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_aot_all_remaining_branches_part_1(_state)
    _test_aot_all_remaining_branches_part_2(_state)
    _test_aot_all_remaining_branches_part_3(_state)
    _test_aot_all_remaining_branches_part_4(_state)
    _test_aot_all_remaining_branches_part_5(_state)
    _test_aot_all_remaining_branches_part_6(_state)
    _test_aot_all_remaining_branches_part_7(_state)
    assert len(_state) > 0
