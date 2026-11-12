"""Tests for test_aot."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.export.aot import compile_function
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_tensor_dict

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_aot() -> None:
    """Test function test_aot."""

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.

        Returns:
            EagerValue: Computed output value from `f`.
        """
        return ensure_eager_tensor(x) + ensure_eager_tensor(1)

    c = compile_function(f)
    assert ensure_callable(c)(1) == 2


def test_aot_tensor_caching() -> None:
    """Test function test_aot_tensor_caching."""

    def f(x: EagerValue, y: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.
            y: Test parameter `y`.

        Returns:
            EagerValue: Computed output value from `f`.
        """
        return ensure_eager_tensor(x) + ensure_eager_tensor(y)

    c = compile_function(f, backend="numpy")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    t2 = Tensor(np.array([2.0]), TensorConfig((1,), DType.Float32, "cpu"))

    # First call will trace and compile
    res = ensure_callable(c)(t1, t2)
    np.testing.assert_allclose(np.asarray(ensure_eager_tensor(res)), [3.0])

    # Second call uses cache
    res2 = ensure_callable(c)(t1, t2)
    np.testing.assert_allclose(np.asarray(ensure_eager_tensor(res2)), [3.0])


def test_aot_list_output() -> None:
    """Test function test_aot_list_output."""

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.

        Returns:
            EagerValue: Computed output value from `f`.
        """
        return [x, ensure_eager_tensor(x) + ensure_eager_tensor(x)]

    c = compile_function(f, backend="numpy")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))

    res = ensure_callable(c)(t1)
    np.testing.assert_allclose(np.asarray(ensure_eager_tensor(res)), [[1.0], [2.0]])


def test_aot_backend_missing_generator() -> None:
    """Test function test_aot_backend_missing_generator."""

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.

        Returns:
            EagerValue: Computed output value from `f`.
        """
        return x

    # backend doesn't exist
    c = compile_function(f, backend="invalid_backend")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    # it should just fallback
    res = ensure_callable(c)(t1)
    assert id(res) == id(t1)


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

            Returns:
                EagerValue: Computed output value from `compile_aot`.
            """
            return lambda *args, **kw: "mock_aot"

    ensure_callable(BackendRegistry.register)("mock_aot_backend", MockBackend)

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.

        Returns:
            EagerValue: Computed output value from `f`.
        """
        return ensure_eager_tensor(x) + ensure_eager_tensor(x)

    c = compile_function(f, backend="mock_aot_backend")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    res = ensure_callable(c)(t1)
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
            """Test function generate.

            Returns:
                EagerValue: Computed output value from `generate`.
            """
            return "def apply_model(params, *args, **kwargs):\n    return 'apply_model'"

        @classmethod
        def get_generator(cls) -> EagerValue:
            """Test function get_generator.

            Returns:
                EagerValue: Computed output value from `get_generator`.
            """
            return cls

    ensure_callable(BackendRegistry.register)("mock_apply", MockBackendApply)

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.

        Returns:
            EagerValue: Computed output value from `f`.
        """
        return x

    c = compile_function(f, backend="mock_apply")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    assert ensure_callable(c)(t1) == "apply_model"


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
            """Test function generate.

            Returns:
                EagerValue: Computed output value from `generate`.
            """
            return "def apply_model(params, *args, **kwargs):\n    raise ValueError('fail')"

        @classmethod
        def get_generator(cls) -> EagerValue:
            """Test function get_generator.

            Returns:
                EagerValue: Computed output value from `get_generator`.
            """
            return cls

    ensure_callable(BackendRegistry.register)("mock_fail", MockBackendFail)

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.

        Returns:
            EagerValue: Computed output value from `f`.
        """
        return "fallback"

    c = compile_function(f, backend="mock_fail")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    assert ensure_callable(c)(t1) == "fallback"


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

            Returns:
                EagerValue: Computed output value from `compile_aot`.
            """
            return lambda *args, **kw: "success_cache"

    ensure_callable(BackendRegistry.register)("mock_cache", MockBackendSuccess)

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.

        Returns:
            EagerValue: Computed output value from `f`.
        """
        return x

    c = compile_function(f, backend="mock_cache")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    assert ensure_callable(c)(t1) == "success_cache"
    assert ensure_callable(c)(t1) == "success_cache"  # Hits cache


def test_aot_list_output_no_tensors() -> None:
    """Test function test_aot_list_output_no_tensors."""

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.

        Returns:
            EagerValue: Computed output value from `f`.
        """
        return [1, 2, 3]

    c = compile_function(f, backend="numpy")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    res = ensure_callable(c)(t1)
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
        ensure_callable(c)(t1)


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
            """Test function generate.

            Returns:
                EagerValue: Computed output value from `generate`.
            """
            return "def apply_model(params, *args, **kwargs):\n    return 'no_mod'"

        @classmethod
        def get_generator(cls) -> EagerValue:
            """Test function get_generator.

            Returns:
                EagerValue: Computed output value from `get_generator`.
            """
            return cls

    ensure_callable(BackendRegistry.register)("mock_no_mod", MockBackendNoModule)

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.

        Returns:
            EagerValue: Computed output value from `f`.
        """
        return x

    c = compile_function(f, backend="mock_no_mod")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    assert ensure_callable(c)(t1) == "no_mod"


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
            """Test function generate.

            Returns:
                EagerValue: Computed output value from `generate`.
            """
            return "def evaluate(args):\n    return 'evaluate_success'"

        @classmethod
        def get_generator(cls) -> EagerValue:
            """Test function get_generator.

            Returns:
                EagerValue: Computed output value from `get_generator`.
            """
            return cls

        @classmethod
        def get_module(cls) -> EagerValue:
            """Test function get_module.

            Returns:
                EagerValue: Computed output value from `get_module`.
            """
            return "mock_module"

    ensure_callable(BackendRegistry.register)("mock_eval", MockBackendEvaluate)

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.

        Returns:
            EagerValue: Computed output value from `f`.
        """
        return x

    c = compile_function(f, backend="mock_eval")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    assert ensure_callable(c)(t1) == "evaluate_success"


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
            """Test function generate.

            Returns:
                EagerValue: Computed output value from `generate`.
            """
            return "x = 1"

        @classmethod
        def get_generator(cls) -> EagerValue:
            """Test function get_generator.

            Returns:
                EagerValue: Computed output value from `get_generator`.
            """
            return cls

    ensure_callable(BackendRegistry.register)("mock_empty", MockBackendEmpty)

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.

        Returns:
            EagerValue: Computed output value from `f`.
        """
        return "fallback_empty"

    c = compile_function(f, backend="mock_empty")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    assert ensure_callable(c)(t1) == "fallback_empty"


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

        Returns:
            EagerValue: Computed output value from `f`.
        """
        return x

    c = compile_function(f, backend="numpy")
    ca1 = ensure_callable(CustomArray)("DType.float32")
    res1 = ensure_callable(c)(ca1)
    assert ensure_eager_tensor(res1).shape == (2, 3)

    ca2 = ensure_callable(CustomArray)("float32")
    res2 = ensure_callable(c)(ca2)
    assert ensure_eager_tensor(res2).shape == (2, 3)

    ca3 = ensure_callable(CustomArray)("unknown_custom_dtype")
    res3 = ensure_callable(c)(ca3)
    assert ensure_eager_tensor(res3).shape == (2, 3)


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
            """Test function generate.

            Returns:
                EagerValue: Computed output value from `generate`.
            """
            return "x = 1"

        @classmethod
        def get_generator(cls) -> EagerValue:
            """Test function get_generator.

            Returns:
                EagerValue: Computed output value from `get_generator`.
            """
            return cls

    ensure_callable(BackendRegistry.register)("mock_empty_tensor", MockBackendEmptyTensor)

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.

        Returns:
            EagerValue: Computed output value from `f`.
        """
        return x

    c = compile_function(f, backend="mock_empty_tensor")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    # Since f(x) returns x (a Tensor), graph.outputs will be populated!
    # But generator returns "x = 1", so apply_model and evaluate are missing!
    res = ensure_callable(c)(t1)
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

        Returns:
            EagerValue: Computed output value from `simple_identity`.
        """
        return x

    # 1. _get_namespace fallback for numpy (lines 130-132)
    class GeneratorNoGetModule:
        """Test class GeneratorNoGetModule."""

        pass

    ns = _get_namespace("numpy", GeneratorNoGetModule)
    assert ensure_eager_tensor("numpy") in ensure_eager_tensor(ns)
    _state.update(ensure_tensor_dict({"BackendRegistry": BackendRegistry, "CompilationError": CompilationError, "GeneratorNoGetModule": GeneratorNoGetModule, "_get_namespace": _get_namespace}))
    _state.update(ensure_tensor_dict({"ns": ns, "pytest": pytest, "simple_identity": simple_identity, "t1": t1}))


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

            Returns:
                EagerValue: Computed output value from `compile_aot`.
            """
            return "not_a_callable"

        def __init__(self, graph: EagerValue) -> None:
            """Test function __init__.

            Args:
                graph: Test parameter `graph`.
            """
            pass

        def generate(self) -> EagerValue:
            """Test function generate.

            Returns:
                EagerValue: Computed output value from `generate`.
            """
            return "def apply_model(x): return x"

    BackendRegistry.register("backend_static_non_callable", BackendStaticNonCallable)
    c_non_call = compile_function(simple_identity, backend="backend_static_non_callable")
    assert ensure_callable(c_non_call)(t1) is not None
    _state.update(ensure_tensor_dict({"BackendStaticNonCallable": BackendStaticNonCallable, "c_non_call": c_non_call}))
    _state.update(ensure_tensor_dict({"BackendRegistry": BackendRegistry, "simple_identity": simple_identity, "t1": t1}))


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
            """Test function generate.

            Returns:
                EagerValue: Computed output value from `generate`.
            """
            return "def apply_model(x): return x"

    BackendRegistry.register("backend_static_raises", BackendStaticRaises)
    c_raises = compile_function(simple_identity, backend="backend_static_raises")
    assert ensure_callable(c_raises)(t1) is not None

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
        ensure_callable(compile_function)(simple_identity, backend="backend_init_fails", strict=True)(t1)
    _state.update(ensure_tensor_dict({"BackendInitFails": BackendInitFails, "BackendStaticRaises": BackendStaticRaises, "c_raises": c_raises}))
    _state.update(ensure_tensor_dict({"BackendRegistry": BackendRegistry, "CompilationError": CompilationError, "pytest": pytest}))
    _state.update(ensure_tensor_dict({"simple_identity": simple_identity, "t1": t1}))


def _test_aot_all_remaining_branches_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_aot_all_remaining_branches.

    Args:
        _state: Shared test state dictionary.
    """
    BackendRegistry, simple_identity, t1 = _state["BackendRegistry"], _state["simple_identity"], _state["t1"]
    c_fallback_init = compile_function(simple_identity, backend="backend_init_fails", strict=False)
    assert ensure_callable(c_fallback_init)(t1) is t1

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

            Returns:
                EagerValue: Computed output value from `compile_aot`.
            """
            if self.mode == "non_callable":
                return 123
            elif self.mode == "not_implemented":
                raise NotImplementedError()
            else:
                raise RuntimeError("instance aot fail")

        def generate(self) -> EagerValue:
            """Test function generate.

            Returns:
                EagerValue: Computed output value from `generate`.
            """
            return "def apply_model(x): return x"

    BackendRegistry.register("backend_instance_aot", BackendInstanceAot)
    # 232->244: returns non-callable
    c_inst_noncall = compile_function(simple_identity, backend="backend_instance_aot")
    assert ensure_callable(c_inst_noncall)(t1) is not None
    _state.update(ensure_tensor_dict({"BackendInstanceAot": BackendInstanceAot, "c_fallback_init": c_fallback_init, "c_inst_noncall": c_inst_noncall}))
    _state.update(ensure_tensor_dict({"BackendRegistry": BackendRegistry, "simple_identity": simple_identity, "t1": t1}))


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
            ensure_callable(super().__init__)(graph)
            self.mode = "not_implemented"

        def compile_aot(self, *args: EagerValue, **kwargs: EagerValue) -> CompiledArtifact:
            """Test function compile_aot.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                CompiledArtifact: Computed output value from `compile_aot`.
            """
            ensure_callable(ensure_eager_tensor(BackendInstanceAot_cls).compile_aot)(self, *args, **kwargs)
            return CompiledArtifact()

    BackendRegistry.register("backend_inst_not_impl", BackendInstNotImpl)
    c_inst_notimpl = compile_function(simple_identity, backend="backend_inst_not_impl")
    assert ensure_callable(c_inst_notimpl)(t1) is not None

    # 238-242: Exception with strict=True and False
    class BackendInstError(BaseGenerator):
        """Test class BackendInstError."""

        def __init__(self, graph: EagerValue) -> None:
            """Test function __init__.

            Args:
                graph: Test parameter `graph`.
            """
            ensure_callable(super().__init__)(graph)
            self.mode = "error"

        def compile_aot(self, *args: EagerValue, **kwargs: EagerValue) -> CompiledArtifact:
            """Test function compile_aot.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                CompiledArtifact: Computed output value from `compile_aot`.
            """
            ensure_callable(ensure_eager_tensor(BackendInstanceAot_cls).compile_aot)(self, *args, **kwargs)
            return CompiledArtifact()

    BackendRegistry.register("backend_inst_error", BackendInstError)
    with pytest.raises(CompilationError, match="AOT compilation failed"):
        ensure_callable(compile_function)(simple_identity, backend="backend_inst_error", strict=True)(t1)

    c_inst_err_fallback = compile_function(simple_identity, backend="backend_inst_error", strict=False)
    assert ensure_callable(c_inst_err_fallback)(t1) is t1
    _state.update(ensure_tensor_dict({"BackendInstError": BackendInstError, "BackendInstNotImpl": BackendInstNotImpl, "c_inst_err_fallback": c_inst_err_fallback, "c_inst_notimpl": c_inst_notimpl}))
    _state.update(ensure_tensor_dict({"BackendRegistry": BackendRegistry, "CompilationError": CompilationError, "pytest": pytest}))
    _state.update(ensure_tensor_dict({"simple_identity": simple_identity, "t1": t1}))


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
            """Test function generate.

            Returns:
                EagerValue: Computed output value from `generate`.
            """
            return "some_var = 1"

    BackendRegistry.register("backend_no_entrypoint", BackendNoEntrypoint)
    with pytest.raises(CompilationError, match="No executable entrypoint found"):
        ensure_callable(compile_function)(simple_identity, backend="backend_no_entrypoint", strict=True)(t1)

    class BackendExecFails:
        """Test class BackendExecFails."""

        def __init__(self, graph: EagerValue) -> None:
            """Test function __init__.

            Args:
                graph: Test parameter `graph`.
            """
            pass

        def generate(self) -> EagerValue:
            """Test function generate.

            Returns:
                EagerValue: Computed output value from `generate`.
            """
            return "def apply_model(x):\n    raise RuntimeError('runtime execution fail')"

    BackendRegistry.register("backend_exec_fails", BackendExecFails)
    with pytest.raises(CompilationError, match="Execution of compiled code failed"):
        ensure_callable(compile_function)(simple_identity, backend="backend_exec_fails", strict=True)(t1)
    _state.update(ensure_tensor_dict({"BackendExecFails": BackendExecFails, "BackendNoEntrypoint": BackendNoEntrypoint}))
    _state.update(ensure_tensor_dict({"BackendRegistry": BackendRegistry, "CompilationError": CompilationError, "pytest": pytest}))
    _state.update(ensure_tensor_dict({"simple_identity": simple_identity, "t1": t1}))


def _test_aot_all_remaining_branches_part_7(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 7 of test_aot_all_remaining_branches.

    Args:
        _state: Shared test state dictionary.
    """
    simple_identity, t1 = _state["simple_identity"], _state["t1"]
    c_exec_fallback = compile_function(simple_identity, backend="backend_exec_fails", strict=False)
    assert ensure_callable(c_exec_fallback)(t1) is t1
    _state.update(ensure_tensor_dict({"c_exec_fallback": c_exec_fallback}))


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
