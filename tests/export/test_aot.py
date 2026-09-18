import numpy as np

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.export.aot import compile_function


def test_aot():
    def f(x):
        return x + 1

    c = compile_function(f)
    assert c(1) == 2


def test_aot_tensor_caching():
    def f(x, y):
        return x + y

    c = compile_function(f, backend="numpy")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    t2 = Tensor(np.array([2.0]), TensorConfig((1,), DType.Float32, "cpu"))

    # First call will trace and compile
    res = c(t1, t2)
    assert res is not None  # Actually, returning evaluated tensor

    # Second call uses cache
    res2 = c(t1, t2)
    assert res2 is not None


def test_aot_list_output():
    def f(x):
        return [x, x + x]

    c = compile_function(f, backend="numpy")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))

    res = c(t1)
    assert res is not None


def test_aot_backend_missing_generator():
    def f(x):
        return x

    # backend doesn't exist
    c = compile_function(f, backend="invalid_backend")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    # it should just fallback
    res = c(t1)


def test_aot_compile_aot_hasattr():
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    class MockBackend:
        @staticmethod
        def compile_aot(graph, **kwargs):
            return lambda *args, **kw: "mock_aot"

    BackendRegistry.register("mock_aot_backend", MockBackend)

    def f(x):
        return x + x

    c = compile_function(f, backend="mock_aot_backend")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    res = c(t1)
    assert res == "mock_aot"


def test_aot_apply_model():
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    class MockBackendApply:
        def __init__(self, graph):
            self.graph = graph

        def generate(self):
            return "def apply_model(params, *args, **kwargs):\n    return 'apply_model'"

        @classmethod
        def get_generator(cls):
            return cls

    BackendRegistry.register("mock_apply", MockBackendApply)

    def f(x):
        return x

    c = compile_function(f, backend="mock_apply")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    assert c(t1) == "apply_model"


def test_aot_execution_failure():
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    class MockBackendFail:
        def __init__(self, graph):
            self.graph = graph

        def generate(self):
            return "def apply_model(params, *args, **kwargs):\n    raise ValueError('fail')"

        @classmethod
        def get_generator(cls):
            return cls

    BackendRegistry.register("mock_fail", MockBackendFail)

    def f(x):
        return "fallback"

    c = compile_function(f, backend="mock_fail")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    assert c(t1) == "fallback"


def test_aot_caching_success():
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    class MockBackendSuccess:
        @staticmethod
        def compile_aot(graph, **kwargs):
            return lambda *args, **kw: "success_cache"

    BackendRegistry.register("mock_cache", MockBackendSuccess)

    def f(x):
        return x

    c = compile_function(f, backend="mock_cache")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    assert c(t1) == "success_cache"
    assert c(t1) == "success_cache"  # Hits cache


def test_aot_list_output_no_tensors():
    def f(x):
        return [1, 2, 3]

    c = compile_function(f, backend="numpy")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    res = c(t1)
    assert res == [1, 2, 3]


def test_aot_tracing_exception():
    def f(x):
        raise ValueError("Trace error")

    c = compile_function(f, backend="numpy")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    # Should fallback and raise ValueError because fn raises it
    import pytest

    with pytest.raises(ValueError):
        c(t1)


def test_aot_fallback():
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    class MockBackendNoModule:
        def __init__(self, graph):
            self.graph = graph

        def generate(self):
            return "def apply_model(params, *args, **kwargs):\n    return 'no_mod'"

        @classmethod
        def get_generator(cls):
            return cls

    BackendRegistry.register("mock_no_mod", MockBackendNoModule)

    def f(x):
        return x

    c = compile_function(f, backend="mock_no_mod")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    assert c(t1) == "no_mod"


def test_aot_evaluate_wrapper():
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    class MockBackendEvaluate:
        def __init__(self, graph):
            self.graph = graph

        def generate(self):
            return "def evaluate(args):\n    return 'evaluate_success'"

        @classmethod
        def get_generator(cls):
            return cls

        @classmethod
        def get_module(cls):
            return "mock_module"

    BackendRegistry.register("mock_eval", MockBackendEvaluate)

    def f(x):
        return x

    c = compile_function(f, backend="mock_eval")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    assert c(t1) == "evaluate_success"


def test_aot_no_func_wrapper():
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    class MockBackendEmpty:
        def __init__(self, graph):
            self.graph = graph

        def generate(self):
            return "x = 1"

        @classmethod
        def get_generator(cls):
            return cls

    BackendRegistry.register("mock_empty", MockBackendEmpty)

    def f(x):
        return "fallback_empty"

    c = compile_function(f, backend="mock_empty")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    assert c(t1) == "fallback_empty"


def test_aot_custom_dtype_string_prefix() -> None:
    """Test _prepare_proxy_args with an input object whose dtype is a string starting with DType."""

    class CustomArray:
        def __init__(self, dt: str) -> None:
            self.shape = (2, 3)
            self.dtype = dt

    def f(x: object) -> object:
        return x

    c = compile_function(f, backend="numpy")
    res1 = c(CustomArray("DType.float32"))
    assert res1 is not None

    res2 = c(CustomArray("float32"))
    assert res2 is not None

    res3 = c(CustomArray("unknown_custom_dtype"))
    assert res3 is not None


def test_aot_no_func_wrapper_tensor():
    from ml_switcheroo_compiler.backends.registry import BackendRegistry

    class MockBackendEmptyTensor:
        def __init__(self, graph):
            self.graph = graph

        def generate(self):
            return "x = 1"

        @classmethod
        def get_generator(cls):
            return cls

    BackendRegistry.register("mock_empty_tensor", MockBackendEmptyTensor)

    def f(x):
        return x

    c = compile_function(f, backend="mock_empty_tensor")
    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))
    # Since f(x) returns x (a Tensor), graph.outputs will be populated!
    # But generator returns "x = 1", so apply_model and evaluate are missing!
    res = c(t1)
    assert res is t1


def test_aot_all_remaining_branches():
    """Test lines 130-132, 213-217, 221-226, 232-242, 280-282, and 288-293 in export/aot.py."""
    import pytest

    from ml_switcheroo_compiler.backends.registry import BackendRegistry
    from ml_switcheroo_compiler.core.errors import CompilationError
    from ml_switcheroo_compiler.export.aot import _get_namespace

    t1 = Tensor(np.array([1.0]), TensorConfig((1,), DType.Float32, "cpu"))

    def simple_identity(x):
        return x

    # 1. _get_namespace fallback for numpy (lines 130-132)
    class GeneratorNoGetModule:
        pass

    ns = _get_namespace("numpy", GeneratorNoGetModule)
    assert "numpy" in ns

    # 2. static compile_aot returns non-callable or raises Exception (lines 213->219 and 216-217)
    class BackendStaticNonCallable:
        @staticmethod
        def compile_aot(graph, **kwargs):
            return "not_a_callable"

        def __init__(self, graph):
            pass

        def generate(self):
            return "def apply_model(x): return x"

    BackendRegistry.register("backend_static_non_callable", BackendStaticNonCallable)
    c_non_call = compile_function(simple_identity, backend="backend_static_non_callable")
    assert c_non_call(t1) is not None

    class BackendStaticRaises:
        @staticmethod
        def compile_aot(graph, **kwargs):
            raise RuntimeError("static aot error")

        def __init__(self, graph):
            pass

        def generate(self):
            return "def apply_model(x): return x"

    BackendRegistry.register("backend_static_raises", BackendStaticRaises)
    c_raises = compile_function(simple_identity, backend="backend_static_raises")
    assert c_raises(t1) is not None

    # 3. generator_cls(graph) fails with strict=True and strict=False (lines 221-226)
    class BackendInitFails:
        def __init__(self, graph):
            raise RuntimeError("init failure")

    BackendRegistry.register("backend_init_fails", BackendInitFails)
    with pytest.raises(CompilationError, match="Failed to instantiate generator"):
        compile_function(simple_identity, backend="backend_init_fails", strict=True)(t1)

    c_fallback_init = compile_function(simple_identity, backend="backend_init_fails", strict=False)
    assert c_fallback_init(t1) is t1

    # 4. Instance compile_aot hook (lines 232-242)
    class BackendInstanceAot:
        def __init__(self, graph):
            self.mode = "non_callable"

        def compile_aot(self, graph, **kwargs):
            if self.mode == "non_callable":
                return 123
            elif self.mode == "not_implemented":
                raise NotImplementedError()
            else:
                raise RuntimeError("instance aot fail")

        def generate(self):
            return "def apply_model(x): return x"

    BackendRegistry.register("backend_instance_aot", BackendInstanceAot)
    # 232->244: returns non-callable
    c_inst_noncall = compile_function(simple_identity, backend="backend_instance_aot")
    assert c_inst_noncall(t1) is not None

    # 235-236: NotImplementedError
    class BackendInstNotImpl(BackendInstanceAot):
        def __init__(self, graph):
            super().__init__(graph)
            self.mode = "not_implemented"

    BackendRegistry.register("backend_inst_not_impl", BackendInstNotImpl)
    c_inst_notimpl = compile_function(simple_identity, backend="backend_inst_not_impl")
    assert c_inst_notimpl(t1) is not None

    # 238-242: Exception with strict=True and False
    class BackendInstError(BackendInstanceAot):
        def __init__(self, graph):
            super().__init__(graph)
            self.mode = "error"

    BackendRegistry.register("backend_inst_error", BackendInstError)
    with pytest.raises(CompilationError, match="AOT compilation failed"):
        compile_function(simple_identity, backend="backend_inst_error", strict=True)(t1)

    c_inst_err_fallback = compile_function(simple_identity, backend="backend_inst_error", strict=False)
    assert c_inst_err_fallback(t1) is t1

    # 5. Missing entrypoint with strict=True (lines 280-282)
    class BackendNoEntrypoint:
        def __init__(self, graph):
            pass

        def generate(self):
            return "some_var = 1"

    BackendRegistry.register("backend_no_entrypoint", BackendNoEntrypoint)
    with pytest.raises(CompilationError, match="No executable entrypoint found"):
        compile_function(simple_identity, backend="backend_no_entrypoint", strict=True)(t1)

    # 6. Execution of compiled code failed (lines 288-293)
    class BackendExecFails:
        def __init__(self, graph):
            pass

        def generate(self):
            return "def apply_model(x):\n    raise RuntimeError('runtime execution fail')"

    BackendRegistry.register("backend_exec_fails", BackendExecFails)
    with pytest.raises(CompilationError, match="Execution of compiled code failed"):
        compile_function(simple_identity, backend="backend_exec_fails", strict=True)(t1)

    c_exec_fallback = compile_function(simple_identity, backend="backend_exec_fails", strict=False)
    assert c_exec_fallback(t1) is t1
