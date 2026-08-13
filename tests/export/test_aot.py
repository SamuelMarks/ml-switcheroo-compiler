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
